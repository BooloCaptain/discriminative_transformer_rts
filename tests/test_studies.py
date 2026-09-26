"""Tests for the study's declarations: the arms, their axes, and the properties they must keep.

These are offline -- they plan cells and read declarations without building a dataset -- except
where a run over the fixture dataset is cheaper than describing one. The arms themselves are
verified against the recorded artifacts by ``scripts/verify_experiment_layer.py``, which is slow
and belongs with the artifacts rather than in the unit suite.

What is pinned here is the contract *between* the arms and the artifacts: that the declared
selector set is the recorded keys, that the two arms keep the contexts a shared score cache
depends on, and that a rung withholds what it says it withholds.
"""

from __future__ import annotations

import json

import pytest

from rts import config, models, populations, studies
from rts.experiment import ROLE_DATASET, ROLE_FEATURES, ROLE_MODEL, ROLE_POPULATION, ROLE_SPLIT


def _recorded(name: str) -> dict:
    return json.loads((config.ARTIFACTS / name).read_text())


# --- the headline arm -------------------------------------------------------


def test_the_study_arm_declares_exactly_the_recorded_selector_keys():
    """A renamed element would silently stop reproducing the artifact, so pin the key set."""
    recorded = _recorded("results_full.json")
    expected = set(recorded["results"]) | set(recorded["ablations"])
    assert set(studies.study_arm().models.names()) == expected


def test_the_shuffle_controls_are_the_recorded_ablation_keys():
    recorded = set(_recorded("results_full.json")["ablations"])
    assert {name for name, _ in studies.ABLATION_MODELS} == recorded
    # And each entry's flags are what produce its name, so the two cannot drift apart.
    for name, kwargs in studies.ABLATION_MODELS:
        assert models.LexicalSelector(**kwargs).name == name


def test_the_population_of_the_headline_arm_is_the_recorded_one():
    recorded = _recorded("results_full.json")
    assert list(studies.study_arm().populations.names()) == [recorded["population"]]


def test_the_headline_arm_reports_the_recorded_budget_grid():
    assert list(studies.study_arm().knobs.budgets) == _recorded("results_full.json")["budgets"]
    assert studies.study_arm().knobs.seed == _recorded("results_full.json")["seed"]


def test_comment_semif_can_be_left_out_without_touching_the_rest():
    arm = studies.study_arm(include_semif=False)
    assert "semif_reranker" not in arm.models.names()
    assert "coverage" in arm.models.names()


# --- the sparse arm --------------------------------------------------------


def test_the_sparse_arm_counts_every_recorded_sparse_key():
    recorded = _recorded("results_full.json")
    arm = studies.sparse_arm()
    for key in recorded["sparse_arm"]:
        name, _, threshold = key.partition("@")
        assert name in arm.models.names()
        assert f"sparse{threshold}" in arm.populations.names()


def test_the_sparse_arm_carries_its_own_budget_set():
    """A different budget grid makes it a different experiment; ``budgets`` is a knob."""
    arm = studies.sparse_arm()
    assert list(arm.knobs.budgets) == list(studies.SPARSE_BUDGETS)
    assert list(studies.study_arm().knobs.budgets) == list(config.DEFAULT_BUDGETS)


def test_the_two_study_arms_keep_the_contexts_a_shared_score_cache_depends_on():
    """Sharing score matrices is only sound if the contexts agree on everything but the rows.

    The key is (dataset element, dataset name, features element, model element, split element,
    split shape, history policy, candidate mode) -- so these three axes must match between the
    arms, and the population axis must be the only one that differs.
    """
    headline = studies.study_arm()
    sparse = studies.sparse_arm()
    for role in (ROLE_DATASET, ROLE_FEATURES, ROLE_SPLIT):
        assert headline.axis_for(role).names() == sparse.axis_for(role).names()
    assert headline.knobs.candidates == sparse.knobs.candidates
    assert headline.history == sparse.history
    assert headline.axis_for(ROLE_POPULATION).names() != sparse.axis_for(ROLE_POPULATION).names()
    # The sparse arm tabulates the headline selectors and not the shuffle controls.
    assert set(sparse.models.names()) < set(headline.models.names())


# --- the ladder arm --------------------------------------------------------


def test_the_ladder_declares_the_rungs_it_renders():
    assert [name for name, _ in studies.RUNGS] == [
        "L0_all",
        "L1_nohistory",
        "L2_nocoverage",
        "L3_notrace",
    ]
    for rung, removed in studies.RUNGS:
        assert rung in studies.ladder_arm().features.names()


def test_a_rung_withholds_a_family_or_raises():
    """Withholding is strict, which is the fix for the ablation bug the refactor exposed."""
    assert not studies.rung_block(()).suppressed
    with pytest.raises(KeyError):
        studies.rung_block(("traceabilty",))


def test_the_ladder_does_not_sweep_populations_the_recorded_artifact_lacks():
    recorded = _recorded("ladder.json")["mutmut"]
    assert set(studies.ladder_arm().populations.names()) == set(
        recorded["populations"]
    ) | set(recorded["populations_unmeasured"])
    assert list(studies.ladder_arm().knobs.budgets) == [
        float(k) for k in recorded["rungs"]["L0_all"]["semif_margin"]
    ]


# --- a parameterised population --------------------------------------------


def test_a_starved_threshold_is_a_population_named_for_its_threshold():
    population = populations.starved(5)
    assert population.name == "starved5"
    assert "5" in population.note
    assert populations.starved(2).name != population.name
    # The caches are named for the threshold too, so the two line up.
    assert studies.starved_cache(5).name == "semif_scores_starved5_full.jsonl"
    assert studies.starved_cache(2).name == "semif_scores_starved2_full.jsonl"


def test_the_variation_arms_declare_the_recorded_selector_keys():
    """A renamed element would silently stop reproducing a section, so pin the key sets."""
    recorded = _recorded("variations.json")
    for section, names in (
        ("full_starved", studies.starved_arm(2).models.names()),
        ("full_starved5", studies.starved_arm(5).models.names()),
        ("p2", studies.instruction_arm().models.names()),
        ("p5", studies.redundancy_arm().models.names()),
    ):
        assert list(names) == list(recorded[section]["results"]), section


def test_the_variation_arms_pair_at_every_budget_of_their_grid():
    """A lever that helps only at a low budget is a different finding, so each grid point is
    paired -- and a comparison is defined by one probe budget."""
    arm = studies.starved_arm(2)
    references = {c.reference for c in arm.comparisons}
    assert references == set(_recorded("variations.json")["full_starved"]["comparisons"]["references"])
    assert {c.probe_budget for c in arm.comparisons} == set(studies.VARIATION_BUDGETS)


def test_a_sparse_threshold_is_a_population_named_for_its_threshold():
    population = populations.low_pair_recurrence(80)
    assert population.name == "sparse80"
    assert "80" in population.note
    assert "coverage" in {requirement.value for requirement in population.requires()}
    # Two thresholds are two populations, and a sweep must not conflate them.
    assert populations.low_pair_recurrence(80).name != populations.low_pair_recurrence(160).name
