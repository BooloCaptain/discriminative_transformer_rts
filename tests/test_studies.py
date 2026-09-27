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

from rts import config, studies
from rts.data import populations
from rts.experiment import (
    ROLE_DATASET,
    ROLE_FEATURES,
    ROLE_MODEL,
    ROLE_POPULATION,
    ROLE_SPLIT,
    Binding,
    Environment,
)
from rts.model import selectors


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
        assert selectors.LexicalSelector(**kwargs).name == name


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


def test_no_two_model_elements_share_a_selector_instance():
    """One element per variant means one *instance* per variant.

    A selector records state -- ``XGBoostSelector`` keeps ``importances_`` from its last call
    and caches extra score columns -- so an instance used as both a standalone element and a
    rank-average parent would let one cell's state describe another's, and would fit the same
    model twice for identical scores. The arms build the reused ones through factories; this
    is the invariant that keeps a future edit from quietly sharing one again.
    """
    arms = (
        studies.study_arm(),
        studies.ladder_arm(),
        studies.starved_arm(2),
        studies.starved_arm(5),
        studies.starved_seeds_arm(2, model_seed=1),
        studies.instruction_arm(),
        studies.embed_arm("covered"),
        studies.embed_arm("full"),
        studies.redundancy_arm(),
        studies.bugsinpy_arm(0.05),
    )
    for arm in arms:
        # Every built value is kept alive, so ``is`` is a sound comparison; comparing ``id``
        # would let a collected instance's address be reused and report a false share.
        seen: list[tuple[str, object]] = []
        for element in arm.axis_for(ROLE_MODEL).elements:
            # Building a model element returns its selector and touches no data.
            value = element.build(Binding(env=Environment()))
            for member in (value, *getattr(value, "selectors", ())):
                for owner, other in seen:
                    assert member is not other, (
                        f"{arm.name}: {element.name!r} shares the selector instance already "
                        f"owned by {owner!r}"
                    )
                seen.append((element.name, member))


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


# --- score production -------------------------------------------------------


def test_the_production_arm_is_unmeasured_without_the_gpu_tier(tmp_path):
    """The declaration path, which is the whole point of tiering score production.

    Production is addressable like any other arm: with the GPU tier the cell scores, without it
    the cell is reported unmeasured with the reason. It must not score, read or write anything
    on a CPU-only run -- and in particular it must not touch a cache it was not asked to build.
    """
    from rts.experiment import run

    scratch = tmp_path / "scratch.jsonl"
    report = run(studies.semif_production_arm(scratch), tiers=["cpu"], save=False, verbose=False)

    assert report.cells == []
    assert len(report.unmeasured) == 1
    assert report.unmeasured[0]["requirement"] == "tier:gpu"
    assert not scratch.exists()


def test_the_production_element_names_its_prompt_configuration():
    """Two wordings must not share a score matrix, and the score key is the element name."""
    from pathlib import Path

    default = studies.semif_scoring_model_axis(Path("x.jsonl")).names()[0]
    execution = studies.semif_scoring_model_axis(
        Path("x.jsonl"), instruction="execution"
    ).names()[0]
    mirrored = studies.semif_scoring_model_axis(Path("x.jsonl"), feature_mode="full").names()[0]

    assert len({default, execution, mirrored}) == 3
    assert default.startswith("semif_scored_")


def test_the_production_arm_is_addressable_from_the_cli():
    assert "semif.produce" in studies.ARMS
    assert studies.ARMS["semif.produce"]().models.names()[0].startswith("semif_scored")


# --- the shape of the package -----------------------------------------------


def test_no_studies_submodule_imports_a_driver():
    """The declarations must not import a driver, or a driver importing them closes a cycle.

    ``rts.ladder`` is the live risk: it imports ``rts.studies``, so a submodule importing it back
    would make the import order decide whether the package loads at all.
    """
    import ast
    from pathlib import Path

    import rts.studies as package

    root = Path(package.__file__).parent
    offenders: list[str] = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [f"{node.module or ''}.{a.name}" for a in node.names]
                if node.level == 2 and node.module in {"ladder", "pipeline", "variations", "bugsinpy"}:
                    names.append(f"rts.{node.module}")
            else:
                continue
            offenders.extend(
                f"{path.name}: {name}" for name in names if name in {"rts.ladder", "rts.pipeline"}
            )
    assert offenders == []


def test_every_dataset_an_arm_uses_declares_its_test_unit_and_semantics():
    """Both are declarations a consumer reads, and an empty one is a hole rather than a default.

    ``test_unit`` flattens what a test id denotes, and ``semantics`` is where the flattening is
    qualified -- so a dataset that answered neither would make its own results uninterpretable
    without the reader going to the source.
    """
    from rts.data import datasets
    from rts.data.contract import TestUnit

    corpus = [datasets.marshmallow(), datasets.bugsinpy(datasets.available_bugsinpy_projects()[0])]
    for ds in corpus:
        assert isinstance(ds.test_unit(), TestUnit), ds.name
        notes = ds.semantics()
        assert notes, f"{ds.name} declares no semantics"
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in notes.items())
