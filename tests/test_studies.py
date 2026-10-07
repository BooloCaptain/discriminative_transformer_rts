"""Tests for the study's declarations: the conditions, their factors, and the properties they must keep.

These are offline -- they plan design points and read declarations without building a dataset -- except
where a run over the fixture dataset is cheaper than describing one. The conditions themselves are
verified against the recorded artifacts by ``scripts/verify_experiment_layer.py``, which is slow
and belongs with the artifacts rather than in the unit suite.

What is pinned here is the contract *between* the conditions and the artifacts: that the declared
ranker set is the recorded keys, that the two conditions keep the contexts a shared score cache
depends on, and that a rung withholds what it says it withholds.
"""

from __future__ import annotations

import json

import pytest

from rts import config, studies
from rts.data import subsets
from rts.experiment import (
    FACTOR_DATASET,
    FACTOR_FEATURES,
    FACTOR_MODEL,
    FACTOR_SPLIT,
    FACTOR_SUBSET,
    Binding,
    Environment,
)
from rts.model import rankers


def _recorded(name: str) -> dict:
    return json.loads((config.ARTIFACTS / name).read_text())


# --- the headline condition -------------------------------------------------------


def test_the_study_arm_declares_exactly_the_recorded_selector_keys():
    """A renamed level would silently stop reproducing the artifact, so pin the key set."""
    recorded = _recorded("results_full.json")
    expected = set(recorded["results"]) | set(recorded["ablations"])
    assert set(studies.study_condition().models.names()) == expected


def test_the_shuffle_controls_are_the_recorded_ablation_keys():
    recorded = set(_recorded("results_full.json")["ablations"])
    assert {name for name, _ in studies.ABLATION_MODELS} == recorded
    # And each entry's flags are what produce its name, so the two cannot drift apart.
    for name, kwargs in studies.ABLATION_MODELS:
        assert rankers.LexicalRanker(**kwargs).name == name


def test_the_population_of_the_headline_arm_is_the_recorded_one():
    recorded = _recorded("results_full.json")
    assert list(studies.study_condition().subsets.names()) == [recorded["subset"]]


def test_the_headline_arm_reports_the_recorded_budget_grid():
    assert list(studies.study_condition().controls.budgets) == _recorded("results_full.json")["budgets"]
    assert studies.study_condition().controls.seed == _recorded("results_full.json")["seed"]


def test_comment_semif_can_be_left_out_without_touching_the_rest():
    condition = studies.study_condition(include_semif=False)
    assert "semif_reranker" not in condition.models.names()
    assert "coverage" in condition.models.names()


# --- the low-co-occurrence condition --------------------------------------------------------


def test_the_low_cooccurrence_condition_counts_every_recorded_key():
    recorded = _recorded("results_full.json")
    condition = studies.low_cooccurrence_condition()
    for key in recorded["low_cooccurrence_condition"]:
        name, _, threshold = key.partition("@")
        assert name in condition.models.names()
        assert f"low_cooccurrence{threshold}" in condition.subsets.names()


def test_the_low_cooccurrence_condition_carries_its_own_budget_set():
    """A different budget grid makes it a different experiment; ``budgets`` is a control."""
    condition = studies.low_cooccurrence_condition()
    assert list(condition.controls.budgets) == list(studies.LOW_COOCCURRENCE_BUDGETS)
    assert list(studies.study_condition().controls.budgets) == list(config.DEFAULT_BUDGETS)


def test_the_two_study_arms_keep_the_contexts_a_shared_score_cache_depends_on():
    """Sharing score matrices is only sound if the contexts agree on everything but the rows.

    The key is (dataset level, dataset name, features level, model level, split level,
    split shape, history policy, candidate mode) -- so these three factors must match between the
    conditions, and the subset axis must be the only one that differs.
    """
    headline = studies.study_condition()
    low_cooccurrence = studies.low_cooccurrence_condition()
    for role in (FACTOR_DATASET, FACTOR_FEATURES, FACTOR_SPLIT):
        assert headline.factor_for(role).names() == low_cooccurrence.factor_for(role).names()
    assert headline.controls.candidate_policy == low_cooccurrence.controls.candidate_policy
    assert headline.temporal == low_cooccurrence.temporal
    assert headline.factor_for(FACTOR_SUBSET).names() != low_cooccurrence.factor_for(FACTOR_SUBSET).names()
    # The low-co-occurrence condition tabulates the headline rankers and not the shuffle controls.
    assert set(low_cooccurrence.models.names()) < set(headline.models.names())


# --- the ladder condition --------------------------------------------------------


def test_the_ladder_declares_the_rungs_it_renders():
    assert [name for name, _ in studies.RUNGS] == [
        "L0_all",
        "L1_notemporal",
        "L2_nocoverage",
        "L3_noproximity",
    ]
    for rung, _removed in studies.RUNGS:
        assert rung in studies.ladder_condition().features.names()


def test_a_rung_withholds_a_family_or_raises():
    """Withholding is strict, which is the fix for the ablation bug the refactor exposed."""
    assert not studies.rung_block(()).suppressed
    with pytest.raises(KeyError):
        studies.rung_block(("traceabilty",))


def test_the_ladder_does_not_sweep_populations_the_recorded_artifact_lacks():
    recorded = _recorded("ladder.json")["mutmut"]
    assert set(studies.ladder_condition().subsets.names()) == set(
        recorded["subsets"]
    ) | set(recorded["subsets_undefined"])
    assert list(studies.ladder_condition().controls.budgets) == [
        float(k) for k in recorded["rungs"]["L0_all"]["semif_margin"]
    ]


# --- a parameterised subset --------------------------------------------


def test_a_cold_start_threshold_is_named_for_its_threshold():
    subset = subsets.cold_start(5)
    assert subset.name == "cold_start5"
    assert "5" in subset.note
    assert subsets.cold_start(2).name != subset.name
    # The caches are named for the threshold too, so the two line up.
    assert studies.cold_start_cache(5).name == "semif_scores_cold_start5_full.jsonl"
    assert studies.cold_start_cache(2).name == "semif_scores_cold_start2_full.jsonl"


def test_no_two_model_elements_share_a_selector_instance():
    """One level per variant means one *instance* per variant.

    A ranker records state -- ``XGBoostRanker`` keeps ``importances_`` from its last call
    and caches extra score columns -- so an instance used as both a standalone level and a
    rank-average parent would let one design point's state describe another's, and would fit the same
    model twice for identical scores. The conditions build the reused ones through factories; this
    is the invariant that keeps a future edit from quietly sharing one again.
    """
    conditions = (
        studies.study_condition(),
        studies.ladder_condition(),
        studies.cold_start_condition(2),
        studies.cold_start_condition(5),
        studies.cold_start_seeds_condition(2, model_seed=1),
        studies.instruction_condition(),
        studies.embedding_condition("coverage_restricted"),
        studies.embedding_condition("full"),
        studies.redundancy_condition(),
        studies.bugsinpy_condition(0.05),
    )
    for condition in conditions:
        # Every built value is kept alive, so ``is`` is a sound contrast; comparing ``id``
        # would let a collected instance's address be reused and report a false share.
        seen: list[tuple[str, object]] = []
        for level in condition.factor_for(FACTOR_MODEL).levels:
            # Building a model level returns its ranker and touches no data.
            value = level.build(Binding(env=Environment()))
            for member in (value, *getattr(value, "rankers", ())):
                for owner, other in seen:
                    assert member is not other, (
                        f"{condition.name}: {level.name!r} shares the ranker instance already "
                        f"owned by {owner!r}"
                    )
                seen.append((level.name, member))


def test_the_variation_arms_declare_the_recorded_selector_keys():
    """A renamed level would silently stop reproducing a section, so pin the key sets."""
    recorded = _recorded("variations.json")
    for section, names in (
        ("cold_start", studies.cold_start_condition(2).models.names()),
        ("cold_start5", studies.cold_start_condition(5).models.names()),
        ("p2", studies.instruction_condition().models.names()),
        ("p5", studies.redundancy_condition().models.names()),
    ):
        assert list(names) == list(recorded[section]["results"]), section


def test_the_variation_arms_pair_at_every_budget_of_their_grid():
    """A lever that helps only at a low budget is a different finding, so each grid point is
    paired -- and a contrast is defined by one probe budget."""
    condition = studies.cold_start_condition(2)
    references = {c.reference for c in condition.contrasts}
    assert references == set(_recorded("variations.json")["cold_start"]["contrasts"]["references"])
    assert {c.probe_budget for c in condition.contrasts} == set(studies.VARIATION_BUDGETS)


def test_a_low_cooccurrence_threshold_is_named_for_its_threshold():
    subset = subsets.low_cooccurrence(80)
    assert subset.name == "low_cooccurrence80"
    assert "80" in subset.note
    assert "coverage" in {requirement.value for requirement in subset.requires()}
    # Two thresholds are two subsets, and a sweep must not conflate them.
    assert subsets.low_cooccurrence(80).name != subsets.low_cooccurrence(160).name


# --- score production -------------------------------------------------------


def test_the_production_arm_is_unmeasured_without_the_gpu_tier(tmp_path):
    """The declaration path, which is the whole point of tiering score production.

    Production is addressable like any other condition: with the GPU tier the design point scores, without it
    the design point is reported undefined with the reason. It must not score, read or write anything
    on a CPU-only run -- and in particular it must not touch a cache it was not asked to build.
    """
    from rts.experiment import run

    scratch = tmp_path / "scratch.jsonl"
    report = run(studies.semif_production_condition(scratch), tiers=["cpu"], save=False, verbose=False)

    assert report.design_points == []
    assert len(report.undefined) == 1
    assert report.undefined[0]["requirement"] == "tier:gpu"
    assert not scratch.exists()


def test_the_production_element_names_its_prompt_configuration():
    """Two wordings must not share a score matrix, and the score key is the level name."""
    from pathlib import Path

    default = studies.semif_scoring_model_factor(Path("x.jsonl")).names()[0]
    execution = studies.semif_scoring_model_factor(
        Path("x.jsonl"), instruction="execution"
    ).names()[0]
    mirrored = studies.semif_scoring_model_factor(Path("x.jsonl"), feature_mode="full").names()[0]

    assert len({default, execution, mirrored}) == 3
    assert default.startswith("semif_scored_")


def test_the_production_arm_is_addressable_from_the_cli():
    assert "semif.produce" in studies.CONDITIONS
    assert studies.CONDITIONS["semif.produce"]().models.names()[0].startswith("semif_scored")


# --- the shape of the package -----------------------------------------------


def test_no_studies_submodule_imports_a_driver():
    """The declarations must not import a renderer, or a renderer importing them closes a
    cycle.

    ``rts.render.ladder`` is the live risk: it imports ``rts.studies``, so a submodule
    importing it back would make the import order decide whether the package loads at all.
    ``bundles`` is the one driver still outside ``rts.render``, and ``factors.py`` does import
    it -- for its rung definitions, which is exactly the knot its migration unties, so this
    check is scoped to the renderers rather than to "anything driver-shaped".
    """
    import ast
    from pathlib import Path

    import rts.studies as package

    root = Path(package.__file__).parent
    offenders: list[str] = []
    for path in sorted(root.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level == 2:
                    base = f"rts.{node.module}" if node.module else "rts"
                elif node.level == 1:
                    base = f"rts.studies.{node.module}" if node.module else "rts.studies"
                else:
                    base = node.module or ""
                imported = [base, *(f"{base}.{alias.name}" for alias in node.names)]
            else:
                continue
            offenders.extend(
                f"{path.name}: {name}"
                for name in imported
                if name == "rts.render" or name.startswith("rts.render.")
            )
    assert offenders == []


def test_every_dataset_an_arm_uses_declares_its_test_unit_and_semantics():
    """Both are declarations a consumer reads, and an empty one is a hole rather than a default.

    ``test_granularity`` flattens what a test id denotes, and ``annotations`` is where the flattening is
    qualified -- so a dataset that answered neither would make its own results uninterpretable
    without the reader going to the source.
    """
    from rts.data import datasets
    from rts.data.contract import Granularity

    corpus = [datasets.marshmallow(), datasets.bugsinpy(datasets.available_bugsinpy_projects()[0])]
    for ds in corpus:
        assert isinstance(ds.test_granularity(), Granularity), ds.name
        notes = ds.annotations()
        assert notes, f"{ds.name} declares no annotations"
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in notes.items())
