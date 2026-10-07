"""Tests for the experiment layer: factors, design points, availability, comparability, provenance.

These use the fixture-backed dataset, so they need neither a checkout nor a real run -- which
is the property ``docs/refactor.md`` §7 asks of the contract, and the experiment layer inherits it.

The marshmallow and ladder conditions are verified separately by re-running them and diffing against
the recorded artifacts (``scripts/verify_experiment_layer.py``), because that check is slow and
belongs with the artifacts rather than in the unit suite. What is pinned here is the layer's own
behaviour: that a design point is one point in the product, that an unavailable thing is reported rather
than dropped, that temporal features are derived rather than configured, and that a paired contrast
across different subsets is refused.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rts import config, features
from rts.data import splits, subsets
from rts.data.contract import Ordering, Undefined
from rts.experiment import (
    FACTOR_DATASET,
    FACTOR_FEATURES,
    FACTOR_MODEL,
    FACTOR_SPLIT,
    FACTOR_SUBSET,
    Binding,
    Contrast,
    Controls,
    Environment,
    Experiment,
    Factor,
    Level,
    constant,
    run,
    unavailable,
)
from rts.model import rankers
from tests.stub_dataset import StubDataset

PROBE = 0.5


def half_split():
    return Level(
        "half",
        lambda b: splits.make_split(b.dataset, train_fraction=0.5, seed=b.controls.seed),
    )


def stub_experiment(**overrides) -> Experiment:
    fields = dict(
        name="stub",
        datasets=Factor(FACTOR_DATASET, (constant("stub", StubDataset()),)),
        features=Factor(FACTOR_FEATURES, (constant("structured", features.STRUCTURED),)),
        models=Factor(FACTOR_MODEL, (constant("coverage", rankers.CoverageRanker()),)),
        subsets=Factor(
            FACTOR_SUBSET, (constant("detectable", subsets.DETECTABLE),)
        ),
        splits=Factor(FACTOR_SPLIT, (half_split(),)),
        controls=Controls(budgets=(PROBE,), n_bootstrap=20),
    )
    fields.update(overrides)
    return Experiment(**fields)


# --- factors and design points ---------------------------------------------------------


def test_a_cell_is_one_point_in_the_product_and_its_key_names_the_factors():
    experiment = stub_experiment(
        models=Factor(
            FACTOR_MODEL,
            (
                constant("a", rankers.CoverageRanker()),
                constant("b", rankers.FailureRateRanker()),
            ),
        )
    )
    design_points = experiment.design_points()
    assert [c.name(FACTOR_MODEL) for c in design_points] == ["a", "b"]
    assert design_points[0].key == (
        "dataset=stub|features=structured|model=a|subset=detectable|split=half"
    )
    assert design_points[0].factors_dict()[FACTOR_DATASET] == "stub"


def test_an_axis_rejects_duplicate_element_names():
    with pytest.raises(ValueError, match="duplicate level names"):
        Factor(FACTOR_MODEL, (constant("same", object()), constant("same", object())))


def test_an_element_with_an_unknown_tier_is_rejected_when_planning():
    experiment = stub_experiment(
        models=Factor(FACTOR_MODEL, (constant("a", rankers.CoverageRanker(), tier="quantum"),))
    )
    with pytest.raises(ValueError, match="not in tier_order"):
        experiment.design_points()


def test_a_shared_option_that_is_inapplicable_leaves_an_unmeasured_cell():
    """``Factor.map`` is what a dimension-level option is, and inapplicability is a finding."""
    axis = Factor(
        FACTOR_MODEL,
        (constant("coverage", rankers.CoverageRanker()), constant("rate", rankers.FailureRateRanker())),
    )

    def exclude_temporal(level):
        if level.name != "coverage":
            return Undefined("withheld", "this ranker reads no columns")
        return level

    mapped = axis.map(exclude_temporal)
    assert mapped.names() == ("coverage", "rate")
    report = run(stub_experiment(models=mapped), save=False, verbose=False)
    assert len(report.design_points) == 1
    assert len(report.undefined) == 1
    assert report.undefined[0]["factors"][FACTOR_MODEL] == "rate"
    assert report.undefined[0]["requirement"] == "withheld"


# --- availability is reported, never dropped --------------------------------


def test_a_missing_artifact_makes_the_cell_unmeasured_rather_than_raising():
    class Cached(rankers.Ranker):
        name = "cached"

        def requirements(self):
            return ("artifact:/nonexistent/nope.jsonl",)

        def scores(self, ctx):  # pragma: no cover - must not be reached
            raise AssertionError("a design_point whose requirement is unmet must not be measured")

    report = run(
        stub_experiment(models=Factor(FACTOR_MODEL, (Level("cached", lambda b: Cached()),))),
        save=False,
        verbose=False,
    )
    assert report.design_points == []
    assert len(report.undefined) == 1
    assert report.undefined[0]["requirement"] == "artifact:/nonexistent/nope.jsonl"
    assert "does not exist" in report.undefined[0]["note"]


def test_an_unrecognised_requirement_spelling_raises():
    class Typo(rankers.Ranker):
        name = "typo"

        def requirements(self):
            return ("coverge",)

        def scores(self, ctx):  # pragma: no cover - must not be reached
            raise AssertionError

    with pytest.raises(ValueError, match="unknown requirement"):
        run(
            stub_experiment(models=Factor(FACTOR_MODEL, (Level("typo", lambda b: Typo()),))),
            save=False,
            verbose=False,
        )


def test_an_unavailable_population_is_unmeasured_not_an_empty_average():
    """An average over no rows and a subset that cannot exist are different claims."""
    poisoned = unavailable("gone", Undefined("coverage", "no coverage on this dataset"))
    report = run(
        stub_experiment(subsets=Factor(FACTOR_SUBSET, (poisoned,))),
        save=False,
        verbose=False,
    )
    assert report.design_points == []
    assert report.undefined[0]["requirement"] == "coverage"
    assert report.undefined[0]["measured"] is False


def test_applicability_may_depend_on_another_role():
    """A variant can be meaningless in combination with another role's level, not alone."""

    def only_with_coverage_population(binding: Binding) -> Undefined | None:
        if binding.factors.get(FACTOR_SUBSET) == "detectable":
            return None
        return Undefined("artifact:partial", "the cache does not cover this subset")

    report = run(
        stub_experiment(
            models=Factor(
                FACTOR_MODEL,
                (
                    Level(
                        "partial",
                        lambda b: rankers.CoverageRanker(),
                        applies=only_with_coverage_population,
                    ),
                ),
            ),
            subsets=Factor(
                FACTOR_SUBSET,
                (
                    constant("detectable", subsets.DETECTABLE),
                    constant("no_prior_failure", subsets.NO_PRIOR_FAILURE),
                ),
            ),
        ),
        save=False,
        verbose=False,
    )
    assert len(report.design_points) == 1
    assert len(report.undefined) == 1
    assert report.undefined[0]["requirement"] == "artifact:partial"


# --- temporal features are derived, not configured -------------------------------------


def test_history_is_off_when_the_split_shuffles_an_observed_dataset():
    """The rule that ties a split to an ordering, reaching the layer rather than the caller."""
    experiment = stub_experiment(
        datasets=Factor(FACTOR_DATASET, (constant("natural", StubDataset(ordering=Ordering.NATURAL)),)),
        splits=Factor(
            FACTOR_SPLIT,
            (
                Level(
                    "shuffled",
                    lambda b: splits.make_split(b.dataset, train_fraction=0.5, shuffle=True),
                ),
            ),
        ),
        temporal=None,
    )
    report = run(experiment, save=False, verbose=False)
    history = set(features.STRUCTURED.family("temporal"))
    withheld = {u["column"] for u in report.design_points[0].features["undefined"]}
    assert history <= withheld


def test_history_is_on_for_an_observed_dataset_by_default():
    experiment = stub_experiment(
        datasets=Factor(FACTOR_DATASET, (constant("natural", StubDataset(ordering=Ordering.NATURAL)),)),
        temporal=None,
    )
    report = run(experiment, save=False, verbose=False)
    history = set(features.STRUCTURED.family("temporal"))
    withheld = {u["column"] for u in report.design_points[0].features["undefined"]}
    assert not (history & withheld)


def test_enabling_temporal_features_on_a_synthetic_order_warns_rather_than_differing_silently():
    report = run(stub_experiment(temporal=True), save=False, verbose=False)
    codes = {w["code"] for w in report.design_points[0].diagnostics}
    assert "feature.temporal_on_synthetic_order" in codes


# --- comparability ----------------------------------------------------------


def test_a_comparison_over_a_row_changing_role_is_refused():
    for role in (FACTOR_DATASET, FACTOR_SUBSET, FACTOR_SPLIT):
        with pytest.raises(ValueError, match="would pair different subsets"):
            Contrast(role, "whatever", PROBE)


def test_a_comparison_reference_must_exist_on_its_factor():
    with pytest.raises(KeyError, match="contrast reference"):
        stub_experiment(contrasts=(Contrast(FACTOR_MODEL, "absent", PROBE),))


def test_paired_deltas_are_computed_over_the_populations_rows():
    experiment = stub_experiment(
        models=Factor(
            FACTOR_MODEL,
            (
                constant("coverage", rankers.CoverageRanker()),
                constant("rate", rankers.FailureRateRanker()),
            ),
        ),
        contrasts=(Contrast(FACTOR_MODEL, "coverage", PROBE),),
    )
    report = run(experiment, save=False, verbose=False)
    record = next(r for r in report.contrasts if r["design_point"] == "rate")
    assert record["measured"] is True
    # The paired set is the subset's rows, which for this dataset is a single change.
    assert record["n"] == report.find(features="structured", model="coverage").n_rows


# --- cost, provenance, and reuse -------------------------------------------


def test_a_tier_filter_reports_the_cells_it_did_not_spend():
    experiment = stub_experiment(
        models=Factor(
            FACTOR_MODEL,
            (
                constant("cpu_model", rankers.CoverageRanker()),
                constant("gpu_model", rankers.RandomRanker(), tier="gpu"),
            ),
        )
    )
    report = run(experiment, tiers=("cpu",), save=False, verbose=False)
    assert [c.design_point.name(FACTOR_MODEL) for c in report.design_points] == ["cpu_model"]
    assert len(report.undefined) == 1
    assert report.undefined[0]["requirement"] == "tier:gpu"


def test_scores_are_reused_across_populations_that_share_a_context():
    """A subset restricts which rows a metric averages; it does not change the scores."""
    calls: list[str] = []

    class Counting(rankers.Ranker):
        name = "counting"

        def scores(self, ctx):
            calls.append("scored")
            return np.zeros((ctx.ds.n_changes, ctx.ds.n_tests), dtype=np.float32)

    report = run(
        stub_experiment(
            models=Factor(FACTOR_MODEL, (Level("counting", lambda b: Counting()),)),
            subsets=Factor(
                FACTOR_SUBSET,
                (
                    constant("detectable", subsets.DETECTABLE),
                    constant("no_prior_failure", subsets.NO_PRIOR_FAILURE),
                ),
            ),
        ),
        save=False,
        verbose=False,
    )
    assert len(report.design_points) == 2
    assert calls == ["scored"]
    assert sum(1 for c in report.design_points if c.seconds > 0) == 1


def test_an_element_is_materialised_once_per_run_and_shared_across_cells():
    built: list[str] = []

    def counting_dataset(binding: Binding) -> StubDataset:
        built.append("dataset")
        return StubDataset()

    experiment = stub_experiment(
        datasets=Factor(FACTOR_DATASET, (Level("counted", counting_dataset),)),
        models=Factor(
            FACTOR_MODEL,
            tuple(constant(f"m{i}", rankers.CoverageRanker()) for i in range(4)),
        ),
    )
    report = run(experiment, save=False, verbose=False)
    assert len(report.design_points) == 4
    assert built == ["dataset"]


def test_the_report_round_trips_through_its_artifact(tmp_path):
    report = run(stub_experiment(), save=False, verbose=False)
    path = report.save(tmp_path)
    payload = json.loads(path.read_text())
    assert payload["experiment"] == "stub"
    assert payload["n_cells"] == len(report.design_points) + len(report.undefined)
    assert payload["design_points"][0]["measured"] is True
    assert payload["environment"]["controls"]["candidate_policy"] == "full"
    assert list(payload["factors"]) == list(
        (FACTOR_DATASET, FACTOR_FEATURES, FACTOR_MODEL, FACTOR_SUBSET, FACTOR_SPLIT)
    )


def test_a_run_knob_override_is_recorded():
    report = run(
        stub_experiment(),
        controls=Controls(budgets=(0.25,), n_bootstrap=7, seed=config.SEED + 1),
        save=False,
        verbose=False,
    )
    assert report.environment.controls.seed == config.SEED + 1
    assert [r["budget"] for r in report.design_points[0].results] == [0.25]


def test_the_run_seed_reaches_the_model_not_just_the_split():
    """A control that stops at the layer is a control that lies about what it changed."""

    def recall_at(seed: int) -> float:
        report = run(
            stub_experiment(
                models=Factor(FACTOR_MODEL, (constant("random", rankers.RandomRanker()),)),
                controls=Controls(budgets=(PROBE,), n_bootstrap=5, seed=seed),
            ),
            save=False,
            verbose=False,
        )
        return report.design_points[0].results[0]["recall"]

    assert recall_at(config.SEED) != recall_at(config.SEED + 1)


def test_a_caller_can_share_score_matrices_between_runs():
    """What lets one study be two runs -- different budget sets -- without retraining."""
    experiment = stub_experiment()
    shared: dict = {}

    first = run(experiment, scores=shared, save=False, verbose=False)
    assert first.design_points[0].seconds > 0
    assert len(shared) == 1

    second = run(
        experiment,
        controls=Controls(budgets=(0.25,), n_bootstrap=5, seed=experiment.controls.seed),
        scores=shared,
        save=False,
        verbose=False,
    )
    assert second.design_points[0].seconds == 0
    assert [r["budget"] for r in second.design_points[0].results] == [0.25]


def test_a_model_seed_is_not_served_by_another_seed_s_score_cache():
    """The score key must carry the model seed.

    Without it, four refits of one level reuse the first fit and report four identical
    deltas -- a plausible number from a contrast that never happened, which is the failure
    mode ``docs/refactor.md`` §14 records for a name-based ablation.
    """
    experiment = stub_experiment(
        models=Factor(FACTOR_MODEL, (constant("random", rankers.RandomRanker()),))
    )
    shared: dict = {}

    def run_with(model_seed: int):
        return run(
            experiment,
            controls=Controls(
                budgets=(PROBE,), n_bootstrap=5, seed=config.SEED, model_seed=model_seed
            ),
            scores=shared,
            save=False,
            verbose=False,
        )

    first = run_with(1)
    second = run_with(2)
    assert len(shared) == 2, "a different model seed must not reuse another seed's scores"
    a, b = (shared[key] for key in shared)
    assert not np.array_equal(a, b)
    # And the data underneath is untouched: the split is seeded by the run, not the model.
    assert first.design_points[0].split == second.design_points[0].split


def test_the_environment_is_a_value_a_caller_can_inject_through():
    report = run(
        stub_experiment(),
        shared={"anything": 1},
        save=False,
        verbose=False,
    )
    assert report.environment.shared == {"anything": 1}
    assert isinstance(report.environment, Environment)


# --- the model half of the declaration -------------------------------------


def test_a_cached_score_selector_declares_its_cache_and_needs_the_right_loader(tmp_path):
    """A cache is not always a scored-pair log, and reading it with the wrong loader is an error."""
    path = tmp_path / "scores.npy"
    matrix = np.arange(12, dtype=np.float32).reshape(4, 3)
    np.save(path, matrix)

    ds = StubDataset()
    ctx = rankers.Context(
        ds=ds,
        features=features.structured(ds, temporal=True),
        split=splits.make_split(ds, train_fraction=0.5),
        bm25=np.zeros((ds.n_changes, ds.n_tests), dtype=np.float32),
    )

    binary = rankers.CachedScores("embed", path, loader=rankers.load_matrix)
    assert binary.requirements() == (f"artifact:{path}",)
    assert np.array_equal(binary.scores(ctx), matrix)

    # The default loader reads scored pairs, so a numpy file is a decode error rather than a
    # silently wrong matrix -- which is how the embedding condition first failed.
    with pytest.raises(UnicodeDecodeError):
        rankers.CachedScores("embed", path).scores(ctx)


def test_a_rank_average_is_fitted_free_and_declares_its_parents_caches():
    from rts import config as cfg

    parent = rankers.CachedScores("semif_textonly", cfg.SEMIF_SCORES_FILE)
    combined = rankers.RankAverageRanker(
        "rankaverage_xgb_semif", (rankers.LexicalRanker(), parent), candidate_policy="coverage_restricted"
    )
    assert combined.requirements() == (f"artifact:{cfg.SEMIF_SCORES_FILE}",)
    assert combined.candidate_policy == "coverage_restricted"
    with pytest.raises(ValueError, match="two or more parents"):
        rankers.RankAverageRanker("solo", (rankers.LexicalRanker(),))


def test_artifact_backed_selectors_declare_what_they_read():
    assert rankers.CoverageRanker().requirements() == ()
    assert rankers.RandomRanker().requirements() == ()

    semif = rankers.SemIfRanker()
    assert semif.requirements() == (f"artifact:{config.SEMIF_SCORES_FILE}",)

    extra = rankers.XGBoostRanker(extra_score_files={"semif": config.SEMIF_SCORES_FILE})
    assert extra.requirements() == (f"artifact:{config.SEMIF_SCORES_FILE}",)


def test_the_shuffle_controls_build_their_own_bm25_and_name_themselves():
    """The change-shuffle ablation is a ranker, so its name is the recorded artifact's key."""
    ds = StubDataset()
    matrix = features.structured(ds, temporal=True)
    bm25 = np.zeros((ds.n_changes, ds.n_tests), dtype=np.float32)
    ctx = rankers.Context(
        ds=ds,
        features=matrix,
        split=splits.make_split(ds, train_fraction=0.5),
        bm25=bm25,
        seed=config.SEED,
    )

    canonical = rankers.LexicalRanker()
    assert canonical.name == "bm25_lexical"
    # No flags means the context's matrix, untouched -- the ablation must not perturb the condition it
    # is a control for.
    assert canonical.scores(ctx) is ctx.bm25

    shuffled = rankers.LexicalRanker(shuffle_changes=True)
    assert shuffled.name == "bm25_change_shuffled"
    got = shuffled.scores(ctx)
    assert got.shape == (ds.n_changes, ds.n_tests)
    assert not np.array_equal(got, ctx.bm25)

    assert rankers.LexicalRanker(shuffle_tests=True).name == "bm25_test_shuffled"
    assert (
        rankers.LexicalRanker(shuffle_changes=True, shuffle_tests=True).name
        == "bm25_both_shuffled"
    )


# --- reading a report back --------------------------------------------------
#
# The report is the only input a renderer should need: it records the dataset's own
# declaration, its description under the split the run used, and the two subset sizes,
# so a renderer that describes the data cannot accidentally describe a different dataset.


def test_a_report_records_the_dataset_facts_a_renderer_needs():
    report = run(stub_experiment(), save=False, verbose=False)

    described = report.describe()
    assert described["changes"] == StubDataset().n_changes
    assert described["tests"] == len(StubDataset().test_suite)
    # The tail is c2 (a fault) and c3 (no killing test).
    assert described["held_out_faults"] == 1

    # The declaration is the dataset's own, recorded rather than re-derived from the type.
    assert report.metadata()["name"] == "stub"

    # StubDataset declares coverage, so the pair-recurrence statistic is defined.
    assert report.recurrence()["killing_pair_count_median"] >= 0.0


def test_a_report_records_the_population_size_before_and_after_the_fault_filter():
    """``changes`` is the subset in the window; ``faults`` is what was averaged over."""
    report = run(stub_experiment(), save=False, verbose=False)
    assert report.subset_size("detectable") == (1, 1)


def test_a_population_size_names_the_dataset_it_belongs_to():
    """A subset name is not unique across datasets, so the size is keyed by the pair.

    Reporting the first dataset's counts for the second is indistinguishable from a correct
    answer, which is why an unnamed lookup over two datasets refuses rather than guesses.
    """
    report = run(
        stub_experiment(
            datasets=Factor(
                FACTOR_DATASET,
                (constant("one", StubDataset()), constant("two", StubDataset())),
            )
        ),
        save=False,
        verbose=False,
    )
    assert set(report.subset_sizes) == {
        "one|detectable",
        "two|detectable",
    }
    assert report.subset_size("detectable", "one") == (1, 1)
    assert report.subset_size("detectable", "two") == (1, 1)
    with pytest.raises(KeyError, match="several datasets"):
        report.subset_size("detectable")
    with pytest.raises(KeyError, match="no subset"):
        report.subset_size("detectable", "absent")
    # The record stays JSON-serialisable: the key is a string pair, not a tuple.
    json.dumps(report.to_dict())


def test_a_dataset_without_coverage_records_no_recurrence():
    """The statistic is undefined without coverage, which is a fact rather than a failure."""
    ds = StubDataset(coverage=False)
    experiment = stub_experiment(
        datasets=Factor(FACTOR_DATASET, (constant("stub", ds),))
    )
    report = run(experiment, save=False, verbose=False)
    assert report.recurrence() is None
    assert report.describe()["changes"] == ds.n_changes


def test_naming_a_dataset_that_was_not_run_is_an_error_not_a_guess():
    report = run(stub_experiment(), save=False, verbose=False)
    with pytest.raises(KeyError, match="records no dataset/split pair"):
        report.describe(dataset="nope", split="nope")
    # Half a name is ambiguous rather than a default.
    with pytest.raises(ValueError, match="both the dataset and the split"):
        report.describe(dataset="stub")
