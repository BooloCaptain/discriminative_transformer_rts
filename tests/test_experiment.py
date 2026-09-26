"""Tests for the experiment layer: axes, cells, availability, comparability, provenance.

These use the fixture-backed dataset, so they need neither a checkout nor a real run -- which
is the property ``refactor.md`` §7 asks of the contract, and the experiment layer inherits it.

The marshmallow and ladder arms are verified separately by re-running them and diffing against
the recorded artifacts (``scripts/verify_experiment_layer.py``), because that check is slow and
belongs with the artifacts rather than in the unit suite. What is pinned here is the layer's own
behaviour: that a cell is one point in the product, that an unavailable thing is reported rather
than dropped, that history is derived rather than configured, and that a paired comparison
across different populations is refused.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rts import config, features, models, populations, splits
from rts.contract import Ordering, Unmeasured
from rts.experiment import (
    ROLE_DATASET,
    ROLE_FEATURES,
    ROLE_MODEL,
    ROLE_POPULATION,
    ROLE_SPLIT,
    Axis,
    Binding,
    Comparison,
    Element,
    Environment,
    Experiment,
    Knobs,
    constant,
    run,
    unavailable,
)
from tests.stub_dataset import StubDataset

PROBE = 0.5


def half_split():
    return Element(
        "half",
        lambda b: splits.make_split(b.dataset, train_fraction=0.5, seed=b.knobs.seed),
    )


def stub_experiment(**overrides) -> Experiment:
    fields = dict(
        name="stub",
        datasets=Axis(ROLE_DATASET, (constant("stub", StubDataset()),)),
        features=Axis(ROLE_FEATURES, (constant("structured", features.STRUCTURED),)),
        models=Axis(ROLE_MODEL, (constant("coverage", models.CoverageSelector()),)),
        populations=Axis(
            ROLE_POPULATION, (constant("fault_bearing", populations.FAULT_BEARING),)
        ),
        splits=Axis(ROLE_SPLIT, (half_split(),)),
        knobs=Knobs(budgets=(PROBE,), n_bootstrap=20),
    )
    fields.update(overrides)
    return Experiment(**fields)


# --- axes and cells ---------------------------------------------------------


def test_a_cell_is_one_point_in_the_product_and_its_key_names_the_factors():
    experiment = stub_experiment(
        models=Axis(
            ROLE_MODEL,
            (
                constant("a", models.CoverageSelector()),
                constant("b", models.FailureRateSelector()),
            ),
        )
    )
    cells = experiment.cells()
    assert [c.name(ROLE_MODEL) for c in cells] == ["a", "b"]
    assert cells[0].key == (
        "dataset=stub|features=structured|model=a|population=fault_bearing|split=half"
    )
    assert cells[0].factors_dict()[ROLE_DATASET] == "stub"


def test_an_axis_rejects_duplicate_element_names():
    with pytest.raises(ValueError, match="duplicate element names"):
        Axis(ROLE_MODEL, (constant("same", object()), constant("same", object())))


def test_an_element_with_an_unknown_tier_is_rejected_when_planning():
    experiment = stub_experiment(
        models=Axis(ROLE_MODEL, (constant("a", models.CoverageSelector(), tier="quantum"),))
    )
    with pytest.raises(ValueError, match="not in tier_order"):
        experiment.cells()


def test_a_shared_option_that_is_inapplicable_leaves_an_unmeasured_cell():
    """``Axis.map`` is what a dimension-level option is, and inapplicability is a finding."""
    axis = Axis(
        ROLE_MODEL,
        (constant("coverage", models.CoverageSelector()), constant("rate", models.FailureRateSelector())),
    )

    def exclude_history(element):
        if element.name != "coverage":
            return Unmeasured("withheld", "this selector reads no columns")
        return element

    mapped = axis.map(exclude_history)
    assert mapped.names() == ("coverage", "rate")
    report = run(stub_experiment(models=mapped), save=False, verbose=False)
    assert len(report.cells) == 1
    assert len(report.unmeasured) == 1
    assert report.unmeasured[0]["factors"][ROLE_MODEL] == "rate"
    assert report.unmeasured[0]["requirement"] == "withheld"


# --- availability is reported, never dropped --------------------------------


def test_a_missing_artifact_makes_the_cell_unmeasured_rather_than_raising():
    class Cached(models.Selector):
        name = "cached"

        def requirements(self):
            return ("artifact:/nonexistent/nope.jsonl",)

        def scores(self, ctx):  # pragma: no cover - must not be reached
            raise AssertionError("a cell whose requirement is unmet must not be measured")

    report = run(
        stub_experiment(models=Axis(ROLE_MODEL, (Element("cached", lambda b: Cached()),))),
        save=False,
        verbose=False,
    )
    assert report.cells == []
    assert len(report.unmeasured) == 1
    assert report.unmeasured[0]["requirement"] == "artifact:/nonexistent/nope.jsonl"
    assert "does not exist" in report.unmeasured[0]["note"]


def test_an_unrecognised_requirement_spelling_raises():
    class Typo(models.Selector):
        name = "typo"

        def requirements(self):
            return ("coverge",)

        def scores(self, ctx):  # pragma: no cover - must not be reached
            raise AssertionError

    with pytest.raises(ValueError, match="unknown requirement"):
        run(
            stub_experiment(models=Axis(ROLE_MODEL, (Element("typo", lambda b: Typo()),))),
            save=False,
            verbose=False,
        )


def test_an_unavailable_population_is_unmeasured_not_an_empty_average():
    """An average over no rows and a population that cannot exist are different claims."""
    poisoned = unavailable("gone", Unmeasured("coverage", "no coverage on this dataset"))
    report = run(
        stub_experiment(populations=Axis(ROLE_POPULATION, (poisoned,))),
        save=False,
        verbose=False,
    )
    assert report.cells == []
    assert report.unmeasured[0]["requirement"] == "coverage"
    assert report.unmeasured[0]["measured"] is False


def test_applicability_may_depend_on_another_role():
    """A variant can be meaningless in combination with another role's element, not alone."""

    def only_with_coverage_population(binding: Binding) -> Unmeasured | None:
        if binding.factors.get(ROLE_POPULATION) == "fault_bearing":
            return None
        return Unmeasured("artifact:partial", "the cache does not cover this population")

    report = run(
        stub_experiment(
            models=Axis(
                ROLE_MODEL,
                (
                    Element(
                        "partial",
                        lambda b: models.CoverageSelector(),
                        applies=only_with_coverage_population,
                    ),
                ),
            ),
            populations=Axis(
                ROLE_POPULATION,
                (
                    constant("fault_bearing", populations.FAULT_BEARING),
                    constant("no_prior_failure", populations.NO_PRIOR_FAILURE),
                ),
            ),
        ),
        save=False,
        verbose=False,
    )
    assert len(report.cells) == 1
    assert len(report.unmeasured) == 1
    assert report.unmeasured[0]["requirement"] == "artifact:partial"


# --- history is derived, not configured -------------------------------------


def test_history_is_off_when_the_split_shuffles_an_observed_dataset():
    """The rule that ties a split to an ordering, reaching the layer rather than the caller."""
    experiment = stub_experiment(
        datasets=Axis(ROLE_DATASET, (constant("observed", StubDataset(ordering=Ordering.OBSERVED)),)),
        splits=Axis(
            ROLE_SPLIT,
            (
                Element(
                    "shuffled",
                    lambda b: splits.make_split(b.dataset, train_fraction=0.5, shuffle=True),
                ),
            ),
        ),
        history=None,
    )
    report = run(experiment, save=False, verbose=False)
    history = set(features.STRUCTURED.family("history"))
    withheld = {u["column"] for u in report.cells[0].features["unmeasured"]}
    assert history <= withheld


def test_history_is_on_for_an_observed_dataset_by_default():
    experiment = stub_experiment(
        datasets=Axis(ROLE_DATASET, (constant("observed", StubDataset(ordering=Ordering.OBSERVED)),)),
        history=None,
    )
    report = run(experiment, save=False, verbose=False)
    history = set(features.STRUCTURED.family("history"))
    withheld = {u["column"] for u in report.cells[0].features["unmeasured"]}
    assert not (history & withheld)


def test_enabling_history_on_an_imposed_order_warns_rather_than_differing_silently():
    report = run(stub_experiment(history=True), save=False, verbose=False)
    codes = {w["code"] for w in report.cells[0].warnings}
    assert "feature.history_on_imposed_order" in codes


# --- comparability ----------------------------------------------------------


def test_a_comparison_over_a_row_changing_role_is_refused():
    for role in (ROLE_DATASET, ROLE_POPULATION, ROLE_SPLIT):
        with pytest.raises(ValueError, match="would pair different populations"):
            Comparison(role, "whatever", PROBE)


def test_a_comparison_reference_must_exist_on_its_axis():
    with pytest.raises(KeyError, match="comparison reference"):
        stub_experiment(comparisons=(Comparison(ROLE_MODEL, "absent", PROBE),))


def test_paired_deltas_are_computed_over_the_populations_rows():
    experiment = stub_experiment(
        models=Axis(
            ROLE_MODEL,
            (
                constant("coverage", models.CoverageSelector()),
                constant("rate", models.FailureRateSelector()),
            ),
        ),
        comparisons=(Comparison(ROLE_MODEL, "coverage", PROBE),),
    )
    report = run(experiment, save=False, verbose=False)
    record = next(r for r in report.comparisons if r["cell"] == "rate")
    assert record["measured"] is True
    # The paired set is the population's rows, which for this dataset is a single change.
    assert record["n"] == report.find(features="structured", model="coverage").n_rows


# --- cost, provenance, and reuse -------------------------------------------


def test_a_tier_filter_reports_the_cells_it_did_not_spend():
    experiment = stub_experiment(
        models=Axis(
            ROLE_MODEL,
            (
                constant("cpu_model", models.CoverageSelector()),
                constant("gpu_model", models.RandomSelector(), tier="gpu"),
            ),
        )
    )
    report = run(experiment, tiers=("cpu",), save=False, verbose=False)
    assert [c.cell.name(ROLE_MODEL) for c in report.cells] == ["cpu_model"]
    assert len(report.unmeasured) == 1
    assert report.unmeasured[0]["requirement"] == "tier:gpu"


def test_scores_are_reused_across_populations_that_share_a_context():
    """A population restricts which rows a metric averages; it does not change the scores."""
    calls: list[str] = []

    class Counting(models.Selector):
        name = "counting"

        def scores(self, ctx):
            calls.append("scored")
            return np.zeros((ctx.ds.n_changes, ctx.ds.n_tests), dtype=np.float32)

    report = run(
        stub_experiment(
            models=Axis(ROLE_MODEL, (Element("counting", lambda b: Counting()),)),
            populations=Axis(
                ROLE_POPULATION,
                (
                    constant("fault_bearing", populations.FAULT_BEARING),
                    constant("no_prior_failure", populations.NO_PRIOR_FAILURE),
                ),
            ),
        ),
        save=False,
        verbose=False,
    )
    assert len(report.cells) == 2
    assert calls == ["scored"]
    assert sum(1 for c in report.cells if c.seconds > 0) == 1


def test_an_element_is_materialised_once_per_run_and_shared_across_cells():
    built: list[str] = []

    def counting_dataset(binding: Binding) -> StubDataset:
        built.append("dataset")
        return StubDataset()

    experiment = stub_experiment(
        datasets=Axis(ROLE_DATASET, (Element("counted", counting_dataset),)),
        models=Axis(
            ROLE_MODEL,
            tuple(constant(f"m{i}", models.CoverageSelector()) for i in range(4)),
        ),
    )
    report = run(experiment, save=False, verbose=False)
    assert len(report.cells) == 4
    assert built == ["dataset"]


def test_the_report_round_trips_through_its_artifact(tmp_path):
    report = run(stub_experiment(), save=False, verbose=False)
    path = report.save(tmp_path)
    payload = json.loads(path.read_text())
    assert payload["experiment"] == "stub"
    assert payload["n_cells"] == len(report.cells) + len(report.unmeasured)
    assert payload["cells"][0]["measured"] is True
    assert payload["environment"]["knobs"]["candidates"] == "full"
    assert list(payload["axes"]) == list(
        (ROLE_DATASET, ROLE_FEATURES, ROLE_MODEL, ROLE_POPULATION, ROLE_SPLIT)
    )


def test_a_run_knob_override_is_recorded():
    report = run(
        stub_experiment(),
        knobs=Knobs(budgets=(0.25,), n_bootstrap=7, seed=config.SEED + 1),
        save=False,
        verbose=False,
    )
    assert report.environment.knobs.seed == config.SEED + 1
    assert [r["budget"] for r in report.cells[0].results] == [0.25]


def test_the_run_seed_reaches_the_model_not_just_the_split():
    """A knob that stops at the layer is a knob that lies about what it changed."""

    def recall_at(seed: int) -> float:
        report = run(
            stub_experiment(
                models=Axis(ROLE_MODEL, (constant("random", models.RandomSelector()),)),
                knobs=Knobs(budgets=(PROBE,), n_bootstrap=5, seed=seed),
            ),
            save=False,
            verbose=False,
        )
        return report.cells[0].results[0]["recall"]

    assert recall_at(config.SEED) != recall_at(config.SEED + 1)


def test_a_caller_can_share_score_matrices_between_runs():
    """What lets one study be two runs -- different budget sets -- without retraining."""
    experiment = stub_experiment()
    shared: dict = {}

    first = run(experiment, scores=shared, save=False, verbose=False)
    assert first.cells[0].seconds > 0
    assert len(shared) == 1

    second = run(
        experiment,
        knobs=Knobs(budgets=(0.25,), n_bootstrap=5, seed=experiment.knobs.seed),
        scores=shared,
        save=False,
        verbose=False,
    )
    assert second.cells[0].seconds == 0
    assert [r["budget"] for r in second.cells[0].results] == [0.25]


def test_a_model_seed_is_not_served_by_another_seed_s_score_cache():
    """The score key must carry the model seed.

    Without it, four refits of one element reuse the first fit and report four identical
    deltas -- a plausible number from a comparison that never happened, which is the failure
    mode ``refactor.md`` §14 records for a name-based ablation.
    """
    experiment = stub_experiment(
        models=Axis(ROLE_MODEL, (constant("random", models.RandomSelector()),))
    )
    shared: dict = {}

    def run_with(model_seed: int):
        return run(
            experiment,
            knobs=Knobs(
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
    assert first.cells[0].split == second.cells[0].split


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
    ctx = models.Context(
        ds=ds,
        features=features.structured(ds, history=True),
        split=splits.make_split(ds, train_fraction=0.5),
        bm25=np.zeros((ds.n_changes, ds.n_tests), dtype=np.float32),
    )

    binary = models.CachedScores("embed", path, loader=models.load_matrix)
    assert binary.requirements() == (f"artifact:{path}",)
    assert np.array_equal(binary.scores(ctx), matrix)

    # The default loader reads scored pairs, so a numpy file is a decode error rather than a
    # silently wrong matrix -- which is how the embedding arm first failed.
    with pytest.raises(UnicodeDecodeError):
        models.CachedScores("embed", path).scores(ctx)


def test_a_rank_average_is_fitted_free_and_declares_its_parents_caches():
    from rts import config as cfg

    parent = models.CachedScores("semif_textonly", cfg.SEMIF_SCORES_FILE)
    combined = models.RankAverageSelector(
        "rankaverage_xgb_semif", (models.LexicalSelector(), parent), candidates_mode="covered"
    )
    assert combined.requirements() == (f"artifact:{cfg.SEMIF_SCORES_FILE}",)
    assert combined.candidates_mode == "covered"
    with pytest.raises(ValueError, match="two or more parents"):
        models.RankAverageSelector("solo", (models.LexicalSelector(),))


def test_artifact_backed_selectors_declare_what_they_read():
    assert models.CoverageSelector().requirements() == ()
    assert models.RandomSelector().requirements() == ()

    semif = models.SemIfSelector()
    assert semif.requirements() == (f"artifact:{config.SEMIF_SCORES_FILE}",)

    extra = models.XGBoostSelector(extra_score_files={"semif": config.SEMIF_SCORES_FILE})
    assert extra.requirements() == (f"artifact:{config.SEMIF_SCORES_FILE}",)


def test_the_shuffle_controls_build_their_own_bm25_and_name_themselves():
    """The change-shuffle ablation is a selector, so its name is the recorded artifact's key."""
    ds = StubDataset()
    matrix = features.structured(ds, history=True)
    bm25 = np.zeros((ds.n_changes, ds.n_tests), dtype=np.float32)
    ctx = models.Context(
        ds=ds,
        features=matrix,
        split=splits.make_split(ds, train_fraction=0.5),
        bm25=bm25,
        seed=config.SEED,
    )

    canonical = models.LexicalSelector()
    assert canonical.name == "bm25_lexical"
    # No flags means the context's matrix, untouched -- the ablation must not perturb the arm it
    # is a control for.
    assert canonical.scores(ctx) is ctx.bm25

    shuffled = models.LexicalSelector(shuffle_changes=True)
    assert shuffled.name == "bm25_change_shuffled"
    got = shuffled.scores(ctx)
    assert got.shape == (ds.n_changes, ds.n_tests)
    assert not np.array_equal(got, ctx.bm25)

    assert models.LexicalSelector(shuffle_tests=True).name == "bm25_test_shuffled"
    assert (
        models.LexicalSelector(shuffle_changes=True, shuffle_tests=True).name
        == "bm25_both_shuffled"
    )
