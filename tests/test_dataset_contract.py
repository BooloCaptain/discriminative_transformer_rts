"""Tests for the dataset contract (``refactor.md`` §2-§7).

The point of these is not coverage of the marshmallow arm -- that is verified by
re-running ``rts.pipeline`` and ``rts.ladder`` and diffing against the recorded
artifacts. It is to pin the *contract*: that a dataset is cheap to fake, that two can
coexist, that a declaration the harness reads is machine-readable, and that an
unmeasurable quantity stays unmeasured rather than becoming zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from rts import dataset as contract
from rts.dataset import (
    CapabilityMissing,
    Ordering,
    TestUnit,
    Unmeasured,
    make_split,
    namespace,
    population,
)
from tests.stub_dataset import DURATIONS, TEST_IDS, StubDataset

# ``TestUnit`` starts with "Test", so pytest would otherwise try to collect the enum
# itself and warn about its constructor.
TestUnit.__test__ = False  # type: ignore[attr-defined]


# --- a dataset is a value --------------------------------------------------


def test_fixture_dataset_is_complete_and_valid():
    ds = StubDataset()
    assert ds.n_changes == 4
    assert ds.n_tests == 3
    assert ds.describe()["fault_bearing_changes"] == 3
    # Frozen dataclasses and no module-level state, so nothing here mutates a global.
    assert ds.test_ids == list(TEST_IDS)


def test_two_datasets_coexist_with_different_declarations():
    imposed = StubDataset(name="a", coverage=True, ordering=Ordering.IMPOSED)
    observed = StubDataset(name="b", coverage=False, ordering=Ordering.OBSERVED)
    assert imposed.capabilities() != observed.capabilities()
    assert imposed.ordering() is not observed.ordering()
    # Deriving from one must not disturb the other.
    contract.structured_features(imposed, history=True)
    before = observed.warnings.codes()
    contract.structured_features(observed)
    assert observed.warnings.codes() != before  # its own warnings, not the other's
    assert "feature.history_on_imposed_order" not in observed.warnings.codes()


def test_name_may_not_contain_namespacing_separator():
    ds = StubDataset(name="a::b")
    with pytest.raises(ValueError):
        namespace(ds, TEST_IDS[0])


# --- capabilities ----------------------------------------------------------


def test_absent_capability_raises_rather_than_returning_nothing():
    ds = StubDataset(coverage=False)
    with pytest.raises(CapabilityMissing):
        ds.coverage(ds.changes[0])
    with pytest.raises(CapabilityMissing):
        ds.candidates("covered")


def test_absent_capability_makes_its_columns_unmeasured_and_warns():
    ds = StubDataset(coverage=False, durations=False)
    X, names = contract.structured_features(ds, history=True)
    ix = {n: i for i, n in enumerate(names)}
    for name in contract.COVERAGE_COLUMNS + contract.DURATION_COLUMNS:
        # Coerced to zero at the model-input boundary, which is the only place a lossy
        # coercion is legitimate -- and it is recorded.
        assert X[:, :, ix[name]].max() == 0.0
    codes = ds.warnings.codes()
    assert "feature.unmeasured_coverage" in codes
    assert "feature.unmeasured_durations" in codes


def test_present_capability_carries_real_values():
    ds = StubDataset()
    X, names = contract.structured_features(ds, history=True)
    ix = {n: i for i, n in enumerate(names)}
    assert X[:, :, ix["test_duration"]].max() == pytest.approx(2.0)
    assert X[0, :, ix["covers_function"]].sum() == pytest.approx(2.0)


# --- ordering and history --------------------------------------------------


def test_history_is_off_by_default_on_an_imposed_order():
    ds = StubDataset(ordering=Ordering.IMPOSED)
    X, names = contract.structured_features(ds)
    ix = {n: i for i, n in enumerate(names)}
    for name in contract.HISTORY_COLUMNS:
        assert X[:, :, ix[name]].max() == 0.0
    assert "feature.unmeasured_history" in ds.warnings.codes()


def test_history_is_on_by_default_on_an_observed_order():
    ds = StubDataset(ordering=Ordering.OBSERVED)
    X, names = contract.structured_features(ds)
    ix = {n: i for i, n in enumerate(names)}
    # Rate is Laplace-smoothed, so an unseen test starts at 0.5 rather than 0.
    assert X[:, :, ix["test_failure_rate_cum"]].min() > 0.0
    assert "feature.unmeasured_history" not in ds.warnings.codes()


def test_asking_for_history_on_an_imposed_order_warns_about_the_seed():
    ds = StubDataset(ordering=Ordering.IMPOSED)
    contract.structured_features(ds, history=True)
    assert "feature.history_on_imposed_order" in ds.warnings.codes()


def test_history_never_reads_the_current_changes_outcome():
    """Change 3 has no killing test, so changes before it see only changes 0 and 1."""
    ds = StubDataset()
    rate, runs, _age = contract.history_features(ds)
    assert runs[0].sum() == 0.0          # nothing before the first change
    assert runs[1].sum() == 3.0          # all three tests ran once
    # At change 2 the loader has seen two changes: test_one failed twice out of two
    # runs, so its Laplace-smoothed rate is 3/4 rather than the 2/3 a single run implies.
    assert float(rate[2, 0]) == pytest.approx(3 / 4)
    assert runs[2].sum() == 6.0


# --- derived features ------------------------------------------------------


def test_derived_features_parse_the_unified_diff():
    ds = StubDataset()
    first = ds.changes[0]
    assert contract.changed_lines(ds, first) == ("    return 2",)
    assert contract.removed_lines(ds, first) == ("    return 1",)
    assert contract.change_size(ds, first) == 2
    assert contract.change_query_text(ds, first) == "    return 2\n    return 1"


def test_test_size_features_are_zero_when_the_source_is_missing():
    empty = StubDataset(name="empty", sources={})
    missing = "tests/test_alpha.py::test_one"
    assert empty.test_source(missing) is None
    # Unlocatable is not the same as empty: both size features are absent, not 1/0.
    assert contract.test_n_lines(empty, missing) == 0
    assert contract.test_n_tokens(empty, missing) == 0


def test_n_tests_in_test_file_counts_the_pool():
    ds = StubDataset()
    counts = contract.n_tests_in_test_file(ds)
    assert counts[0] == 2 and counts[2] == 1


def test_proximity_features_read_the_change_path():
    ds = StubDataset()
    stem = contract.filename_stem_match(ds)
    # The stem is a substring test, which is what makes it cheap and strong: "beta"
    # matches test_beta.py, and "alpha" does not match test_beta.py.
    assert stem[2, 2] == 1.0 and stem[0, 2] == 0.0
    assert stem[0, 0] == 1.0
    distance = contract.path_distance(ds)
    # pkg/ and tests/ share no component, so the distance is one level each.
    assert distance[0, 0] == 2.0


# --- unmeasured is a state, not a value ------------------------------------


def test_population_is_unavailable_rather_than_empty_without_coverage():
    spec = population("low_pair_recurrence")
    ds = StubDataset(coverage=False)
    result = spec.mask(ds)
    assert isinstance(result, Unmeasured)
    assert result.requirement == contract.REQ_COVERAGE
    with_coverage = spec.mask(StubDataset())
    assert isinstance(with_coverage, np.ndarray) and with_coverage.dtype == bool


def test_available_population_returns_rows_inside_the_window():
    ds = StubDataset()
    rows = np.arange(ds.n_changes, dtype=np.int64)
    fault = population("fault_bearing").rows(ds, rows)
    assert isinstance(fault, np.ndarray)
    assert fault.tolist() == [0, 1, 2]  # change 3 has no killing test


def test_default_population_is_fault_bearing():
    assert "fault_bearing" in contract.POPULATIONS
    assert "starved" in contract.POPULATIONS
    with pytest.raises(KeyError):
        population("nope")


# --- the split -------------------------------------------------------------


def test_shuffling_switches_the_effective_ordering_to_imposed():
    ds = StubDataset(ordering=Ordering.OBSERVED)
    shuffled = make_split(ds, shuffle=True, seed=1)
    assert shuffled.effective_ordering is Ordering.IMPOSED
    assert "split.shuffles_observed_order" in [w.code for w in shuffled.warnings]
    contiguous = make_split(ds, shuffle=False)
    assert contiguous.effective_ordering is Ordering.OBSERVED


def test_contiguous_prefix_warns_on_an_imposed_order():
    ds = StubDataset(ordering=Ordering.IMPOSED)
    split = make_split(ds, train_fraction=0.5)
    assert len(split.train_idx) == 2 and len(split.test_idx) == 2
    assert "split.contiguous_prefix_on_imposed_order" in [w.code for w in split.warnings]


# --- composition -----------------------------------------------------------


def test_pooling_namespaces_test_ids_and_keeps_meanings_apart():
    a = StubDataset(name="a")
    b = StubDataset(name="b")
    pooled = contract.pool([a, b])
    assert pooled.n_changes == 8
    assert len(pooled.test_pool) == 6
    assert all(t.startswith("a::") or t.startswith("b::") for t in pooled.test_pool)
    # The label of a pooled change is namespaced with the same prefix as its pool.
    labels = pooled.labels
    assert labels[0].sum() == 1
    assert pooled.canonical_test_id("a::tests/test_alpha.py::test_one") == (
        "a::tests/test_alpha.py::test_one"
    )


def test_pooling_warns_when_test_units_disagree():
    a = StubDataset(name="a", test_unit=TestUnit.FUNCTION)
    b = StubDataset(name="b", test_unit=TestUnit.CASE)
    pooled = contract.pool([a, b])
    assert "pool.mixed_test_unit" in pooled.warnings.codes()


def test_pooled_dataset_capabilities_are_the_intersection():
    both = contract.pool([StubDataset(name="a"), StubDataset(name="b")])
    one = contract.pool([StubDataset(name="a"), StubDataset(name="b", coverage=False)])
    assert both.capabilities() == frozenset({"coverage", "durations"})
    assert one.capabilities() == frozenset({"durations"})


def test_pooled_ordering_is_observed_only_if_every_constituent_is():
    observed = contract.pool(
        [StubDataset(name="a", ordering=Ordering.OBSERVED),
         StubDataset(name="b", ordering=Ordering.OBSERVED)]
    )
    mixed = contract.pool(
        [StubDataset(name="a", ordering=Ordering.OBSERVED),
         StubDataset(name="b", ordering=Ordering.IMPOSED)]
    )
    assert observed.ordering() is Ordering.OBSERVED
    assert mixed.ordering() is Ordering.IMPOSED


# --- derived datasets ------------------------------------------------------


def test_bundle_dataset_is_a_dataset_over_another_dataset():
    from rts.datasets import Bundle, BundleDataset

    base = StubDataset()
    bundles = [Bundle(rung=1, signal=0, members=(0, 2)), Bundle(rung=1, signal=2, members=(2, 3))]
    wrapped = BundleDataset(base, bundles, rung=1)

    assert wrapped.n_changes == 2
    # Declared, never inherited: a bundle draws its parts from anywhere in the base.
    assert wrapped.ordering() is Ordering.IMPOSED
    # The label is the union of the members', so it stays exact.
    assert wrapped.killing_tests(bundles[0]) == base.killing_tests(base.changes[0]) | (
        base.killing_tests(base.changes[2])
    )
    # The change size the harness derives equals the sum over members, because the
    # concatenation preserves line structure.
    assert contract.change_size(wrapped, bundles[0]) == sum(
        contract.change_size(base, base.changes[m]) for m in bundles[0].members
    )
    assert "derived.concatenated_diff" in wrapped.warnings.codes()


def test_bundle_dataset_narrows_capabilities_and_candidate_pool():
    from rts.datasets import Bundle, BundleDataset

    base = StubDataset()
    bundles = [Bundle(rung=0, signal=0, members=(0,))]
    signal_pool = BundleDataset(base, bundles, rung=0, pool="signal")
    union_pool = BundleDataset(base, bundles, rung=0, pool="union")

    # Durations do not transfer to a bundle; coverage does.
    assert signal_pool.capabilities() == frozenset({"coverage"})
    assert signal_pool.candidates("covered").sum() == len(base.coverage(base.changes[0]))
    assert union_pool.candidates("covered").sum() == len(base.coverage(base.changes[0]))


def test_bundle_dataset_rejects_an_unknown_pool():
    from rts.datasets import Bundle, BundleDataset

    with pytest.raises(ValueError):
        BundleDataset(StubDataset(), [Bundle(0, 0, (0,))], rung=0, pool="nonsense")


# --- evaluation contract ---------------------------------------------------


def test_scores_labels_candidates_rows_is_enough_to_measure():
    from rts import evaluate

    ds = StubDataset()
    rows = np.arange(ds.n_changes, dtype=np.int64)
    scores = ds.labels.astype(np.float32)  # a perfect ranker
    candidates = np.ones((ds.n_changes, ds.n_tests), dtype=bool)
    results = evaluate.evaluate(scores, ds, rows, budgets=(0.5,), n_bootstrap=0,
                                candidates=candidates)
    assert results[0].recall == pytest.approx(1.0)
    assert results[0].n_faults == 3


def test_unavailable_population_raises_rather_than_averaging_over_nothing():
    from rts import evaluate

    ds = StubDataset(coverage=False)
    rows = np.arange(ds.n_changes, dtype=np.int64)
    scores = np.zeros((ds.n_changes, ds.n_tests), dtype=np.float32)
    evaluation = evaluate.evaluate_rows(
        scores, ds, rows, budgets=(0.5,), n_bootstrap=0, population="low_pair_recurrence"
    )
    assert not evaluation.measured
    assert isinstance(evaluation.unmeasured, Unmeasured)
    with pytest.raises(evaluate.UnmeasuredPopulation):
        evaluate.evaluate(
            scores, ds, rows, budgets=(0.5,), n_bootstrap=0, population="low_pair_recurrence"
        )


def test_named_population_restricts_the_averaging_rows():
    from rts import evaluate

    ds = StubDataset()
    rows = np.arange(ds.n_changes, dtype=np.int64)
    scores = np.ones((ds.n_changes, ds.n_tests), dtype=np.float32)
    evaluation = evaluate.evaluate_rows(
        scores, ds, rows, budgets=(0.5,), n_bootstrap=0, population="no_prior_failure"
    )
    assert evaluation.population == "no_prior_failure"
    assert evaluation.n_rows <= 3
