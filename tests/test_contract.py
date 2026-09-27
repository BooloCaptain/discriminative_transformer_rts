"""Tests for the dataset contract and everything derived from it.

These do not cover the marshmallow arm -- that is verified by re-running the pipeline and
the ladder and diffing against the recorded artifacts. They pin the *contract*: that a
dataset is cheap to fake, that two coexist, that a declaration the harness reads is
machine-readable, and that an unmeasurable quantity stays unmeasured rather than becoming
zero.

Two deliberate assertions about *where* things live, because both were defects:

* ``test_warnings_are_returned_not_stored`` -- a warning belongs to the computation that
  produced it. The first pass stored warnings on the dataset, which made the record
  describe how a function was called rather than what the data is.
* ``test_describe_requires_an_explicit_split`` -- the split is evaluation configuration.
  Reporting it from a default published one run's partition as a property of the data, and
  put it in the recorded artifacts.
"""

from __future__ import annotations

import numpy as np
import pytest

from rts import features
from rts.data import accessors, composition, contract, populations, reporting, splits
from rts.data.contract import (
    Capability,
    CapabilityMissing,
    Ordering,
    TestUnit,
    Unmeasured,
)
from tests.stub_dataset import DURATIONS, TEST_IDS, StubDataset

# ``TestUnit`` starts with "Test", so pytest would otherwise try to collect the enum.
TestUnit.__test__ = False  # type: ignore[attr-defined]


# --- a dataset is a value --------------------------------------------------


def test_fixture_dataset_is_complete_and_valid():
    ds = StubDataset()
    assert ds.n_changes == 4
    assert ds.n_tests == 3
    assert ds.test_ids == list(TEST_IDS)
    assert accessors.fault_idx(ds).tolist() == [0, 1, 2]


def test_two_datasets_coexist_with_different_declarations():
    imposed = StubDataset(name="a", coverage=True, ordering=Ordering.IMPOSED)
    observed = StubDataset(name="b", coverage=False, ordering=Ordering.OBSERVED)
    assert imposed.capabilities() != observed.capabilities()
    assert imposed.available_requirements() != observed.available_requirements()
    # Deriving from one must not disturb the other.
    features.structured(imposed, history=True)
    features.structured(observed)
    assert "coverage" not in {c.value for c in observed.capabilities()}


def test_name_may_not_contain_the_namespacing_separator():
    with pytest.raises(ValueError):
        composition.namespace(StubDataset(name="a::b"), TEST_IDS[0])


def test_capabilities_are_enums_not_strings():
    ds = StubDataset()
    assert ds.capabilities() == frozenset({Capability.COVERAGE, Capability.DURATIONS})
    assert ds.has_capability(Capability.COVERAGE)
    assert ds.has_capability("coverage")  # the string spelling still resolves
    # An unrecognised name is a typo, not an absent capability: a quiet False would gate a
    # whole feature block off and report it as unmeasured.
    with pytest.raises(ValueError):
        ds.has_capability("nonsense")
    assert ds.available_requirements() == frozenset(
        {contract.Requirement.LABELS, contract.Requirement.DIFF_TEXT, contract.Requirement.COVERAGE,
         contract.Requirement.DURATIONS}
    )
    assert StubDataset(durations=False).missing_requirements(
        [contract.Requirement.DURATIONS]
    ) == (contract.Requirement.DURATIONS,)


# --- capabilities ----------------------------------------------------------


def test_absent_capability_raises_rather_than_returning_nothing():
    ds = StubDataset(coverage=False)
    with pytest.raises(CapabilityMissing):
        ds.coverage(ds.changes[0])
    with pytest.raises(CapabilityMissing):
        accessors.covered(ds)
    with pytest.raises(CapabilityMissing):
        accessors.candidates(ds, "covered")


def test_absent_capability_marks_its_columns_unmeasured_and_says_so():
    ds = StubDataset(coverage=False, durations=False)
    matrix = features.structured(ds, history=True)
    unmeasured = {c for c, _ in matrix.unmeasured}
    assert unmeasured == set(features.STRUCTURED.family("coverage")) | {
        "test_duration"
    }
    # Coerced to zero at the model-input boundary -- the one place a lossy coercion is
    # legitimate -- and *reported*.
    for column in unmeasured:
        assert matrix.column(column).max() == 0.0
    assert not matrix.measured("covers_function")
    assert matrix.reason("covers_function").requirement == "coverage"
    assert any(w.code == "feature.unmeasured" for w in matrix.warnings)


def test_present_capability_carries_real_values():
    ds = StubDataset()
    matrix = features.structured(ds, history=True)
    assert matrix.column("test_duration").max() == pytest.approx(2.0)
    assert matrix.column("covers_function")[0].sum() == pytest.approx(2.0)
    assert not matrix.unmeasured


# --- ordering and history --------------------------------------------------


def test_history_is_off_by_default_on_an_imposed_order():
    matrix = features.structured(StubDataset(ordering=Ordering.IMPOSED))
    for column in features.STRUCTURED.family("history"):
        assert matrix.column(column).max() == 0.0
        assert matrix.is_unmeasured(column)


def test_history_is_on_by_default_on_an_observed_order():
    matrix = features.structured(StubDataset(ordering=Ordering.OBSERVED))
    # The rate is Laplace-smoothed, so an unseen test starts at 0.5 rather than 0.
    assert matrix.column("test_failure_rate_cum").min() > 0.0
    assert not any(
        matrix.is_unmeasured(c) for c in features.STRUCTURED.family("history")
    )


def test_asking_for_history_on_an_imposed_order_warns_about_the_seed():
    matrix = features.structured(StubDataset(ordering=Ordering.IMPOSED), history=True)
    codes = [w.code for w in matrix.warnings]
    assert "feature.history_on_imposed_order" in codes
    warning = next(w for w in matrix.warnings if w.code.startswith("feature.history_on"))
    assert warning.requirement == contract.Policy.OBSERVED_ORDER.value


def test_history_never_reads_the_current_changes_outcome():
    """Change 3 has no killing test, so changes before it see only changes 0 and 1."""
    ds = StubDataset()
    rate, runs, _age = features.derived.history_features(
        accessors.labels(ds), accessors.runs(ds)
    )
    assert runs[0].sum() == 0.0  # nothing before the first change
    assert runs[1].sum() == 3.0  # all three tests ran once
    # At change 2 the loader has seen two changes: test_one failed twice out of two runs,
    # so its Laplace-smoothed rate is 3/4 rather than the 2/3 a single run implies.
    assert float(rate[2, 0]) == pytest.approx(3 / 4)
    assert runs[2].sum() == 6.0


# --- derived features ------------------------------------------------------


def test_derived_features_parse_the_unified_diff():
    ds = StubDataset()
    first = ds.changes[0]
    assert features.derived.changed_lines(ds, first) == ("    return 2",)
    assert features.derived.removed_lines(ds, first) == ("    return 1",)
    assert features.derived.change_size(ds, first) == 2
    assert features.derived.change_query_text(ds, first) == "    return 2\n    return 1"


def test_test_size_features_distinguish_missing_from_empty():
    empty = StubDataset(name="empty", sources={})
    missing = TEST_IDS[0]
    assert empty.test_source(missing) is None
    # Unlocatable is not the same as empty: both size features are absent, not 1 and 0.
    assert features.derived.test_n_lines(empty, missing) == 0
    assert features.derived.test_n_tokens(empty, missing) == 0
    assert features.derived.test_n_lines_in("") == 1
    assert features.derived.test_n_tokens_in("") == 0


def test_proximity_features_read_the_change_path():
    ds = StubDataset()
    stem = features.derived.filename_stem_match(
        accessors.change_paths(ds), ds.test_ids
    )
    # The stem is a substring test, which is what makes it cheap and strong: "beta"
    # matches test_beta.py, and "alpha" does not.
    assert stem[2, 2] == 1.0 and stem[0, 2] == 0.0
    distance = features.derived.path_distance(accessors.change_paths(ds), ds.test_ids)
    # pkg/ and tests/ share no component, so the distance is one level each.
    assert distance[0, 0] == 2.0


def test_n_tests_in_test_file_counts_the_pool():
    counts = features.derived.n_tests_in_test_file(StubDataset().test_ids)
    assert counts[0, 0] == 2 and counts[0, 2] == 1


def test_multi_file_changes_are_reported_not_warned_as_a_side_effect():
    ds = StubDataset(extra_files={"c0": ("pkg/other.py",)})
    assert accessors.multi_file_changes(ds) == (0,)
    paths = accessors.change_paths(ds)
    assert paths[0] == "pkg/alpha.py"  # the first path, deterministically
    assert any(
        w.code == "dataset.multi_file_changes_flattened" for w in reporting.audit(ds)
    )


# --- warnings are values ---------------------------------------------------


def test_warnings_are_returned_not_stored():
    ds = StubDataset(coverage=False)
    assert not hasattr(ds, "warnings")
    first = features.structured(ds, history=True)
    assert first.warnings
    # A second, differently-configured derivation does not append to the first, and the
    # dataset itself accumulates nothing.
    second = features.structured(ds, history=False)
    assert not hasattr(ds, "warnings")
    assert [w.code for w in first.warnings] != [] and isinstance(second.warnings, tuple)
    with_coverage = features.structured(StubDataset(), history=True)
    assert not any(w.code == "feature.unmeasured" for w in with_coverage.warnings)


def test_warnings_deduplicate_by_code_and_scope():
    warnings = contract.Warnings()
    for _ in range(3):
        warnings.add("x", "r", "note", scope="s")
    assert len(warnings) == 1
    warnings.add("x", "r", "note", scope="other")
    assert len(warnings) == 2
    merged = contract.Warnings.from_(warnings, warnings)
    assert len(merged) == 2  # merging the same collection changes nothing
    assert merged.codes() == ["x", "x"]


# --- unmeasured is a state, not a value ------------------------------------


def test_population_is_unavailable_rather_than_empty_without_coverage():
    spec = populations.population("low_pair_recurrence")
    ds = StubDataset(coverage=False)
    result = spec.mask(ds)
    assert isinstance(result, Unmeasured)
    assert result.requirement == "coverage"
    assert isinstance(spec.mask(StubDataset()), np.ndarray)


def test_population_requirements_are_derived_from_what_it_reads():
    # Nothing hand-writes a requirement set: each population's requirements come from the
    # material its predicate names, through the one catalogue.
    spec = populations.population("starved")
    assert spec.needs == ("pair_failures", "pair_runs", "paths", "killing")
    assert {r.value for r in spec.requires()} == {"labels", "diff_text"}
    assert accessors.requirements_for(("coverage", "pairs")) == frozenset(
        {contract.Requirement.COVERAGE}
    )
    with pytest.raises(KeyError):
        accessors.requirements_for(("nonsense",))


def test_available_population_returns_rows_inside_the_window():
    ds = StubDataset()
    rows = np.arange(ds.n_changes, dtype=np.int64)
    fault = populations.population("fault_bearing").rows(ds, rows)
    assert isinstance(fault, np.ndarray)
    assert fault.tolist() == [0, 1, 2]  # change 3 has no killing test


def test_starved_and_sparse_have_one_implementation_each():
    ds = StubDataset()
    # The declared population and the parameterised helper are the same predicate.
    assert populations.sparse_mask(ds).tolist() == populations.population(
        "low_pair_recurrence"
    ).mask(ds).tolist()
    assert populations.starved_mask(ds, max_failures=1).tolist() == populations.population(
        "no_prior_failure"
    ).mask(ds).tolist()
    # The extra knob the two implementations had drifted apart on is still reachable.
    assert populations.starved_mask(ds, max_failures=5, max_runs=1).dtype == bool


# --- the split -------------------------------------------------------------


def test_shuffling_switches_the_effective_ordering_to_imposed():
    ds = StubDataset(ordering=Ordering.OBSERVED)
    shuffled = splits.make_split(ds, shuffle=True, seed=1)
    assert shuffled.effective_ordering is Ordering.IMPOSED
    codes = [w.code for w in shuffled.warnings]
    assert "split.shuffles_observed_order" in codes
    # A shuffle warning is about the *run's* order, not the dataset's, and it says so.
    warning = next(w for w in shuffled.warnings if w.code.startswith("split.shuffles"))
    assert warning.requirement == contract.Policy.EFFECTIVE_ORDER.value
    assert splits.make_split(ds, shuffle=False).effective_ordering is Ordering.OBSERVED


def test_contiguous_prefix_warns_on_an_imposed_order():
    ds = StubDataset(ordering=Ordering.IMPOSED)
    split = splits.make_split(ds, train_fraction=0.5)
    assert len(split.train_idx) == 2 and len(split.test_idx) == 2
    warning = next(w for w in split.warnings if w.code.startswith("split.contiguous"))
    assert warning.requirement == contract.Policy.OBSERVED_ORDER.value


def test_make_split_reproduces_a_contiguous_prefix():
    ds = StubDataset()
    split = splits.make_split(ds, train_fraction=0.75)
    assert split.train_idx.tolist() == [0, 1, 2]
    assert split.test_idx.tolist() == [3]


def test_describe_requires_an_explicit_split():
    ds = StubDataset()
    with pytest.raises(TypeError):
        reporting.describe(ds)  # type: ignore[call-arg]
    split = splits.make_split(ds, train_fraction=0.5)
    described = reporting.describe(ds, split)
    assert described["train_changes"] == 2 and described["test_changes"] == 2
    # One of the four changes survives, so only three have a fault at all.
    assert described["fault_bearing_changes"] == 3


# --- composition -----------------------------------------------------------


def test_pooling_namespaces_test_ids_and_keeps_meanings_apart():
    pooled = composition.pool([StubDataset(name="a"), StubDataset(name="b")])
    assert pooled.n_changes == 8
    assert len(pooled.test_pool) == 6
    assert all(t.startswith("a::") or t.startswith("b::") for t in pooled.test_pool)
    labels = accessors.labels(pooled)
    assert labels[0].sum() == 1
    assert pooled.canonical_test_id("a::tests/test_alpha.py::test_one") == (
        "a::tests/test_alpha.py::test_one"
    )


def test_pooled_change_identity_is_namespaced_and_stable():
    pooled = composition.pool([StubDataset(name="a"), StubDataset(name="b")])
    changes = pooled.changes
    # Both constituents have a change called "c0"; pooling must not confuse them.
    assert changes[0].key == "a::c0" and changes[4].key == "b::c0"
    assert pooled.change_id(changes[0]) != pooled.change_id(changes[4])
    assert accessors.change_index(pooled)["b::c0"] == 4
    # Identity survives a garbage collection: the map is not keyed by object id.
    import gc

    gc.collect()
    assert pooled.change_id(changes[0]) == "a::c0"


def test_pooling_warns_when_test_units_disagree():
    pooled = composition.pool(
        [StubDataset(name="a", test_unit=TestUnit.FUNCTION), StubDataset(name="b", test_unit=TestUnit.CASE)]
    )
    assert pooled.mixed_test_units()
    assert any(w.code == "pool.mixed_test_unit" for w in reporting.audit(pooled))


def test_pooled_capabilities_are_the_intersection():
    both = composition.pool([StubDataset(name="a"), StubDataset(name="b")])
    one = composition.pool([StubDataset(name="a"), StubDataset(name="b", coverage=False)])
    assert both.capabilities() == frozenset(
        {Capability.COVERAGE, Capability.DURATIONS}
    )
    assert one.capabilities() == frozenset({Capability.DURATIONS})


def test_pooled_ordering_is_observed_only_if_every_constituent_is():
    observed = composition.pool(
        [
            StubDataset(name="a", ordering=Ordering.OBSERVED),
            StubDataset(name="b", ordering=Ordering.OBSERVED),
        ]
    )
    mixed = composition.pool(
        [
            StubDataset(name="a", ordering=Ordering.OBSERVED),
            StubDataset(name="b", ordering=Ordering.IMPOSED),
        ]
    )
    assert observed.ordering() is Ordering.OBSERVED
    assert mixed.ordering() is Ordering.IMPOSED


# --- derived datasets ------------------------------------------------------


def test_bundle_dataset_is_a_dataset_over_another_dataset():
    from rts.data.datasets import Bundle, BundleDataset

    base = StubDataset()
    bundles = [
        Bundle(rung=1, signal=0, members=(0, 2)),
        Bundle(rung=1, signal=2, members=(2, 3)),
    ]
    wrapped = BundleDataset(base, bundles, rung=1)

    assert wrapped.n_changes == 2
    # Declared, never inherited: a bundle draws its parts from anywhere in the base.
    assert wrapped.ordering() is Ordering.IMPOSED
    # The label is the union of the members', so it stays exact.
    assert wrapped.killing_tests(bundles[0]) == base.killing_tests(
        base.changes[0]
    ) | base.killing_tests(base.changes[2])
    assert any(w.code == "derived.concatenated_diff" for w in reporting.audit(wrapped))


def test_bundle_dataset_narrows_capabilities():
    from rts.data.datasets import Bundle, BundleDataset

    base = StubDataset()
    bundles = [Bundle(rung=0, signal=0, members=(0,))]
    signal_pool = BundleDataset(base, bundles, rung=0, pool="signal")
    union_pool = BundleDataset(base, bundles, rung=0, pool="union")
    # Durations do not transfer to a bundle; coverage does.
    assert signal_pool.capabilities() == frozenset({Capability.COVERAGE})
    assert accessors.candidates(signal_pool, "covered").sum() == len(
        base.coverage(base.changes[0])
    )
    assert accessors.candidates(union_pool, "covered").sum() == len(
        base.coverage(base.changes[0])
    )


def test_bundle_dataset_rejects_an_unknown_pool():
    from rts.data.datasets import Bundle, BundleDataset

    with pytest.raises(ValueError):
        BundleDataset(StubDataset(), [Bundle(0, 0, (0,))], rung=0, pool="nonsense")


# --- the evaluation contract ----------------------------------------------


def test_scores_labels_candidates_rows_is_enough_to_measure():
    from rts import evaluate

    ds = StubDataset()
    split = splits.make_split(ds)
    rows = np.arange(ds.n_changes, dtype=np.int64)
    scores = accessors.labels(ds).astype(np.float32)  # a perfect ranker
    candidates = np.ones((ds.n_changes, ds.n_tests), dtype=bool)
    results = evaluate.evaluate(
        scores, ds, rows, budgets=(0.5,), n_bootstrap=0, candidates=candidates
    )
    assert results[0].recall == pytest.approx(1.0)
    assert results[0].n_faults == 3
    assert split.test_idx.tolist() == [3]


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
            scores, ds, rows, budgets=(0.5,), n_bootstrap=0,
            population="low_pair_recurrence",
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


def test_accessor_arrays_are_read_only():
    ds = StubDataset()
    for array in (accessors.labels(ds), accessors.runs(ds), accessors.fault_mask(ds)):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0, 0] = 1


def test_test_source_is_memoised_on_the_dataset():
    calls = {"n": 0}

    class Counting(StubDataset):
        def test_source(self, test):
            calls["n"] += 1
            return super().test_source(test)

    ds = Counting()
    first = accessors.test_source(ds, TEST_IDS[0])
    second = accessors.test_source(ds, TEST_IDS[0])
    assert first == second
    assert calls["n"] == 1
    # The primitive itself is not memoised, so a second dataset reads its own text.
    assert accessors.test_source(Counting(), TEST_IDS[0]) == first
