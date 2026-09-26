"""Reproduction checks for the marshmallow arm ("keep the existing numbers intact").

The documented numbers are the deliverable, so the refactor is verified against them rather
than trusted. These checks are cheap: they read the recorded artifacts and assert the
harness derives the same quantities, without re-running a pipeline. The pipeline and ladder
are re-run and diffed separately, which is what actually gates the change.

Skipped when the checkout or the recorded artifacts are absent, so the contract tests stay
runnable anywhere.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rts import accessors, config, datasets, features, populations, reporting, splits
from rts.contract import Capability, Ordering, TestUnit

pytestmark = pytest.mark.skipif(
    not config.SUT.exists() or not (config.MUTANTS_DIR / "mutmut-stats.json").exists(),
    reason="requires the marshmallow checkout with mutmut artifacts",
)


@pytest.fixture(scope="module")
def ds():
    return datasets.marshmallow()


@pytest.fixture(scope="module")
def split(ds):
    return splits.make_split(ds)


def test_dataset_declares_its_limits_rather_than_only_documenting_them(ds):
    assert ds.ordering() is Ordering.IMPOSED
    assert ds.test_unit() is TestUnit.FUNCTION
    assert ds.capabilities() == frozenset({Capability.COVERAGE, Capability.DURATIONS})
    notes = ds.semantics()
    # The three study limitations are machine-readable, not prose-only.
    assert "imposed" in notes["ordering"]
    assert "defined by coverage" in notes["killing_tests"]
    assert notes["ran_tests"].startswith("tests mutmut actually executed")


def test_describe_matches_the_recorded_artifact(ds, split):
    recorded = config.ARTIFACTS / "results_full.json"
    if not recorded.exists():
        pytest.skip("no recorded results_full.json to compare against")
    payload = json.loads(recorded.read_text())
    assert reporting.describe(ds, split) == payload["dataset"]


def test_structured_matrix_reproduces_the_recorded_shapes(ds):
    matrix = features.structured(ds, history=True)
    assert matrix.X.shape == (ds.n_changes, ds.n_tests, len(features.STRUCTURED.columns))
    assert matrix.columns == tuple(features.STRUCTURED.columns)
    assert not matrix.unmeasured
    # A coverage indicator holds only 0/1, and the prior is 1/n_covering where covered.
    covered = matrix.column("covers_function")
    assert set(np.unique(covered)) <= {0.0, 1.0}
    n_covering = matrix.column("n_covering_tests")
    prior = matrix.column("coverage_rank_prior")
    assert np.allclose(prior[covered > 0.5], 1.0 / n_covering[covered > 0.5], atol=1e-6)
    # Exactly one change size per change, replicated across the row.
    assert np.allclose(matrix.column("change_size").std(axis=1), 0.0)


def test_history_default_is_off_and_leaves_every_other_column_alone(ds):
    off = features.structured(ds)
    on = features.structured(ds, history=True)
    history = features.STRUCTURED.family("history")
    for name in history:
        assert off.column(name).max() == 0.0
        assert off.is_unmeasured(name)
    for name in features.STRUCTURED.columns:
        if name in history:
            continue
        assert np.array_equal(off.column(name), on.column(name)), name
    # The opt-in is what raises the caveat, and it travels with the matrix that needed it.
    assert "feature.history_on_imposed_order" in [w.code for w in on.warnings]
    assert "feature.history_on_imposed_order" not in [w.code for w in off.warnings]


def test_every_killing_test_is_inside_the_coverage_set(ds):
    """The invariant the ``covered`` candidate mask relies on to be lossless."""
    covered = accessors.covered(ds)
    outside = sum(
        1
        for i, change in enumerate(ds.changes)
        for test in ds.killing_tests(change)
        if test not in covered[i]
    )
    assert outside == 0


def test_bm25_matches_the_previously_recorded_value(ds):
    """A single scalar that would move if the text pipeline changed at all."""
    matrix = features.text.build_bm25_scores(ds)
    assert matrix.shape == (ds.n_changes, ds.n_tests)
    assert float(matrix.mean()) == pytest.approx(0.7897728681564331, rel=1e-12)


def test_recorded_population_sizes_are_reproduced(ds, split):
    """The starved and sparse arms' sizes, which several recorded tables rest on."""
    assert len(populations.population("fault_bearing").rows(ds, split.test_idx)) == 464
    assert len(populations.population("starved").rows(ds, split.test_idx)) == 43
    assert int(populations.sparse_mask(ds).sum()) == 2


def test_label_source_is_a_source_property_not_a_global(ds):
    assert datasets.marshmallow(labels="mutmut").name == "marshmallow"
    with pytest.raises(ValueError):
        datasets.marshmallow(labels="nonsense")


def test_score_cache_canonicalisation_is_delegated_to_the_dataset(ds):
    # The mutmut pool is not rebuilt, so ids pass through unchanged; under the full label
    # source the dataset collapses the unstable parametrization instead.
    assert ds.canonical_test_id("tests/test_x.py::test_y") == "tests/test_x.py::test_y"


def test_the_split_reports_the_same_window_the_documented_arm_used(ds, split):
    assert len(split.train_idx) == 2121
    assert len(split.test_idx) == 530
    assert len(accessors.test_fault_idx(ds, split.test_idx)) == 464
    # And the split is a value, so a second one is independent of the dataset.
    other = splits.make_split(ds, train_fraction=0.5)
    assert len(other.train_idx) == 1326 and len(other.test_idx) == 1325
    assert len(split.test_idx) == 530
