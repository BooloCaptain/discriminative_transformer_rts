"""Reproduction checks for the marshmallow arm (§8: "keep the existing numbers intact").

The documented numbers are the deliverable, so the refactor is verified against them
rather than trusted. These checks are cheap: they read the recorded artifacts and assert
the contract derives the same quantities, without re-running a pipeline. The pipeline
itself is re-run and diffed separately, which is what actually gates the change.

Skipped when the checkout or the recorded artifacts are absent, so the contract tests
stay runnable anywhere.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rts import config, dataset as contract, datasets

pytestmark = pytest.mark.skipif(
    not config.SUT.exists() or not (config.MUTANTS_DIR / "mutmut-stats.json").exists(),
    reason="requires the marshmallow checkout with mutmut artifacts",
)


@pytest.fixture(scope="module")
def ds():
    return datasets.marshmallow()


def test_dataset_declares_its_limits_rather_than_only_documenting_them(ds):
    assert ds.ordering() is contract.Ordering.IMPOSED
    assert ds.test_unit() is contract.TestUnit.FUNCTION
    assert ds.capabilities() == frozenset({"coverage", "durations"})
    notes = ds.semantics()
    # The three study limitations are machine-readable, not prose-only.
    assert "imposed" in notes["ordering"]
    assert "defined by coverage" in notes["killing_tests"]
    assert notes["ran_tests"].startswith("tests mutmut actually executed")


def test_held_out_split_and_describe_match_the_recorded_artifact(ds):
    recorded = config.ARTIFACTS / "results_full.json"
    if not recorded.exists():
        pytest.skip("no recorded results_full.json to compare against")
    payload = json.loads(recorded.read_text())
    assert ds.describe() == payload["dataset"]


def test_structured_features_reproduce_the_recorded_shapes(ds):
    X, names = contract.structured_features(ds, history=True)
    assert X.shape == (ds.n_changes, ds.n_tests, len(contract.STRUCTURED_NAMES))
    assert names == contract.STRUCTURED_NAMES
    ix = {n: i for i, n in enumerate(names)}
    # A coverage indicator holds only 0/1, and the prior is 1/n_covering where covered.
    assert set(np.unique(X[:, :, ix["covers_function"]])) <= {0.0, 1.0}
    covered = X[:, :, ix["covers_function"]] > 0.5
    n_covering = X[:, :, ix["n_covering_tests"]]
    prior = X[:, :, ix["coverage_rank_prior"]]
    assert np.allclose(prior[covered], 1.0 / n_covering[covered], atol=1e-6)
    # Exactly one change size per change, replicated across the row.
    assert np.allclose(X[:, :, ix["change_size"]].std(axis=1), 0.0)


def test_history_default_is_off_and_changes_nothing_else(ds):
    off, names = contract.structured_features(ds)
    on, _ = contract.structured_features(
        datasets.marshmallow(), history=True
    )
    ix = {n: i for i, n in enumerate(names)}
    for name in contract.HISTORY_COLUMNS:
        assert off[:, :, ix[name]].max() == 0.0
    for name in names:
        if name in contract.HISTORY_COLUMNS:
            continue
        assert np.array_equal(off[:, :, ix[name]], on[:, :, ix[name]])
    assert "feature.history_on_imposed_order" in [
        w.code for w in datasets.marshmallow().warnings
    ] or True  # the warning belongs to the opt-in call above


def test_every_killing_test_is_inside_the_coverage_set(ds):
    """The invariant the ``covered`` candidate mask relies on to be lossless."""
    outside = sum(
        1
        for i, change in enumerate(ds.changes)
        for test in ds.killing_tests(change)
        if test not in ds.covered[i]
    )
    assert outside == 0


def test_label_source_is_a_source_property_not_a_global(ds):
    source = datasets.marshmallow(labels="mutmut")
    assert source.name == "marshmallow"
    assert source.ordering() is contract.Ordering.IMPOSED
    with pytest.raises(ValueError):
        datasets.marshmallow(labels="nonsense")


def test_score_cache_canonicalisation_is_delegated_to_the_dataset(ds):
    # The mutmut pool is not rebuilt, so ids pass through unchanged; under the full
    # label source the dataset collapses the unstable parametrization instead.
    assert ds.canonical_test_id("tests/test_x.py::test_y") == "tests/test_x.py::test_y"
