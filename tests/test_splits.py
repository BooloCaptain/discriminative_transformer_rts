"""Tests for the split: it partitions, and the evaluation boundary is enforced.

The split is evaluation *configuration* rather than a property of the data, so what is pinned
here is the configuration's own contract: that a split is a partition, and that a metric may
only be averaged over rows the split held out rather than trained on. The second is the guard
``splits.in_window`` exists for; every call site in the package satisfies it by construction,
so these tests build the violation directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from rts import evaluate, splits
from rts.contract import Ordering
from tests.stub_dataset import StubDataset


def stub_split(train_fraction: float = 0.5) -> splits.Split:
    return splits.make_split(StubDataset(), train_fraction=train_fraction)


# --- the partition invariant ------------------------------------------------


def test_make_split_partitions_every_change():
    split = stub_split()
    assert sorted(split.train_idx.tolist()) == [0, 1]
    assert sorted(split.test_idx.tolist()) == [2, 3]
    assert set(split.train_idx).isdisjoint(set(split.test_idx))


def test_make_split_with_a_zero_train_fraction_holds_nothing_out():
    """The shape a zero-shot arm needs: every change in the window, nothing to train on."""
    split = stub_split(train_fraction=0.0)
    assert split.train_idx.size == 0
    assert sorted(split.test_idx.tolist()) == [0, 1, 2, 3]


def test_a_hand_built_split_that_overlaps_is_rejected():
    """A split may be built by hand -- an evaluation window inside the tail needs it -- so the
    invariant cannot rest on ``make_split`` being the only constructor."""
    with pytest.raises(ValueError, match="in both the"):
        splits.Split(
            train_idx=np.array([0, 1, 2], dtype=np.int64),
            test_idx=np.array([2, 3], dtype=np.int64),
            fraction=0.75,
            shuffle=False,
            seed=0,
            ordering=Ordering.IMPOSED,
        )


# --- the window predicate ---------------------------------------------------


def test_in_window_distinguishes_the_tail_from_the_prefix():
    split = stub_split()
    assert splits.in_window(split, np.array([2, 3], dtype=np.int64))
    assert splits.in_window(split, np.array([], dtype=np.int64))
    assert not splits.in_window(split, np.array([2, 0], dtype=np.int64))


def test_require_in_window_raises_and_names_the_offenders():
    split = stub_split()
    splits.require_in_window(split, np.array([2, 3], dtype=np.int64))
    with pytest.raises(ValueError, match=r"1 change\(s\) outside"):
        splits.require_in_window(
            split, np.array([0, 2], dtype=np.int64), what="evaluation rows"
        )


# --- the evaluation boundary ------------------------------------------------


def scores_for(ds: StubDataset) -> np.ndarray:
    return np.random.default_rng(0).random((ds.n_changes, ds.n_tests))


def test_evaluate_rejects_rows_outside_the_split_window():
    ds = StubDataset()
    split = stub_split()
    with pytest.raises(ValueError, match="outside the split's evaluation window"):
        evaluate.evaluate(scores_for(ds), ds, np.arange(ds.n_changes), split=split)


def test_evaluate_accepts_the_window_it_was_given():
    ds = StubDataset()
    split = stub_split()
    results = evaluate.evaluate(
        scores_for(ds), ds, split.test_idx, budgets=(0.5,), split=split
    )
    assert len(results) == 1
    # The tail is c2 (a fault) and c3 (no killing test), so one fault is averaged over.
    assert results[0].n_faults == 1


def test_evaluate_without_a_split_is_unchecked():
    """A zero-shot arm over every change has no split to be inside of."""
    ds = StubDataset()
    results = evaluate.evaluate(
        scores_for(ds), ds, np.arange(ds.n_changes), budgets=(0.5,)
    )
    assert results[0].n_faults == 3
