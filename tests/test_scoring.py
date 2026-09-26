"""Tests for score production: a cache that is a cell's *output* rather than its precondition.

The point of ``models.ProducedScores`` and ``semif_runner.score_context`` is that the study's
most expensive step becomes a cell, so what is pinned here is the produce-or-read rule, the
distinction from ``CachedScores``, and the completeness check that stops a half-written cache
from being read as a finished one. No GPU and no model: the producer is injected, which is the
capability the design exists to make possible.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rts import features, models, semif, semif_runner, splits
from tests.stub_dataset import StubDataset


def stub_context(ds: StubDataset) -> models.Context:
    return models.Context(
        ds=ds,
        features=features.structured(ds, history=True),
        split=splits.make_split(ds, train_fraction=0.5),
        bm25=np.zeros((ds.n_changes, ds.n_tests), dtype=np.float32),
    )


# --- the selector -----------------------------------------------------------


def test_a_produced_cache_is_not_a_declared_requirement():
    """The whole reason the class exists: declaring it would make the cell unmeasured first."""
    assert models.CachedScores("semif", "x.jsonl").requirements() == ("artifact:x.jsonl",)
    assert models.ProducedScores("semif", "x.jsonl", lambda ctx, out: (None, {})).requirements() == ()


def test_produced_scores_read_an_existing_cache_without_producing(tmp_path):
    path = tmp_path / "matrix.npy"
    np.save(path, np.zeros((4, 3), dtype=np.float32))
    produced: list = []

    def produce(ctx, out):
        produced.append(out)
        return np.ones((4, 3)), {}

    selector = models.ProducedScores("m", path, produce, loader=models.load_matrix)
    ds = StubDataset()
    got = selector.scores(stub_context(ds))

    assert produced == []
    assert selector.last_stats["produced"] is False
    assert np.array_equal(got, np.zeros((4, 3), dtype=np.float32))


def test_produced_scores_produce_and_cache_when_the_cache_is_absent(tmp_path):
    path = tmp_path / "absent.npy"

    def produce(ctx, out):
        matrix = np.full((ctx.ds.n_changes, ctx.ds.n_tests), 7.0, dtype=np.float32)
        np.save(out, matrix)
        return matrix, {"pairs": 12}

    selector = models.ProducedScores("m", path, produce, loader=models.load_matrix)
    ds = StubDataset()
    got = selector.scores(stub_context(ds))

    assert path.exists()
    assert np.all(got == 7.0)
    assert selector.last_stats == {"pairs": 12, "produced": True}


# --- the context-driven scoring core ---------------------------------------


def fake_writer(ds, pair_set, out, *, partial: bool = False):
    """A stand-in for ``score_to_cache`` that writes the cache the real one would."""

    def write(model, tokenizer, ds_, pair_set_, out_, **kwargs):
        index = pair_set_.index[:1] if partial else pair_set_.index
        with out_.open("w") as fh:
            for row, col in index:
                fh.write(
                    json.dumps(
                        {
                            "change_row": row,
                            "test_col": col,
                            "change_id": ds_.change_id(ds_.changes[row]),
                            "test_nodeid": ds_.test_ids[col],
                            "score": float(row + col),
                        }
                    )
                    + "\n"
                )
        return {"pairs": len(index), "pairs_per_second": 1.0}

    return write


def test_score_context_returns_the_matrix_the_cache_holds(tmp_path, monkeypatch):
    """Read back from the cache, so "produced" and "read later" cannot drift."""
    ds = StubDataset()
    split = splits.make_split(ds, train_fraction=0.5)
    candidates = np.ones((ds.n_changes, ds.n_tests), dtype=bool)
    path = tmp_path / "semif.jsonl"

    monkeypatch.setattr(semif_runner, "score_to_cache", fake_writer(ds, None, path))
    matrix, stats = semif_runner.score_context(
        ds, split.test_idx, candidates, path, model=object(), tokenizer=object()
    )

    # The tail is scored; the training prefix is not, so it keeps the sentinel.
    assert stats["pairs_expected"] == len(split.test_idx) * ds.n_tests
    assert stats["pairs_in_cache"] == stats["pairs_expected"]
    assert np.array_equal(matrix, semif.load_scores(path, ds))
    assert matrix[2, 0] == 2.0
    assert matrix[0, 0] < -1e8


def test_score_context_refuses_a_partial_cache(tmp_path, monkeypatch):
    """A torn cache is otherwise indistinguishable from a finished one."""
    ds = StubDataset()
    split = splits.make_split(ds, train_fraction=0.5)
    candidates = np.ones((ds.n_changes, ds.n_tests), dtype=bool)
    path = tmp_path / "partial.jsonl"

    monkeypatch.setattr(semif_runner, "score_to_cache", fake_writer(ds, None, path, partial=True))
    with pytest.raises(RuntimeError, match="missing"):
        semif_runner.score_context(
            ds, split.test_idx, candidates, path, model=object(), tokenizer=object()
        )


def test_score_context_scores_exactly_the_rows_it_was_given(tmp_path, monkeypatch):
    """Restricting the rows is what makes a starved cache cheap; the mask does the same for tests."""
    ds = StubDataset()
    split = splits.make_split(ds, train_fraction=0.5)
    rows = split.test_idx[:1]
    candidates = np.zeros((ds.n_changes, ds.n_tests), dtype=bool)
    candidates[2, :2] = True
    path = tmp_path / "one_row.jsonl"

    monkeypatch.setattr(semif_runner, "score_to_cache", fake_writer(ds, None, path))
    _, stats = semif_runner.score_context(
        ds, rows, candidates, path, model=object(), tokenizer=object()
    )
    assert stats["pairs_expected"] == 2
