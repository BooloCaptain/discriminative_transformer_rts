"""Individual derived features: one quantity each, computed from the primitives.

Two layers, deliberately:

* ``*_in`` functions are pure over the material they read -- a diff string, a test's
  source text, a label matrix. They are what a feature group calls, so a group never
  reaches back into a dataset while it is being built.
* The dataset-level functions (``changed_lines(ds, change)``, ...) are the convenience
  spelling for consumers outside the feature layer -- ``bundles``, ``semif``, ``embed``
  all build change text -- and they are thin wrappers over the pure ones, so there is
  one implementation of each quantity rather than one per caller.

Every quantity here is defined for *any* dataset, which is what makes an identical
column mean an identical thing across datasets.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from ..contract import Dataset, TestId

# --- change text -----------------------------------------------------------


def changed_lines_in(diff_text: str) -> tuple[str, ...]:
    """Added lines of a unified diff.

    The ``+++`` header is excluded. A content line whose text begins with ``++`` is
    excluded too -- that is the historical spelling of this parser and the documented
    numbers depend on it, so it is stated rather than quietly fixed.
    """
    return tuple(
        line[1:]
        for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def removed_lines_in(diff_text: str) -> tuple[str, ...]:
    """Removed lines of a unified diff, excluding the ``---`` header."""
    return tuple(
        line[1:]
        for line in diff_text.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )


def change_size_in(diff_text: str) -> int:
    """Added plus removed lines. One function, not one per dataset, so a cross-dataset
    comparison of "change size" compares the same quantity."""
    return len(changed_lines_in(diff_text)) + len(removed_lines_in(diff_text))


def change_query_text_in(diff_text: str) -> str:
    """The change side of a pair: added and removed lines, weighted by repetition.

    Including the removed lines matters because a mutation's meaning often comes from
    what it replaced.
    """
    added = "\n".join(changed_lines_in(diff_text))
    removed = "\n".join(removed_lines_in(diff_text))
    return f"{added}\n{removed}"


# --- test text -------------------------------------------------------------


def test_n_lines_in(text: str | None) -> int:
    """Lines in a test's source. ``None`` means unlocatable, which is not zero-length."""
    return 0 if text is None else text.count("\n") + 1


def test_n_tokens_in(text: str | None) -> int:
    """Whitespace tokens in a test's source. ``None`` means unlocatable."""
    return 0 if text is None else len(text.split())


# --- dataset-level convenience --------------------------------------------


def changed_lines(ds: Dataset, change: Any) -> tuple[str, ...]:
    return changed_lines_in(ds.diff_text(change))


def removed_lines(ds: Dataset, change: Any) -> tuple[str, ...]:
    return removed_lines_in(ds.diff_text(change))


def change_size(ds: Dataset, change: Any) -> int:
    return change_size_in(ds.diff_text(change))


def change_query_text(ds: Dataset, change: Any) -> str:
    return change_query_text_in(ds.diff_text(change))


def test_n_lines(ds: Dataset, test: TestId) -> int:
    return test_n_lines_in(ds.test_source(test))


def test_n_tokens(ds: Dataset, test: TestId) -> int:
    return test_n_tokens_in(ds.test_source(test))


# --- per-test and per-pair matrices ---------------------------------------


def durations_column(durations: Mapping[TestId, float], test_ids: Sequence[TestId]) -> np.ndarray:
    """``[n_changes, n_tests]`` view of a per-test constant, for broadcasting."""
    values = np.array([durations.get(t, 0.0) for t in test_ids], dtype=np.float32)
    return np.broadcast_to(values[None, :], (1, len(test_ids)))


def test_n_lines_row(test_ids: Sequence[TestId], sources: Mapping[str, str | None]) -> np.ndarray:
    values = np.array([test_n_lines_in(sources.get(t)) for t in test_ids], dtype=np.float32)
    return np.broadcast_to(values[None, :], (1, len(test_ids)))


def test_n_tokens_row(test_ids: Sequence[TestId], sources: Mapping[str, str | None]) -> np.ndarray:
    values = np.array([test_n_tokens_in(sources.get(t)) for t in test_ids], dtype=np.float32)
    return np.broadcast_to(values[None, :], (1, len(test_ids)))


def n_tests_in_test_file(test_ids: Sequence[TestId]) -> np.ndarray:
    """How many tests share a file with this one. Cheap context for whether a test is a
    focused unit test or one of many in a large module suite."""
    counts: dict[str, int] = defaultdict(int)
    for nodeid in test_ids:
        counts[nodeid.split("::")[0]] += 1
    values = np.array([counts[t.split("::")[0]] for t in test_ids], dtype=np.float32)
    return np.broadcast_to(values[None, :], (1, len(test_ids)))


def _dirs_and_stems(paths: Sequence[str]) -> tuple[list[str], list[str]]:
    dirs = [p.rsplit("/", 1)[0] for p in paths]
    stems = [p.rsplit("/", 1)[-1].removesuffix(".py") for p in paths]
    return dirs, stems


def _path_parts(directory: str) -> list[str]:
    """Components of a directory path. Pass a directory, not a file path."""
    return [part for part in directory.split("/") if part]


def _dir_distance(a: list[str], b: list[str]) -> int:
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return len(a) + len(b) - 2 * common


def path_distance(paths: Sequence[str], test_ids: Sequence[TestId]) -> np.ndarray:
    """Directory-tree distance between each change's file and each test's file."""
    n_c, n_t = len(paths), len(test_ids)
    change_dirs, _ = _dirs_and_stems(paths)
    test_dirs = [str(test_ids[i].split("::")[0]).rsplit("/", 1)[0] for i in range(n_t)]
    out = np.zeros((n_c, n_t), dtype=np.float32)
    cache: dict[tuple[str, str], int] = {}
    for i in range(n_c):
        for j in range(n_t):
            key = (change_dirs[i], test_dirs[j])
            d = cache.get(key)
            if d is None:
                d = _dir_distance(_path_parts(change_dirs[i]), _path_parts(test_dirs[j]))
                cache[key] = d
            out[i, j] = d
    return out


def filename_stem_match(paths: Sequence[str], test_ids: Sequence[TestId]) -> np.ndarray:
    """``test_utils.py`` for ``utils.py``: a strong, cheap naming signal."""
    n_c, n_t = len(paths), len(test_ids)
    _, change_stems = _dirs_and_stems(paths)
    test_names = [test_ids[i].split("::")[0].rsplit("/", 1)[-1] for i in range(n_t)]
    out = np.zeros((n_c, n_t), dtype=np.float32)
    for i in range(n_c):
        for j in range(n_t):
            out[i, j] = 1.0 if change_stems[i] and change_stems[i] in test_names[j] else 0.0
    return out


def coverage_columns(
    covered_by: Sequence[frozenset[TestId]],
    test_index: Mapping[TestId, int],
    n_changes: int,
    n_tests: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(mask, n_covering, rank_prior)`` -- the three coverage columns.

    Restricting to the index matters even though most datasets pre-filter, because a
    pooled dataset namespaces its ids and a dataset whose coverage map is external may
    not. A test outside the pool cannot be a column.
    """
    mask = np.zeros((n_changes, n_tests), dtype=np.float32)
    for i, tests in enumerate(covered_by):
        for test in tests:
            j = test_index.get(test)
            if j is not None:
                mask[i, j] = 1.0
    n_covering = mask.sum(axis=1, keepdims=True).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        prior = np.where(n_covering > 0, 1.0 / np.maximum(n_covering, 1.0), 0.0)
    return (
        mask,
        np.broadcast_to(n_covering, (n_changes, n_tests)),
        np.broadcast_to(prior, (n_changes, n_tests)),
    )


def history_features(
    label_matrix: np.ndarray, run_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative ``(failure_rate, runs, last_failure_age)`` over the canonical order.

    Every value is emitted *before* the change's own outcome is folded in, so nothing
    peeks at the label it is used to predict. The features are only interpretable when
    the order is real; whether to compute them at all is the caller's decision.
    """
    n_c, n_t = label_matrix.shape
    alpha = 1.0  # Laplace prior so an unseen test starts at 0.5
    running_fails = np.zeros(n_t, dtype=np.float64)
    running_runs = np.zeros(n_t, dtype=np.float64)
    last_failure = np.full(n_t, -1, dtype=np.int64)

    failure_rate = np.zeros((n_c, n_t), dtype=np.float32)
    runs_cum = np.zeros((n_c, n_t), dtype=np.float32)
    last_failure_age = np.zeros((n_c, n_t), dtype=np.float32)

    for i in range(n_c):
        failure_rate[i] = (running_fails + alpha) / (running_runs + 2 * alpha)
        runs_cum[i] = running_runs
        age = np.where(last_failure >= 0, i - last_failure, n_c)
        last_failure_age[i] = age
        # Update only after emitting features for change i.
        running_fails += label_matrix[i]
        running_runs += run_matrix[i]
        last_failure = np.where(label_matrix[i] == 1, i, last_failure)

    return failure_rate, runs_cum, last_failure_age


__all__ = [
    "change_query_text",
    "change_query_text_in",
    "change_size",
    "change_size_in",
    "changed_lines",
    "changed_lines_in",
    "coverage_columns",
    "durations_column",
    "filename_stem_match",
    "history_features",
    "n_tests_in_test_file",
    "path_distance",
    "removed_lines",
    "removed_lines_in",
    "test_n_lines",
    "test_n_lines_in",
    "test_n_lines_row",
    "test_n_tokens",
    "test_n_tokens_in",
    "test_n_tokens_row",
]
