"""Derived accessors: harness functions over the dataset contract.

These were methods on :class:`rts.data.contract.Dataset` in the first pass, which is why
that class grew to twenty-odd members and why one file had to hold everything.
They are free functions because almost all of them have between zero and four
consumers: keeping them on the contract made the interface look far larger than the
contract actually is, and hid which accessors are load-bearing.

What stayed on the contract is the *index-alignment convention* -- ``test_ids`` and
``test_index`` -- because every matrix in the harness obeys it, so it is part of what a
dataset supplies rather than a statistic computed from it.

Three defects the move to free functions fixes:

* **Nine hand-written copies of the memo idiom** are now one call to
  :meth:`rts.data.contract.Dataset.cached`. A subclass that omitted them all, as
  ``PooledDataset`` did, no longer silently loses every cache -- ``n_changes`` rebuilt
  the entire change list just to take a length.
* **Mutable internals were handed out by reference.** ``labels``, ``execution_matrix`` and
  ``fault_mask`` are now frozen on construction, so ``ds.labels[i, j] = 1`` raises
  instead of corrupting every later reader.
* **An expensive ``test_source`` was read twice per test** (once for the line count,
  once for the token count) with nothing in the contract to share a cache. There is now
  one memoised accessor that everything reads.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .contract import Capability, CapabilityMissing, Dataset, Requirement, TestId

# --- labels and runs -------------------------------------------------------


def _build_label_matrix(ds: Dataset) -> np.ndarray:
    out = np.zeros((ds.n_changes, ds.n_tests), dtype=np.uint8)
    index = ds.test_index
    for i, change in enumerate(ds.changes):
        for test in ds.killing_tests(change):
            j = index.get(test)
            if j is not None:
                out[i, j] = 1
    out.setflags(write=False)
    return out


def _build_execution_matrix(ds: Dataset) -> np.ndarray:
    out = np.zeros((ds.n_changes, ds.n_tests), dtype=np.uint8)
    index = ds.test_index
    for i, change in enumerate(ds.changes):
        for test in ds.executed_tests(change):
            j = index.get(test)
            if j is not None:
                out[i, j] = 1
    out.setflags(write=False)
    return out


def labels(ds: Dataset) -> np.ndarray:
    """``[n_changes, n_tests]`` uint8: 1 where the test failed against the change.

    Read-only: this is the evaluation contract's label matrix, and a caller that
    mutated it would change every later measurement.
    """
    return ds.cached("labels", lambda: _build_label_matrix(ds))


def execution_matrix(ds: Dataset) -> np.ndarray:
    """``[n_changes, n_tests]`` uint8: 1 where the test was actually executed.

    Harness-internal rather than part of the advertised surface: the only readers are
    the cumulative temporal block and the pair-recurrence subset, and both want it
    for the same reason -- to separate "passed" from "never ran".
    """
    return ds.cached("execution", lambda: _build_execution_matrix(ds))


# --- paths and coverage ----------------------------------------------------


def change_paths(ds: Dataset) -> tuple[str, ...]:
    """One path per change, for the single-file derived features.

    ``files`` is plural because a change may touch several; this is the flattened view
    those features read. It does not warn -- which change was flattened is a fact about
    the data, reported by :func:`multi_file_changes` and recorded by
    :func:`rts.reporting.audit`, rather than a side effect of reading.
    """
    def build() -> tuple[str, ...]:
        out = []
        for change in ds.changes:
            paths = ds.files(change)
            out.append(paths[0] if paths else "")
        return tuple(out)

    return ds.cached("change_paths", build)


def multi_file_changes(ds: Dataset) -> tuple[int, ...]:
    """Indices of changes that touch more than one file, and are thus flattened."""
    return tuple(i for i, c in enumerate(ds.changes) if len(ds.files(c)) != 1)


def coverage_sets(ds: Dataset) -> tuple[frozenset[TestId], ...]:
    """Tests covering each change, index-aligned. Requires ``coverage``."""
    if not ds.has_capability(Capability.COVERAGE):
        raise CapabilityMissing(ds.name, Capability.COVERAGE)
    return ds.cached("coverage_sets", lambda: tuple(ds.coverage(c) for c in ds.changes))


# --- faults ----------------------------------------------------------------


def fault_mask(ds: Dataset) -> np.ndarray:
    """Boolean per change: has at least one killing test."""
    def build() -> np.ndarray:
        out = np.array([bool(ds.killing_tests(c)) for c in ds.changes], dtype=bool)
        out.setflags(write=False)
        return out

    return ds.cached("fault_mask", build)


def fault_idx(ds: Dataset) -> np.ndarray:
    """Changes that can be caught: those with a killing test."""
    return np.flatnonzero(fault_mask(ds)).astype(np.int64)


def test_fault_idx(ds: Dataset, rows: np.ndarray) -> np.ndarray:
    """Fault-bearing changes inside the evaluation window ``rows``."""
    mask = fault_mask(ds)
    return np.array([int(i) for i in rows if mask[int(i)]], dtype=np.int64)


def change_index(ds: Dataset) -> dict[str, int]:
    """Change id -> row. The key is :meth:`rts.data.contract.Dataset.change_id`."""
    return ds.cached("change_index", lambda: {ds.change_id(c): i for i, c in enumerate(ds.changes)})


# --- candidate sets --------------------------------------------------------


def candidate_sets(ds: Dataset, mode: str = "full") -> np.ndarray:
    """Which ``(change, test)`` pairs are eligible for selection.

    ``full``    -- every test in the pool, the realistic RTS setting.
    ``covered`` -- only tests covering the change, plus that change's killing tests as
                   a safety net. Requires ``coverage``.
    ``own``     -- the dataset's own per-change pool, for a corpus where each change's
                   candidate sets come from its own suite ``(Dataset.own_candidate_pool)``.

    All rankers are evaluated on the same mask so the comparison stays fair.
    """
    if mode == "full":
        return np.ones((ds.n_changes, ds.n_tests), dtype=bool)
    if mode == "own":
        pool = ds.own_candidate_pool()
        if pool is None:
            raise CapabilityMissing(ds.name, "own_candidate_pool")
        return pool
    if mode != "coverage_restricted":
        raise ValueError(f"unknown candidate mode: {mode!r}")
    if not ds.has_capability(Capability.COVERAGE):
        raise CapabilityMissing(ds.name, Capability.COVERAGE)
    mask = np.zeros((ds.n_changes, ds.n_tests), dtype=bool)
    index = ds.test_index
    coverage_by = coverage_sets(ds)
    for i, change in enumerate(ds.changes):
        for test in coverage_by[i]:
            j = index.get(test)
            if j is not None:
                mask[i, j] = True
        for test in ds.killing_tests(change):
            j = index.get(test)
            if j is not None:
                mask[i, j] = True
    return mask


def candidate_counts(ds: Dataset, candidate_mask: np.ndarray) -> np.ndarray:
    return candidate_mask.sum(axis=1).astype(np.int64)


# --- pair recurrence -------------------------------------------------------


def pair_cooccurrence_counts(ds: Dataset) -> dict[tuple[str, str], int]:
    """How often each ``(file, test)`` combination recurs across changes.

    A pair that occurs once is a combination the structured models have no history for.
    Requires ``coverage``.
    """
    counts: dict[tuple[str, str], int] = {}
    coverage_by = coverage_sets(ds)
    for i in range(ds.n_changes):
        path = change_paths(ds)[i]
        for test in coverage_by[i]:
            key = (path, test)
            counts[key] = counts.get(key, 0) + 1
    return counts


def low_cooccurrence_mask(ds: Dataset, max_pair_count: int = 1) -> np.ndarray:
    """Changes whose every ``(file, test)`` combination recurs at most this often.

    The "low co-occurrence" evaluation condition: a proxy for software evolution where a
    file and a test are not repeatedly paired. Note that it also removes exactly the
    repeated co-occurrences the structured temporal features depend on, so it is a
    robustness check, not a neutral split.
    """
    counts = pair_cooccurrence_counts(ds)
    coverage_by = coverage_sets(ds)
    paths = change_paths(ds)
    mask = np.zeros(ds.n_changes, dtype=bool)
    for i in range(ds.n_changes):
        pairs = [(paths[i], t) for t in coverage_by[i]]
        if not pairs:
            continue
        mask[i] = max(counts[p] for p in pairs) <= max_pair_count
    return mask


def pair_history_counts(ds: Dataset) -> tuple[dict, dict]:
    """Per ``(file, test)``: how many times the test ran, and failed, for that file.

    Counts are over the whole dataset. For a change's own killing pair the failure count
    therefore includes the change itself, so ``failures <= 1`` means the pair has *no
    prior failure history*.
    """
    execution, label_matrix = execution_matrix(ds), labels(ds)
    paths = change_paths(ds)
    test_ids = ds.test_ids
    run_counts: dict[tuple[str, str], int] = {}
    fail_counts: dict[tuple[str, str], int] = {}
    for i, path in enumerate(paths):
        for j in np.flatnonzero(execution[i]):
            key = (path, test_ids[int(j)])
            run_counts[key] = run_counts.get(key, 0) + 1
        for j in np.flatnonzero(label_matrix[i]):
            key = (path, test_ids[int(j)])
            fail_counts[key] = fail_counts.get(key, 0) + 1
    return run_counts, fail_counts


# --- test source -----------------------------------------------------------


def test_source(ds: Dataset, test: TestId) -> str | None:
    """The test's source text, memoised on the dataset.

    The contract's raw accessor returns ``None`` for an unlocatable test and ``""`` for an
    empty one, and keeps the two apart. This wrapper exists purely so that the several
    features derived from one test's text share a single read: without it the block read
    every test twice, which is free for an ``lru_cache``-backed checkout and not free
    for a dataset that reads from disk or a remote.

    The cache is keyed by test id *on this dataset instance*, which is the opposite of
    the module-level node-id cache the design forbids: two datasets over two checkouts
    can hold the same node id and must not see each other's text.
    """
    return ds.cached(f"test_source:{test}", lambda: ds.test_source(test))


# --- inputs --------------------------------------------------------------


@dataclass(frozen=True)
class Input:
    """One named thing a computation may be handed, and what reading it requires."""

    name: str
    requires: frozenset[Requirement]
    resolve: Callable[[Dataset], Any]
    note: str = ""


def _pair_history(ds: Dataset) -> tuple[dict, dict]:
    return pair_history_counts(ds)


def _catalogue() -> dict[str, Input]:
    r = Requirement
    entries = (
        Input("changes", frozenset(), lambda ds: ds.changes),
        Input("test_ids", frozenset(), lambda ds: ds.test_ids),
        Input("paths", frozenset({r.DIFF_TEXT}), change_paths),
        Input("diff", frozenset({r.DIFF_TEXT}), lambda ds: ds.diff_text),
        Input("test_source", frozenset(), lambda ds: lambda t: test_source(ds, t)),
        Input("labels", frozenset({r.LABELS}), labels),
        Input("execution", frozenset({r.LABELS}), execution_matrix),
        Input("faults", frozenset({r.LABELS}), fault_mask),
        Input(
            "killing",
            frozenset({r.LABELS}),
            lambda ds: tuple(frozenset(ds.killing_tests(c)) for c in ds.changes),
            "per-change killing tests, sorted",
        ),
        Input("coverage", frozenset({r.COVERAGE}), coverage_sets),
        Input("durations", frozenset({r.DURATIONS}), lambda ds: ds.durations()),
        Input(
            "pair_runs",
            frozenset({r.LABELS}),
            lambda ds: _pair_history(ds)[0],
            "executions per (file, test)",
        ),
        Input(
            "pair_failures",
            frozenset({r.LABELS}),
            lambda ds: _pair_history(ds)[1],
            "failures per (file, test)",
        ),
        Input("pairs", frozenset({r.COVERAGE}), pair_cooccurrence_counts),
    )
    return {entry.name: entry for entry in entries}


#: The one catalogue of inputs a computation may name. A block group's ``needs``, a
#: subset's ``needs`` and the derived features all resolve through it, so a
#: requirement set is *derived* from what is read rather than asserted beside it.
INPUTS: dict[str, Input] = _catalogue()


def requirements_for(needs: Sequence[str]) -> frozenset[Requirement]:
    """The requirements implied by a list of input names."""
    out: set[Requirement] = set()
    for name in needs:
        try:
            out |= INPUTS[name].requires
        except KeyError:
            raise KeyError(
                f"unknown input {name!r}; known: {sorted(INPUTS)}"
            ) from None
    return frozenset(out)


def inputs(ds: Dataset) -> dict[str, Any]:
    """Everything a computation may be handed, resolved for this dataset.

    Only what the dataset can satisfy is present. A computation that names absent
    inputs is never called, so it cannot raise a capability error deep inside
    arithmetic -- which is what makes :func:`requirements_for` a derivation rather than
    an assertion.
    """
    available = ds.available_requirements()
    return {
        entry.name: entry.resolve(ds)
        for entry in INPUTS.values()
        if entry.requires <= available
    }


def resolve(ds: Dataset, needs: Sequence[str]) -> dict[str, Any]:
    """Resolve exactly ``needs``. Raises if the dataset cannot supply any of them."""
    _, missing = satisfies(ds, needs)
    if missing:
        raise CapabilityMissing(ds.name, missing[0].value)
    everything = inputs(ds)
    return {name: everything[name] for name in needs}


def satisfies(ds: Dataset, needs: Sequence[str]) -> tuple[bool, tuple[Requirement, ...]]:
    """Whether ``ds`` can supply ``needs``, and which requirements are missing."""
    required = requirements_for(needs)
    missing = ds.missing_requirements(required)
    return (not missing), missing


__all__ = [
    "INPUTS",
    "Input",
    "candidate_counts",
    "candidate_sets",
    "change_index",
    "change_paths",
    "coverage_sets",
    "fault_idx",
    "fault_mask",
    "labels",
    "inputs",
    "multi_file_changes",
    "pair_cooccurrence_counts",
    "pair_history_counts",
    "requirements_for",
    "resolve",
    "execution_matrix",
    "satisfies",
    "low_cooccurrence_mask",
    "test_fault_idx",
    "test_source",
]
