"""Assemble the synthetic change history into an evaluable dataset.

Shape of the problem
--------------------
* **Changes** are mutants. The mutant population has no intrinsic temporal
  order, so history is *imposed*: a fixed-seed permutation defines the sequence.
  This is a deliberate, documented limitation (see ``implementation.md``); it is
  also why the recency baseline is expected to be uninformative.
* **Candidates** are the full collected suite (1187 tests). RTS ranks the whole
  suite for each change.
* **Labels** mark the tests that actually failed. mutmut executes only the tests
  associated with the mutated function, so a test outside that set is *assumed*
  not to fail. That assumption is mutmut's, inherited here, and recorded in
  ``ran`` so downstream code can distinguish "passed" from "never ran".

Note on negatives: survived mutants (340 of them) contribute no killing test and
therefore no fault. They are retained because they exercise the ranking problem
and give the structured models negative signal, but they are excluded from the
recall denominator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import artifacts, config, source
from .artifacts import Change


@dataclass
class Dataset:
    changes: list[Change]
    test_ids: list[str]
    test_index: dict[str, int]

    # [n_changes, n_tests]
    labels: np.ndarray  # 1 where the test failed against this change
    ran: np.ndarray  # 1 where mutmut actually executed the test

    # Per-change derived views
    covered: list[frozenset[str]]  # tests covering the mutated function
    files: list[str]
    func_keys: list[str]

    train_idx: np.ndarray
    test_idx: np.ndarray

    # --- convenience ------------------------------------------------------

    @property
    def n_changes(self) -> int:
        return len(self.changes)

    @property
    def n_tests(self) -> int:
        return len(self.test_ids)

    @property
    def fault_idx(self) -> np.ndarray:
        """Changes that can actually be caught, i.e. have at least one killing test."""
        return np.array(
            [i for i, c in enumerate(self.changes) if c.killing_tests], dtype=np.int64
        )

    @property
    def test_fault_idx(self) -> np.ndarray:
        """Fault-bearing changes inside the held-out window."""
        held = set(self.test_idx.tolist())
        return np.array([i for i in self.fault_idx if i in held], dtype=np.int64)

    def killing_tests(self, i: int) -> tuple[str, ...]:
        return self.changes[i].killing_tests

    @property
    def change_index(self) -> dict[str, int]:
        return {c.change_id: i for i, c in enumerate(self.changes)}

    def candidate_counts(self, candidates: np.ndarray) -> np.ndarray:
        return candidates.sum(axis=1).astype(np.int64)

    def pair_counts(self) -> dict[tuple[str, str], int]:
        """How often each (file, test) combination recurs across changes.

        Used by the sparse-change filter: a pair that occurs once is a
        combination the structured models have no history for.
        """
        counts: dict[tuple[str, str], int] = {}
        for i, change in enumerate(self.changes):
            for test in self.covered[i]:
                key = (change.file, test)
                counts[key] = counts.get(key, 0) + 1
        return counts

    def sparse_mask(self, max_pair_count: int = 1) -> np.ndarray:
        """Changes whose every (file, test) combination recurs at most this often.

        This is the "sparse" evaluation arm from ``plan.md``: a proxy for software
        evolution where a file and a test are not repeatedly paired. Note that it
        also removes exactly the repeated co-occurrences the structured history
        features depend on, so it is a robustness check, not a neutral split.
        """
        counts = self.pair_counts()
        mask = np.zeros(self.n_changes, dtype=bool)
        for i, change in enumerate(self.changes):
            pairs = [(change.file, t) for t in self.covered[i]]
            if not pairs:
                continue
            mask[i] = max(counts[p] for p in pairs) <= max_pair_count
        return mask


def build(
    train_fraction: float = config.DEFAULT_TRAIN_FRACTION,
    seed: int = config.SEED,
) -> Dataset:
    changes = artifacts.build_changes()
    if not changes:
        raise SystemExit("no changes reconstructed; run mutmut first (see implementation.md)")

    # Impose a synthetic temporal order.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(changes))
    changes = [changes[i] for i in order]

    test_ids = artifacts.all_test_nodeids()
    test_index = {t: i for i, t in enumerate(test_ids)}
    coverage = artifacts.coverage_map()

    n_c, n_t = len(changes), len(test_ids)
    labels = np.zeros((n_c, n_t), dtype=np.uint8)
    ran = np.zeros((n_c, n_t), dtype=np.uint8)
    covered: list[frozenset[str]] = []

    for i, change in enumerate(changes):
        for test in change.killing_tests:
            j = test_index.get(test)
            if j is not None:
                labels[i, j] = 1
        for test in change.ran_tests:
            j = test_index.get(test)
            if j is not None:
                ran[i, j] = 1
        covered.append(frozenset(t for t in coverage.get(change.func_key, ()) if t in test_index))

    split = int(round(n_c * train_fraction))
    train_idx = np.arange(split, dtype=np.int64)
    test_idx = np.arange(split, n_c, dtype=np.int64)

    return Dataset(
        changes=changes,
        test_ids=test_ids,
        test_index=test_index,
        labels=labels,
        ran=ran,
        covered=covered,
        files=[c.file for c in changes],
        func_keys=[c.func_key for c in changes],
        train_idx=train_idx,
        test_idx=test_idx,
    )


def candidate_mask(ds: Dataset, mode: str = "full") -> np.ndarray:
    """Which (change, test) pairs are eligible for selection.

    ``full``    -- every collected test, the realistic RTS setting.
    ``covered`` -- only tests covering the mutated function, plus that change's
                   killing tests as a safety net. This is the candidate set the
                   SemIf reranker can afford to score, and restricting to it
                   cannot lose a fault: every killing test covers the mutated
                   function (verified across all 2311 faults).

    All selectors are evaluated on the same mask so the comparison stays fair.
    """
    if mode == "full":
        return np.ones((ds.n_changes, ds.n_tests), dtype=bool)
    if mode != "covered":
        raise ValueError(f"unknown candidate mode: {mode!r}")

    mask = np.zeros((ds.n_changes, ds.n_tests), dtype=bool)
    for i, covered in enumerate(ds.covered):
        for test in covered:
            mask[i, ds.test_index[test]] = True
        # Safety net: never exclude a known killing test.
        for test in ds.changes[i].killing_tests:
            j = ds.test_index.get(test)
            if j is not None:
                mask[i, j] = True
    return mask


def describe(ds: Dataset) -> dict:
    per_change_kill = np.array([len(c.killing_tests) for c in ds.changes])
    faults = ds.fault_idx
    return {
        "changes": ds.n_changes,
        "tests": ds.n_tests,
        "killed": int(sum(c.killed for c in ds.changes)),
        "survived": int(sum(c.survived for c in ds.changes)),
        "fault_bearing_changes": len(faults),
        "train_changes": len(ds.train_idx),
        "test_changes": len(ds.test_idx),
        "held_out_faults": len(ds.test_fault_idx),
        "killing_tests_per_fault_median": float(np.median(per_change_kill[faults])),
        "killing_tests_per_fault_max": int(per_change_kill[faults].max()),
        "mean_covered_tests_per_change": float(
            np.mean([len(c) for c in ds.covered])
        ),
        "changes_with_empty_coverage": int(sum(not c for c in ds.covered)),
        "sparse_changes": int(ds.sparse_mask().sum()),
    }


def save(ds: Dataset, out_dir=None) -> None:
    out = config.ensure_artifacts_dir() if out_dir is None else out_dir
    np.savez_compressed(
        out / "dataset.npz",
        labels=ds.labels,
        ran=ds.ran,
        train_idx=ds.train_idx,
        test_idx=ds.test_idx,
    )
    import json

    (out / "dataset.json").write_text(
        json.dumps(
            {
                "test_ids": ds.test_ids,
                "change_ids": [c.change_id for c in ds.changes],
                "files": ds.files,
                "func_keys": ds.func_keys,
                "covered": [sorted(c) for c in ds.covered],
                "killing_tests": [list(c.killing_tests) for c in ds.changes],
                "ran_tests": [list(c.ran_tests) for c in ds.changes],
            }
        )
    )
    print(f"[dataset] wrote {out / 'dataset.npz'} and dataset.json")


if __name__ == "__main__":
    ds = build()
    for key, value in describe(ds).items():
        print(f"{key:>32}: {value}")

    # Sanity: every killing test must be inside the coverage set for that change.
    outside = 0
    for i, change in enumerate(ds.changes):
        for test in change.killing_tests:
            if test not in ds.covered[i]:
                outside += 1
    print(f"{'killing tests outside coverage':>32}: {outside}")
