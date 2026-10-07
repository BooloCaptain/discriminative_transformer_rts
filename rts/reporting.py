"""Reporting: describing a dataset, and auditing what a consumer should know.

Two rules decide what is here.

**A dataset cannot describe a run.** ``describe`` takes a :class:`~rts.data.splits.Split`
*explicitly*, because ``train_changes`` / ``held_out_faults`` are statements about a run, and a
defaulted split would publish one run's partition as a property of the data. There is
deliberately no ``save`` here either: persisting a dataset is not a statistic, and the earlier
version's ``save`` wrote evaluation configuration into a dataset artifact.

**Diagnostics are values, not log lines.** Facts a consumer should act on -- changes that touch
several files and are therefore flattened, a pool whose constituents disagree about what a test
is, a split that discards the temporal reading -- are *returned* by :func:`audit`, derived once
from the dataset and the split, rather than emitted as a side effect of reading the dataset. A
record that described the call history instead of the data was the failure mode.

Statistics that are about the data rather than about a sweep (change/test counts, pair
recurrence) live beside these, so a report can record them without rebuilding anything.
"""

from __future__ import annotations

import numpy as np

from .data import accessors
from .data.contract import Dataset, Diagnostic, Diagnostics, Policy
from .data.splits import Split


def audit(ds: Dataset, split: Split | None = None) -> tuple[Diagnostic, ...]:
    """What a consumer should know about ``ds`` before trusting a number from it.

    Deliberately silent about absent capabilities: a feature block already reports each column
    it had to coerce, naming the capability, and :meth:`Dataset.declaration` reports the
    capability set. A third voice saying the same thing is noise rather than propagation.
    """
    collected = Diagnostics()

    multi = accessors.multi_file_changes(ds)
    if multi:
        collected.add(
            "dataset.multi_file_changes_flattened",
            "diff_text",
            f"{len(multi)} change(s) touch more than one file; single-file derived "
            "features read the first path",
            scope="reporting:paths",
        )

    # Divergences the dataset declares about itself: a pool whose constituents disagree on
    # what a test is, a bundle whose diff text is a concatenation. Declared at the dataset
    # rather than inferred here from its type.
    collected.extend(ds.integrity_notes())

    if split is not None:
        collected.extend(split.diagnostics)
        if split.shuffle and ds.ordering().value == "natural":
            collected.add(
                "dataset.temporal_off_for_shuffled_run",
                Policy.EFFECTIVE_ORDER.value,
                "the split shuffled a natural-order dataset, so the temporal features are off "
                "for this run even though the dataset's own order is real",
                scope="reporting:split",
            )

    return tuple(collected)


def describe(ds: Dataset, split: Split) -> dict:
    """The dataset's shape, under an explicitly chosen split.

    The split is required rather than defaulted: ``train_changes`` / ``test_changes`` /
    ``held_out_faults`` are statements about the run, and a default would let a caller publish
    one run's partition as if it were a property of the data.
    """
    killing_per_change = np.array(
        [len(ds.killing_tests(c)) for c in ds.changes], dtype=np.int64
    )
    faults = accessors.fault_idx(ds)
    out: dict = {
        "changes": ds.n_changes,
        "tests": ds.n_tests,
    }
    out.update(ds.source_counts())
    out.update(
        {
            "detectable_changes": int(len(faults)),
            "train_changes": int(len(split.train_idx)),
            "test_changes": int(len(split.test_idx)),
            "held_out_faults": int(len(accessors.test_fault_idx(ds, split.test_idx))),
            "killing_tests_per_fault_median": float(np.median(killing_per_change[faults])),
            "killing_tests_per_fault_max": int(killing_per_change[faults].max()),
        }
    )
    if ds.has_capability("coverage"):
        coverage_by = accessors.coverage_sets(ds)
        out["mean_coverage_set_size"] = float(np.mean([len(c) for c in coverage_by]))
        out["changes_with_empty_coverage"] = int(sum(not c for c in coverage_by))
    return out


def recurrence(ds: Dataset) -> dict:
    """How often ``(file, test)`` pairs recur, which is what makes temporal features flattered.

    A statistic about the *data* rather than about a sweep, which is why it lives here and not in
    the experiment layer: it needs no scores, no split and no ranker. It is recorded beside a
    temporal result because a pair that recurs a median of 159 times means the cumulative
    features were measured against a sequence that revisits a pair far more often than real
    evolution would.
    """
    counts = accessors.pair_cooccurrence_counts(ds)
    faults = accessors.fault_idx(ds)
    coverage_by = accessors.coverage_sets(ds)
    paths = accessors.change_paths(ds)
    fault_pairs = np.array(
        [
            counts.get((paths[i], sorted(ds.killing_tests(ds.changes[i]))[0]), 0)
            for i in faults
        ]
    )
    maxpc = np.array(
        [
            max((counts[(paths[i], t)] for t in coverage_by[i]), default=0)
            for i in range(ds.n_changes)
        ]
    )
    return {
        "killing_pair_count_median": float(np.median(fault_pairs)),
        "killing_pair_count_mean": float(fault_pairs.mean()),
        "killing_pair_count_max": float(fault_pairs.max()),
        "per_change_max_pair_count_p05": float(np.percentile(maxpc, 5)),
        "per_change_max_pair_count_p50": float(np.percentile(maxpc, 50)),
    }


__all__ = ["audit", "describe", "recurrence"]
