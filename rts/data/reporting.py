"""Reporting: describing a dataset, and auditing what a consumer should know.

Reporting is separated from the contract for a specific reason. The first pass put
``describe`` and ``save`` on the dataset, which meant they had to *choose* a train/test
split to report -- so evaluation configuration was published as dataset metadata and
ended up inside the recorded artifacts. Both now take a :class:`~rts.splits.Split`
explicitly, because a dataset cannot know which of its changes were held out: that is
not a fact about the data.

The same mistake in miniature is why :func:`audit` exists. Facts a consumer should act
on -- changes that touch several files and are therefore flattened, a pool whose
constituents disagree about what a test is, a split that discards the temporal reading --
used to be emitted as a side effect of *reading* the dataset, which made the record
describe the call history rather than the data. They are derived here, once, from the
dataset and the split, and returned as values.
"""

from __future__ import annotations

import json

import numpy as np

from .. import config
from . import accessors
from .contract import Dataset, Policy, Warning, Warnings
from .splits import Split, make_split


def audit(ds: Dataset, split: Split | None = None) -> tuple[Warning, ...]:
    """What a consumer should know about ``ds`` before trusting a number from it.

    Deliberately silent about absent capabilities: a feature block already reports each
    column it had to coerce, naming the capability, and :meth:`Dataset.declaration`
    reports the capability set. A third voice saying the same thing is noise rather than
    propagation.
    """
    collected = Warnings()

    multi = accessors.multi_file_changes(ds)
    if multi:
        collected.add(
            "dataset.multi_file_changes_flattened",
            "diff_text",
            f"{len(multi)} change(s) touch more than one file; single-file derived "
            "features read the first path",
            scope="reporting:paths",
        )

    # Divergences the dataset declares about itself: a pool whose constituents disagree
    # on what a test is, a bundle whose diff_text is a concatenation. Declared at the
    # dataset rather than inferred here from its type.
    collected.extend(ds.integrity_notes())

    if split is not None:
        collected.extend(split.warnings)
        if split.shuffle and ds.ordering().value == "observed":
            collected.add(
                "dataset.history_off_for_shuffled_run",
                Policy.EFFECTIVE_ORDER.value,
                "the split shuffled an observed dataset, so history features are off for "
                "this run even though the dataset's own order is real",
                scope="reporting:split",
            )

    return tuple(collected)


def describe(ds: Dataset, split: Split) -> dict:
    """The dataset's shape, under an explicitly chosen split.

    The split is required rather than defaulted: ``train_changes`` / ``test_changes`` /
    ``held_out_faults`` are statements about the run, and a default would let a caller
    publish one run's partition as if it were a property of the data.
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
            "fault_bearing_changes": int(len(faults)),
            "train_changes": int(len(split.train_idx)),
            "test_changes": int(len(split.test_idx)),
            "held_out_faults": int(len(accessors.test_fault_idx(ds, split.test_idx))),
            "killing_tests_per_fault_median": float(np.median(killing_per_change[faults])),
            "killing_tests_per_fault_max": int(killing_per_change[faults].max()),
        }
    )
    if ds.has_capability("coverage"):
        covered_by = accessors.covered(ds)
        out["mean_covered_tests_per_change"] = float(
            np.mean([len(c) for c in covered_by])
        )
        out["changes_with_empty_coverage"] = int(sum(not c for c in covered_by))
        from .populations import sparse_mask

        out["sparse_changes"] = int(sparse_mask(ds).sum())
    return out


def recurrence(ds: Dataset) -> dict:
    """How often ``(file, test)`` pairs recur, which is what makes history features flattered.

    A statistic about the *data* rather than about a sweep, which is why it lives here and not
    in the experiment layer: it needs no scores, no split and no selector. It is recorded
    beside every history-bearing result because a pair that recurs a median of 159 times means
    the cumulative features were measured against a sequence that revisits a pair far more often
    than real evolution would.
    """
    counts = accessors.pair_counts(ds)
    faults = accessors.fault_idx(ds)
    covered_by = accessors.covered(ds)
    paths = accessors.change_paths(ds)
    fault_pairs = np.array(
        [
            counts.get((paths[i], sorted(ds.killing_tests(ds.changes[i]))[0]), 0)
            for i in faults
        ]
    )
    maxpc = np.array(
        [
            max((counts[(paths[i], t)] for t in covered_by[i]), default=0)
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


def describe_starved(ds: Dataset, split: Split, mask: np.ndarray) -> dict:
    """Distribution shape of a filtered subset, to show it is narrow and low."""
    run_counts, fail_counts = accessors.pair_history_counts(ds)
    paths = accessors.change_paths(ds)
    idx = np.flatnonzero(mask)
    keys = []
    for i in idx:
        tests = sorted(ds.killing_tests(ds.changes[i]))
        if tests:
            keys.append((paths[i], tests[0]))
    failures = np.array([fail_counts.get(k, 0) for k in keys], dtype=np.float64)
    runs = np.array([run_counts.get(k, 0) for k in keys], dtype=np.float64)
    held = set(split.test_idx.tolist())
    return {
        "changes": int(mask.sum()),
        "held_out_changes": int(sum(1 for i in idx if i in held)),
        "held_out_faults": int(
            sum(1 for i in idx if i in held and ds.killing_tests(ds.changes[i]))
        ),
        "failure_count_mean": float(failures.mean()) if failures.size else float("nan"),
        "failure_count_sd": float(failures.std()) if failures.size else float("nan"),
        "failure_count_max": int(failures.max()) if failures.size else 0,
        "run_count_mean": float(runs.mean()) if runs.size else float("nan"),
        "run_count_max": int(runs.max()) if runs.size else 0,
    }


def save(ds: Dataset, split: Split, out_dir=None) -> None:
    """Persist the dataset and the split that was used with it, side by side."""
    out = config.ensure_artifacts_dir() if out_dir is None else out_dir
    payload = {
        **ds.declaration(),
        "audit": [w.to_dict() for w in audit(ds, split)],
        "split": {
            "fraction": split.fraction,
            "shuffle": split.shuffle,
            "seed": split.seed,
            "effective_ordering": split.effective_ordering.value,
        },
        "test_ids": ds.test_ids,
        "change_ids": [ds.change_id(c) for c in ds.changes],
        "files": [list(ds.files(c)) for c in ds.changes],
        "killing_tests": [sorted(ds.killing_tests(c)) for c in ds.changes],
        "ran_tests": [sorted(ds.ran_tests(c)) for c in ds.changes],
        "covered": (
            [sorted(t) for t in accessors.covered(ds)]
            if ds.has_capability("coverage")
            else None
        ),
    }
    np.savez_compressed(
        out / "dataset.npz",
        labels=accessors.labels(ds),
        runs=accessors.runs(ds),
        train_idx=split.train_idx,
        test_idx=split.test_idx,
    )
    (out / "dataset.json").write_text(json.dumps(payload))
    print(f"[dataset] wrote {out / 'dataset.npz'} and dataset.json")


def split_for(ds: Dataset, **kwargs) -> Split:
    """The study's default split, spelled once so call sites read the same."""
    return make_split(ds, **kwargs)


__all__ = ["audit", "describe", "describe_starved", "recurrence", "save", "split_for"]
