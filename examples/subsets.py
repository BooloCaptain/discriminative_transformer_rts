"""The study's evaluation subsets, declared over the kernel mechanism.

``detectable`` is the kernel's default averaging subset; the three here are this study's
deployment proxies. They are values composed onto the kernel's registry, so the kernel needs
no knowledge of "cold start" or "low co-occurrence" -- those are study concepts, not
properties of a dataset.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from rts.data import accessors
from rts.data.contract import Dataset
from rts.data.subsets import DEFAULT, Subset, SubsetRegistry


def _no_prior_failure(
    inputs: Mapping[str, Any],
    max_failures: int,
    max_runs: int | None = None,
) -> np.ndarray:
    """Changes whose killing ``(file, test)`` pair has at most ``max_failures`` failures.

    Counts include the change itself, so ``max_failures=1`` means the pair has *no prior
    failure history*. Keyed on failure history rather than coverage co-occurrence, because
    failure history is what the structured models actually exploit.
    """
    fails = inputs["pair_failures"]
    run_counts = inputs["pair_runs"]
    paths = inputs["paths"]
    killing = inputs["killing"]
    mask = np.zeros(len(paths), dtype=bool)
    for i, tests in enumerate(killing):
        if not tests:
            continue
        key = (paths[i], sorted(tests)[0])
        if fails.get(key, 0) > max_failures:
            continue
        if max_runs is not None and run_counts.get(key, 0) > max_runs:
            continue
        mask[i] = True
    return mask


def _low_pair_recurrence(
    inputs: Mapping[str, Any], max_pair_count: int
) -> np.ndarray:
    """Changes whose every ``(file, test)`` pair recurs at most this often.

    The "low co-occurrence" condition: a proxy for software evolution where a file and a test
    are not repeatedly paired. It also removes exactly the repeated co-occurrences the history
    features depend on, so it is a robustness check, not a neutral split.
    """
    counts = inputs["pairs"]
    coverage = inputs["coverage"]
    paths = inputs["paths"]
    mask = np.zeros(len(paths), dtype=bool)
    for i, tests in enumerate(coverage):
        pairs = [(paths[i], t) for t in tests]
        if not pairs:
            continue
        mask[i] = max(counts[p] for p in pairs) <= max_pair_count
    return mask


# --- the study's subsets ----------------------------------------------


NO_PRIOR_FAILURE = Subset(
    name="no_prior_failure",
    note="killing (file, test) pairs with no earlier failure",
    needs=("pair_failures", "paths", "killing"),
    predicate=lambda inputs: _no_prior_failure(inputs, max_failures=1),
)

COLD_START = Subset(
    name="cold_start",
    note="cold-start deployment proxy: killing pairs with almost no failure history",
    needs=("pair_failures", "pair_runs", "paths", "killing"),
    predicate=lambda inputs: _no_prior_failure(inputs, max_failures=2),
)

LOW_COOCCURRENCE = Subset(
    name="low_cooccurrence",
    note="changes whose (file, test) pairs rarely recur",
    needs=("pairs", "coverage", "paths"),
    predicate=lambda inputs: _low_pair_recurrence(inputs, max_pair_count=1),
)


#: The study's subsets: the kernel default plus the study's three proxies.
STUDY = DEFAULT + SubsetRegistry((NO_PRIOR_FAILURE, COLD_START, LOW_COOCCURRENCE))


# --- parameterised entry points onto the same predicates ------------------


def cold_start_mask(
    ds: Dataset, max_failures: int = 2, max_runs: int | None = None
) -> np.ndarray:
    """The cold-start proxy, parameterised. Calls the subset's own predicate."""
    return _no_prior_failure(accessors.inputs(ds), max_failures, max_runs)


def low_cooccurrence_mask(ds: Dataset, max_pair_count: int = 1) -> np.ndarray:
    """The low-co-occurrence proxy, parameterised. Calls the subset's own predicate."""
    return _low_pair_recurrence(accessors.inputs(ds), max_pair_count)


def cold_start(max_failures: int) -> Subset:
    """The cold-start proxy as a declared subset, for one threshold.

    Counts include the change itself, so ``max_failures=1`` means the killing pair has no prior
    failure history. The name carries the threshold, because two thresholds are two different
    subsets and a sweep must not conflate them.
    """
    return Subset(
        name=f"cold_start{max_failures}",
        note=(
            f"changes whose killing (file, test) pair has at most {max_failures} failures, "
            "counting the change itself"
        ),
        needs=("pair_failures", "pair_runs", "paths", "killing"),
        predicate=lambda inputs: _no_prior_failure(inputs, max_failures),
    )


def low_cooccurrence(max_pair_count: int) -> Subset:
    """The low-co-occurrence proxy as a declared subset, for one threshold.

    The name carries the threshold, because two thresholds are two different subsets and a
    sweep must not conflate them.
    """
    return Subset(
        name=f"low_cooccurrence{max_pair_count}",
        note=(
            f"changes whose (file, test) pairs all recur at most {max_pair_count} times; the "
            "proxy for software evolution where a file and a test are not repeatedly paired"
        ),
        needs=("pairs", "coverage", "paths"),
        predicate=lambda inputs: _low_pair_recurrence(inputs, max_pair_count),
    )


__all__ = [
    "COLD_START",
    "LOW_COOCCURRENCE",
    "NO_PRIOR_FAILURE",
    "STUDY",
    "cold_start",
    "cold_start_mask",
    "low_cooccurrence",
    "low_cooccurrence_mask",
]
