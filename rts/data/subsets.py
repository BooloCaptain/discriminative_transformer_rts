"""Evaluation subsets: named subsets of the evaluation window.

A subset is to ``rows`` what a derived feature is to pairs -- a harness function
over the contract, defined once and applicable to any dataset. Three things follow, and
the first pass got all three wrong:

**Requirements are derived, not asserted.** A subset names the inputs its
predicate reads; the requirements come from :data:`rts.data.accessors.INPUTS`. The old
version hand-wrote a ``requires`` set beside a predicate and nothing checked the two
agreed. Declaring too little raised a capability error from inside the predicate instead
of returning :class:`Undefined`; declaring too much reported a working subset
unavailable. Both were silent, and both are now unrepresentable -- the predicate is
handed the inputs, so it cannot read what it did not declare.

**Unavailable is not empty.** A subset the dataset cannot support returns
:class:`Undefined` carrying the unmet requirement, never an empty array.

**One implementation per predicate.** ``no_prior_failure`` and ``cold_start_mask`` were two
implementations of one idea that had already drifted apart (only one supported
``max_runs``). There is now one function each, with a declared subset and a
parameterised helper as two entry points onto it.

The registry is a *value* (:class:`SubsetRegistry`) rather than a module-level dict
that callers mutate or extend by editing this file. A caller with a different vocabulary
builds its own registry and passes the :class:`Subset` object directly; the study's
own set is :data:`STUDY`, and it lives here rather than in the contract because
"cold start" and "low pair recurrence" are study concepts, not properties of a dataset.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import accessors
from .contract import Dataset, Requirement, Undefined

Predicate = Callable[[Mapping[str, Any]], np.ndarray]


@dataclass(frozen=True)
class Subset:
    """A named subset of a dataset's change rows that a metric is averaged over."""

    name: str
    note: str
    needs: tuple[str, ...]
    predicate: Predicate

    def requires(self) -> frozenset[Requirement]:
        """The requirements implied by the inputs this subset reads."""
        return accessors.requirements_for(self.needs)

    def unavailable(self, ds: Dataset) -> Undefined | None:
        """Why this subset cannot exist on ``ds``, or ``None`` if it can."""
        missing = ds.missing_requirements(self.requires())
        if not missing:
            return None
        return Undefined(
            requirement=missing[0].value,
            note=(
                f"subset {self.name!r} reads {', '.join(self.needs)}, which needs "
                f"{', '.join(m.value for m in missing)}; dataset {ds.name!r} does not "
                "provide it"
            ),
        )

    def mask(self, ds: Dataset) -> np.ndarray | Undefined:
        """Boolean mask over changes, or :class:`Undefined` when unavailable."""
        unavailable = self.unavailable(ds)
        if unavailable is not None:
            return unavailable
        inputs = accessors.inputs(ds)
        mask = np.asarray(self.predicate(inputs), dtype=bool)
        if mask.shape != (ds.n_changes,):
            raise ValueError(
                f"subset {self.name!r} produced shape {mask.shape}, expected "
                f"{(ds.n_changes,)}"
            )
        return mask

    def rows(self, ds: Dataset, rows: np.ndarray) -> np.ndarray | Undefined:
        """``rows`` restricted to the subset, or :class:`Undefined`."""
        mask = self.mask(ds)
        if isinstance(mask, Undefined):
            return mask
        return np.array([int(r) for r in rows if mask[int(r)]], dtype=np.int64)

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "needs": list(self.needs),
            "requires": sorted(r.value for r in self.requires()),
            "note": self.note,
        }


# --- predicate implementations (one each) ---------------------------------


def _fault_bearing(inputs: Mapping[str, Any]) -> np.ndarray:
    return inputs["faults"]


def _no_prior_failure(
    inputs: Mapping[str, Any],
    max_failures: int,
    max_runs: int | None = None,
) -> np.ndarray:
    """Changes whose killing ``(file, test)`` pair has at most ``max_failures`` failures.

    Counts include the change itself, so ``max_failures=1`` means the pair has *no prior
    failure history*. Keyed on failure history rather than coverage co-occurrence,
    because failure history is what the structured models actually exploit.
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

    The "low co-occurrence" condition: a proxy for software evolution where a file and a test are not
    repeatedly paired. It also removes exactly the repeated co-occurrences the history
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


DETECTABLE = Subset(
    name="detectable",
    note="changes with at least one killing test; the default averaging subset",
    needs=("faults",),
    predicate=_fault_bearing,
)

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


@dataclass(frozen=True)
class SubsetRegistry:
    """An immutable set of subsets, resolvable by name.

    A value rather than a global dict: an experiment with its own vocabulary builds its
    own registry, and two registries cannot leak into each other. ``__add__`` composes
    them, so a study can extend the base set without editing it.
    """

    subsets: tuple[Subset, ...] = ()

    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.subsets)

    def has(self, name: str) -> bool:
        return any(p.name == name for p in self.subsets)

    def get(self, name: str) -> Subset:
        for subset in self.subsets:
            if subset.name == name:
                return subset
        raise KeyError(f"unknown subset {name!r}; known: {list(self.names())}")

    def __add__(self, other: SubsetRegistry) -> SubsetRegistry:
        seen = {p.name for p in self.subsets}
        return SubsetRegistry(
            self.subsets
            + tuple(p for p in other.subsets if p.name not in seen)
        )

    def describe(self) -> list[dict]:
        return [p.metadata() for p in self.subsets]


#: The study's subsets. This is the set the recorded artifacts' subset names
#: resolve against; nothing else in the package depends on these specific names.
STUDY = SubsetRegistry(
    (DETECTABLE, NO_PRIOR_FAILURE, COLD_START, LOW_COOCCURRENCE)
)


def subset(name: str, registry: SubsetRegistry | None = None) -> Subset:
    """Resolve a subset by name against ``registry`` (default :data:`STUDY`)."""
    return (registry or STUDY).get(name)


def resolve(spec: Subset | str, registry: SubsetRegistry | None = None) -> Subset:
    """Accept either a subset or a name, for APIs that take both."""
    return spec if isinstance(spec, Subset) else subset(spec, registry)


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
    subsets and a sweep must not conflate them -- and the recorded caches are named for the
    threshold too (``semif_scores_cold_start5_full.jsonl``), so the two line up.
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

    The module already has ``LOW_COOCCURRENCE`` as a fixed threshold and ``low_cooccurrence_mask`` as
    a parameterised helper; this is the two joined, so an experiment can sweep thresholds as
    subset *levels* rather than looping outside the grid. The name carries the threshold,
    because two thresholds are two different subsets and a sweep must not conflate them.
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
    "DETECTABLE",
    "LOW_COOCCURRENCE",
    "NO_PRIOR_FAILURE",
    "Subset",
    "SubsetRegistry",
    "COLD_START",
    "STUDY",
    "low_cooccurrence",
    "subset",
    "resolve",
    "low_cooccurrence_mask",
    "cold_start",
    "cold_start_mask",
]
