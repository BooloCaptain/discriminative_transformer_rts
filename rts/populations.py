"""Evaluation populations: named subsets of the evaluation window.

A population is to ``rows`` what a derived feature is to pairs -- a harness function
over the contract, defined once and applicable to any dataset. Three things follow, and
the first pass got all three wrong:

**Requirements are derived, not asserted.** A population names the material its
predicate reads; the requirements come from :data:`rts.accessors.MATERIAL`. The old
version hand-wrote a ``requires`` set beside a predicate and nothing checked the two
agreed. Declaring too little raised a capability error from inside the predicate instead
of returning :class:`Unmeasured`; declaring too much reported a working population
unavailable. Both were silent, and both are now unrepresentable -- the predicate is
handed the material, so it cannot read what it did not declare.

**Unavailable is not empty.** A population the dataset cannot support returns
:class:`Unmeasured` carrying the unmet requirement, never an empty array.

**One implementation per predicate.** ``no_prior_failure`` and ``starved_mask`` were two
implementations of one idea that had already drifted apart (only one supported
``max_runs``). There is now one function each, with a declared population and a
parameterised helper as two entry points onto it.

The registry is a *value* (:class:`PopulationRegistry`) rather than a module-level dict
that callers mutate or extend by editing this file. A caller with a different vocabulary
builds its own registry and passes the :class:`Population` object directly; the study's
own set is :data:`STUDY`, and it lives here rather than in the contract because
"starved" and "low pair recurrence" are study concepts, not properties of a dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from . import accessors
from .contract import Dataset, Requirement, Unmeasured

Predicate = Callable[[Mapping[str, Any]], np.ndarray]


@dataclass(frozen=True)
class Population:
    """A named subset of a dataset's change rows that a metric is averaged over."""

    name: str
    note: str
    needs: tuple[str, ...]
    predicate: Predicate

    def requires(self) -> frozenset[Requirement]:
        """The requirements implied by the material this population reads."""
        return accessors.requirements_for(self.needs)

    def unavailable(self, ds: Dataset) -> Unmeasured | None:
        """Why this population cannot exist on ``ds``, or ``None`` if it can."""
        missing = ds.missing_requirements(self.requires())
        if not missing:
            return None
        return Unmeasured(
            requirement=missing[0].value,
            note=(
                f"population {self.name!r} reads {', '.join(self.needs)}, which needs "
                f"{', '.join(m.value for m in missing)}; dataset {ds.name!r} does not "
                "provide it"
            ),
        )

    def mask(self, ds: Dataset) -> np.ndarray | Unmeasured:
        """Boolean mask over changes, or :class:`Unmeasured` when unavailable."""
        unavailable = self.unavailable(ds)
        if unavailable is not None:
            return unavailable
        material = accessors.material(ds)
        mask = np.asarray(self.predicate(material), dtype=bool)
        if mask.shape != (ds.n_changes,):
            raise ValueError(
                f"population {self.name!r} produced shape {mask.shape}, expected "
                f"{(ds.n_changes,)}"
            )
        return mask

    def rows(self, ds: Dataset, rows: np.ndarray) -> np.ndarray | Unmeasured:
        """``rows`` restricted to the population, or :class:`Unmeasured`."""
        mask = self.mask(ds)
        if isinstance(mask, Unmeasured):
            return mask
        return np.array([int(r) for r in rows if mask[int(r)]], dtype=np.int64)

    def declaration(self) -> dict:
        return {
            "name": self.name,
            "needs": list(self.needs),
            "requires": sorted(r.value for r in self.requires()),
            "note": self.note,
        }


# --- predicate implementations (one each) ---------------------------------


def _fault_bearing(material: Mapping[str, Any]) -> np.ndarray:
    return material["faults"]


def _no_prior_failure(
    material: Mapping[str, Any],
    max_failures: int,
    max_runs: int | None = None,
) -> np.ndarray:
    """Changes whose killing ``(file, test)`` pair has at most ``max_failures`` failures.

    Counts include the change itself, so ``max_failures=1`` means the pair has *no prior
    failure history*. Keyed on failure history rather than coverage co-occurrence,
    because failure history is what the structured models actually exploit.
    """
    fails = material["pair_failures"]
    run_counts = material["pair_runs"]
    paths = material["paths"]
    killing = material["killing"]
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
    material: Mapping[str, Any], max_pair_count: int
) -> np.ndarray:
    """Changes whose every ``(file, test)`` pair recurs at most this often.

    The "sparse" arm: a proxy for software evolution where a file and a test are not
    repeatedly paired. It also removes exactly the repeated co-occurrences the history
    features depend on, so it is a robustness check, not a neutral split.
    """
    counts = material["pairs"]
    coverage = material["coverage"]
    paths = material["paths"]
    mask = np.zeros(len(paths), dtype=bool)
    for i, tests in enumerate(coverage):
        pairs = [(paths[i], t) for t in tests]
        if not pairs:
            continue
        mask[i] = max(counts[p] for p in pairs) <= max_pair_count
    return mask


# --- the study's populations ----------------------------------------------


FAULT_BEARING = Population(
    name="fault_bearing",
    note="changes with at least one killing test; the default averaging population",
    needs=("faults",),
    predicate=_fault_bearing,
)

NO_PRIOR_FAILURE = Population(
    name="no_prior_failure",
    note="killing (file, test) pairs with no earlier failure",
    needs=("pair_failures", "paths", "killing"),
    predicate=lambda material: _no_prior_failure(material, max_failures=1),
)

STARVED = Population(
    name="starved",
    note="data-starved deployment proxy: killing pairs with almost no failure history",
    needs=("pair_failures", "pair_runs", "paths", "killing"),
    predicate=lambda material: _no_prior_failure(material, max_failures=2),
)

LOW_PAIR_RECURRENCE = Population(
    name="low_pair_recurrence",
    note="changes whose (file, test) pairs rarely recur",
    needs=("pairs", "coverage", "paths"),
    predicate=lambda material: _low_pair_recurrence(material, max_pair_count=1),
)


@dataclass(frozen=True)
class PopulationRegistry:
    """An immutable set of populations, resolvable by name.

    A value rather than a global dict: an experiment with its own vocabulary builds its
    own registry, and two registries cannot leak into each other. ``__add__`` composes
    them, so a study can extend the base set without editing it.
    """

    populations: tuple[Population, ...] = ()

    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.populations)

    def has(self, name: str) -> bool:
        return any(p.name == name for p in self.populations)

    def get(self, name: str) -> Population:
        for population in self.populations:
            if population.name == name:
                return population
        raise KeyError(f"unknown population {name!r}; known: {list(self.names())}")

    def __add__(self, other: "PopulationRegistry") -> "PopulationRegistry":
        seen = {p.name for p in self.populations}
        return PopulationRegistry(
            self.populations
            + tuple(p for p in other.populations if p.name not in seen)
        )

    def describe(self) -> list[dict]:
        return [p.declaration() for p in self.populations]


#: The study's populations. This is the set the recorded artifacts' population names
#: resolve against; nothing else in the package depends on these specific names.
STUDY = PopulationRegistry(
    (FAULT_BEARING, NO_PRIOR_FAILURE, STARVED, LOW_PAIR_RECURRENCE)
)


def population(name: str, registry: PopulationRegistry | None = None) -> Population:
    """Resolve a population by name against ``registry`` (default :data:`STUDY`)."""
    return (registry or STUDY).get(name)


def resolve(spec: "Population | str", registry: PopulationRegistry | None = None) -> Population:
    """Accept either a population or a name, for APIs that take both."""
    return spec if isinstance(spec, Population) else population(spec, registry)


# --- parameterised entry points onto the same predicates ------------------


def starved_mask(
    ds: Dataset, max_failures: int = 2, max_runs: int | None = None
) -> np.ndarray:
    """The data-starved proxy, parameterised. Calls the population's own predicate."""
    return _no_prior_failure(accessors.material(ds), max_failures, max_runs)


def sparse_mask(ds: Dataset, max_pair_count: int = 1) -> np.ndarray:
    """The sparse proxy, parameterised. Calls the population's own predicate."""
    return _low_pair_recurrence(accessors.material(ds), max_pair_count)


__all__ = [
    "FAULT_BEARING",
    "LOW_PAIR_RECURRENCE",
    "NO_PRIOR_FAILURE",
    "Population",
    "PopulationRegistry",
    "STARVED",
    "STUDY",
    "population",
    "resolve",
    "sparse_mask",
    "starved_mask",
]
