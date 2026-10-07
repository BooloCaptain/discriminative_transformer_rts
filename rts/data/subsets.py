"""Evaluation subsets: named subsets of the evaluation window.

A subset is to ``rows`` what a derived feature is to pairs -- a harness function over the
contract, defined once and applicable to any dataset. Three things follow:

**Requirements are derived, not asserted.** A subset names the inputs its predicate reads; the
requirements come from :data:`rts.data.accessors.INPUTS`. Declaring too little would raise a
capability error from inside the predicate instead of returning :class:`Undefined`; declaring too
much would report a working subset unavailable. The predicate is handed the inputs, so it cannot
read what it did not declare.

**Unavailable is not empty.** A subset the dataset cannot support returns :class:`Undefined`
carrying the unmet requirement, never an empty array.

**The registry is a value.** :class:`SubsetRegistry` is not a module-level dict that callers
mutate or extend by editing this file: a caller with its own vocabulary builds its own registry
and passes a :class:`Subset` object directly. The kernel ships one subset -- the default
averaging set, :data:`DETECTABLE` -- and a study declares the rest beside itself.
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


#: The default averaging subset: changes with at least one killing test. Recall over all
#: changes and recall over fault-bearing changes are different quantities, so the subset is
#: part of a result rather than an implicit choice buried in it.
DETECTABLE = Subset(
    name="detectable",
    note="changes with at least one killing test; the default averaging subset",
    needs=("faults",),
    predicate=_fault_bearing,
)


@dataclass(frozen=True)
class SubsetRegistry:
    """An immutable set of subsets, resolvable by name.

    A value rather than a global dict: an experiment with its own vocabulary builds its own
    registry, and two registries cannot leak into each other. ``__add__`` composes them, so a
    study can extend the default set without editing it.
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


#: The kernel's default registry. A study composes its own vocabulary onto this one.
DEFAULT = SubsetRegistry((DETECTABLE,))


def subset(name: str, registry: SubsetRegistry | None = None) -> Subset:
    """Resolve a subset by name against ``registry`` (default :data:`DEFAULT`)."""
    return (registry or DEFAULT).get(name)


def resolve(spec: Subset | str, registry: SubsetRegistry | None = None) -> Subset:
    """Accept either a subset or a name, for APIs that take both."""
    return spec if isinstance(spec, Subset) else subset(spec, registry)


__all__ = [
    "DEFAULT",
    "DETECTABLE",
    "Subset",
    "SubsetRegistry",
    "resolve",
    "subset",
]
