"""Composition: namespacing, pooling, and derived datasets over other datasets.

Because datasets are values behind one contract, pooling is just iteration -- but two
details are load-bearing and both had to be fixed from the first pass.

**Identity.** A pooled change is a wrapper (:class:`PooledChange`) carrying which
dataset it came from and its id *there*, so a pooled change's identity is stable and
namespaced. The first pass keyed an ownership map by ``id(change)``, which worked only
because the map happened to hold strong references to keep the ids from being reused --
a guarantee that could be lost by an innocuous edit. It is now a plain dataclass, and
pooling two projects that both contain a bug numbered ``3`` cannot confuse them.

**Meaning.** Namespacing fixes collisions, not semantics. Pooling datasets whose
:meth:`~rts.data.contract.Dataset.test_unit` differs is defensible only when the difference is
immaterial, so the pool *reports* the disagreement rather than refusing it, and
:func:`rts.data.reporting.audit` turns it into a warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .contract import (
    Capability,
    CapabilityMissing,
    Dataset,
    Ordering,
    TestId,
    TestUnit,
    Warning,
)


def namespace(ds: Dataset, test: TestId) -> str:
    """``dataset::nodeid``. Two projects may both contain ``tests/test_utils.py::test_x``."""
    if "::" in ds.name:
        raise ValueError(f"dataset name {ds.name!r} contains '::', which namespacing reserves")
    return f"{ds.name}::{test}"


def unnamespace(test: TestId) -> tuple[str, TestId]:
    name, _, nodeid = test.partition("::")
    if not nodeid:
        raise ValueError(f"test id {test!r} is not namespaced as 'dataset::nodeid'")
    return name, nodeid


@dataclass(frozen=True)
class PooledChange:
    """One constituent change, with the dataset it belongs to and its id there."""

    dataset: str
    change_id: str
    own: Any

    @property
    def key(self) -> str:
        """The namespaced identity: stable, hashable, and collision-free."""
        return f"{self.dataset}::{self.change_id}"


class PooledDataset(Dataset):
    """Several datasets as one. Pooling is iteration, so it composes by construction."""

    def __init__(self, datasets: Sequence[Dataset], name: str | None = None):
        if not datasets:
            raise ValueError("cannot pool zero datasets")
        self._datasets = list(datasets)
        self._by_name = {d.name: d for d in self._datasets}
        if len(self._by_name) != len(self._datasets):
            raise ValueError(
                f"pooled datasets must have distinct names; got "
                f"{[d.name for d in self._datasets]}"
            )
        self._name = name or "+".join(d.name for d in self._datasets)
        if "::" in self._name:
            raise ValueError(f"pool name {self._name!r} must not contain '::'")
        self._pool = tuple(
            namespace(ds, t) for ds in self._datasets for t in ds.test_ids
        )

    # --- constituents -----------------------------------------------------

    @property
    def datasets(self) -> tuple[Dataset, ...]:
        return tuple(self._datasets)

    def mixed_test_units(self) -> tuple[TestUnit, ...]:
        """The distinct test units among constituents, if they disagree."""
        units = {d.test_unit() for d in self._datasets}
        return tuple(units) if len(units) > 1 else ()

    # --- primitives -------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def changes(self) -> Sequence[PooledChange]:
        return tuple(
            PooledChange(ds.name, ds.change_id(c), c)
            for ds in self._datasets
            for c in ds.changes
        )

    def _owner_of(self, change: Any) -> tuple[Dataset, Any]:
        if not isinstance(change, PooledChange):
            raise KeyError(
                "change does not belong to this pooled dataset; take it from .changes"
            )
        try:
            return self._by_name[change.dataset], change.own
        except KeyError:
            raise KeyError(
                f"change names dataset {change.dataset!r}, which is not in this pool"
            ) from None

    def files(self, change: Any) -> tuple[str, ...]:
        ds, own = self._owner_of(change)
        return ds.files(own)

    def diff_text(self, change: Any) -> str:
        ds, own = self._owner_of(change)
        return ds.diff_text(own)

    def killing_tests(self, change: Any) -> frozenset[TestId]:
        ds, own = self._owner_of(change)
        return frozenset(namespace(ds, t) for t in ds.killing_tests(own))

    def ran_tests(self, change: Any) -> frozenset[TestId]:
        ds, own = self._owner_of(change)
        return frozenset(namespace(ds, t) for t in ds.ran_tests(own))

    @property
    def test_pool(self) -> Sequence[TestId]:
        return self._pool

    def test_source(self, test: TestId) -> str | None:
        ds, nodeid = self._resolve(test)
        return ds.test_source(nodeid)

    # --- optional primitives ---------------------------------------------

    def capabilities(self) -> frozenset[Capability]:
        """The intersection: a pool has material only where every part does."""
        return frozenset.intersection(*(d.capabilities() for d in self._datasets))

    def coverage(self, change: Any) -> frozenset[TestId]:
        ds, own = self._owner_of(change)
        if not ds.has_capability(Capability.COVERAGE):
            raise CapabilityMissing(self.name, Capability.COVERAGE)
        return frozenset(namespace(ds, t) for t in ds.coverage(own))

    def durations(self) -> Mapping[TestId, float]:
        out: dict[TestId, float] = {}
        for ds in self._datasets:
            for test, value in ds.durations().items():
                out[namespace(ds, test)] = value
        return out

    # --- declarations -----------------------------------------------------

    def ordering(self) -> Ordering:
        """Imposed unless every constituent is observed: pooling interleaves sequences."""
        if all(d.ordering() is Ordering.OBSERVED for d in self._datasets):
            return Ordering.OBSERVED
        return Ordering.IMPOSED

    def test_unit(self) -> TestUnit:
        units = {d.test_unit() for d in self._datasets}
        return next(iter(units)) if len(units) == 1 else TestUnit.CASE

    def semantics(self) -> Mapping[str, str]:
        return {
            "pooled": ", ".join(d.name for d in self._datasets),
            "note": "rows are concatenated in constituent order; test ids are namespaced",
        }

    def own_candidate_pool(self) -> np.ndarray | None:
        """The constituents' own pools, laid out against the pooled test order.

        ``None`` unless *every* constituent has one: a per-change pool is only meaningful
        when each change's candidates come from its own suite, and if one constituent cannot
        say what its pool is, a pooled answer would be a guess.
        """
        masks = [d.own_candidate_pool() for d in self._datasets]
        if any(mask is None for mask in masks):
            return None
        rows = sum(len(d.changes) for d in self._datasets)
        out = np.zeros((rows, len(self._pool)), dtype=bool)
        row = col = 0
        for ds, mask in zip(self._datasets, masks):
            assert mask is not None  # narrowed by the check above
            n_rows, n_cols = len(ds.changes), len(ds.test_ids)
            out[row : row + n_rows, col : col + n_cols] = mask
            row += n_rows
            col += n_cols
        return out

    def integrity_notes(self) -> Sequence[Warning]:
        units = self.mixed_test_units()
        if not units:
            return ()
        return (
            Warning(
                code="pool.mixed_test_unit",
                requirement="test_unit",
                note=(
                    "pooled datasets disagree on test_unit "
                    f"({sorted(u.value for u in units)}); the difference is assumed "
                    "immaterial"
                ),
                scope="pool",
            ),
        )

    def canonical_test_id(self, test: TestId) -> TestId:
        ds, nodeid = self._resolve(test)
        return namespace(ds, ds.canonical_test_id(nodeid))

    def change_id(self, change: Any) -> str:
        """The namespaced id, so two projects' bug ``3`` cannot collide."""
        if isinstance(change, PooledChange):
            return change.key
        ds, own = self._owner_of(change)
        return f"{ds.name}::{ds.change_id(own)}"

    def coverage_key(self, change: Any) -> str:
        ds, own = self._owner_of(change)
        return f"{ds.name}::{ds.coverage_key(own)}"

    def source_counts(self) -> Mapping[str, int]:
        out: dict[str, int] = {}
        for ds in self._datasets:
            for key, value in ds.source_counts().items():
                out[key] = out.get(key, 0) + value
        return out

    def _resolve(self, test: TestId) -> tuple[Dataset, TestId]:
        name, nodeid = unnamespace(test)
        try:
            return self._by_name[name], nodeid
        except KeyError:
            raise KeyError(f"test {test!r} does not belong to any pooled dataset") from None


def pool(datasets: Sequence[Dataset], name: str | None = None) -> PooledDataset:
    """Combine datasets into one evaluation population, namespacing test ids."""
    return PooledDataset(list(datasets), name=name)


__all__ = ["PooledChange", "PooledDataset", "namespace", "pool", "unnamespace"]
