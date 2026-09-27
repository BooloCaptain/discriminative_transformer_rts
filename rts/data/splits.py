"""The train/test split: evaluation configuration, not a property of the data.

The split is the one thing in this package that reads a dataset and belongs
unambiguously to evaluation. It is a value rather than a method on the dataset for
exactly that reason: a dataset cannot report which of its changes were held out,
because that is not a fact about the dataset. Reporting it as one -- which the first
pass did, via ``describe()`` and ``save()`` reading a default split -- published
evaluation configuration as dataset metadata and put it in the recorded artifacts.

The rule that ties a split to an ordering: **the effective ordering of a run is the
dataset's ordering, unless the split shuffles, in which case it is imposed.** Shuffling
an observed dataset therefore switches history features off by default, with no second
switch to forget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config
from .contract import Dataset, Ordering, Policy, Warning


@dataclass(frozen=True)
class Split:
    """A partition of a dataset's canonical order into a train prefix and test tail."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    fraction: float
    shuffle: bool
    seed: int
    ordering: Ordering
    warnings: tuple[Warning, ...] = ()

    def __post_init__(self) -> None:
        """A split is a *partition*, so the two halves may not overlap.

        The construction sites in this package cannot violate this -- ``make_split`` slices one
        permutation -- but a split may also be built by hand, which is what an evaluation window
        *inside* the held-out tail needs. The overlap is the failure mode that would silently
        train on rows that are also evaluated, so it is checked here rather than left to be
        noticed in a number.
        """
        overlap = np.intersect1d(self.train_idx, self.test_idx)
        if overlap.size:
            raise ValueError(
                f"a split partitions a dataset, but {overlap.size} change(s) are in both the "
                f"train prefix and the evaluation window (e.g. {overlap[:5].tolist()})"
            )

    @property
    def effective_ordering(self) -> Ordering:
        return Ordering.IMPOSED if self.shuffle else self.ordering

    @property
    def rows(self) -> np.ndarray:
        """The evaluation window: every change, in canonical order.

        A ``rows`` set is deliberately distinct from a *population*: ``rows`` says which
        changes are in the window at all, and a population is a named subset of them
        that a metric is averaged over.
        """
        return np.arange(len(self.train_idx) + len(self.test_idx), dtype=np.int64)


def make_split(
    ds: Dataset,
    train_fraction: float = config.DEFAULT_TRAIN_FRACTION,
    shuffle: bool = False,
    seed: int = config.SEED,
) -> Split:
    """Partition ``ds``'s canonical order into a contiguous train prefix and test tail.

    ``shuffle=False`` takes a contiguous prefix -- a well-defined operation on any
    dataset, so it only *warns* when the ordering is imposed, because the partition is
    valid but carries no temporal reading. ``shuffle=True`` permutes first and is
    permitted on any dataset at the user's risk, because it discards the temporal
    reading that history features rest on.
    """
    n = ds.n_changes
    warnings: list[Warning] = []
    order = np.arange(n, dtype=np.int64)
    if shuffle:
        order = np.random.default_rng(seed).permutation(n)
        observed = ds.ordering() is Ordering.OBSERVED
        warnings.append(
            Warning(
                code="split.shuffles_observed_order" if observed else "split.shuffles",
                requirement=Policy.EFFECTIVE_ORDER.value,
                note=(
                    "the split shuffles, so the effective ordering is imposed and history "
                    "features are off for this run"
                    + (
                        " even though the dataset's own order is real"
                        if observed
                        else ""
                    )
                ),
                scope="split",
            )
        )
    elif ds.ordering() is Ordering.IMPOSED:
        warnings.append(
            Warning(
                code="split.contiguous_prefix_on_imposed_order",
                requirement=Policy.OBSERVED_ORDER.value,
                note=(
                    "a contiguous prefix on an imposed order is a well-defined partition "
                    "but carries no temporal reading"
                ),
                scope="split",
            )
        )

    cut = int(round(n * train_fraction))
    return Split(
        train_idx=order[:cut],
        test_idx=order[cut:],
        fraction=train_fraction,
        shuffle=shuffle,
        seed=seed,
        ordering=ds.ordering(),
        warnings=tuple(warnings),
    )


def in_window(split: Split, rows: np.ndarray) -> bool:
    """Whether ``rows`` (change indices) are a subset of the split's evaluation window."""
    window = set(split.test_idx.tolist())
    return all(int(r) in window for r in rows)


def require_in_window(split: Split, rows: np.ndarray, *, what: str = "rows") -> None:
    """Raise unless ``rows`` lie inside the split's evaluation window.

    The evaluation boundary's guard. A metric may only be averaged over rows the split did not
    hold out **as training data**, and this is the one place that is checked. Every call site in
    this package satisfies it by construction -- the layer evaluates exactly ``split.test_idx``
    -- so it changes no number; it exists for the call sites that pass rows *directly*, which is
    what evaluating inside the held-out tail does.

    A violation is a programming error rather than a fact about the data, so it raises rather
    than returning :class:`~rts.contract.Unmeasured`.
    """
    if in_window(split, rows):
        return
    outside = sorted(set(int(r) for r in rows) - set(split.test_idx.tolist()))
    raise ValueError(
        f"{what} name {len(outside)} change(s) outside the split's evaluation window "
        f"(e.g. {outside[:5]}); a metric may only be averaged over rows the split held out"
    )


__all__ = ["Split", "in_window", "make_split", "require_in_window"]
