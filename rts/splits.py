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

from . import config
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
    """Whether ``rows`` are a subset of the split's evaluation window."""
    window = set(split.test_idx.tolist())
    return all(int(r) in window for r in rows)


__all__ = ["Split", "in_window", "make_split"]
