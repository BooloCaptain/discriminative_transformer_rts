"""The dataset interface: one contract that every change source implements.

This module is the *harness* half of the refactor described in ``refactor.md``.
A dataset supplies changes, a test pool, and the outcome relation between them; the
harness supplies every statistic computed from those. The split is load-bearing:

**Primitives** are dataset-specific, because their *extraction* is.
:meth:`Dataset.diff_text` is the archetype: a mutation is reconstructed by diffing
generated source, a real change is a git patch, and a derived change is a
concatenation of others. Nothing generic can obtain it.

**Derived features** are harness functions over the primitives, defined once below.
Every dataset inherits them, so an identical statistic means an identical quantity
across datasets -- which is what makes a cross-dataset comparison meaningful.

Beyond the primitives a dataset *declares* three things:

* :meth:`Dataset.capabilities` --- what optional material it has (``coverage``,
  ``durations``). Absence becomes an explicit, visible fact rather than an all-zero
  column a model will happily split on.
* :meth:`Dataset.ordering` --- ``observed`` or ``imposed``. History features are only
  interpretable when the change sequence is real, so they are **off by default for an
  imposed dataset**: a cumulative feature over an arbitrary order does not merely fail
  to be interpretable, it manufactures a leak.
* :meth:`Dataset.test_unit` and :meth:`Dataset.semantics` --- a machine-readable enum
  for everything the harness asks a question about, plus human-readable notes.

Two further rules apply to everything, not only to populations:

* **Unmeasured is a state, not a value, and it propagates.** A quantity that cannot be
  defined stays :class:`Unmeasured` and carries the requirement that was not met.
* **Warnings are structured and propagated**, whether or not anything currently acts
  on them.

Three sets are deliberately distinct, because conflating them loses information:
``candidates`` (rankable pairs), ``rows`` (changes in the evaluation window) and
``population`` (a named subset of rows that a metric is averaged over).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from . import config

TestId = str

#: Optional material a dataset may or may not have.
KNOWN_CAPABILITIES = frozenset({"coverage", "durations"})

#: Requirement ids a population or derived feature may depend on.
REQ_LABELS = "labels"
REQ_COVERAGE = "coverage"
REQ_ORDERING_OBSERVED = "ordering.observed"
REQ_DIFF_TEXT = "diff_text"


class TestUnit(str, Enum):
    """What one :data:`TestId` denotes. The harness-checkable half of §2.5."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    CASE = "case"


class Ordering(str, Enum):
    """Whether the change sequence is real (``observed``) or imposed by the harness."""

    OBSERVED = "observed"
    IMPOSED = "imposed"


class CapabilityMissing(Exception):
    """Raised when a capability is requested from a dataset that does not declare it."""

    def __init__(self, dataset: str, capability: str):
        super().__init__(
            f"dataset {dataset!r} does not declare the {capability!r} capability; "
            f"check capabilities() before requesting it"
        )
        self.dataset = dataset
        self.capability = capability


# --- Unmeasured values ----------------------------------------------------


@dataclass(frozen=True)
class Unmeasured:
    """A quantity that could not be defined, plus the requirement that was unmet.

    Reporting an average over a population that cannot exist as ``0.0`` is not a
    rounding error: it is a false statement about the data. Carrying the state
    explicitly means a meaningless number never reaches a table or a comparison.
    """

    requirement: str
    note: str

    def __bool__(self) -> bool:
        return False

    def to_dict(self) -> dict:
        return {"unmeasured": True, "requirement": self.requirement, "note": self.note}


def is_unmeasured(value: Any) -> bool:
    return isinstance(value, Unmeasured)


def first_unmeasured(*values: Any) -> Unmeasured | None:
    """The first :class:`Unmeasured` among ``values``, or ``None`` if all are measured."""
    for value in values:
        if isinstance(value, Unmeasured):
            return value
    return None


# --- Structured warnings --------------------------------------------------


@dataclass(frozen=True)
class Warning:
    """A structured warning: checkable code, unmet requirement, note, and scope."""

    code: str
    requirement: str
    note: str
    scope: str = ""

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "requirement": self.requirement,
            "note": self.note,
            "scope": self.scope,
        }


class Warnings:
    """An ordered, de-duplicated collection of :class:`Warning`.

    A declaration that is only printed is invisible to the layers that should be able
    to act on it, so warnings are values that travel with a result rather than log
    lines. Recording one is not conditional on a consumer existing.
    """

    def __init__(self, items: Iterable[Warning] = ()):  # noqa: D401
        self._items: list[Warning] = []
        self._seen: set[tuple[str, str]] = set()
        for item in items:
            self.add(item)

    def add(self, code: str, requirement: str, note: str, scope: str = "") -> Warning:
        warning = Warning(code=code, requirement=requirement, note=note, scope=scope)
        key = (code, scope)
        if key not in self._seen:
            self._seen.add(key)
            self._items.append(warning)
        return warning

    def extend(self, other: Iterable[Warning]) -> None:
        for warning in other:
            key = (warning.code, warning.scope)
            if key not in self._seen:
                self._seen.add(key)
                self._items.append(warning)

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def codes(self) -> list[str]:
        return [w.code for w in self._items]

    def to_list(self) -> list[dict]:
        return [w.to_dict() for w in self._items]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Warnings({len(self._items)}: {', '.join(self.codes())})"


# --- The split ------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    """A train/test partition of a dataset's canonical order.

    The split is evaluation configuration, not a property of the data. One rule ties
    it to the ordering: **the effective ordering of a run is the dataset's ordering,
    unless the split shuffles, in which case it is** :attr:`Ordering.IMPOSED`.
    Shuffling an observed dataset therefore switches history features off by the §2.4
    default, with no second switch to forget.
    """

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


def make_split(
    ds: "Dataset",
    train_fraction: float = config.DEFAULT_TRAIN_FRACTION,
    shuffle: bool = False,
    seed: int = config.SEED,
) -> Split:
    """Partition ``ds``'s canonical order into a contiguous train prefix and test tail.

    ``shuffle=False`` takes a contiguous prefix -- a well-defined operation on any
    dataset, so it only *warns* when the ordering is imposed. ``shuffle=True``
    permutes first and is permitted on any dataset at the user's risk, because it
    discards the temporal reading that history features rest on.
    """
    n = ds.n_changes
    warnings = Warnings()
    order = np.arange(n, dtype=np.int64)
    if shuffle:
        order = np.random.default_rng(seed).permutation(n)
        warnings.add(
            "split.shuffles_observed_order" if ds.ordering() is Ordering.OBSERVED
            else "split.shuffles",
            REQ_ORDERING_OBSERVED,
            "the split shuffles, so the effective ordering is imposed and history "
            "features are off by the §2.4 default",
            scope="split",
        )
    elif ds.ordering() is Ordering.IMPOSED:
        warnings.add(
            "split.contiguous_prefix_on_imposed_order",
            REQ_ORDERING_OBSERVED,
            "a contiguous prefix on an imposed order is a well-defined partition but "
            "carries no temporal reading",
            scope="split",
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


# --- Populations ----------------------------------------------------------


@dataclass(frozen=True)
class Population:
    """A named subset of a dataset's change rows that a metric is averaged over.

    A population is to ``rows`` what a derived feature is to pairs: a harness
    function over the contract. It therefore *declares its requirements*, and where a
    dataset cannot meet them the population is **unavailable** -- never silently
    empty.
    """

    name: str
    note: str
    requires: frozenset[str]
    predicate: Callable[["Dataset"], np.ndarray]

    def unavailable(self, ds: "Dataset") -> Unmeasured | None:
        """Return why this population cannot exist on ``ds``, or ``None``."""
        missing = sorted(self.requires - ds.available_requirements())
        if not missing:
            return None
        return Unmeasured(
            requirement=missing[0],
            note=(
                f"population {self.name!r} requires {', '.join(missing)}, which "
                f"dataset {ds.name!r} does not provide"
            ),
        )

    def mask(self, ds: "Dataset") -> np.ndarray | Unmeasured:
        """Boolean mask over changes, or :class:`Unmeasured` when unavailable."""
        unavailable = self.unavailable(ds)
        if unavailable is not None:
            return unavailable
        return self.predicate(ds)

    def rows(self, ds: "Dataset", rows: np.ndarray) -> np.ndarray | Unmeasured:
        """``rows`` restricted to the population, or :class:`Unmeasured`."""
        mask = self.mask(ds)
        if isinstance(mask, Unmeasured):
            return mask
        return rows[mask[rows]]


def fault_bearing(ds: "Dataset") -> np.ndarray:
    return ds.fault_mask


def no_prior_failure(ds: "Dataset", max_failures: int = 1) -> np.ndarray:
    """Changes whose killing ``(file, test)`` pair has at most ``max_failures`` failures.

    Counts include the change itself, so ``max_failures=1`` means the pair has no
    prior failure history. Readable from ``killing_tests`` and the order, as §5
    requires of a population predicate.
    """
    runs, fails = ds.pair_history_counts()
    mask = np.zeros(ds.n_changes, dtype=bool)
    for i, change in enumerate(ds.changes):
        killing = sorted(ds.killing_tests(change))
        if not killing:
            continue
        if fails.get((ds.change_paths[i], killing[0]), 0) <= max_failures:
            mask[i] = True
    return mask


def low_pair_recurrence(ds: "Dataset", max_pair_count: int = 1) -> np.ndarray:
    """Changes whose every ``(file, test)`` pair recurs at most this often.

    Needs ``coverage``: without it there is no pair set to count. That requirement is
    what makes the population unavailable rather than silently empty on a dataset
    without coverage.
    """
    return ds.sparse_mask(max_pair_count=max_pair_count)


POPULATIONS: dict[str, Population] = {
    "fault_bearing": Population(
        name="fault_bearing",
        note="changes with at least one killing test; the default averaging population",
        requires=frozenset({REQ_LABELS}),
        predicate=fault_bearing,
    ),
    "no_prior_failure": Population(
        name="no_prior_failure",
        note="killing (file, test) pairs with no earlier failure",
        requires=frozenset({REQ_LABELS}),
        predicate=lambda ds: no_prior_failure(ds, max_failures=1),
    ),
    "starved": Population(
        name="starved",
        note="data-starved deployment proxy: killing pairs with almost no failure history",
        requires=frozenset({REQ_LABELS}),
        predicate=lambda ds: no_prior_failure(ds, max_failures=2),
    ),
    "low_pair_recurrence": Population(
        name="low_pair_recurrence",
        note="changes whose (file, test) pairs rarely recur",
        requires=frozenset({REQ_LABELS, REQ_COVERAGE}),
        predicate=low_pair_recurrence,
    ),
}


def population(name: str) -> Population:
    try:
        return POPULATIONS[name]
    except KeyError:
        raise KeyError(f"unknown population {name!r}; known: {sorted(POPULATIONS)}") from None


# --- The dataset contract -------------------------------------------------


class Dataset(ABC):
    """What a dataset *supplies*: primitives, then declarations. Everything else is derived.

    A dataset is a plain value behind this contract: constructing one has no side
    effects, and two may coexist in a process -- which is why nothing here is stored
    at module level.
    """

    # --- primitives -------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable dataset name. Must contain no ``::``, which namespacing reserves."""

    @property
    @abstractmethod
    def changes(self) -> Sequence[Any]:
        """The changes, in canonical order. The order is part of the dataset's identity."""

    @abstractmethod
    def files(self, change: Any) -> tuple[str, ...]:
        """Repo-relative paths the change touches. Plural because a change may touch several."""

    @abstractmethod
    def diff_text(self, change: Any) -> str:
        """The change as a **unified diff**. The harness recovers added and removed lines."""

    @abstractmethod
    def killing_tests(self, change: Any) -> frozenset[TestId]:
        """The label: tests that failed against the change."""

    @abstractmethod
    def ran_tests(self, change: Any) -> frozenset[TestId]:
        """What was actually executed. Not the same as ``killing_tests``: separates
        "passed" from "never ran"."""

    @property
    @abstractmethod
    def test_pool(self) -> Sequence[TestId]:
        """Every test in this dataset's suite."""

    @abstractmethod
    def test_source(self, test: TestId) -> str | None:
        """The test's source text, or ``None`` when it cannot be obtained.

        ``None`` and ``""`` are different states: an unlocatable test contributes no
        size features at all, an empty one contributes zero-length ones.
        """

    # --- optional primitives (§2.3) ---------------------------------------

    def capabilities(self) -> frozenset[str]:
        """Subset of :data:`KNOWN_CAPABILITIES`. Defaults to none."""
        return frozenset()

    def coverage(self, change: Any) -> frozenset[TestId]:
        """Tests covering the change. Only call when ``"coverage"`` is declared."""
        raise CapabilityMissing(self.name, "coverage")

    def durations(self) -> Mapping[TestId, float]:
        """Per-test wall-clock durations. Only call when ``"durations"`` is declared."""
        raise CapabilityMissing(self.name, "durations")

    # --- declarations (§2.4, §2.5) ----------------------------------------

    @abstractmethod
    def ordering(self) -> Ordering:
        """Whether the change sequence is real. Declared, never inferred."""

    def test_unit(self) -> TestUnit:
        """What one :data:`TestId` denotes. Flattens; see :meth:`semantics`."""
        return TestUnit.FUNCTION

    def semantics(self) -> Mapping[str, str]:
        """Free-form notes keyed by attribute, qualifying anything an enum flattens."""
        return {}

    def canonical_test_id(self, test: TestId) -> TestId:
        """Canonical spelling of a test id, for matching against external caches.

        A dataset whose pool was rebuilt from a fresh collection may need to collapse
        unstable parametrization ids; one that was not does not.
        """
        return test

    def change_id(self, change: Any) -> str:
        """The change's stable identity, and the key of :attr:`change_index`.

        This is what score caches are keyed by, so it must be stable across processes
        and independent of the change's position in the canonical order.
        """
        return str(getattr(change, "change_id", change))

    def coverage_key(self, change: Any) -> str:
        """The key a coverage map is indexed by. Only meaningful with ``coverage``.

        Deliberately distinct from :meth:`change_id`. mutmut's coverage map is keyed by
        the *mutated function*, not by the mutant, so conflating the two silently
        mis-maps every lookup -- which is exactly the failure a single "key" accessor
        would have made easy to write.
        """
        return str(getattr(change, "func_key", self.change_id(change)))

    # --- required capabilities -------------------------------------------

    def has_capability(self, capability: str) -> bool:
        return capability in self.capabilities()

    def available_requirements(self) -> frozenset[str]:
        """Requirement ids this dataset can satisfy."""
        available = {REQ_LABELS, REQ_DIFF_TEXT}
        capabilities = self.capabilities()
        if "coverage" in capabilities:
            available.add(REQ_COVERAGE)
        if self.ordering() is Ordering.OBSERVED:
            available.add(REQ_ORDERING_OBSERVED)
        return frozenset(available)

    # --- warnings ---------------------------------------------------------

    @property
    def warnings(self) -> Warnings:
        """Warnings collected while deriving things from this dataset.

        Instance state, not module state: a dataset is a value, so warnings belong to
        the value that produced them.
        """
        warnings = self.__dict__.get("_warnings")
        if warnings is None:
            warnings = Warnings()
            self.__dict__["_warnings"] = warnings
        return warnings

    # --- derived: identity of test ids -----------------------------------

    @property
    def test_ids(self) -> list[TestId]:
        """The pool as an index-aligned list. Column ``j`` of every matrix is this test."""
        cached = self.__dict__.get("_test_ids")
        if cached is None:
            cached = list(self.test_pool)
            self.__dict__["_test_ids"] = cached
        return cached

    @property
    def test_index(self) -> dict[TestId, int]:
        cached = self.__dict__.get("_test_index")
        if cached is None:
            cached = {t: i for i, t in enumerate(self.test_ids)}
            self.__dict__["_test_index"] = cached
        return cached

    @property
    def n_tests(self) -> int:
        return len(self.test_ids)

    @property
    def n_changes(self) -> int:
        return len(self.changes)

    @property
    def change_index(self) -> dict[str, int]:
        return {self.change_id(c): i for i, c in enumerate(self.changes)}

    # --- derived: labels -------------------------------------------------

    @property
    def labels(self) -> np.ndarray:
        """``[n_changes, n_tests]`` uint8: 1 where the test failed against the change."""
        cached = self.__dict__.get("_labels")
        if cached is None:
            out = np.zeros((self.n_changes, self.n_tests), dtype=np.uint8)
            for i, change in enumerate(self.changes):
                for test in self.killing_tests(change):
                    j = self.test_index.get(test)
                    if j is not None:
                        out[i, j] = 1
            self.__dict__["_labels"] = out
            cached = out
        return cached

    @property
    def ran(self) -> np.ndarray:
        """``[n_changes, n_tests]`` uint8: 1 where the test was actually executed."""
        cached = self.__dict__.get("_ran")
        if cached is None:
            out = np.zeros((self.n_changes, self.n_tests), dtype=np.uint8)
            for i, change in enumerate(self.changes):
                for test in self.ran_tests(change):
                    j = self.test_index.get(test)
                    if j is not None:
                        out[i, j] = 1
            self.__dict__["_ran"] = out
            cached = out
        return cached

    @property
    def change_paths(self) -> list[str]:
        """One path per change, for datasets whose changes touch a single file.

        ``files`` is plural because a change may touch several; this is the flattened
        view the single-file derived features read, and it warns when it has to
        choose.
        """
        cached = self.__dict__.get("_change_paths")
        if cached is None:
            cached = []
            for change in self.changes:
                paths = self.files(change)
                if len(paths) != 1:
                    self.warnings.add(
                        "derived.multi_file_change_flattened",
                        REQ_DIFF_TEXT,
                        f"change {change!r} touches {len(paths)} files; single-file derived "
                        "features use the first",
                        scope="change_paths",
                    )
                cached.append(paths[0] if paths else "")
            self.__dict__["_change_paths"] = cached
        return cached

    @property
    def covered(self) -> list[frozenset[TestId]]:
        """Tests covering each change, as an index-aligned list."""
        cached = self.__dict__.get("_covered")
        if cached is None:
            cached = [self.coverage(c) for c in self.changes]
            self.__dict__["_covered"] = cached
        return cached

    @property
    def fault_mask(self) -> np.ndarray:
        cached = self.__dict__.get("_fault_mask")
        if cached is None:
            cached = np.array(
                [bool(self.killing_tests(c)) for c in self.changes], dtype=bool
            )
            self.__dict__["_fault_mask"] = cached
        return cached

    @property
    def fault_idx(self) -> np.ndarray:
        """Changes that can be caught, i.e. have at least one killing test."""
        return np.flatnonzero(self.fault_mask).astype(np.int64)

    # --- derived: the study's default split ------------------------------

    @property
    def split(self) -> Split:
        """The study's default split of the canonical order.

        Evaluation owns the split (§5) and may build any :class:`Split` it likes with
        :func:`make_split`. These two accessors exist so the documented default is
        one object rather than a convention repeated at every call site.
        """
        cached = self.__dict__.get("_split")
        if cached is None:
            cached = make_split(self, seed=self.order_seed)
            self.__dict__["_split"] = cached
        return cached

    @property
    def order_seed(self) -> int:
        """Seed used to materialise the canonical order of an imposed dataset."""
        return config.SEED

    @property
    def train_idx(self) -> np.ndarray:
        return self.split.train_idx

    @property
    def test_idx(self) -> np.ndarray:
        return self.split.test_idx

    @property
    def test_fault_idx(self) -> np.ndarray:
        """Fault-bearing changes inside the held-out window."""
        held = set(self.test_idx.tolist())
        return np.array([i for i in self.fault_idx if i in held], dtype=np.int64)

    # --- derived: candidate sets and pair statistics ---------------------

    def candidates(self, mode: str = "full") -> np.ndarray:
        """Which ``(change, test)`` pairs are eligible for selection.

        ``full``    -- every test in the pool, the realistic RTS setting.
        ``covered`` -- only tests covering the change, plus that change's killing
                       tests as a safety net. Requires the ``coverage`` capability.

        All selectors are evaluated on the same mask so the comparison stays fair.
        """
        if mode == "full":
            return np.ones((self.n_changes, self.n_tests), dtype=bool)
        if mode != "covered":
            raise ValueError(f"unknown candidate mode: {mode!r}")
        if not self.has_capability("coverage"):
            raise CapabilityMissing(self.name, "coverage")
        mask = np.zeros((self.n_changes, self.n_tests), dtype=bool)
        for i, change in enumerate(self.changes):
            for test in self.covered[i]:
                j = self.test_index.get(test)
                if j is not None:
                    mask[i, j] = True
            for test in self.killing_tests(change):
                j = self.test_index.get(test)
                if j is not None:
                    mask[i, j] = True
        return mask

    def candidate_counts(self, candidates: np.ndarray) -> np.ndarray:
        return candidates.sum(axis=1).astype(np.int64)

    def pair_counts(self) -> dict[tuple[str, str], int]:
        """How often each ``(file, test)`` combination recurs across changes.

        A pair that occurs once is a combination the structured models have no
        history for. Requires ``coverage``.
        """
        counts: dict[tuple[str, str], int] = {}
        for i, change in enumerate(self.changes):
            path = self.change_paths[i]
            for test in self.covered[i]:
                key = (path, test)
                counts[key] = counts.get(key, 0) + 1
        return counts

    def sparse_mask(self, max_pair_count: int = 1) -> np.ndarray:
        """Changes whose every ``(file, test)`` combination recurs at most this often.

        This is the "sparse" evaluation arm from ``plan.md``: a proxy for software
        evolution where a file and a test are not repeatedly paired. Note that it also
        removes exactly the repeated co-occurrences the structured history features
        depend on, so it is a robustness check, not a neutral split.
        """
        counts = self.pair_counts()
        mask = np.zeros(self.n_changes, dtype=bool)
        for i, change in enumerate(self.changes):
            pairs = [(self.change_paths[i], t) for t in self.covered[i]]
            if not pairs:
                continue
            mask[i] = max(counts[p] for p in pairs) <= max_pair_count
        return mask

    def pair_history_counts(self) -> tuple[dict, dict]:
        """Per ``(file, test)``: how many times the test ran, and failed, for that file.

        Counts are over the whole dataset. For a change's own killing pair the failure
        count therefore includes the change itself, so ``failures <= 1`` means the pair
        has *no prior failure history*.
        """
        runs: dict[tuple[str, str], int] = {}
        fails: dict[tuple[str, str], int] = {}
        for i, path in enumerate(self.change_paths):
            for j in np.flatnonzero(self.ran[i]):
                key = (path, self.test_ids[int(j)])
                runs[key] = runs.get(key, 0) + 1
            for j in np.flatnonzero(self.labels[i]):
                key = (path, self.test_ids[int(j)])
                fails[key] = fails.get(key, 0) + 1
        return runs, fails

    # --- derived: description --------------------------------------------

    def source_counts(self) -> Mapping[str, int]:
        """Counts the *source* can report and the contract cannot derive.

        mutmut reports killed/survived verdicts; a corpus of real bugs reports nothing
        of the sort. Keeping them behind a hook is what stops :meth:`describe` from
        reading an attribute only one source happens to have -- which is exactly how a
        generic helper silently becomes source-specific.
        """
        return {}

    def describe(self) -> dict:
        per_change_kill = np.array([len(self.killing_tests(c)) for c in self.changes])
        faults = self.fault_idx
        out = {
            "changes": self.n_changes,
            "tests": self.n_tests,
        }
        out.update(self.source_counts())
        out.update(
            {
                "fault_bearing_changes": len(faults),
                "train_changes": len(self.train_idx),
                "test_changes": len(self.test_idx),
                "held_out_faults": len(self.test_fault_idx),
                "killing_tests_per_fault_median": float(np.median(per_change_kill[faults])),
                "killing_tests_per_fault_max": int(per_change_kill[faults].max()),
            }
        )
        if self.has_capability("coverage"):
            out["mean_covered_tests_per_change"] = float(
                np.mean([len(c) for c in self.covered])
            )
            out["changes_with_empty_coverage"] = int(sum(not c for c in self.covered))
            out["sparse_changes"] = int(self.sparse_mask().sum())
        return out

    def describe_starved(self, mask: np.ndarray) -> dict:
        """Distribution shape of a filtered subset, to show it is narrow and low."""
        runs, fails = self.pair_history_counts()
        idx = np.flatnonzero(mask)
        fc = np.array(
            [
                fails.get((self.change_paths[i], sorted(self.killing_tests(self.changes[i]))[0]), 0)
                for i in idx
                if self.killing_tests(self.changes[i])
            ]
        )
        rc = np.array(
            [
                runs.get((self.change_paths[i], sorted(self.killing_tests(self.changes[i]))[0]), 0)
                for i in idx
                if self.killing_tests(self.changes[i])
            ]
        )
        held = set(self.test_idx.tolist())
        return {
            "changes": int(mask.sum()),
            "held_out_changes": int(sum(1 for i in idx if i in held)),
            "held_out_faults": int(
                sum(1 for i in idx if i in held and self.killing_tests(self.changes[i]))
            ),
            "failure_count_mean": float(fc.mean()) if fc.size else float("nan"),
            "failure_count_sd": float(fc.std()) if fc.size else float("nan"),
            "failure_count_max": int(fc.max()) if fc.size else 0,
            "run_count_mean": float(rc.mean()) if rc.size else float("nan"),
            "run_count_max": int(rc.max()) if rc.size else 0,
        }

    # --- helpers ----------------------------------------------------------


# --- Derived features: harness functions over the primitives ---------------


def changed_lines(ds: Dataset, change: Any) -> tuple[str, ...]:
    """Added lines of the change's unified diff.

    The ``+++`` header is excluded. A content line whose text begins with ``++`` is
    excluded too -- that is the historical spelling of this parser and the documented
    numbers depend on it, so it is stated rather than quietly fixed.
    """
    return tuple(
        line[1:]
        for line in ds.diff_text(change).splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def removed_lines(ds: Dataset, change: Any) -> tuple[str, ...]:
    """Removed lines of the change's unified diff, excluding the ``---`` header."""
    return tuple(
        line[1:]
        for line in ds.diff_text(change).splitlines()
        if line.startswith("-") and not line.startswith("---")
    )


def change_size(ds: Dataset, change: Any) -> int:
    """Added plus removed lines. A single function, not one per dataset, so a
    cross-dataset comparison of "change size" is comparing the same quantity."""
    return len(changed_lines(ds, change)) + len(removed_lines(ds, change))


def change_query_text(ds: Dataset, change: Any) -> str:
    """The change side of a pair: added and removed lines, weighted by repetition.

    Including the removed lines matters because a mutation's meaning often comes from
    what it replaced.
    """
    added = "\n".join(changed_lines(ds, change))
    removed = "\n".join(removed_lines(ds, change))
    return f"{added}\n{removed}"


def test_n_lines(ds: Dataset, test: TestId) -> int:
    source = ds.test_source(test)
    if source is None:
        return 0
    return source.count("\n") + 1


def test_n_tokens(ds: Dataset, test: TestId) -> int:
    source = ds.test_source(test)
    if source is None:
        return 0
    return len(source.split())


def n_tests_in_test_file(ds: Dataset) -> np.ndarray:
    """How many tests share a file with this one. Cheap context for whether a test is
    a focused unit test or one of many in a large module suite."""
    counts: dict[str, int] = defaultdict(int)
    for nodeid in ds.test_ids:
        counts[nodeid.split("::")[0]] += 1
    return np.array([counts[t.split("::")[0]] for t in ds.test_ids], dtype=np.float32)


def _dirs_and_stems(paths: Sequence[str]) -> tuple[list[str], list[str]]:
    dirs = [p.rsplit("/", 1)[0] for p in paths]
    stems = [p.rsplit("/", 1)[-1].removesuffix(".py") for p in paths]
    return dirs, stems


def _path_parts(directory: str) -> list[str]:
    """Components of a directory path. Pass a directory, not a file path."""
    return [part for part in directory.split("/") if part]


def _dir_distance(a: list[str], b: list[str]) -> int:
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return len(a) + len(b) - 2 * common


def path_distance(ds: Dataset) -> np.ndarray:
    """Directory-tree distance between each change's file and each test's file."""
    n_c, n_t = ds.n_changes, ds.n_tests
    change_dirs, _ = _dirs_and_stems(ds.change_paths)
    test_dirs = [str(ds.test_ids[i].split("::")[0]).rsplit("/", 1)[0] for i in range(n_t)]
    out = np.zeros((n_c, n_t), dtype=np.float32)
    cache: dict[tuple[str, str], int] = {}
    for i in range(n_c):
        for j in range(n_t):
            key = (change_dirs[i], test_dirs[j])
            d = cache.get(key)
            if d is None:
                d = _dir_distance(_path_parts(change_dirs[i]), _path_parts(test_dirs[j]))
                cache[key] = d
            out[i, j] = d
    return out


def filename_stem_match(ds: Dataset) -> np.ndarray:
    """``test_utils.py`` for ``utils.py``: a strong, cheap naming signal."""
    n_c, n_t = ds.n_changes, ds.n_tests
    _, change_stems = _dirs_and_stems(ds.change_paths)
    test_names = [ds.test_ids[i].split("::")[0].rsplit("/", 1)[-1] for i in range(n_t)]
    out = np.zeros((n_c, n_t), dtype=np.float32)
    for i in range(n_c):
        for j in range(n_t):
            out[i, j] = 1.0 if change_stems[i] and change_stems[i] in test_names[j] else 0.0
    return out


def history_features(
    ds: Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative ``(failure_rate, runs, last_failure_age)`` over the canonical order.

    Every value is emitted *before* the change's own outcome is folded in, so nothing
    peeks at the label it is used to predict. The features are only interpretable when
    the order is real; whether to compute them at all is the caller's decision (§2.4).
    """
    n_c, n_t = ds.n_changes, ds.n_tests
    alpha = 1.0  # Laplace prior so an unseen test starts at 0.5
    running_fails = np.zeros(n_t, dtype=np.float64)
    running_runs = np.zeros(n_t, dtype=np.float64)
    last_failure = np.full(n_t, -1, dtype=np.int64)

    failure_rate = np.zeros((n_c, n_t), dtype=np.float32)
    runs_cum = np.zeros((n_c, n_t), dtype=np.float32)
    last_failure_age = np.zeros((n_c, n_t), dtype=np.float32)

    labels, ran = ds.labels, ds.ran
    for i in range(n_c):
        failure_rate[i] = (running_fails + alpha) / (running_runs + 2 * alpha)
        runs_cum[i] = running_runs
        age = np.where(last_failure >= 0, i - last_failure, n_c)
        last_failure_age[i] = age
        # Update only after emitting features for change i.
        running_fails += labels[i]
        running_runs += ran[i]
        last_failure = np.where(labels[i] == 1, i, last_failure)

    return failure_rate, runs_cum, last_failure_age


#: The columns of :func:`structured_features`, in order.
STRUCTURED_NAMES = [
    "covers_function",
    "n_covering_tests",
    "coverage_rank_prior",
    "path_distance",
    "n_tests_in_test_file",
    "filename_stem_match",
    "test_duration",
    "test_n_lines",
    "test_n_tokens",
    "change_size",
    "change_added_lines",
    "change_removed_lines",
    "test_failure_rate_cum",
    "test_runs_cum",
    "test_last_failure_age",
]

#: Columns that are unmeasured when the ``coverage`` capability is absent.
COVERAGE_COLUMNS = ("covers_function", "n_covering_tests", "coverage_rank_prior")
#: Columns that are unmeasured when the ``durations`` capability is absent.
DURATION_COLUMNS = ("test_duration",)
#: Columns that are unmeasured when history features are off.
HISTORY_COLUMNS = ("test_failure_rate_cum", "test_runs_cum", "test_last_failure_age")


def coverage_columns(ds: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(mask, n_covering, rank_prior)`` for the coverage block."""
    n_c, n_t = ds.n_changes, ds.n_tests
    mask = np.zeros((n_c, n_t), dtype=np.float32)
    for i in range(n_c):
        for test in ds.covered[i]:
            j = ds.test_index.get(test)
            if j is not None:
                mask[i, j] = 1.0
    n_covering = mask.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        prior = np.where(n_covering > 0, 1.0 / np.maximum(n_covering, 1.0), 0.0)
    return mask, n_covering, prior


def structured_features(
    ds: Dataset,
    history: bool | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return ``(X, names)`` with ``X`` of shape ``[n_changes, n_tests, n_features]``.

    Every column is either static per pair or cumulative over strictly earlier
    changes, and every column is derived from the primitives -- so an identical
    column means an identical quantity on any dataset.

    ``history`` selects the cumulative-history block. ``None`` means *the §2.4
    default*: on for an :attr:`Ordering.OBSERVED` dataset, off for an
    :attr:`Ordering.IMPOSED` one, where a cumulative feature over an arbitrary order
    manufactures a leak. Study entry points that reproduce the documented numbers opt
    in explicitly, and the resulting warning travels with the dataset.

    Where a block cannot be defined at all -- no coverage capability, no durations, or
    history switched off -- the columns are **unmeasured**, and the model-input
    boundary is the one place the lossy coercion to zero is legitimate. The coercion
    is recorded as a warning rather than performed silently.
    """
    n_c, n_t = ds.n_changes, ds.n_tests
    n_f = len(STRUCTURED_NAMES)
    X = np.zeros((n_c, n_t, n_f), dtype=np.float32)
    ix = {name: i for i, name in enumerate(STRUCTURED_NAMES)}

    # --- static per-test ---
    if ds.has_capability("durations"):
        durations_map = ds.durations()
        X[:, :, ix["test_duration"]] = np.array(
            [durations_map.get(t, 0.0) for t in ds.test_ids], dtype=np.float32
        )[None, :]
    else:
        ds.warnings.add(
            "feature.unmeasured_durations",
            "capability:durations",
            "dataset declares no durations; test_duration is unmeasured and coerced to 0 "
            "at the model-input boundary",
            scope="structured_features",
        )

    X[:, :, ix["test_n_lines"]] = np.array(
        [test_n_lines(ds, t) for t in ds.test_ids], dtype=np.float32
    )[None, :]
    X[:, :, ix["test_n_tokens"]] = np.array(
        [test_n_tokens(ds, t) for t in ds.test_ids], dtype=np.float32
    )[None, :]
    X[:, :, ix["n_tests_in_test_file"]] = n_tests_in_test_file(ds)[None, :]

    # --- static per change ---
    X[:, :, ix["change_size"]] = np.array(
        [change_size(ds, c) for c in ds.changes], dtype=np.float32
    )[:, None]
    X[:, :, ix["change_added_lines"]] = np.array(
        [len(changed_lines(ds, c)) for c in ds.changes], dtype=np.float32
    )[:, None]
    X[:, :, ix["change_removed_lines"]] = np.array(
        [len(removed_lines(ds, c)) for c in ds.changes], dtype=np.float32
    )[:, None]

    # --- coverage / proximity ---
    if ds.has_capability("coverage"):
        mask, n_covering, prior = coverage_columns(ds)
        X[:, :, ix["covers_function"]] = mask
        X[:, :, ix["n_covering_tests"]] = n_covering
        X[:, :, ix["coverage_rank_prior"]] = prior
    else:
        ds.warnings.add(
            "feature.unmeasured_coverage",
            REQ_COVERAGE,
            "dataset declares no coverage; covers_function, n_covering_tests and "
            "coverage_rank_prior are unmeasured and coerced to 0 at the model-input boundary",
            scope="structured_features",
        )

    X[:, :, ix["path_distance"]] = path_distance(ds)
    X[:, :, ix["filename_stem_match"]] = filename_stem_match(ds)

    # --- cumulative history ---
    use_history = ds.ordering() is Ordering.OBSERVED if history is None else bool(history)
    if use_history:
        failure_rate, runs_cum, last_failure_age = history_features(ds)
        X[:, :, ix["test_failure_rate_cum"]] = failure_rate
        X[:, :, ix["test_runs_cum"]] = runs_cum
        X[:, :, ix["test_last_failure_age"]] = last_failure_age
        if ds.ordering() is Ordering.IMPOSED:
            ds.warnings.add(
                "feature.history_on_imposed_order",
                REQ_ORDERING_OBSERVED,
                f"history features are enabled on an imposed order (order_seed="
                f"{ds.order_seed}); the values are only interpretable under that seed and "
                "should be reported as a spread over seeds, not as one number",
                scope="structured_features",
            )
    else:
        ds.warnings.add(
            "feature.unmeasured_history",
            REQ_ORDERING_OBSERVED,
            "history features are off (the default for an imposed dataset); "
            "test_failure_rate_cum, test_runs_cum and test_last_failure_age are "
            "unmeasured and coerced to 0 at the model-input boundary",
            scope="structured_features",
        )

    return X, list(STRUCTURED_NAMES)


# --- Composition ---------------------------------------------------------


def namespace(ds: Dataset, test: TestId) -> str:
    """``dataset::nodeid``. Two projects may both contain ``tests/test_utils.py::test_x``."""
    if "::" in ds.name:
        raise ValueError(f"dataset name {ds.name!r} contains '::', which namespacing reserves")
    return f"{ds.name}::{test}"


def pool(datasets: Sequence[Dataset], name: str | None = None) -> "PooledDataset":
    """Combine datasets into one evaluation population, namespacing test ids.

    What namespacing does *not* fix is meaning: pooling datasets whose ``test_unit``
    differs is defensible only when the difference is immaterial, so it is warned
    about rather than refused (§4).
    """
    return PooledDataset(list(datasets), name=name)


class PooledDataset(Dataset):
    """Several datasets as one. Pooling is just iteration, so it composes by construction."""

    def __init__(self, datasets: Sequence[Dataset], name: str | None = None):
        if not datasets:
            raise ValueError("cannot pool zero datasets")
        self._datasets = list(datasets)
        self._name = name or "+".join(d.name for d in self._datasets)
        # ``changes`` returns the constituents' own change objects, so ownership is
        # resolved by object identity rather than by index: a caller holding a change
        # from ``changes`` gets back the dataset it came from.
        self._owner: dict[int, tuple[Dataset, Any]] = {}
        rows: list[tuple[int, int]] = []
        pool_ids: list[str] = []
        for d, ds in enumerate(self._datasets):
            for i in range(ds.n_changes):
                rows.append((d, i))
                self._owner[id(ds.changes[i])] = (ds, ds.changes[i])
            pool_ids.extend(namespace(ds, t) for t in ds.test_ids)
        self._rows = rows
        self._pool = tuple(pool_ids)
        units = {d.test_unit() for d in self._datasets}
        if len(units) > 1:
            self.warnings.add(
                "pool.mixed_test_unit",
                "test_unit",
                f"pooled datasets disagree on test_unit ({sorted(u.value for u in units)}); "
                "the difference is assumed immaterial",
                scope="pool",
            )

    @property
    def datasets(self) -> list[Dataset]:
        return self._datasets

    @property
    def name(self) -> str:
        return self._name

    @property
    def changes(self) -> Sequence[Any]:
        return [self._datasets[d].changes[i] for d, i in self._rows]

    def _owner_of(self, change: Any) -> tuple[Dataset, Any]:
        try:
            return self._owner[id(change)]
        except (KeyError, TypeError):
            raise KeyError(
                "change does not belong to this pooled dataset; take it from .changes"
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

    def capabilities(self) -> frozenset[str]:
        if not self._datasets:
            return frozenset()
        return frozenset.intersection(*(d.capabilities() for d in self._datasets))

    def coverage(self, change: Any) -> frozenset[TestId]:
        ds, own = self._owner_of(change)
        return frozenset(namespace(ds, t) for t in ds.coverage(own))

    def durations(self) -> Mapping[TestId, float]:
        out: dict[TestId, float] = {}
        for ds in self._datasets:
            for test, value in ds.durations().items():
                out[namespace(ds, test)] = value
        return out

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

    def canonical_test_id(self, test: TestId) -> TestId:
        ds, nodeid = self._resolve(test)
        return namespace(ds, ds.canonical_test_id(nodeid))

    def change_id(self, change: Any) -> str:
        ds, own = self._owner_of(change)
        return f"{ds.name}::{ds.change_id(own)}"

    def coverage_key(self, change: Any) -> str:
        ds, own = self._owner_of(change)
        return f"{ds.name}::{ds.coverage_key(own)}"

    def _resolve(self, test: TestId) -> tuple[Dataset, TestId]:
        name, _, nodeid = test.partition("::")
        for ds in self._datasets:
            if ds.name == name:
                return ds, nodeid
        raise KeyError(f"test {test!r} does not belong to any pooled dataset")


# --- Persistence ---------------------------------------------------------


def save(ds: Dataset, out_dir=None) -> None:
    out = config.ensure_artifacts_dir() if out_dir is None else out_dir
    payload = {
        "name": ds.name,
        "ordering": ds.ordering().value,
        "test_unit": ds.test_unit().value,
        "capabilities": sorted(ds.capabilities()),
        "semantics": dict(ds.semantics()),
        "warnings": ds.warnings.to_list(),
        "test_ids": ds.test_ids,
        "change_ids": [ds.change_id(c) for c in ds.changes],
        "files": [list(ds.files(c)) for c in ds.changes],
        "covered": [sorted(c) for c in ds.covered] if ds.has_capability("coverage") else None,
        "killing_tests": [sorted(ds.killing_tests(c)) for c in ds.changes],
        "ran_tests": [sorted(ds.ran_tests(c)) for c in ds.changes],
    }
    np.savez_compressed(
        out / "dataset.npz", labels=ds.labels, ran=ds.ran,
        train_idx=ds.train_idx, test_idx=ds.test_idx,
    )
    (out / "dataset.json").write_text(json.dumps(payload))
    print(f"[dataset] wrote {out / 'dataset.npz'} and dataset.json")


# Back-compatible entry points: these were module-level functions before the
# contract moved onto the dataset, and callers still use the function spelling.
def candidate_mask(ds: Dataset, mode: str = "full") -> np.ndarray:
    return ds.candidates(mode)


def pair_history_counts(ds: Dataset) -> tuple[dict, dict]:
    return ds.pair_history_counts()


def sparse_mask(ds: Dataset, max_pair_count: int = 1) -> np.ndarray:
    return ds.sparse_mask(max_pair_count=max_pair_count)


def starved_mask(ds: Dataset, max_failures: int = 2, max_runs: int | None = None) -> np.ndarray:
    """Changes whose killing ``(file, test)`` pair has almost no history.

    This targets the data-starved deployment regime: a huge codebase where a file may
    not have changed in years and a long-running test may have been executed against it
    only once or twice. ``max_failures=2`` means at most one prior failure of this pair;
    ``max_failures=1`` means none. ``max_runs`` optionally also caps how often the test
    has been run for this file.

    Keyed on *failure* history rather than coverage co-occurrence, because failure
    history is what the structured models actually exploit.
    """
    runs, fails = ds.pair_history_counts()
    mask = np.zeros(ds.n_changes, dtype=bool)
    for i, change in enumerate(ds.changes):
        killing = sorted(ds.killing_tests(change))
        if not killing:
            continue
        key = (ds.change_paths[i], killing[0])
        if fails.get(key, 0) > max_failures:
            continue
        if max_runs is not None and runs.get(key, 0) > max_runs:
            continue
        mask[i] = True
    return mask


def describe(ds: Dataset) -> dict:
    return ds.describe()


def describe_starved(ds: Dataset, mask: np.ndarray) -> dict:
    return ds.describe_starved(mask)


if __name__ == "__main__":
    from . import datasets

    ds = datasets.marshmallow()
    for key, value in describe(ds).items():
        print(f"{key:>32}: {value}")
    print(f"{'warnings':>32}: {ds.warnings!r}")

    # Sanity: every killing test must be inside the coverage set for that change.
    outside = sum(
        1
        for i, change in enumerate(ds.changes)
        for test in ds.killing_tests(change)
        if test not in ds.covered[i]
    )
    print(f"{'killing tests outside coverage':>32}: {outside}")
