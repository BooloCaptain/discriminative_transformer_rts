"""The dataset contract: primitives, declarations, and the values they speak in.

This module answers one question: *what must a dataset supply, and what does it
declare about what it supplies?* It deliberately does **not** answer what can be
computed from those supplies. Every derived quantity lives in :mod:`rts.accessors`,
:mod:`rts.features` or :mod:`rts.populations`, because those are harness functions
that must not know which dataset they are reading.

**Primitives** are dataset-specific, because their *extraction* is.
:meth:`Dataset.diff_text` is the archetype: a mutation is reconstructed by diffing
generated source, a real change is a git patch, and a derived change is a
concatenation of others. Nothing generic can obtain it.

**Declarations** are the machine-readable answers to the questions the harness asks:
:meth:`Dataset.capabilities` (what optional material exists),
:meth:`Dataset.ordering` (is the sequence real), :meth:`Dataset.test_unit` (what one
test id denotes), and :meth:`Dataset.semantics` (the human-readable qualifications an
enum necessarily flattens).

Two vocabularies are defined here and used everywhere else. :class:`Capability` is
material a dataset may or may not have; :class:`Requirement` is what a computation
needs in order to be defined at all. They share spellings deliberately -- a
capability *is* the requirement of the same name -- and :func:`requirement_for` is the
single place that conversion happens, so the two cannot drift apart.

**Unmeasured is a state, not a value, and it propagates.** A quantity that cannot be
defined -- rather than merely being zero or absent -- stays :class:`Unmeasured` and
carries the requirement that was not met. An average over an empty set and an average
over a population that cannot exist are indistinguishable once both are ``0.0``, and
reporting the second as the first is a false statement about the data.

**Warnings are values, not log lines.** They are returned by the computation that
produced them and travel with its result, so a layer that should decline a comparison
can see why. Nothing records a warning as a side effect of reading a dataset: the
record would then describe how someone called a function rather than what the data is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

import numpy as np

TestId = str

T = TypeVar("T")


class TestUnit(str, Enum):
    """What one test id denotes. The harness-checkable half of the semantics split."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    CASE = "case"


class Ordering(str, Enum):
    """Whether the change sequence is real (``observed``) or imposed by the harness."""

    OBSERVED = "observed"
    IMPOSED = "imposed"


class Capability(str, Enum):
    """Optional material a dataset may declare. Absence is a fact, not a zero column."""

    COVERAGE = "coverage"
    DURATIONS = "durations"


class Requirement(str, Enum):
    """What a computation needs in order to be defined.

    A requirement is either a :class:`Capability` of the same spelling, or something
    every dataset supplies (labels, diff text). Computations declare the *material*
    they read; the material names map onto requirements in exactly one place
    (:data:`MATERIAL_REQUIREMENTS`), so a requirement set is derived rather than
    asserted and cannot silently disagree with the code that reads it.
    """

    LABELS = "labels"
    DIFF_TEXT = "diff_text"
    COVERAGE = "coverage"
    DURATIONS = "durations"


def requirement_for(capability: Capability) -> Requirement:
    """The requirement a capability satisfies. The only capability->requirement map."""
    return Requirement(capability.value)


class Policy(str, Enum):
    """A requirement about the *order* rather than about material.

    Kept separate from :class:`Requirement` because the two are genuinely different
    questions, and collapsing them made one id carry three meanings. ``OBSERVED_ORDER``
    asks whether the dataset's sequence is real, which is what history features rest
    on. ``EFFECTIVE_ORDER`` asks whether this *run's* order is still real, which a
    shuffling split discards even for a dataset that has one. A warning about a shuffle
    is therefore not a claim that the dataset lacks an observed order, and no longer
    says so.
    """

    OBSERVED_ORDER = "ordering.observed"
    EFFECTIVE_ORDER = "ordering.effective"


#: Material a computation may be handed is catalogued in :mod:`rts.accessors`, not here.
#: The contract defines the *vocabulary* (a capability is a requirement of the same
#: name) and what a dataset declares; which named material a computation reads, and what
#: that implies, is a harness concern. Keeping one catalogue in one place is what stops
#: the three parallel spellings this used to have -- a capability name, a requirement id,
#: and an inline ``"capability:durations"`` -- from disagreeing.


class CapabilityMissing(Exception):
    """Raised when a capability is requested from a dataset that does not declare it."""

    def __init__(self, dataset: str, capability: Capability | str):
        name = capability.value if isinstance(capability, Capability) else str(capability)
        super().__init__(
            f"dataset {dataset!r} does not declare the {name!r} capability; "
            f"check capabilities() before requesting it"
        )
        self.dataset = dataset
        self.capability = name


# --- Unmeasured values ----------------------------------------------------


@dataclass(frozen=True)
class Unmeasured:
    """A quantity that could not be defined, plus the requirement that was unmet."""

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

    A value, not a sink: warnings are returned by the computation that produced them
    rather than accumulating on the dataset they were derived from. A declaration that
    is only printed is invisible to the layers that should be able to act on it, and a
    record stored on the dataset would describe the *calls* rather than the data.
    """

    def __init__(self, items: Iterable[Warning] = ()):
        self._items: list[Warning] = []
        self._seen: set[tuple[str, str]] = set()
        self.extend(items)

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

    @classmethod
    def from_(cls, *sources: Iterable[Warning]) -> "Warnings":
        """Merge several warning collections into one."""
        out = cls()
        for source in sources:
            out.extend(source)
        return out

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


# --- The dataset contract -------------------------------------------------


class Dataset(ABC):
    """What a dataset *supplies* and *declares*. Nothing here is a statistic.

    A dataset is a plain value: constructing one has no side effects, two may coexist
    in a process, and reading from one has no side effect on any other. The only
    mutable state is a private memo table, which exists so that the derived accessors
    in :mod:`rts.accessors` can share one caching mechanism instead of each
    re-implementing it.
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
        """What was actually executed. Not the same as ``killing_tests``: the difference
        is what separates "passed" from "never ran"."""

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

    # --- optional primitives ---------------------------------------------

    def capabilities(self) -> frozenset[Capability]:
        """Optional material this dataset has. Defaults to none."""
        return frozenset()

    def coverage(self, change: Any) -> frozenset[TestId]:
        """Tests covering the change. Only call when ``coverage`` is declared."""
        raise CapabilityMissing(self.name, Capability.COVERAGE)

    def durations(self) -> Mapping[TestId, float]:
        """Per-test wall-clock durations. Only call when ``durations`` is declared."""
        raise CapabilityMissing(self.name, Capability.DURATIONS)

    def own_candidate_pool(self) -> np.ndarray | None:
        """A per-change candidate mask, when a change's pool is not the whole suite.

        ``None`` -- the default -- means the harness's ``full``/``covered`` modes are the
        only sensible ones, which is true when one suite serves every change. A pooled
        corpus of projects is the case this exists for: a bug's candidates are its *own*
        project's tests, so a budget of a fraction of the union of eight suites would mean
        eight different things.

        An ndarray rather than a path because it is derived from the labels-like material
        the dataset already holds; returning it here keeps the knowledge of *what a pool
        is* with the dataset that has one.
        """
        return None

    # --- declarations -----------------------------------------------------

    @abstractmethod
    def ordering(self) -> Ordering:
        """Whether the change sequence is real. Declared, never inferred."""

    def test_unit(self) -> TestUnit:
        """What one test id denotes. Flattens; see :meth:`semantics`."""
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
        """The change's stable identity, and the key of the change index.

        This is what score caches are keyed by, so it must be stable across processes
        and independent of the change's position in the canonical order.
        """
        return str(getattr(change, "change_id", change))

    def coverage_key(self, change: Any) -> str:
        """The key a coverage map is indexed by. Only meaningful with ``coverage``.

        Deliberately distinct from :meth:`change_id`. mutmut's coverage map is keyed by
        the *mutated function*, not by the mutant, so conflating the two silently
        mis-maps every lookup.
        """
        return str(getattr(change, "func_key", self.change_id(change)))

    def source_counts(self) -> Mapping[str, int]:
        """Counts the *source* can report and the contract cannot derive.

        mutmut reports killed/survived verdicts; a corpus of real bugs reports nothing
        of the sort. Keeping them here is what stops the reporting layer's ``describe``
        from reading an attribute only one source happens to have.
        """
        return {}

    def integrity_notes(self) -> Sequence[Warning]:
        """Facts about this dataset a consumer should know before trusting a number.

        A dataset that diverges from a contract *obligation* records it here -- a
        bundle's ``diff_text`` is a concatenation of diffs rather than one valid diff; a
        pool's constituents may disagree about what a test is. Declaring it where the
        divergence is made keeps the alternative (reporting inferring it from a type
        check) from putting knowledge of every dataset kind in the reporting layer.
        """
        return ()

    @property
    def order_seed(self) -> int:
        """Seed used to materialise the canonical order of an imposed dataset."""
        from . import config

        return config.SEED

    # --- capability queries ----------------------------------------------

    def has_capability(self, capability: Capability | str) -> bool:
        """Whether this dataset declares ``capability``.

        An unrecognised name raises rather than returning ``False``: the vocabulary is
        closed, so ``has_capability("coverge")`` is a typo, and a quiet ``False`` would
        silently report a capability as absent and gate a whole feature block off.
        """
        name = capability.value if isinstance(capability, Capability) else str(capability)
        return Capability(name) in self.capabilities()

    def available_requirements(self) -> frozenset[Requirement]:
        """Requirement ids this dataset can satisfy."""
        available = {Requirement.LABELS, Requirement.DIFF_TEXT}
        for capability in self.capabilities():
            available.add(requirement_for(capability))
        return frozenset(available)

    def missing_requirements(
        self, requirements: Iterable[Requirement]
    ) -> tuple[Requirement, ...]:
        return tuple(sorted(set(requirements) - self.available_requirements(), key=str))

    # --- infrastructure ---------------------------------------------------

    def cached(self, key: str, factory: Callable[[], T]) -> T:
        """Memoise ``factory`` on this dataset under ``key``.

        One mechanism, so the derived accessors do not each re-implement the same
        eight-line idiom -- nine copies of which previously meant that a subclass could
        silently omit all of them, which the pooled dataset did.
        """
        table = self.__dict__.get("_memo")
        if table is None:
            table = {}
            self.__dict__["_memo"] = table
        if key not in table:
            table[key] = factory()
        return table[key]

    # --- index alignment --------------------------------------------------

    @property
    def test_ids(self) -> list[TestId]:
        """The pool as an index-aligned list. Column ``j`` of every matrix is this test.

        Kept on the contract because it is the alignment convention every matrix in the
        harness obeys, not a statistic computed from one.
        """
        return self.cached("test_ids", lambda: list(self.test_pool))

    @property
    def test_index(self) -> dict[TestId, int]:
        """Inverse of :attr:`test_ids`, for mapping a label onto its column."""
        return self.cached("test_index", lambda: {t: i for i, t in enumerate(self.test_ids)})

    @property
    def n_tests(self) -> int:
        return len(self.test_ids)

    @property
    def n_changes(self) -> int:
        return self.cached("n_changes", lambda: len(self.changes))

    # --- audit ------------------------------------------------------------

    def declaration(self) -> dict:
        """The dataset's declarations as a plain dict, for an artifact payload."""
        return {
            "name": self.name,
            "ordering": self.ordering().value,
            "test_unit": self.test_unit().value,
            "capabilities": sorted(c.value for c in self.capabilities()),
            "semantics": dict(self.semantics()),
        }


__all__ = [
    "Capability",
    "CapabilityMissing",
    "Dataset",
    "Ordering",
    "Policy",
    "Requirement",
    "TestId",
    "TestUnit",
    "Unmeasured",
    "Warning",
    "Warnings",
    "first_unmeasured",
    "is_unmeasured",
    "requirement_for",
]
