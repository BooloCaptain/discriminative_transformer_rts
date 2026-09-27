"""Concrete datasets: three pieces of machinery behind the contract.

A concrete dataset is a **source** (raw material for one SUT or revision), a **sample
generator** (turns source records into samples) and the dataset itself (implements the
primitives over those samples). One pair of generators per data format; **one dataset
per evaluation unit** -- so the eight BugsInPy projects are eight datasets sharing one
source class and one generator, because their suites, pools and meanings are separate
even though they ship the same schema.

Transformation is not a special case. :class:`BundleDataset` wraps a dataset and is
itself a dataset, rather than a parallel code path, which is the shape every other
reshaping experiment (holding out a file, taking one change per function) also takes.
Its feature block is declared in :mod:`rts.features.bundle` like any other, so a bundle
is a dataset with features rather than a feature generator with a dataset attached.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .. import config
from . import accessors, test_source
from .composition import pool
from .contract import (
    Capability,
    CapabilityMissing,
    Dataset,
    Ordering,
    TestId,
    TestUnit,
    Warning,
)
from .sources import Bug, BugsInPySource, MutmutSource, available_bugsinpy_projects


class DerivedDataset(Dataset):
    """Base for datasets that are a function of others.

    A derived dataset **states its own ordering rather than inheriting one**: a
    transformation that maps each derived sample to exactly one point in its base's
    sequence may declare ``observed``; one that draws its parts from anywhere in the base
    is ``imposed``. The harness cannot verify either claim, so the declaration carries
    the responsibility.
    """

    def __init__(self, base: Dataset):
        self._base = base

    @property
    def base(self) -> Dataset:
        return self._base

    def semantics(self) -> Mapping[str, str]:
        notes = dict(self._base.semantics())
        notes["derived_from"] = self._base.name
        return notes


# --- Marshmallow: the mutation-testing SUT ---------------------------------


class MarshmallowDataset(Dataset):
    """One SUT under mutation testing: mutants as changes, the collected suite as pool.

    Three properties are stated as declarations rather than only in prose, because they
    are the study's documented limitations:

    * **The change order is imposed.** The mutant population has no intrinsic temporal
      order, so a fixed-seed permutation defines the sequence. The seed is recorded in
      :meth:`semantics`, and history features are off by default for exactly this reason.
    * **``killing_tests`` is mutmut's label, and under the ``mutmut`` label source it is
      defined by coverage.** mutmut executes only the tests associated with the mutated
      function, so a test outside that set is *assumed* not to fail.
    * **Survivors are retained.** A survived mutant contributes no fault, and so no recall
      denominator, but it is the models' negative training signal.
    """

    def __init__(
        self,
        labels: str = "mutmut",
        sut: Path | None = None,
        order_seed: int = config.SEED,
    ):
        self._source = MutmutSource(labels=labels, sut=sut)
        change_set = self._source.changes()

        # Impose a synthetic temporal order. For an imposed dataset a permutation is free
        # -- every order is equally valid, and the seed is only a reproducibility knob --
        # so applying it at construction is honest and is what makes a run reproduce
        # without threading a seed through every consumer.
        rng = np.random.default_rng(order_seed)
        order = rng.permutation(len(change_set.changes))
        self._changes = tuple(change_set.changes[int(i)] for i in order)
        self._pool = tuple(change_set.pool)
        self._order_seed = int(order_seed)
        self._coverage = self._source.coverage_map()

    # --- primitives -------------------------------------------------------

    @property
    def name(self) -> str:
        return "marshmallow" if self._source.labels == "mutmut" else f"marshmallow_{self._source.labels}"

    @property
    def changes(self) -> Sequence[Any]:
        return self._changes

    def files(self, change: Any) -> tuple[str, ...]:
        return (change.file,)

    def diff_text(self, change: Any) -> str:
        return change.diff_text

    def killing_tests(self, change: Any) -> frozenset[TestId]:
        return frozenset(change.killing_tests)

    def ran_tests(self, change: Any) -> frozenset[TestId]:
        return frozenset(change.ran_tests)

    @property
    def test_pool(self) -> Sequence[TestId]:
        return self._pool

    def test_source(self, test: TestId) -> str | None:
        return test_source.test_source(test, sut=self._source.sut)

    # --- optional primitives ---------------------------------------------

    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.COVERAGE, Capability.DURATIONS})

    def coverage(self, change: Any) -> frozenset[TestId]:
        tests = self._coverage.get(change.func_key, ())
        index = self.test_index
        return frozenset(t for t in tests if t in index)

    def durations(self) -> Mapping[TestId, float]:
        return self._source.duration_by_test()

    # --- declarations -----------------------------------------------------

    def ordering(self) -> Ordering:
        return Ordering.IMPOSED

    def test_unit(self) -> TestUnit:
        return TestUnit.FUNCTION

    def coverage_key(self, change: Any) -> str:
        return change.func_key

    def canonical_test_id(self, test: TestId) -> TestId:
        return self._source.canonical_test_id(test)

    @property
    def order_seed(self) -> int:
        return self._order_seed

    def source_counts(self) -> Mapping[str, int]:
        """mutmut's own verdicts, which no other source can report."""
        return {
            "killed": int(sum(c.killed for c in self._changes)),
            "survived": int(sum(c.survived for c in self._changes)),
        }

    def semantics(self) -> Mapping[str, str]:
        notes = {
            "change": "one mutant; the diff is reconstructed by diffing generated source",
            "ordering": "imposed: mutant population has no intrinsic temporal order; "
            f"canonical order is a permutation at seed {self._order_seed}",
            "killing_tests": "tests that failed; under the mutmut label source these are the "
            "tests mutmut selected, so the label is defined by coverage",
            "ran_tests": "tests mutmut actually executed; the complement of killing_tests "
            "inside this set passed, and everything outside it never ran",
            "test_unit": "one test function; pytest parametrization is collapsed to the base id",
            "survivors": "retained as negative training signal, excluded from the recall denominator",
            "coverage": "mutmut's function-level test association",
            "durations": "wall-clock from the mutmut stats file; hardware-dependent, so "
            "available but not comparable with a duration measured elsewhere",
        }
        if self._source.labels == "full":
            notes["killing_tests"] = (
                "tests that failed when the full suite was executed against the mutant; "
                "not defined by coverage"
            )
            notes["pool"] = (
                "canonical pool of the full-suite run; wall-clock parametrization ids are "
                "collapsed so the pool is stable across collections"
            )
        return notes

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"MarshmallowDataset(labels={self._source.labels!r}, n_changes={self.n_changes})"


# --- BugsInPy: real bugs, one dataset per project ---------------------------


class BugsInPyDataset(Dataset):
    """One BugsInPy project: real bugs, real failing tests, no synthetic history.

    This is the arm whose **labels are not defined by coverage**. BugsInPy's failing tests
    come from the projects' own bug reports, so no feature is circular with respect to
    them -- which is what makes it structurally the traceability ladder's hardest rung on
    real data.

    What it does *not* have is coverage and durations: obtaining them would mean running
    every project's suite at every bug commit. So it declares neither capability, and the
    harness reports those feature columns as unmeasured rather than as an all-zero column
    a model would happily split on.
    """

    def __init__(self, project: str, root: Path | None = None):
        self._source = BugsInPySource(project, root=root)
        self._bugs = self._source.bugs()
        self._sources = self._source.test_sources()
        pool: set[str] = set()
        for bug in self._bugs:
            pool.update(bug.pool)
        self._pool = tuple(sorted(pool))

    # --- primitives -------------------------------------------------------

    @property
    def name(self) -> str:
        return self._source.project

    @property
    def changes(self) -> Sequence[Any]:
        return self._bugs

    def files(self, change: Any) -> tuple[str, ...]:
        return tuple(change.changed_files)

    def diff_text(self, change: Any) -> str:
        return change.change_text

    def killing_tests(self, change: Any) -> frozenset[TestId]:
        return frozenset(change.failing)

    def ran_tests(self, change: Any) -> frozenset[TestId]:
        # No execution record survives in the built dataset: the failing tests came from
        # the project's own bug report, and nothing re-ran the suite. Every pooled test is
        # therefore an eligible candidate, and none of them is known to have passed.
        return frozenset(self._pool)

    @property
    def test_pool(self) -> Sequence[TestId]:
        return self._pool

    def test_source(self, test: TestId) -> str | None:
        return self._sources.get(test)

    def own_candidate_pool(self) -> np.ndarray:
        """A bug's candidates are its own project's enumerated tests.

        Not the whole suite: pooling eight projects means a budget of a fraction of their
        union would mean eight different things.
        """
        mask = np.zeros((len(self._bugs), len(self._pool)), dtype=bool)
        index = {test: j for j, test in enumerate(self._pool)}
        for i, bug in enumerate(self._bugs):
            for test in bug.pool:
                j = index.get(test)
                if j is not None:
                    mask[i, j] = True
        return mask

    # --- declarations -----------------------------------------------------

    def capabilities(self) -> frozenset[Capability]:
        # Deliberately empty: no coverage, no durations. A dataset that faked either would
        # make the arm's central claim ("no circular feature") false.
        return frozenset()

    def ordering(self) -> Ordering:
        return Ordering.IMPOSED

    def test_unit(self) -> TestUnit:
        return TestUnit.CASE

    def change_id(self, change: Any) -> str:
        return change.bug_id

    def integrity_notes(self) -> Sequence[Warning]:
        # ``files`` may be empty or name several paths, so the single-file derived features
        # flatten it. Declared here rather than inferred, because it is a fact about this
        # source's schema rather than about one change.
        if not any(len(b.changed_files) != 1 for b in self._bugs):
            return ()
        return (
            Warning(
                code="dataset.multi_file_changes_flattened",
                requirement="diff_text",
                note="bug commits may touch several files or none; single-file derived "
                "features read the first path",
                scope="bugsinpy",
            ),
        )

    def semantics(self) -> Mapping[str, str]:
        return {
            "change": "one bug-inducing commit; diff_text is the commit's unified diff, "
            "source files only",
            "ordering": "imposed: the builder emits bugs in file order, which is not a "
            "temporal sequence",
            "killing_tests": "the bug report's failing tests; NOT defined by coverage, "
            "which is the arm's whole point",
            "ran_tests": "every pooled test, because nothing was executed; the dataset "
            "records no passed/never-ran distinction",
            "test_unit": "one test case as named by the project's run_test.sh or enumerated "
            "from the suite; parametrization is not collapsed",
            "pool": "tests enumerated by parsing the project's test files with ast, plus any "
            "failing test the enumeration missed",
            "coverage": "absent: obtaining it would mean running every suite at every bug commit",
            "durations": "absent, for the same reason",
            "changed_files": "may be empty or name several files; single-file derived "
            "features use the first",
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"BugsInPyDataset({self.name!r}, n_changes={self.n_changes}, n_tests={self.n_tests})"


# --- Bundles: a derived dataset over another dataset ------------------------


@dataclass(frozen=True)
class Bundle:
    """One change formed by bundling several of a base dataset's changes.

    ``signal`` is the change whose killing tests are the bundle's label, so the label is
    exact rather than approximated: survived distractors have no killing tests by
    construction, and the union of the bundle's kill set is therefore exactly the
    signal's.
    """

    rung: int
    signal: int
    members: tuple[int, ...]

    @property
    def change_id(self) -> str:
        return f"rung{self.rung}#{self.signal}"


class BundleDataset(DerivedDataset):
    """Several base changes presented as one, with the base's answer held fixed.

    Broadening the *change* while holding the *answer* fixed is what makes this the
    manipulation of interest: coverage plus filename matching solves a one-line mutant,
    and the question is whether anything survives a realistic commit.

    Two honest qualifications, both recorded:

    * ``diff_text`` is a concatenation of unified diffs. Its **line structure** is
      preserved, so the derived size features mean exactly what they mean on the base --
      but it is not itself a single valid unified diff, which
      :meth:`integrity_notes` declares.
    * ``capabilities`` is narrower than the base's. Durations do not transfer: a bundle
      spans several mutations and has no wall-clock of its own.
    """

    def __init__(
        self,
        base: Dataset,
        bundles: Sequence[Bundle],
        rung: int,
        pool: str = "signal",
        name: str | None = None,
    ):
        super().__init__(base)
        if pool not in ("signal", "union"):
            raise ValueError(f"unknown pool {pool!r}; expected 'signal' or 'union'")
        self._bundles = tuple(bundles)
        self._rung = rung
        self._pool_mode = pool
        self._name = name or f"{base.name}_bundle_rung{rung}_{pool}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def changes(self) -> Sequence[Any]:
        return self._bundles

    def files(self, change: Any) -> tuple[str, ...]:
        paths = accessors.change_paths(self._base)
        return tuple(sorted({paths[m] for m in change.members}))

    def diff_text(self, change: Any) -> str:
        return "\n".join(
            self._base.diff_text(self._base.changes[m]) for m in change.members
        )

    def killing_tests(self, change: Any) -> frozenset[TestId]:
        out: set[str] = set()
        for m in change.members:
            out.update(self._base.killing_tests(self._base.changes[m]))
        return frozenset(out)

    def ran_tests(self, change: Any) -> frozenset[TestId]:
        out: set[str] = set()
        for m in change.members:
            out.update(self._base.ran_tests(self._base.changes[m]))
        return frozenset(out)

    @property
    def test_pool(self) -> Sequence[TestId]:
        return self._base.test_pool

    def test_source(self, test: TestId) -> str | None:
        return self._base.test_source(test)

    def capabilities(self) -> frozenset[Capability]:
        # Coverage transfers (a bundle is covered by the union of its members'); durations
        # do not (a bundle has no wall-clock of its own).
        if self._base.has_capability(Capability.COVERAGE):
            return frozenset({Capability.COVERAGE})
        return frozenset()

    def coverage(self, change: Any) -> frozenset[TestId]:
        if not self._base.has_capability(Capability.COVERAGE):
            raise CapabilityMissing(self.name, Capability.COVERAGE)
        members = change.members if self._pool_mode == "union" else (change.signal,)
        out: set[str] = set()
        for m in members:
            out.update(self._base.coverage(self._base.changes[m]))
        return frozenset(out)

    def ordering(self) -> Ordering:
        # Declared, never inherited: a bundle draws its parts from anywhere in the base,
        # so it does not map to one point in the base's sequence.
        return Ordering.IMPOSED

    def test_unit(self) -> TestUnit:
        return self._base.test_unit()

    def change_id(self, change: Any) -> str:
        return change.change_id

    @property
    def rung(self) -> int:
        return self._rung

    @property
    def pool_mode(self) -> str:
        return self._pool_mode

    def integrity_notes(self) -> Sequence[Warning]:
        return (
            Warning(
                code="derived.concatenated_diff",
                requirement="diff_text",
                note=(
                    "diff_text concatenates the members' unified diffs; line structure is "
                    "preserved and the size features are exact, but it is not one valid diff"
                ),
                scope="bundle",
            ),
        )

    def semantics(self) -> Mapping[str, str]:
        notes = super().semantics()
        notes.update(
            {
                "change": "a bundle of mutations; the signal's diff first",
                "ordering": "imposed: bundling draws members from anywhere in the base order",
                "candidate_pool": (
                    "the signal change's covered set"
                    if self._pool_mode == "signal"
                    else "the union of the members' covered sets"
                ),
                "durations": "not transferred from the base; a bundle has no wall-clock of its own",
            }
        )
        return notes


# --- Constructors ----------------------------------------------------------


def marshmallow(
    labels: str = "mutmut",
    sut: Path | None = None,
    order_seed: int = config.SEED,
) -> MarshmallowDataset:
    """The study's mutation-testing dataset. ``labels='mutmut'`` reproduces the documented numbers."""
    return MarshmallowDataset(labels=labels, sut=sut, order_seed=order_seed)


def bugsinpy(project: str, root: Path | None = None) -> BugsInPyDataset:
    return BugsInPyDataset(project, root=root)


def bugsinpy_all(root: Path | None = None) -> list[BugsInPyDataset]:
    """One dataset per project, in sorted project order."""
    return [BugsInPyDataset(p, root=root) for p in available_bugsinpy_projects(root)]


def bugsinpy_pooled(root: Path | None = None) -> Dataset:
    """Every project as one evaluation population. Pooling is iteration, so it composes."""
    return pool(bugsinpy_all(root=root), name="bugsinpy")


def bundles(
    base: Dataset,
    rung: int,
    seed: int = config.SEED,
    pool: str = "signal",
) -> BundleDataset:
    """Build the rung's bundles over ``base``. Rung definitions live in ``rts.bundles``.

    The import is deferred because ``rts.bundles`` imports this module: the rung *definitions*
    are a study choice and live with the driver, while the bundled dataset is a dataset.
    """
    from .. import bundles as bundle_module

    picked = bundle_module.make_bundles(base, rung, seed)
    return BundleDataset(base, picked, rung=rung, pool=pool)


__all__ = [
    "Bug",
    "Bundle",
    "BundleDataset",
    "BugsInPyDataset",
    "DerivedDataset",
    "MarshmallowDataset",
    "bugsinpy",
    "bugsinpy_all",
    "bugsinpy_pooled",
    "bundles",
    "marshmallow",
]
