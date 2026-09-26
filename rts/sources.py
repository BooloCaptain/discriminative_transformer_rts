"""Sources: the raw material for one SUT or revision, holding no module-level state.

A *source* is the first of the three pieces a concrete dataset is assembled from
(§3 of the refactor design). It knows how to read one on-disk format and nothing
about evaluation; one pair of sources covers a whole family of datasets, which is
why the eight BugsInPy projects share :class:`BugsInPySource` rather than eight
copies of the same reader.

Keeping the pin (which checkout, which label source, which project) in the
constructor rather than in a module global is what makes two datasets able to
coexist in one process -- the property §7's testability argument rests on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import artifacts, config
from .artifacts import Change, ChangeSet, Layout, MutmutArtifacts

BUGSINPY_DIR = config.ARTIFACTS / "bugsinpy"
BUGSINPY_SUMMARY = "summary.json"


class MutmutSource:
    """Raw material for one mutmut checkout, under one label source.

    ``labels`` selects which outcome log defines the fault labels:

    * ``mutmut`` -- mutmut's selected tests only (the historical, documented default)
    * ``full``   -- every collected test (the corrected labels)

    This used to be a process-wide global (``config.LABELS``), on the argument that
    every selector, feature and evaluation must agree on it. The argument is right;
    the mechanism was not. Agreement is now achieved by handing *one source* to every
    consumer of a run, which cannot disagree with itself and does not prevent a second
    source existing beside it.
    """

    def __init__(self, labels: str = "mutmut", sut: Path | None = None):
        if labels not in artifacts.LABEL_SOURCES:
            raise ValueError(
                f"unknown label source: {labels!r}; expected one of {artifacts.LABEL_SOURCES}"
            )
        self.labels = labels
        self.sut = Path(sut) if sut is not None else config.SUT
        self.layout = Layout(sut=self.sut)
        self.artifacts = MutmutArtifacts(self.layout)

    # --- raw accessors ----------------------------------------------------

    def changes(self, require_outcomes: bool = True) -> ChangeSet:
        """Every mutant as a change, plus the pool it was run against."""
        return self.artifacts.build_changes(self.labels, require_outcomes=require_outcomes)

    def test_pool(self) -> tuple[str, ...]:
        return self.artifacts.test_pool(self.labels)

    def coverage_map(self) -> dict[str, list[str]]:
        return self.artifacts.coverage_map(self.labels)

    def duration_by_test(self) -> dict[str, float]:
        return self.artifacts.duration_by_test()

    def canonical_test_id(self, test: str) -> str:
        """Collapse unstable parametrization ids, but only when the pool was rebuilt."""
        return artifacts.canonical_nodeid(test) if self.labels == "full" else test

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"MutmutSource(labels={self.labels!r}, sut={str(self.sut)!r})"


@dataclass(frozen=True)
class Bug:
    """One real bug, as recorded by the BugsInPy dataset builder.

    ``change_text`` is a unified diff (``git show --unified=3`` of the bug-inducing
    commit, source files only), so it satisfies the contract's obligation on
    ``diff_text`` without a transformation step.
    """

    project: str
    bug_id: str
    change_text: str
    pool: tuple[str, ...]
    failing: tuple[str, ...]
    changed_files: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.project}/{self.bug_id}"

    @property
    def change_id(self) -> str:
        return self.key


class BugsInPySource:
    """Raw material for one BugsInPy project.

    One project is one dataset, because a project's suite is what tests are run
    against and pooling is a decision an experiment makes rather than a property of
    the data (§1). The eight projects ship the same schema, so this one class reads
    all of them -- it only differs by which project it is pointed at.
    """

    def __init__(self, project: str, root: Path | None = None):
        self.project = project
        self.root = Path(root) if root is not None else BUGSINPY_DIR
        self.path = self.root / f"{project}.json"
        if not self.path.exists():
            raise FileNotFoundError(
                f"missing {self.path}; run scripts/build_bugsinpy_dataset.py first"
            )
        self._data: dict | None = None

    @property
    def data(self) -> dict:
        if self._data is None:
            self._data = json.loads(self.path.read_text())
        return self._data

    def test_sources(self) -> dict[str, str]:
        """node id -> test source, for every test the project enumerates."""
        return dict(self.data.get("tests", {}))

    def bugs(self) -> tuple[Bug, ...]:
        """Every usable bug in the project.

        A failing test is always forced into its bug's pool. ``run_test.sh`` names the
        failing tests directly and the pool is enumerated by parsing test files with
        ``ast``, so the two can disagree -- a parametrized variant, or a test file the
        enumeration skipped. Dropping such a label would silently make a bug uncaught
        by every selector, so it is added instead.
        """
        tests = self.test_sources()
        out: list[Bug] = []
        for raw in self.data.get("bugs", []):
            failing = [t for t in raw["failing"] if t in tests]
            if not failing:
                continue
            pool = {t for t in raw["pool"] if t in tests}
            pool |= {t for t in failing if t not in pool}
            if not pool:
                continue
            out.append(
                Bug(
                    project=self.project,
                    bug_id=raw["bug_id"],
                    change_text=raw["change_text"],
                    pool=tuple(sorted(pool)),
                    failing=tuple(failing),
                    changed_files=tuple(raw.get("changed_files", [])),
                )
            )
        return tuple(out)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"BugsInPySource(project={self.project!r})"


def available_bugsinpy_projects(root: Path | None = None) -> list[str]:
    """Every project with a built dataset file, sorted."""
    directory = Path(root) if root is not None else BUGSINPY_DIR
    return sorted(
        path.stem for path in directory.glob("*.json") if path.name != BUGSINPY_SUMMARY
    )


__all__ = [
    "Bug",
    "BugsInPySource",
    "Change",
    "ChangeSet",
    "MutmutSource",
    "available_bugsinpy_projects",
]
