"""A fixture-backed dataset, to show the contract is cheap to satisfy.

"A fixture-backed dataset returning ten changes over three tests is complete and valid,
and two of them can exist at once." Nothing here touches a real checkout or a real test
run, and the whole file is importable without one.
"""

from __future__ import annotations

from dataclasses import dataclass

from rts.data.contract import Capability, Dataset, Granularity, Ordering

TEST_IDS = (
    "tests/test_alpha.py::test_one",
    "tests/test_alpha.py::test_two",
    "tests/test_beta.py::test_three",
)

#: (change id, file, diff, killing tests)
CHANGES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "c0",
        "pkg/alpha.py",
        "--- original\n+++ mutated\n@@ -1,1 +1,1 @@\n-    return 1\n+    return 2\n",
        ("tests/test_alpha.py::test_one",),
    ),
    (
        "c1",
        "pkg/alpha.py",
        "--- original\n+++ mutated\n@@ -1,1 +1,1 @@\n-    return 2\n+    return 3\n",
        ("tests/test_alpha.py::test_one", "tests/test_alpha.py::test_two"),
    ),
    (
        "c2",
        "pkg/beta.py",
        "--- original\n+++ mutated\n@@ -1,2 +1,2 @@\n-    x = 1\n-    return x\n+    x = 2\n+    return x\n",
        ("tests/test_beta.py::test_three",),
    ),
    (
        "c3",
        "pkg/beta.py",
        "--- original\n+++ mutated\n@@ -1,1 +1,1 @@\n-    return 0\n+    return 1\n",
        (),
    ),
)

SOURCES = {
    "tests/test_alpha.py::test_one": "def test_one():\n    assert alpha() == 1\n",
    "tests/test_alpha.py::test_two": "def test_two():\n    assert alpha() != 3\n",
    "tests/test_beta.py::test_three": "def test_three():\n    assert beta() == 0\n",
}

COVERAGE = {
    "c0": frozenset({"tests/test_alpha.py::test_one", "tests/test_alpha.py::test_two"}),
    "c1": frozenset({"tests/test_alpha.py::test_one"}),
    "c2": frozenset({"tests/test_beta.py::test_three"}),
    "c3": frozenset({"tests/test_beta.py::test_three"}),
}

DURATIONS = {
    "tests/test_alpha.py::test_one": 1.0,
    "tests/test_alpha.py::test_two": 2.0,
    "tests/test_beta.py::test_three": 0.5,
}


@dataclass(frozen=True)
class StubChange:
    change_id: str
    file: str
    diff: str
    killing: tuple[str, ...]
    #: Extra files this change touches, so a test can exercise the plural accessor.
    extra_files: tuple[str, ...] = ()


class StubDataset(Dataset):
    """Four changes over three tests, with or without the optional capabilities."""

    def __init__(
        self,
        name: str = "stub",
        coverage: bool = True,
        durations: bool = True,
        ordering: Ordering = Ordering.SYNTHETIC,
        test_granularity: Granularity = Granularity.FUNCTION,
        changes: tuple[tuple[str, str, str, tuple[str, ...]], ...] = CHANGES,
        sources=SOURCES,
        extra_files: dict[str, tuple[str, ...]] | None = None,
    ):
        self._name = name
        self._coverage = coverage
        self._durations = durations
        self._ordering = ordering
        self._test_unit = test_granularity
        extras = dict(extra_files or {})
        self._changes = tuple(
            StubChange(
                change_id=cid,
                file=file,
                diff=diff,
                killing=killed,
                extra_files=extras.get(cid, ()),
            )
            for cid, file, diff, killed in changes
        )
        self._sources = dict(sources)

    # --- primitives -------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def changes(self):
        return self._changes

    def files(self, change):
        return (change.file,) + tuple(change.extra_files)

    def diff_text(self, change) -> str:
        return change.diff

    def killing_tests(self, change):
        return frozenset(change.killing)

    def executed_tests(self, change):
        # Every test ran, so the label and the run set differ only by outcome.
        return frozenset(TEST_IDS)

    @property
    def test_suite(self):
        return TEST_IDS

    def test_source(self, test):
        return self._sources.get(test)

    # --- optional ---------------------------------------------------------

    def capabilities(self):
        caps = set()
        if self._coverage:
            caps.add(Capability.COVERAGE)
        if self._durations:
            caps.add(Capability.DURATIONS)
        return frozenset(caps)

    def coverage(self, change):
        if not self._coverage:
            return super().coverage(change)
        return COVERAGE[change.change_id]

    def durations(self):
        if not self._durations:
            return super().durations()
        return dict(DURATIONS)

    # --- declarations -----------------------------------------------------

    def ordering(self) -> Ordering:
        return self._ordering

    def test_granularity(self) -> Granularity:
        return self._test_unit

    def annotations(self):
        return {"change": "stub", "ordering": self._ordering.value}
