"""Raw mutmut artifacts, and the sample generator that turns them into changes.

This module is the *format* half of the mutmut source: it knows where the
artifacts live and how to join them, but it holds no module-level state. Every
loader is parameterised by the :class:`Layout` it reads from and, where the
answer depends on it, by the *label source* -- ``mutmut`` (the tests mutmut
selected) or ``full`` (all collected tests). That choice used to be a process
global; it is now an argument, so two checkouts or two label sources can be
loaded in one process.

mutmut writes four artifacts we join here:

* ``mutants/mutmut-stats.json`` --- ``tests_by_mangled_function_name`` maps a
  mangled function key to the tests that cover it, plus per-test durations.
* ``mutants/src/**/*.py.meta`` --- one verdict (exit code) per mutant.
* ``mutants/src/**/*.py.spans`` --- line spans of each mutant's trampoline block
  inside the mutated file, including an ``__mutmut_orig`` baseline per function.
* ``mutmut-test-outcomes.jsonl`` --- per (mutant, test) outcome, captured by our
  own pytest plugin since mutmut does not record it.

Key formats differ between artifacts and are reconciled here:

* verdict / outcome keys:  ``marshmallow.utils.x_is_generator__mutmut_1``
* coverage keys:           ``marshmallow.utils.x_is_generator``
* span keys:               ``x_is_generator__mutmut_1`` (file-local)
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .. import config

MUTANT_SUFFIX_RE = re.compile(r"__mutmut_(\d+|orig)$")

# ``tests/test_deserialization.py`` parametrizes over ``dt.datetime.now()``, so two node ids
# embed the collection wall-clock and change on every fresh collection:
#     test_invalid_datetime_deserialization[13:45:28 2026-09-24]
#     test_invalid_datetime_deserialization[09-24-2026 13:45:28]
# The stats file was written by one collection, so any *other* collection produces ids that
# cannot be looked up: they can never be selected, and they read as out-of-coverage killers.
# Canonicalising both sides to a placeholder makes the pool stable across collections. Applied
# only under the ``full`` label source, where the pool is rebuilt from a fresh collection.
_TS_PARAMS = (
    re.compile(r"\[\d{2}:\d{2}:\d{2} \d{4}-\d{2}-\d{2}\]"),
    re.compile(r"\[\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}\]"),
)


def canonical_nodeid(nodeid: str) -> str:
    """Collapse wall-clock parametrization ids to ``[<TS>]``."""
    out = nodeid
    for pattern in _TS_PARAMS:
        out = pattern.sub("[<TS>]", out)
    return out


def _canonicalize_ids(ids) -> list[str]:
    return [canonical_nodeid(t) for t in ids]


@dataclass(frozen=True)
class Layout:
    """Where one mutmut checkout's artifacts live.

    Passed in rather than read from :mod:`config`, so a test-time dataset needs
    neither a real checkout nor a real mutmut run (§7 of the refactor design).
    """

    sut: Path

    @property
    def mutants_dir(self) -> Path:
        return self.sut / "mutants"

    @property
    def mutated_src(self) -> Path:
        return self.mutants_dir / "src"

    @property
    def stats_file(self) -> Path:
        return self.mutants_dir / "mutmut-stats.json"

    @property
    def outcomes_file(self) -> Path:
        return self.sut / "mutmut-test-outcomes.jsonl"

    @property
    def full_suite_outcomes_file(self) -> Path:
        return self.sut / "mutmut-full-suite-outcomes.jsonl"

    @property
    def full_suite_tests_file(self) -> Path:
        return self.sut / "mutmut-full-suite-tests.json"

    def outcome_log(self, labels: str) -> Path:
        """The outcome log for a label source."""
        return self.full_suite_outcomes_file if labels == "full" else self.outcomes_file

    def require(self, labels: str) -> None:
        """Fail loudly with an actionable message when the label source is not built."""
        if labels == "full" and not self.full_suite_tests_file.exists():
            raise SystemExit(
                f"missing {self.full_suite_tests_file}; run "
                "scripts/data/emit_full_suite_outcomes.py first"
            )


@dataclass(frozen=True)
class Change:
    """One synthetic change, i.e. one mutant."""

    change_id: str
    file: str
    module: str
    func_key: str
    short_name: str
    start_line: int
    end_line: int
    orig_code: str
    mutated_code: str
    diff_text: str
    exit_code: int | None
    killing_tests: tuple[str, ...]
    executed_tests: tuple[str, ...]

    @property
    def killed(self) -> bool:
        return self.exit_code in config.KILLED_EXIT_CODES

    @property
    def survived(self) -> bool:
        return self.exit_code == config.SURVIVED_EXIT_CODE

    @property
    def func_name(self) -> str:
        """Human-readable function name, e.g. ``Validator._repr_args``."""
        tail = self.func_key.rsplit(".", 1)[-1]
        return tail.replace("x\u01c1", ".").removeprefix("x_")


@dataclass(frozen=True)
class ChangeSet:
    """Every change from one source, plus the pool they were run against.

    Returning the pool alongside the changes replaces the module global that
    previously carried it, so a label source's pool cannot leak between two
    datasets that share a process.
    """

    changes: tuple[Change, ...]
    pool: tuple[str, ...]
    labels: str


# --- Raw artifact loaders -------------------------------------------------


class MutmutArtifacts:
    """Loaders over one mutmut checkout. Caches per instance, never per module."""

    def __init__(self, layout: Layout):
        self.layout = layout
        self._stats: dict | None = None

    # -- stats -------------------------------------------------------------

    def stats(self) -> dict:
        if self._stats is None:
            self._stats = json.loads(self.layout.stats_file.read_text())
        return self._stats

    def coverage_map(self, labels: str = "mutmut") -> dict[str, list[str]]:
        """mangled function key -> tests that cover it."""
        raw = self.stats()["tests_by_mangled_function_name"]
        if labels == "full":
            return {k: _canonicalize_ids(v) for k, v in raw.items()}
        return raw

    def duration_by_test(self) -> dict[str, float]:
        return self.stats()["duration_by_test"]

    def test_suite(self, labels: str = "mutmut") -> tuple[str, ...]:
        """Every collected test, which is the candidate set for every change.

        Under the ``full`` label source this is the canonical pool recorded by the
        full-suite run (1189 tests: the two wall-clock parametrizations merged, and
        the three tests mutmut deselects included). Otherwise it is the stats
        file's 1187 tests, unchanged.
        """
        if labels == "full":
            self.layout.require(labels)
            return tuple(json.loads(self.layout.full_suite_tests_file.read_text())["test_ids"])
        return tuple(sorted(self.duration_by_test()))

    # -- verdicts, spans, outcomes ----------------------------------------

    def verdicts(self) -> dict[str, int | None]:
        """mutant name -> exit code."""
        verdicts: dict[str, int | None] = {}
        for meta_path in sorted(self.layout.mutated_src.rglob("*.py.meta")):
            data = json.loads(meta_path.read_text())
            verdicts.update(data.get("exit_code_by_key", {}))
        return verdicts

    def spans(self) -> dict[str, dict[str, list[int]]]:
        """mutated source path (relative to mutants/) -> {short mutant name: [start, end]}."""
        spans: dict[str, dict[str, list[int]]] = {}
        for span_path in sorted(self.layout.mutated_src.rglob("*.py.spans")):
            rel = span_path.relative_to(self.layout.mutants_dir)
            rel_py = str(rel)[: -len(".spans")]
            spans[rel_py] = json.loads(span_path.read_text())["spans"]
        return spans

    def outcomes(self, labels: str = "mutmut") -> dict[str, dict[str, str]]:
        """mutant name -> {test nodeid: outcome}, from whichever label source is active.

        Under the ``full`` label source the log contains failures only (every test
        runs, so the complement is "passed"); ``build_changes`` reconstructs ``ran``
        as the whole pool.
        """
        outcomes: dict[str, dict[str, str]] = {}
        path = self.layout.outcome_log(labels)
        if not path.exists():
            return outcomes
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                mutant = record["mutant"]
                if mutant in config.NON_MUTANT_SENTINELS:
                    continue
                # A test that fails in setup/teardown still did not pass.
                previous = outcomes.setdefault(mutant, {}).get(record["nodeid"])
                if previous == "failed":
                    continue
                outcomes[mutant][record["nodeid"]] = record["outcome"]
        return outcomes

    # -- reconstruction ----------------------------------------------------

    def build_changes(
        self, labels: str = "mutmut", require_outcomes: bool = True
    ) -> ChangeSet:
        """Reconstruct every mutant as a :class:`Change`."""
        if labels not in config.LABEL_SOURCES:
            raise ValueError(f"unknown label source: {labels!r}")
        self.layout.require(labels)

        verdicts = self.verdicts()
        spans_by_file = self.spans()
        outcomes = self.outcomes(labels)
        pool = self.test_suite(labels)

        # Under full-suite labels every collected test ran against every mutant, so a
        # mutant with no failure records is a *survivor*, not a missing run. The run
        # set is recorded alongside the pool so the two can be told apart, and
        # survivors are kept because they are the dataset's negative signal.
        ran_set: set[str] | None = None
        if labels == "full":
            meta = json.loads(self.layout.full_suite_tests_file.read_text())
            ran_set = set(meta["mutants"])

        changes: list[Change] = []
        missing_span: list[str] = []

        for change_id, exit_code in verdicts.items():
            short = short_name_of(change_id)
            func_key = func_key_of(change_id)

            # Locate the file via the module prefix.
            module = func_key.rsplit(".", 1)[0] if "." in func_key else ""
            file_spans = None
            rel_py = ""
            for candidate in self.candidate_relpaths(module):
                if candidate in spans_by_file:
                    file_spans = spans_by_file[candidate]
                    rel_py = candidate
                    break
            if file_spans is None or short not in file_spans:
                missing_span.append(change_id)
                continue

            mutated_path = self.layout.mutants_dir / rel_py
            lines = mutated_path.read_text().splitlines(keepends=True)
            span = file_spans[short]

            orig_short = func_key.rsplit(".", 1)[-1] + "__mutmut_orig"
            orig_span = file_spans.get(orig_short)
            if orig_span is None:
                missing_span.append(change_id)
                continue

            mutated_code = canonicalize(snippet_for(lines, span), short)
            orig_code = canonicalize(snippet_for(lines, orig_span), orig_short)

            diff_text = "".join(
                difflib.unified_diff(
                    orig_code.splitlines(keepends=True),
                    mutated_code.splitlines(keepends=True),
                    fromfile="original",
                    tofile="mutated",
                    n=1,
                )
            )

            per_test = outcomes.get(change_id, {})
            if labels == "full":
                if ran_set is not None and change_id not in ran_set:
                    continue
            elif require_outcomes and not per_test:
                continue
            killing = tuple(sorted(t for t, o in per_test.items() if o == "failed"))
            # Under full-suite labels every collected test ran against every mutant, so
            # "ran" is the whole pool; the log records only failures. Under mutmut labels
            # the log is the authoritative record of what was actually selected.
            ran = pool if labels == "full" else tuple(sorted(per_test))

            changes.append(
                Change(
                    change_id=change_id,
                    file=rel_py,
                    module=module,
                    func_key=func_key,
                    short_name=short,
                    start_line=span[0],
                    end_line=span[1],
                    orig_code=orig_code,
                    mutated_code=mutated_code,
                    diff_text=diff_text,
                    exit_code=exit_code,
                    killing_tests=killing,
                    executed_tests=ran,
                )
            )

        if missing_span:
            print(f"[artifacts] skipped {len(missing_span)} mutants without span data")
        changes.sort(key=lambda c: c.change_id)
        return ChangeSet(changes=tuple(changes), pool=pool, labels=labels)

    def source_root_rel(self) -> str:
        """Source root as a path relative to ``mutants/``, e.g. ``src``."""
        return self.layout.mutated_src.relative_to(self.layout.mutants_dir).as_posix()

    def candidate_relpaths(self, module: str) -> list[str]:
        """Possible mutated-file paths for a module, relative to ``mutants/``."""
        root = self.source_root_rel()
        base = module.replace(".", "/")
        return [f"{root}/{base}.py", f"{root}/{base}/__init__.py"]


# --- Naming helpers -------------------------------------------------------


def module_from_relpath(rel_py: str) -> str:
    """``src/marshmallow/utils.py`` -> ``marshmallow.utils``."""
    path = Path(rel_py)
    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def short_name_of(change_id: str) -> str:
    """``marshmallow.utils.x_is_generator__mutmut_1`` -> ``x_is_generator__mutmut_1``.

    Mangled nesting uses ``xǁ`` instead of ``.``, so the last dot separates module
    from function reliably.
    """
    return change_id.rsplit(".", 1)[-1]


def func_key_of(change_id: str) -> str:
    """Strip the ``__mutmut_N`` suffix to get the coverage key."""
    return MUTANT_SUFFIX_RE.sub("", change_id)


def canonicalize(snippet: str, mangled_name: str) -> str:
    """Rename the trampoline function so the diff shows the real mutation only.

    Without this every diff would contain a spurious ``def`` line rename.
    """
    pattern = re.compile(r"((?:async\s+)?def\s+)" + re.escape(mangled_name) + r"(\s*\()")
    return pattern.sub(r"\1func\2", snippet)


def snippet_for(lines: list[str], span: list[int]) -> str:
    start, end = span
    return "".join(lines[start - 1 : end])


def summary(changes) -> dict:
    killed = [c for c in changes if c.killed]
    return {
        "changes": len(changes),
        "killed": len(killed),
        "survived": sum(1 for c in changes if c.survived),
        "with_killing_tests": sum(1 for c in changes if c.killing_tests),
        "files": len({c.file for c in changes}),
        "functions": len({c.func_key for c in changes}),
        "empty_diff": sum(1 for c in changes if not c.diff_text.strip()),
    }


def default_layout() -> Layout:
    """The study's pinned checkout. Constructed here so the pin lives in one place."""
    return Layout(sut=config.SUT)


if __name__ == "__main__":
    cs = MutmutArtifacts(default_layout()).build_changes()
    for key, value in summary(cs.changes).items():
        print(f"{key:>20}: {value}")
    if cs.changes:
        sample = next((c for c in cs.changes if c.killed), cs.changes[0])
        print("\n--- sample change ---")
        print("id      :", sample.change_id)
        print("file    :", sample.file)
        print("func    :", sample.func_name)
        print("exit    :", sample.exit_code, "| killing_tests:", len(sample.killing_tests))
        print("diff    :")
        print(sample.diff_text)
