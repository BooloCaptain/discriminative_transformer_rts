"""Load mutmut artifacts and reconstruct one Change per mutant.

mutmut writes three artifacts we join here:

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
from functools import lru_cache
from pathlib import Path

from . import config

MUTANT_SUFFIX_RE = re.compile(r"__mutmut_(\d+|orig)$")


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
    ran_tests: tuple[str, ...]

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

    @property
    def changed_lines(self) -> tuple[str, ...]:
        """Added lines from the diff, excluding the hunk header."""
        return tuple(
            line[1:]
            for line in self.diff_text.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )

    @property
    def removed_lines(self) -> tuple[str, ...]:
        return tuple(
            line[1:]
            for line in self.diff_text.splitlines()
            if line.startswith("-") and not line.startswith("---")
        )

    @property
    def change_size(self) -> int:
        return len(self.changed_lines) + len(self.removed_lines)


# --- Raw artifact loaders -------------------------------------------------


@lru_cache(maxsize=1)
def load_stats() -> dict:
    return json.loads(config.STATS_FILE.read_text())


def coverage_map() -> dict[str, list[str]]:
    """mangled function key -> tests that cover it."""
    return load_stats()["tests_by_mangled_function_name"]


def duration_by_test() -> dict[str, float]:
    return load_stats()["duration_by_test"]


def all_test_nodeids() -> list[str]:
    """Every collected test, which is the candidate set for every change."""
    return sorted(duration_by_test())


def load_verdicts() -> dict[str, int | None]:
    """mutant name -> exit code."""
    verdicts: dict[str, int | None] = {}
    for meta_path in sorted(config.MUTATED_SRC.rglob("*.py.meta")):
        data = json.loads(meta_path.read_text())
        verdicts.update(data.get("exit_code_by_key", {}))
    return verdicts


def load_spans() -> dict[str, dict[str, list[int]]]:
    """mutated source path (relative to mutants/) -> {short mutant name: [start, end]}."""
    spans: dict[str, dict[str, list[int]]] = {}
    for span_path in sorted(config.MUTATED_SRC.rglob("*.py.spans")):
        rel = span_path.relative_to(config.MUTANTS_DIR)
        rel_py = str(rel)[: -len(".spans")]
        spans[rel_py] = json.loads(span_path.read_text())["spans"]
    return spans


def load_outcomes() -> dict[str, dict[str, str]]:
    """mutant name -> {test nodeid: outcome}."""
    outcomes: dict[str, dict[str, str]] = {}
    if not config.OUTCOMES_FILE.exists():
        return outcomes
    with config.OUTCOMES_FILE.open() as fh:
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


# --- Reconstruction -------------------------------------------------------


def module_from_relpath(rel_py: str) -> str:
    """``src/marshmallow/utils.py`` -> ``marshmallow.utils``."""
    path = Path(rel_py)
    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def source_root_rel() -> str:
    """Source root as a path relative to ``mutants/``, e.g. ``src``."""
    return config.MUTATED_SRC.relative_to(config.MUTANTS_DIR).as_posix()


def candidate_relpaths(module: str) -> list[str]:
    """Possible mutated-file paths for a module, relative to ``mutants/``."""
    root = source_root_rel()
    base = module.replace(".", "/")
    return [f"{root}/{base}.py", f"{root}/{base}/__init__.py"]


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


def build_changes(require_outcomes: bool = True) -> list[Change]:
    """Reconstruct every mutant as a Change."""
    verdicts = load_verdicts()
    spans_by_file = load_spans()
    outcomes = load_outcomes()

    # Index spans by short name within each file for O(1) lookup.
    changes: list[Change] = []
    missing_span: list[str] = []

    for change_id, exit_code in verdicts.items():
        short = short_name_of(change_id)
        func_key = func_key_of(change_id)

        # Locate the file via the module prefix.
        module = func_key.rsplit(".", 1)[0] if "." in func_key else ""
        file_spans = None
        rel_py = ""
        for candidate in candidate_relpaths(module):
            if candidate in spans_by_file:
                file_spans = spans_by_file[candidate]
                rel_py = candidate
                break
        if file_spans is None or short not in file_spans:
            missing_span.append(change_id)
            continue

        mutated_path = config.MUTANTS_DIR / rel_py
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
        if require_outcomes and not per_test:
            continue
        killing = tuple(sorted(t for t, o in per_test.items() if o == "failed"))
        ran = tuple(sorted(per_test))

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
                ran_tests=ran,
            )
        )

    if missing_span:
        print(f"[artifacts] skipped {len(missing_span)} mutants without span data")
    changes.sort(key=lambda c: c.change_id)
    return changes


def summary(changes: list[Change]) -> dict:
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


if __name__ == "__main__":
    cs = build_changes()
    for key, value in summary(cs).items():
        print(f"{key:>20}: {value}")
    if cs:
        sample = next((c for c in cs if c.killed), cs[0])
        print("\n--- sample change ---")
        print("id      :", sample.change_id)
        print("file    :", sample.file)
        print("func    :", sample.func_name)
        print("exit    :", sample.exit_code, "| killing tests:", len(sample.killing_tests))
        print("diff    :")
        print(sample.diff_text)
