"""Build a real-bug RTS dataset from BugsInPy (W2a in ``plan_next_steps.md``).

Why BugsInPy
------------
Every number in this study is currently measured on labels that are *defined by coverage*:
mutmut runs only the tests that cover the mutated function. BugsInPy supplies **real
bug-inducing changes and real failing tests**, so it is the cheapest route to labels that are
not circular with respect to the coverage feature.

What it gives, and what it does not
-----------------------------------
It gives: real change text (the bug-inducing commit), real failing tests (``run_test.sh`` per
bug), real test suites, several failing tests per change, and 17 independent projects.

It does **not** give the boundary regime -- these are still co-located unit tests. It is the
"honest labels, real text, no synthetic history" arm, not the embedded arm.

No test execution is required
-----------------------------
Labels come from the per-bug ``run_test.sh``, which names the failing tests directly, so no
project virtualenv is needed. The candidate pool is enumerated by parsing the test files with
``ast`` rather than by running ``pytest --collect-only``, which would require every project's
dependencies. The cost of that shortcut is that parametrized variants collapse to their base
function; noted as a limitation rather than hidden.

Output: ``artifacts/bugsinpy/<project>.json`` per project, each holding the pool once plus
per-bug change text and failing tests.

Usage
-----
    python scripts/build_bugsinpy_dataset.py [--projects tqdm black ...] [--limit N]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
BUGSINPY = WORKSPACE / "sut" / "external" / "BugsInPy"
REPOS = WORKSPACE / "sut" / "external" / "repos"
OUT_DIR = WORKSPACE / "artifacts" / "bugsinpy"

# Small, pure-Python projects with parseable labels. Heavy projects (pandas, keras,
# matplotlib, spacy, scrapy) are excluded: their clones and pools are an order of magnitude
# larger and buy nothing for a feasibility probe.
DEFAULT_PROJECTS = ["tqdm", "cookiecutter", "httpie", "PySnooper", "sanic", "thefuck", "black", "tornado"]

# Directories that hold tests, per project layout.
TEST_DIR_HINTS = ("tests", "test", "testing")


def sh(cmd: list[str], cwd: Path | None = None, check: bool = True) -> str:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {proc.stderr.strip()[:300]}")
    return proc.stdout


def project_url(project: str) -> str:
    info = (BUGSINPY / "projects" / project / "project.info").read_text()
    match = re.search(r'github_url="([^"]+)"', info)
    if not match:
        raise RuntimeError(f"no github_url for {project}")
    return match.group(1)


def ensure_repo(project: str) -> Path:
    repo = REPOS / project
    if repo.exists():
        return repo
    REPOS.mkdir(parents=True, exist_ok=True)
    url = project_url(project)
    print(f"  cloning {url} ...", flush=True)
    sh(["git", "clone", "-q", url, str(repo)])
    return repo


def parse_bug_info(project: str, bug_dir: Path) -> dict:
    info = (bug_dir / "bug.info").read_text()
    fields = dict(re.findall(r'(\w+)="([^"]*)"', info))
    return {
        "bug_id": bug_dir.name,
        "buggy_commit": fields.get("buggy_commit_id", ""),
        "fixed_commit": fields.get("fixed_commit_id", ""),
        "test_file": fields.get("test_file", ""),
    }


def parse_failing_tests(bug_dir: Path) -> list[str]:
    """Extract failing-test node ids from ``run_test.sh``.

    Two runner styles appear in BugsInPy:
        pytest   <file>::<test>                      (already a node id)
        unittest <module>.<Class>.<test>             (needs converting to file::Class::test)
    """
    script = bug_dir / "run_test.sh"
    if not script.exists():
        return []
    ids: list[str] = []
    for line in script.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.extend(_ids_from_command(line))
    return sorted(set(ids))


def _ids_from_command(line: str) -> list[str]:
    out: list[str] = []
    for token in line.split():
        if token.startswith("-"):
            continue
        if "::" in token:
            # pytest style; may be "file::Class::test" or "file::test".
            out.append(_normalise_path(token))
        elif re.fullmatch(r"[A-Za-z_][\w.]*\.test_\w+", token) or re.fullmatch(
            r"[A-Za-z_][\w.]*\.[A-Z]\w*\.test_\w+", token
        ):
            parts = token.split(".")
            test = parts[-1]
            rest = parts[:-1]
            # A class component starts with an upper-case letter.
            if rest and rest[-1][:1].isupper():
                cls = rest[-1]
                module = ".".join(rest[:-1])
                out.append(f"{module.replace('.', '/')}.py::{cls}::{test}")
            else:
                module = ".".join(rest)
                out.append(f"{module.replace('.', '/')}.py::{test}")
    return out


def _normalise_path(nodeid: str) -> str:
    return nodeid.replace("\\", "/").lstrip("./")


def enumerate_tests(repo: Path) -> dict[str, str]:
    """node id -> test-function source, by parsing every test file with ``ast``."""
    found: dict[str, str] = {}
    for path in sorted(repo.rglob("*.py")):
        if any(part in {".git", "build", "dist", ".tox", "node_modules"} for part in path.parts):
            continue
        rel = path.relative_to(repo).as_posix()
        if not (path.name.startswith("test_") or path.name.endswith("_test.py") or "/test" in f"/{rel}"):
            continue
        try:
            source = path.read_text(errors="replace")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue
        lines = source.splitlines()

        def walk(body, prefix: str) -> None:
            for node in body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("test"):
                        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                        end = node.end_lineno or node.lineno
                        nodeid = f"{rel}::{prefix}{node.name}" if prefix else f"{rel}::{node.name}"
                        found.setdefault(nodeid, "\n".join(lines[start - 1 : end]))
                elif isinstance(node, ast.ClassDef):
                    walk(node.body, f"{node.name}::")

        walk(tree.body, "")
    return found


def change_text(repo: Path, commit: str, max_files: int = 8, max_lines: int = 200) -> str:
    """The bug-inducing diff: the commit itself, source files only.

    ``bug_patch.txt`` is the *fix*, so using it would invert the RTS question (applying it
    makes the failing tests pass). The change that introduces the fault is the buggy commit.
    """
    try:
        diff = sh(["git", "show", "--format=", "--unified=3", commit], cwd=repo, check=False)
    except RuntimeError:
        return ""
    chunks: list[str] = []
    keep = False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            keep = not any(
                marker in line for marker in ("/test", "test_", "_test", "tests/")
            )
            if len(chunks) >= max_files:
                break
        if keep:
            chunks.append(line)
    return "\n".join(chunks[:max_lines])


def build_project(project: str, limit: int | None) -> dict:
    print(f"\n=== {project} ===", flush=True)
    repo = ensure_repo(project)
    bug_dirs = sorted((BUGSINPY / "projects" / project / "bugs").iterdir(), key=lambda p: p.name)
    if limit:
        bug_dirs = bug_dirs[:limit]

    test_sources: dict[str, str] = {}
    bugs: list[dict] = []
    for bug_dir in bug_dirs:
        meta = parse_bug_info(project, bug_dir)
        if not meta["buggy_commit"]:
            continue
        failing = parse_failing_tests(bug_dir)
        if not failing:
            continue
        sh(["git", "checkout", "-q", meta["buggy_commit"]], cwd=repo, check=False)
        pool = enumerate_tests(repo)
        test_sources.update(pool)
        text = change_text(repo, meta["buggy_commit"])
        if not text.strip():
            continue
        bugs.append(
            {
                "bug_id": meta["bug_id"],
                "buggy_commit": meta["buggy_commit"],
                "fixed_commit": meta["fixed_commit"],
                "change_text": text,
                "failing": failing,
                "pool": sorted(pool),
                "changed_files": sorted(
                    {l.split(" b/")[-1] for l in text.splitlines() if l.startswith("diff --git")}
                ),
            }
        )
        print(
            f"  bug {meta['bug_id']:>4}: pool={len(pool):4d} failing={len(failing)} "
            f"change_lines={len(text.splitlines())}",
            flush=True,
        )

    return {
        "project": project,
        "url": project_url(project),
        "tests": test_sources,
        "bugs": bugs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects", nargs="*", default=DEFAULT_PROJECTS)
    parser.add_argument("--limit", type=int, default=None, help="max bugs per project")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for project in args.projects:
        try:
            data = build_project(project, args.limit)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  [skip] {project}: {exc}")
            continue
        if not data["bugs"]:
            print(f"  [skip] {project}: no usable bugs")
            continue
        out = OUT_DIR / f"{project}.json"
        out.write_text(json.dumps(data))
        summary[project] = {
            "bugs": len(data["bugs"]),
            "pool": len(data["tests"]),
            "failing_per_bug": round(
                sum(len(b["failing"]) for b in data["bugs"]) / len(data["bugs"]), 2
            ),
            "bytes": out.stat().st_size,
        }
        print(f"  wrote {out.relative_to(WORKSPACE)}  {summary[project]}")

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== summary ===")
    total_bugs = sum(v["bugs"] for v in summary.values())
    total_pairs = sum(v["bugs"] * v["pool"] for v in summary.values())
    for name, row in summary.items():
        print(f"  {name:14s} bugs={row['bugs']:3d} pool={row['pool']:4d} pairs={row['bugs']*row['pool']:6d}")
    print(f"  TOTAL          bugs={total_bugs:3d} pairs={total_pairs:6d}")
    print(f"  SemIf cost at 30 pairs/s: {total_pairs/30/60:.1f} min")


if __name__ == "__main__":
    main()
