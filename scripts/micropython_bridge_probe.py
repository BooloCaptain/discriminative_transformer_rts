"""MicroPython bridge probe (W2c scoping spike, gate T0 on the target structure).

Why MicroPython
---------------
The target regime is a test suite that drives an embedded system **across a boundary**, so the
changed code does not run in the test process. MicroPython is the closest public match:
``tests/run-tests.py`` executes the interpreter under test with ``subprocess.Popen`` and even
``pty.openpty()``, and the code under test lives in ``py/``, ``extmod/`` and ``ports/`` -- C,
outside the host process. Coverage therefore cannot cross the boundary, and the test suite is
~1650 script-style tests, which is the "thousands of tests" scale.

What this measures, and why it is a proxy
----------------------------------------
MicroPython ships no bug dataset, and building it to obtain real failing tests needs a C
toolchain per revision. So this probe uses a **co-change proxy**: a commit that modifies both
source and a test file is evidence that the test is related to the change. That is weaker than
a failing-test label, and it is stated as a limitation.

The measurement is nevertheless exactly the one the hypothesis needs. For each such commit it
asks a real RTS question -- *rank the whole test suite by the change text, and see where the
related test lands* -- and reports:

* **recall@budget**: is the co-changed test in the top k of the full pool? This is the same
  metric the marshmallow study reports, on the target structure and at the target pool size.
* **lexical bridge rate**: does the co-changed test share any token with the change at all?

If recall is at the random floor and the bridge rate is near zero, then no text model can work
in this regime and the direction should be dropped or re-specified. If recall is above the
floor but well below what the marshmallow benchmark shows, the regime is the one the
hypothesis predicts and building a real label set is justified.

Usage
-----
    python scripts/micropython_bridge_probe.py [--commits 200]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from rts import config, features  # noqa: E402

REPO = WORKSPACE / "sut" / "external" / "micropython"
OUT = config.ARTIFACTS / "micropython_bridge_probe.json"
BUDGETS = (0.001, 0.01, 0.05, 0.1)

SOURCE_DIRS = ("py/", "extmod/", "shared/")
TEST_DIR = "tests/"
TEST_SUFFIXES = (".py", ".py.exp")


def sh(args: list[str], cwd: Path = REPO) -> str:
    proc = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    return proc.stdout


def is_test_path(path: str) -> bool:
    return path.startswith(TEST_DIR) and path.endswith(TEST_SUFFIXES) and "/assets/" not in path


def mine_commits(limit: int) -> list[dict]:
    """Commits that touch source and at least one test, with the file lists."""
    raw = sh(
        [
            "git", "log", "--no-merges", f"-n{limit * 8}", "--format=@@%H",
            "--name-only", "--", *SOURCE_DIRS, TEST_DIR,
        ]
    )
    commits: list[dict] = []
    sha = None
    files: list[str] = []

    def flush() -> None:
        if sha is None:
            return
        src = [f for f in files if f.startswith(SOURCE_DIRS)]
        tst = [f for f in files if is_test_path(f)]
        if src and tst:
            commits.append({"sha": sha, "source": src, "tests": tst})

    for line in raw.splitlines():
        if line.startswith("@@"):
            flush()
            sha, files = line[2:].strip(), []
        elif line.strip():
            files.append(line.strip())
    flush()
    return commits[:limit]


def pool_at_head() -> dict[str, str]:
    """Test file -> content, at HEAD. A stable pool makes the ranking comparable."""
    out: dict[str, str] = {}
    for path in sorted(REPO.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO).as_posix()
        if not is_test_path(rel) or rel.endswith(".exp"):
            continue
        try:
            out[rel] = path.read_text(errors="replace")
        except OSError:
            continue
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commits", type=int, default=200)
    args = parser.parse_args()

    pool = pool_at_head()
    print(f"[mpy] pool: {len(pool)} test files")
    if not pool:
        raise SystemExit("no tests found; is the MicroPython clone present?")

    commits = mine_commits(args.commits)
    print(f"[mpy] co-change commits mined: {len(commits)}")

    docs = [pool[k] for k in pool]
    keys = list(pool)
    index_of = {k: i for i, k in enumerate(keys)}
    scorer = features.BM25Scorer().fit(docs)

    rows: list[dict] = []
    for commit in commits:
        diff = sh(["git", "show", "--format=", "--unified=0", commit["sha"], "--", *SOURCE_DIRS])
        if not diff.strip():
            continue
        change_tokens = set(features.tokenize(diff))
        if not change_tokens:
            continue
        scores = scorer.score(diff)
        order = np.argsort(-scores, kind="stable")
        rank_of = {int(idx): r for r, idx in enumerate(order)}

        targets = [t for t in commit["tests"] if t in index_of]
        if not targets:
            continue
        target_cols = {index_of[t] for t in targets}
        best_rank = min(rank_of[c] for c in target_cols)
        overlaps = {t: len(change_tokens & set(features.tokenize(pool[t]))) for t in targets}

        rows.append(
            {
                "sha": commit["sha"],
                "pool": len(keys),
                "n_targets": len(targets),
                "best_rank": best_rank,
                "targets": targets,
                "overlap_max": max(overlaps.values()),
                "change_tokens": len(change_tokens),
            }
        )

    if not rows:
        raise SystemExit("no usable commits")

    n_pool = len(keys)
    print(f"\n[mpy] {len(rows)} usable commits, pool {n_pool} tests")
    print("  lexical RTS on MicroPython co-change (full pool, per-change budget)")
    results: dict[str, dict] = {}
    for budget in BUDGETS:
        k = max(1, math.ceil(budget * n_pool))
        hits = [int(r["best_rank"] < k) for r in rows]
        results[f"{budget:.3f}"] = {
            "k": k,
            "recall": float(np.mean(hits)),
            "random_floor": k / n_pool,
            "n": len(hits),
        }
        print(
            f"    b{budget:<6.3f} k={k:4d}  recall={np.mean(hits):.4f}   "
            f"random floor={k/n_pool:.4f}   lift={(np.mean(hits)/(k/n_pool)):.2f}x"
        )

    ranks = np.array([r["best_rank"] for r in rows], dtype=np.float64)
    bridge = float(np.mean([r["overlap_max"] > 0 for r in rows]))
    print(f"\n  median rank of the co-changed test : {np.median(ranks):.0f} of {n_pool}")
    print(f"  mean percentile                    : {(ranks/n_pool).mean():.3f} (0.5 = random)")
    print(f"  co-changed test shares ANY token   : {bridge:.1%}")

    OUT.write_text(
        json.dumps(
            {
                "repo": "micropython",
                "pool": n_pool,
                "commits_mined": len(commits),
                "commits_used": len(rows),
                "budgets": results,
                "median_rank": float(np.median(ranks)),
                "mean_percentile": float((ranks / n_pool).mean()),
                "bridge_rate": bridge,
                "label_proxy": "co-change (commit touches source and the test file)",
                "boundary": (
                    "tests/run-tests.py drives the interpreter under test with subprocess.Popen "
                    "and pty.openpty(); the changed code is C in py/, extmod/, ports/ and does "
                    "not run in the test process, so coverage cannot cross the boundary."
                ),
                "rows": rows[:200],
            },
            indent=2,
        )
    )
    print(f"\n[mpy] wrote {OUT.relative_to(WORKSPACE)}")


if __name__ == "__main__":
    main()
