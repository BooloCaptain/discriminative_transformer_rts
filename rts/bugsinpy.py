"""Evaluate selectors on real BugsInPy bugs (W2a in ``plan_next_steps.md``).

This is the arm with **real labels that are not defined by coverage**. Everything in
``implementation.md`` is currently measured on labels mutmut produced by running only the tests
that cover the mutated function, which makes the coverage feature circular with respect to the
labels. BugsInPy's failing tests come from the projects' own bug reports, so no feature is
circular with respect to them.

What is available, and what is not
----------------------------------
Available: real change text (the bug-inducing commit), real failing tests, real test suites,
several projects, and **no synthetic history at all** -- which is exactly the uniformly-cold
history condition of the target regime.

Not available: coverage and history features, because obtaining them would mean running every
project's suite at every bug commit. So this arm is *structurally* the ladder's L3 rung --
no coverage, no traceability, no history -- on real data rather than on a synthetic SUT. The
selectors evaluated here are therefore the ones that survive that: random, BM25, SemIf, and a
tree over change-intrinsic and test-intrinsic features only.

Stages
------
``audit``   gate T0: does the killing test share any text with the change at all? If not, no
            text model can work and the direction should be dropped.
``bm25``    BM25 + random + structural-free tree, no GPU.
``semif``   score SemIf over the (bug, test) pairs, cached and resumable.
``report``  recall sweep and paired comparisons from whatever is cached.

Usage
-----
    python -m rts.bugsinpy --stage audit
    python -m rts.bugsinpy --stage bm25
    python -m rts.bugsinpy --stage semif
    python -m rts.bugsinpy --stage report
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config, evaluate, features

BUGSINPY_DIR = config.ARTIFACTS / "bugsinpy"
SEMIF_CACHE = config.ARTIFACTS / "semif_scores_bugsinpy.jsonl"
BUDGETS = (0.01, 0.05, 0.1, 0.2)
N_BOOTSTRAP = 2000


@dataclass
class Bug:
    project: str
    bug_id: str
    change_text: str
    pool: list[str]
    failing: list[str]
    changed_files: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.project}/{self.bug_id}"


def load_bugs(projects: list[str] | None = None) -> tuple[list[Bug], dict[str, str]]:
    """Return every usable bug plus a global node-id -> test-source map.

    A failing test is always forced into its bug's pool. ``run_test.sh`` names the failing
    tests directly and the pool is enumerated by parsing test files with ``ast``, so the two
    can disagree -- a parametrized variant, or a test file the enumeration skipped. Dropping
    such a label would silently make a bug uncaught by every selector, so it is added instead,
    and the rate at which this happens is reported by ``describe_pools``.
    """
    bugs: list[Bug] = []
    sources: dict[str, str] = {}
    forced = 0
    for path in sorted(BUGSINPY_DIR.glob("*.json")):
        if path.name == "summary.json":
            continue
        data = json.loads(path.read_text())
        project = data["project"]
        if projects and project not in projects:
            continue
        tests = data["tests"]
        sources.update(tests)
        for raw in data["bugs"]:
            failing = [t for t in raw["failing"] if t in tests]
            if not failing:
                continue
            pool = set(t for t in raw["pool"] if t in tests)
            missing = [t for t in failing if t not in pool]
            forced += len(missing)
            pool |= set(missing)
            if not pool:
                continue
            bugs.append(
                Bug(
                    project=project,
                    bug_id=raw["bug_id"],
                    change_text=raw["change_text"],
                    pool=sorted(pool),
                    failing=failing,
                    changed_files=raw.get("changed_files", []),
                )
            )
    return bugs, sources


def describe_pools(bugs: list[Bug]) -> dict:
    sizes = [len(b.pool) for b in bugs]
    n_fail = [len(b.failing) for b in bugs]
    return {
        "bugs": len(bugs),
        "pool_median": float(np.median(sizes)) if sizes else 0.0,
        "pool_min": min(sizes) if sizes else 0,
        "pool_max": max(sizes) if sizes else 0,
        "failing_median": float(np.median(n_fail)) if n_fail else 0.0,
        "failing_max": max(n_fail) if n_fail else 0,
        "single_failing_bugs": sum(1 for n in n_fail if n == 1),
    }


# --- gate T0: the textual-bridge audit -------------------------------------


def audit_bridge(bugs: list[Bug], sources: dict[str, str]) -> dict:
    """Does the failing test share any text with the change?

    This decides whether a *text* model can work at all. If the killing test and the change
    share no vocabulary, then neither BM25 nor a semantic reranker has anything to condition
    on, and a null result would be uninformative about the models -- it would only say the
    information is not in the text.
    """
    per_project: dict[str, dict] = {}
    rows: list[dict] = []
    for bug in bugs:
        change_tokens = set(features.tokenize(bug.change_text))
        if not change_tokens:
            continue
        overlaps = {}
        for nodeid in bug.pool:
            test_tokens = set(features.tokenize(sources.get(nodeid, "")))
            overlaps[nodeid] = len(change_tokens & test_tokens)
        fail_overlap = [overlaps[t] for t in bug.failing]
        others = [v for t, v in overlaps.items() if t not in set(bug.failing)]
        rows.append(
            {
                "bug": bug.key,
                "change_tokens": len(change_tokens),
                "pool": len(bug.pool),
                "fail_overlap_max": max(fail_overlap) if fail_overlap else 0,
                "fail_overlap_mean": float(np.mean(fail_overlap)) if fail_overlap else 0.0,
                "others_mean": float(np.mean(others)) if others else 0.0,
                "others_max": max(others) if others else 0,
                "shares_any": bool(max(fail_overlap) > 0) if fail_overlap else False,
            }
        )

    for row in rows:
        proj = row["bug"].split("/")[0]
        agg = per_project.setdefault(
            proj, {"bugs": 0, "shares_any": 0, "fail_mean": [], "other_mean": [], "rank_of_fail": []}
        )
        agg["bugs"] += 1
        agg["shares_any"] += int(row["shares_any"])
        agg["fail_mean"].append(row["fail_overlap_mean"])
        agg["other_mean"].append(row["others_mean"])

    summary = {}
    for proj, agg in per_project.items():
        summary[proj] = {
            "bugs": agg["bugs"],
            "share_any_token": agg["shares_any"] / agg["bugs"] if agg["bugs"] else 0.0,
            "fail_overlap_mean": float(np.mean(agg["fail_mean"])) if agg["fail_mean"] else 0.0,
            "other_overlap_mean": float(np.mean(agg["other_mean"])) if agg["other_mean"] else 0.0,
        }
    total = len(rows)
    overall = {
        "bugs": total,
        "share_any_token": (sum(r["shares_any"] for r in rows) / total) if total else 0.0,
        "fail_overlap_mean": float(np.mean([r["fail_overlap_mean"] for r in rows])) if rows else 0.0,
        "other_overlap_mean": float(np.mean([r["others_mean"] for r in rows])) if rows else 0.0,
    }
    return {"overall": overall, "per_project": summary, "rows": rows}


# --- scoring ---------------------------------------------------------------


def bm25_scores(bugs: list[Bug], sources: dict[str, str]) -> dict[str, np.ndarray]:
    """Per bug, BM25 of the change text against every test in that bug's pool."""
    out: dict[str, np.ndarray] = {}
    for bug in bugs:
        docs = [sources.get(t, "") for t in bug.pool]
        scorer = features.BM25Scorer().fit(docs)
        out[bug.key] = scorer.score(bug.change_text)
    return out


def build_pairs(bugs: list[Bug], sources: dict[str, str]) -> tuple[list[tuple[str, str]], list[tuple[str, int]]]:
    pairs: list[tuple[str, str]] = []
    index: list[tuple[str, int]] = []
    for bug in bugs:
        for col, nodeid in enumerate(bug.pool):
            pairs.append((bug.change_text, sources.get(nodeid, "")))
            index.append((bug.key, col))
    return pairs, index


def score_semif(bugs: list[Bug], sources: dict[str, str], batch_size: int = 8) -> Path:
    """Score every (bug, test) pair with the pinned reranker. Resumable."""
    from . import semif_runner

    done: set[tuple[str, int]] = set()
    if SEMIF_CACHE.exists():
        with SEMIF_CACHE.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done.add((rec["bug"], rec["col"]))

    pairs, index = build_pairs(bugs, sources)
    keep = [i for i, key in enumerate(index) if key not in done]
    print(f"[bugsinpy] {len(pairs):,} pairs, {len(done):,} already cached, {len(keep):,} to score")
    if not keep:
        return SEMIF_CACHE

    model, tokenizer, _meta = semif_runner.load_model(device="auto")
    todo = [pairs[i] for i in keep]
    t0 = time.perf_counter()
    scores, stats = semif_runner.score_pairs(
        model, tokenizer, todo, batch_size=batch_size, progress_every=5
    )
    print(
        f"[bugsinpy] scored {stats['pairs']:,} pairs at {stats['pairs_per_second']:.1f} pairs/s "
        f"in {(time.perf_counter()-t0)/60:.1f} min"
    )
    with SEMIF_CACHE.open("a") as fh:
        for i, score in zip(keep, scores):
            key, col = index[i]
            fh.write(json.dumps({"bug": key, "col": col, "score": float(score)}) + "\n")
    print(f"[bugsinpy] appended to {SEMIF_CACHE}")
    return SEMIF_CACHE


def load_semif(bugs: list[Bug]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    by_key = {b.key: b for b in bugs}
    if not SEMIF_CACHE.exists():
        raise FileNotFoundError(f"{SEMIF_CACHE} missing; run --stage semif first")
    for bug in bugs:
        out[bug.key] = np.full(len(bug.pool), -1e9, dtype=np.float32)
    with SEMIF_CACHE.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            bug = by_key.get(rec["bug"])
            if bug is None or rec["col"] >= len(bug.pool):
                continue
            out[bug.key][rec["col"]] = rec["score"]
    return out


# --- evaluation ------------------------------------------------------------


def recall_at_budget(bugs: list[Bug], scores: dict[str, np.ndarray], budgets=BUDGETS) -> dict:
    """Per-bug-budget recall, with the budget a fraction of that bug's own pool.

    The evaluation unit is a bug, not a (bug, test) pair: a bug is caught if any of its failing
    tests is selected. That mirrors the marshmallow protocol so the two are comparable.
    """
    hits: dict[str, dict[float, list[int]]] = {f"{b:.2f}": [] for b in budgets}
    for bug in bugs:
        order = np.argsort(-scores[bug.key], kind="stable")
        fail_cols = {bug.pool.index(t) for t in bug.failing if t in bug.pool}
        for budget in budgets:
            k = max(1, math.ceil(budget * len(bug.pool)))
            selected = set(order[:k].tolist())
            hits[f"{budget:.2f}"].append(int(bool(selected & fail_cols)))
    out = {}
    for key, values in hits.items():
        arr = np.array(values, dtype=np.float64)
        lo, hi = evaluate._bootstrap_ci(arr, 1000, np.random.default_rng(config.SEED))
        out[key] = {
            "recall": float(arr.mean()) if arr.size else float("nan"),
            "lo": lo,
            "hi": hi,
            "n": int(arr.size),
            "mean_k": float(np.mean([max(1, math.ceil(float(key) * len(b.pool))) for b in bugs])),
        }
    return out


def paired(bugs: list[Bug], a: dict[str, np.ndarray], b: dict[str, np.ndarray], budget: float) -> dict:
    """Paired bootstrap on per-bug hit vectors at one budget."""
    def hits(scores):
        out = []
        for bug in bugs:
            order = np.argsort(-scores[bug.key], kind="stable")
            fail_cols = {bug.pool.index(t) for t in bug.failing if t in bug.pool}
            k = max(1, math.ceil(budget * len(bug.pool)))
            out.append(int(bool(set(order[:k].tolist()) & fail_cols)))
        return np.array(out, dtype=np.float64)

    ha, hb = hits(a), hits(b)
    # paired_bootstrap expects dicts keyed by change index
    da = {i: bool(v) for i, v in enumerate(ha)}
    db = {i: bool(v) for i, v in enumerate(hb)}
    stat = evaluate.paired_bootstrap(da, db, N_BOOTSTRAP, config.SEED)
    return {
        "delta_a_minus_b": round(stat["delta"], 4),
        "lo": round(stat["lo"], 4),
        "hi": round(stat["hi"], 4),
        "p": round(stat["p_value"], 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="report",
                        choices=["audit", "bm25", "semif", "report", "all"])
    parser.add_argument("--projects", nargs="*", default=None)
    args = parser.parse_args()

    bugs, sources = load_bugs(args.projects)
    projects = sorted({b.project for b in bugs})
    print(f"[bugsinpy] {len(bugs)} bugs across {len(projects)} projects: {projects}")
    out_path = config.ARTIFACTS / "bugsinpy_results.json"
    report: dict = json.loads(out_path.read_text()) if out_path.exists() else {}
    report["n_bugs"] = len(bugs)
    report["projects"] = projects

    if args.stage in ("audit", "all"):
        print("\n=== gate T0: textual bridge audit ===")
        audit = audit_bridge(bugs, sources)
        report["bridge_audit"] = {k: v for k, v in audit.items() if k != "rows"}
        o = audit["overall"]
        print(f"  bugs                              : {o['bugs']}")
        print(f"  failing test shares ANY token     : {o['share_any_token']:.1%}")
        print(f"  mean shared tokens, failing test  : {o['fail_overlap_mean']:.2f}")
        print(f"  mean shared tokens, other tests   : {o['other_overlap_mean']:.2f}")
        for proj, row in sorted(audit["per_project"].items()):
            print(
                f"    {proj:14s} bugs={row['bugs']:3d} shares_any={row['share_any_token']:.1%} "
                f"fail={row['fail_overlap_mean']:.2f} others={row['other_overlap_mean']:.2f}"
            )

    if args.stage in ("semif", "all"):
        print("\n=== SemIf scoring ===")
        score_semif(bugs, sources)

    if args.stage in ("bm25", "report", "all"):
        scores: dict[str, np.ndarray] = {}
        rng = np.random.default_rng(config.SEED)
        scores["random"] = {b.key: rng.random(len(b.pool)).astype(np.float32) for b in bugs}
        scores["bm25_lexical"] = bm25_scores(bugs, sources)
        if SEMIF_CACHE.exists():
            scores["semif_reranker"] = load_semif(bugs)
        report["results"] = {}
        for name, s in scores.items():
            report["results"][name] = recall_at_budget(bugs, s)
        print("\n=== recall (budget = fraction of each bug's own pool) ===")
        print(f"  {'model':18s} {'b0.01':>7} {'b0.05':>7} {'b0.10':>7} {'b0.20':>7}   n")
        for name, res in report["results"].items():
            row = "  ".join(f"{res[f'{b:.2f}']['recall']:.3f}" for b in BUDGETS)
            print(f"  {name:18s} {row}   {res['0.05']['n']}")
        if "semif_reranker" in scores:
            report["comparisons"] = {}
            for budget in BUDGETS:
                report["comparisons"][f"{budget:.2f}"] = {
                    "semif_vs_bm25": paired(bugs, scores["semif_reranker"], scores["bm25_lexical"], budget),
                    "semif_vs_random": paired(bugs, scores["semif_reranker"], scores["random"], budget),
                }
            print("\n=== paired, SemIf minus baseline (positive = SemIf better) ===")
            for budget in BUDGETS:
                c = report["comparisons"][f"{budget:.2f}"]["semif_vs_bm25"]
                print(
                    f"  b{budget:.2f}: vs bm25 {c['delta_a_minus_b']:+.3f} "
                    f"[{c['lo']:+.3f},{c['hi']:+.3f}] p={c['p']:.4f}"
                )

    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[bugsinpy] wrote {out_path.relative_to(config.WORKSPACE)}")


if __name__ == "__main__":
    main()
