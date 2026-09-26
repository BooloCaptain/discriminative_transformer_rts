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

Not available: coverage and durations, because obtaining them would mean running every
project's suite at every bug commit. So the datasets declare neither capability, and the harness
reports those derived features as *unmeasured* rather than as an all-zero column. This arm is
therefore *structurally* the ladder's L3 rung -- no coverage, no traceability, no history -- on
real data rather than on a synthetic SUT. The selectors evaluated here are the ones that survive
that: random, BM25, SemIf, and a tree over change-intrinsic and test-intrinsic features only.

One dataset per project
-----------------------
Per §1 of ``refactor.md``, granularity is maximal: **one project is one dataset**, because a
project's suite is what tests are run against and pooling is a decision an experiment makes
rather than a property of the data. All eight ship the same schema, so they share one source
class and one generator. The headline numbers come from the pooled population, which is the
composition of the eight.

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
import time
from pathlib import Path

import numpy as np

from . import config, dataset, datasets as dataset_module, evaluate, features

BUGSINPY_DIR = config.ARTIFACTS / "bugsinpy"
SEMIF_CACHE = config.ARTIFACTS / "semif_scores_bugsinpy.jsonl"
BUDGETS = (0.01, 0.05, 0.1, 0.2)
N_BOOTSTRAP = 2000


# --- loading ---------------------------------------------------------------


def load_datasets(projects: list[str] | None = None) -> list[dataset_module.BugsInPyDataset]:
    """One dataset per project, in sorted project order."""
    all_projects = dataset_module.available_bugsinpy_projects()
    if projects:
        missing = sorted(set(projects) - set(all_projects))
        if missing:
            raise FileNotFoundError(f"no built dataset for project(s): {missing}")
        all_projects = [p for p in all_projects if p in set(projects)]
    return [dataset_module.bugsinpy(p) for p in all_projects]


def bug_key(ds: dataset_module.BugsInPyDataset, bug) -> str:
    """``project/bug_id``. This is the key the on-disk SemIf cache and the audit use."""
    return f"{ds.name}/{ds.change_id(bug)}"


def pooled_row(ds: dataset_module.BugsInPyDataset, bug, pooled: dataset.Dataset) -> int:
    """The bug's row in the pooled matrix, whose change ids are namespaced.

    Resolved through the pooled dataset's own index rather than a module-level cache:
    the row layout belongs to the pooled value, and two pools must be able to coexist.
    """
    return pooled.change_index[f"{ds.name}::{ds.change_id(bug)}"]


def pooled_dataset(project_datasets: list[dataset_module.BugsInPyDataset]) -> dataset.Dataset:
    """The eight projects as one evaluation population, namespacing test ids."""
    return dataset.pool(project_datasets, name="bugsinpy")


def bug_pools(
    project_datasets: list[dataset_module.BugsInPyDataset],
    pooled: dataset.Dataset,
) -> dict[str, tuple[str, ...]]:
    """``project/bug_id`` -> its own pool, in the pooled dataset's namespaced id space."""
    return {
        bug_key(ds, bug): tuple(dataset.namespace(ds, t) for t in bug.pool)
        for ds in project_datasets
        for bug in ds.changes
    }


def candidate_matrix(
    project_datasets: list[dataset_module.BugsInPyDataset], pooled: dataset.Dataset
) -> np.ndarray:
    """``[n_bugs, n_tests]``: a bug is rankable only within its own project's pool.

    This is what makes the budget a fraction of *that bug's* pool rather than of the
    union of eight suites, which would make the budget mean eight different things.
    """
    mask = np.zeros((pooled.n_changes, pooled.n_tests), dtype=bool)
    index = pooled.test_index
    for ds in project_datasets:
        for bug in ds.changes:
            row = pooled_row(ds, bug, pooled)
            for nodeid in bug.pool:
                col = index.get(dataset.namespace(ds, nodeid))
                if col is not None:
                    mask[row, col] = True
    return mask


def describe_pools(project_datasets: list[dataset_module.BugsInPyDataset]) -> dict:
    bugs = [change for ds in project_datasets for change in ds.changes]
    sizes = [len(b.pool) for b in bugs]
    n_fail = [len(b.failing) for b in bugs]
    return {
        "bugs": len(bugs),
        "projects": len(project_datasets),
        "pool_median": float(np.median(sizes)) if sizes else 0.0,
        "pool_min": min(sizes) if sizes else 0,
        "pool_max": max(sizes) if sizes else 0,
        "failing_median": float(np.median(n_fail)) if n_fail else 0.0,
        "failing_max": max(n_fail) if n_fail else 0,
        "single_failing_bugs": sum(1 for n in n_fail if n == 1),
    }


# --- gate T0: the textual-bridge audit -------------------------------------


def audit_bridge(project_datasets: list[dataset_module.BugsInPyDataset]) -> dict:
    """Does the failing test share any text with the change?

    This decides whether a *text* model can work at all. If the killing test and the change
    share no vocabulary, then neither BM25 nor a semantic reranker has anything to condition
    on, and a null result would be uninformative about the models -- it would only say the
    information is not in the text.
    """
    per_project: dict[str, dict] = {}
    rows: list[dict] = []
    for ds in project_datasets:
        for bug in ds.changes:
            change_tokens = set(features.tokenize(ds.diff_text(bug)))
            if not change_tokens:
                continue
            pool = list(bug.pool)
            failing = set(bug.failing)
            overlaps = {}
            for nodeid in pool:
                test_tokens = set(features.tokenize(ds.test_source(nodeid) or ""))
                overlaps[nodeid] = len(change_tokens & test_tokens)
            fail_overlap = [overlaps[t] for t in pool if t in failing]
            subset = [t for t in pool if t in failing]
            others = [overlaps[t] for t in pool if t not in failing]
            rows.append(
                {
                    "bug": bug_key(ds, bug),
                    "change_tokens": len(change_tokens),
                    "pool": len(pool),
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


def bm25_scores(
    project_datasets: list[dataset_module.BugsInPyDataset], pooled: dataset.Dataset
) -> np.ndarray:
    """``[n_bugs, n_tests]`` BM25 of the change text against the bug's own pool.

    The scorer is still fitted per bug, on that bug's pool: a global index over eight
    projects would make a term's idf depend on the other seven, and that is a different
    quantity from the one the marshmallow arm reports.
    """
    index = pooled.test_index
    out = np.full((pooled.n_changes, pooled.n_tests), -1e9, dtype=np.float32)
    for ds in project_datasets:
        for bug in ds.changes:
            row = pooled_row(ds, bug, pooled)
            docs = [ds.test_source(t) or "" for t in bug.pool]
            scorer = features.BM25Scorer().fit(docs)
            values = scorer.score(ds.diff_text(bug))
            for nodeid, value in zip(bug.pool, values):
                col = index.get(dataset.namespace(ds, nodeid))
                if col is not None:
                    out[row, col] = value
    return out


def random_scores(
    project_datasets: list[dataset_module.BugsInPyDataset], pooled: dataset.Dataset
) -> np.ndarray:
    """Uniform scores, drawn per bug over that bug's own pool.

    Drawn per bug rather than over the whole matrix so the baseline is the same
    sequence of random numbers the arm has always used: a uniform draw over a 71x3000
    matrix would give each bug a different set of ranks, and the recorded baseline
    would move for a reason that has nothing to do with the refactor.
    """
    rng = np.random.default_rng(config.SEED)
    index = pooled.test_index
    out = np.full((pooled.n_changes, pooled.n_tests), -1e9, dtype=np.float32)
    for ds in project_datasets:
        for bug in ds.changes:
            row = pooled_row(ds, bug, pooled)
            values = rng.random(len(bug.pool)).astype(np.float32)
            for nodeid, value in zip(bug.pool, values):
                col = index.get(dataset.namespace(ds, nodeid))
                if col is not None:
                    out[row, col] = value
    return out


def build_pairs(
    project_datasets: list[dataset_module.BugsInPyDataset],
) -> tuple[list[tuple[str, str]], list[tuple[str, int]]]:
    pairs: list[tuple[str, str]] = []
    index: list[tuple[str, int]] = []
    for ds in project_datasets:
        for bug in ds.changes:
            for col, nodeid in enumerate(bug.pool):
                pairs.append((ds.diff_text(bug), ds.test_source(nodeid) or ""))
                index.append((bug_key(ds, bug), col))
    return pairs, index


def score_semif(
    project_datasets: list[dataset_module.BugsInPyDataset], batch_size: int = 8
) -> Path:
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

    pairs, index = build_pairs(project_datasets)
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


def load_semif(
    project_datasets: list[dataset_module.BugsInPyDataset], pooled: dataset.Dataset
) -> np.ndarray:
    """Load the SemIf cache into the pooled matrix layout."""
    pools = bug_pools(project_datasets, pooled)
    by_bug = {bug_key(ds, bug): ds for ds in project_datasets for bug in ds.changes}
    rows = {bug_key(ds, bug): pooled_row(ds, bug, pooled) for ds in project_datasets for bug in ds.changes}
    index = pooled.test_index
    out = np.full((pooled.n_changes, pooled.n_tests), -1e9, dtype=np.float32)
    if not SEMIF_CACHE.exists():
        raise FileNotFoundError(f"{SEMIF_CACHE} missing; run --stage semif first")
    for line in SEMIF_CACHE.open():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        pool = pools.get(rec["bug"])
        row = rows.get(rec["bug"])
        if pool is None or row is None or rec["col"] >= len(pool):
            continue
        col = index.get(pool[rec["col"]])
        if col is None:
            continue
        out[row, col] = rec["score"]
    return out


# --- evaluation ------------------------------------------------------------


def recall_at_budget(
    scores: np.ndarray,
    pooled: dataset.Dataset,
    candidates: np.ndarray,
    budgets=BUDGETS,
    rows: np.ndarray | None = None,
) -> dict:
    """Per-bug-budget recall, with the budget a fraction of that bug's own pool.

    The evaluation unit is a bug, not a (bug, test) pair: a bug is caught if any of its failing
    tests is selected. That mirrors the marshmallow protocol so the two are comparable -- and it
    is now literally the same code, via the harness's metric sweep over a label matrix.

    The evaluation window is every bug: this arm has no training stage, since a zero-shot text
    model has nothing to fit, so there is no split to hold anything out for.
    """
    if rows is None:
        rows = np.arange(pooled.n_changes, dtype=np.int64)
    cand_counts = candidates[rows].sum(axis=1)
    out: dict = {}
    for budget in budgets:
        # One budget at a time so each gets a fresh bootstrap generator, which is how
        # these intervals were originally drawn.
        result = evaluate.evaluate(
            scores, pooled, rows, budgets=(budget,), n_bootstrap=1000,
            seed=config.SEED, candidates=candidates,
        )[0]
        out[f"{budget:.2f}"] = {
            "recall": float(result.recall),
            "lo": float(result.recall_lo),
            "hi": float(result.recall_hi),
            "n": int(result.n_faults),
            # The mean selected count is itself a mean over changes, so it is reported
            # unrounded; BudgetResult.k is the rounded per-change figure.
            "mean_k": float(
                np.mean([evaluate.budget_k(budget, int(c)) for c in cand_counts])
            ),
        }
    return out


def paired(
    pooled: dataset.Dataset,
    candidates: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    budget: float,
) -> dict:
    """Paired bootstrap on per-bug hit vectors at one budget."""
    rows = np.arange(pooled.n_changes, dtype=np.int64)
    ha = evaluate.per_change_hits(a, pooled, rows, budget, candidates)
    hb = evaluate.per_change_hits(b, pooled, rows, budget, candidates)
    stat = evaluate.paired_bootstrap(ha, hb, N_BOOTSTRAP, config.SEED)
    return {
        "delta_a_minus_b": round(stat["delta"], 4),
        "lo": round(stat["lo"], 4),
        "hi": round(stat["hi"], 4),
        "p": round(stat["p_value"], 5),
    }


def collect_scores(
    project_datasets: list[dataset_module.BugsInPyDataset],
    pooled: dataset.Dataset,
) -> dict[str, np.ndarray]:
    """Every selector this arm can afford, in the pooled matrix layout."""
    scores: dict[str, np.ndarray] = {
        "random": random_scores(project_datasets, pooled),
        "bm25_lexical": bm25_scores(project_datasets, pooled),
    }
    if SEMIF_CACHE.exists():
        scores["semif_reranker"] = load_semif(project_datasets, pooled)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="report",
                        choices=["audit", "bm25", "semif", "report", "all"])
    parser.add_argument("--projects", nargs="*", default=None)
    args = parser.parse_args()

    project_datasets = load_datasets(args.projects)
    pooled = pooled_dataset(project_datasets)
    candidates = candidate_matrix(project_datasets, pooled)
    projects = [d.name for d in project_datasets]
    print(f"[bugsinpy] {pooled.n_changes} bugs across {len(projects)} datasets: {projects}")
    print(f"[bugsinpy] pooled: {pooled.n_changes} rows x {pooled.n_tests} tests; "
          f"capabilities={sorted(pooled.capabilities())} ordering={pooled.ordering().value}")
    out_path = config.ARTIFACTS / "bugsinpy_results.json"
    report: dict = json.loads(out_path.read_text()) if out_path.exists() else {}
    report["n_bugs"] = pooled.n_changes
    report["projects"] = projects
    report["dataset_declaration"] = {
        "pooled_name": pooled.name,
        "per_project": {
            d.name: {
                "ordering": d.ordering().value,
                "test_unit": d.test_unit().value,
                "capabilities": sorted(d.capabilities()),
                "n_changes": d.n_changes,
                "n_tests": d.n_tests,
            }
            for d in project_datasets
        },
        "warnings": [w.to_dict() for w in pooled.warnings],
    }

    if args.stage in ("audit", "all"):
        print("\n=== gate T0: textual bridge audit ===")
        audit = audit_bridge(project_datasets)
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
        score_semif(project_datasets)

    if args.stage in ("bm25", "report", "all"):
        scores = collect_scores(project_datasets, pooled)
        report["results"] = {}
        for name, matrix in scores.items():
            report["results"][name] = recall_at_budget(matrix, pooled, candidates)
        print("\n=== recall (budget = fraction of each bug's own pool) ===")
        print(f"  {'model':18s} {'b0.01':>7} {'b0.05':>7} {'b0.10':>7} {'b0.20':>7}   n")
        for name, res in report["results"].items():
            row = "  ".join(f"{res[f'{b:.2f}']['recall']:.3f}" for b in BUDGETS)
            print(f"  {name:18s} {row}   {res['0.05']['n']}")

        # Per project, because "SemIf ties BM25" is a claim about the pooled population
        # and it matters whether it holds everywhere or is carried by one project.
        report["per_project_recall_at_0.05"] = {}
        for ds in project_datasets:
            own = np.array(
                [pooled_row(ds, bug, pooled) for bug in ds.changes], dtype=np.int64
            )
            entry = {}
            for name, matrix in scores.items():
                res = evaluate.evaluate(
                    matrix, pooled, own, budgets=(0.05,), n_bootstrap=0,
                    candidates=candidates,
                )[0]
                entry[name] = round(float(res.recall), 4)
            report["per_project_recall_at_0.05"][ds.name] = entry
        if report["per_project_recall_at_0.05"]:
            print("\n=== recall at b0.05, per project dataset ===")
            for proj, entry in report["per_project_recall_at_0.05"].items():
                cells = "  ".join(f"{k}={v:.3f}" for k, v in entry.items())
                print(f"  {proj:14s} {cells}")

        if "semif_reranker" in scores:
            report["comparisons"] = {}
            for budget in BUDGETS:
                report["comparisons"][f"{budget:.2f}"] = {
                    "semif_vs_bm25": paired(
                        pooled, candidates, scores["semif_reranker"], scores["bm25_lexical"], budget
                    ),
                    "semif_vs_random": paired(
                        pooled, candidates, scores["semif_reranker"], scores["random"], budget
                    ),
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
