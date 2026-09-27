"""Render the BugsInPy arm (W2a in ``docs/plan_next_steps.md``) into ``artifacts/bugsinpy_results.json``.

This is the arm with **real labels that are not defined by coverage**. Everything else in
``docs/implementation.md`` is measured on labels mutmut produced by running only the tests that cover
the mutated function, which makes the coverage feature circular with respect to them. BugsInPy's
failing tests come from the projects' own bug reports, so no feature is circular -- and the
corpus declares neither coverage nor durations, which makes it structurally the ladder's hardest
rung (L3: no coverage, no traceability, no history) on real data rather than a synthetic SUT.

What this module is, after the experiment layer
-----------------------------------------------
The sweep is *declared* in :mod:`rts.studies.bugsinpy`: the pooled dataset, the three
per-change-scope selectors, the per-project populations and the budget grid are values there.
This module runs that arm and renders the artifact whose shape the recorded numbers and
``docs/implementation.md`` are written against. It contains no experiment logic.

Two things stay here because they are not cell semantics:

* **The bridge audit (gate T0).** "Does the killing test share any token with the change?" is a
  statistic about the *dataset* -- no scores, no split, no selector -- so it belongs beside the
  per-project declaration rather than on the layer, exactly as ``reporting.recurrence`` does.
* **Score production.** *Measuring* SemIf is a cell; *producing* its cache is a GPU job in a
  bespoke record format (a column is a position within a bug's own pool), so it remains the
  ``--stage semif`` driver path. The model element declares the cache as a requirement either
  way, so an absent cache is an unmeasured cell rather than a crash.

Usage::

    python -m rts.bugsinpy                 # run the arm and write the artifact
    python -m rts.bugsinpy --stage audit    # the T0 gate alone
    python -m rts.bugsinpy --stage semif    # score the pairs (GPU, resumable)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import config, evaluate, features
from .data import accessors, composition, contract
from .data import datasets as dataset_module
from .studies import bugsinpy as declarations

SEMIF_CACHE = declarations.SEMIF_CACHE
BUDGETS = declarations.BUGSINPY_BUDGETS
PROBE = 0.05


# --- loading ---------------------------------------------------------------


def load_datasets(projects: list[str] | None = None) -> list[dataset_module.BugsInPyDataset]:
    """One dataset per selected project, in sorted project order."""
    return [dataset_module.bugsinpy(name) for name in declarations.project_names(projects)]


def bug_key(ds: dataset_module.BugsInPyDataset, bug) -> str:
    """``project/bug_id``. This is the key the on-disk SemIf cache and the audit use."""
    return f"{ds.name}/{ds.change_id(bug)}"


def pooled_dataset(project_datasets: list[dataset_module.BugsInPyDataset]) -> contract.Dataset:
    """The given projects as one evaluation population, namespacing test ids.

    Built from the datasets the caller passed rather than from every project on disk: otherwise
    the renderer would describe a different corpus from the one the arm measured.
    """
    return composition.pool(project_datasets, name="bugsinpy")


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
            change_tokens = set(features.text.tokenize(ds.diff_text(bug)))
            if not change_tokens:
                continue
            pool = list(bug.pool)
            failing = set(bug.failing)
            overlaps = {}
            for nodeid in pool:
                test_tokens = set(features.text.tokenize(ds.test_source(nodeid) or ""))
                overlaps[nodeid] = len(change_tokens & test_tokens)
            fail_overlap = [overlaps[t] for t in pool if t in failing]
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
            proj, {"bugs": 0, "shares_any": 0, "fail_mean": [], "other_mean": []}
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


# --- score production (GPU, resumable) -------------------------------------


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
    """Score every (bug, test) pair with the pinned reranker. Resumable.

    Kept on the driver path rather than moved onto the layer: it writes a record format nothing
    else in the study uses (``{bug, col, score}``, where ``col`` is a position within that bug's
    own pool), and the reader for it now lives with the arm's declarations. Its cost is GPU
    hours, so a production *cell* would need the same tier story ``semif.produce`` has; that is
    a deliberate next step, not an oversight.
    """
    from .model import semif_runner

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


# --- rendering --------------------------------------------------------------


def _cell(report, model: str, population: str = "fault_bearing"):
    for cell in report.cells:
        if cell.selector == model and cell.population == population:
            return cell
    return None


def _paired_out(record: dict) -> dict:
    """The recorded ``delta_a_minus_b`` shape, rounded as the artifact rounds."""
    return {
        "delta_a_minus_b": round(record["delta"], 4),
        "lo": round(record["lo"], 4),
        "hi": round(record["hi"], 4),
        "p": round(record["p_value"], 5),
    }


def render(
    reports: dict,
    project_datasets: list[dataset_module.BugsInPyDataset],
    pooled: dataset_module.Dataset,
) -> dict:
    """The recorded artifact's shape, built from the reports and the datasets.

    The numbers come from the reports -- the run's own records -- and the two quantities that are
    properties of the *data* rather than of the run come from the dataset the renderer already
    holds: the per-change pool sizes (which give ``mean_k``) and the bridge audit. The
    declaration and the audit are the same values the arm was measured on, since both are built
    from the same datasets.
    """
    # ``budget_k`` is a function of the budget and that change's candidate count, so the mean
    # selected count needs the pool sizes, and the report records the *rounded* per-change figure
    # rather than the mean.
    pools = accessors.candidates(pooled, "own")
    candidate_counts = pools.sum(axis=1)

    # The declared grid, restricted to what this run actually produced -- so a smoke run over one
    # project at one budget renders that budget instead of keying off a report that is absent.
    budgets = tuple(budget for budget in declarations.BUGSINPY_BUDGETS if budget in reports)
    if PROBE not in reports:
        raise ValueError(
            f"the per-project breakdown is defined at the probe budget {PROBE}; "
            f"this run covers {sorted(reports)}"
        )

    results: dict[str, dict] = {}
    for model in declarations.MODEL_ORDER:
        results[model] = {}
        for budget in budgets:
            cell = _cell(reports[budget], model)
            row = next(r for r in cell.results if r["budget"] == budget)
            results[model][f"{budget:.2f}"] = {
                "recall": row["recall"],
                "lo": row["recall_lo"],
                "hi": row["recall_hi"],
                "n": row["n_faults"],
                "mean_k": float(
                    np.mean([evaluate.budget_k(budget, int(c)) for c in candidate_counts])
                ),
            }

    comparisons: dict[str, dict] = {}
    for budget in budgets:
        at_reference: dict[str, dict] = {}
        for record in reports[budget].comparisons:
            if not record.get("measured"):
                continue
            if record["group"].get("population") != "fault_bearing":
                continue
            if record["cell"] != "semif_reranker":
                continue
            at_reference[record["reference"]] = record
        comparisons[f"{budget:.2f}"] = {
            "semif_vs_bm25": _paired_out(at_reference["bm25_lexical"]),
            "semif_vs_random": _paired_out(at_reference["random"]),
        }

    # Per project, because "SemIf ties BM25" is a claim about the pooled population and it
    # matters whether it holds everywhere or is carried by one project.
    per_project: dict[str, dict] = {}
    for ds in project_datasets:
        entry: dict[str, float] = {}
        for model in declarations.MODEL_ORDER:
            cell = _cell(reports[PROBE], model, population=ds.name)
            row = next(r for r in cell.results if r["budget"] == PROBE)
            entry[model] = round(float(row["recall"]), 4)
        per_project[ds.name] = entry

    return {
        "n_bugs": pooled.n_changes,
        "projects": [d.name for d in project_datasets],
        "bridge_audit": {
            key: value for key, value in audit_bridge(project_datasets).items() if key != "rows"
        },
        "results": results,
        "comparisons": comparisons,
        "dataset_declaration": {
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
            "warnings": [w.to_dict() for w in pooled.integrity_notes()],
        },
        "per_project_recall_at_0.05": per_project,
    }


def run(
    projects: list[str] | None = None,
    budgets: tuple[float, ...] = BUDGETS,
    *,
    save: bool = True,
    verbose: bool = True,
) -> dict:
    """Run the arm at every budget, render the artifact, and (by default) write it."""
    project_datasets = load_datasets(projects)
    pooled = pooled_dataset(project_datasets)
    print(
        f"[bugsinpy] {pooled.n_changes} bugs across {len(project_datasets)} datasets: "
        f"{[d.name for d in project_datasets]}"
    )
    print(
        f"[bugsinpy] pooled: {pooled.n_changes} rows x {pooled.n_tests} tests; "
        f"capabilities={sorted(pooled.capabilities())} ordering={pooled.ordering().value}"
    )
    reports = declarations.run_arm(
        budgets=budgets, projects=projects, save=False, verbose=verbose
    )
    payload = render(reports, project_datasets, pooled)

    print("\n=== recall (budget = fraction of each bug's own pool) ===")
    print(f"  {'model':18s} " + " ".join(f"{f'b{b:.2f}':>7}" for b in budgets) + "   n")
    for model in declarations.MODEL_ORDER:
        row = "  ".join(f"{payload['results'][model][f'{b:.2f}']['recall']:.3f}" for b in budgets)
        print(f"  {model:18s} {row}   {payload['results'][model][f'{budgets[0]:.2f}']['n']}")

    print("\n=== paired, SemIf minus baseline (positive = SemIf better) ===")
    for budget in budgets:
        c = payload["comparisons"][f"{budget:.2f}"]["semif_vs_bm25"]
        print(
            f"  b{budget:.2f}: vs bm25 {c['delta_a_minus_b']:+.3f} "
            f"[{c['lo']:+.3f},{c['hi']:+.3f}] p={c['p']:.4f}"
        )

    audit = payload["bridge_audit"]["overall"]
    print("\n=== gate T0: textual bridge audit ===")
    print(f"  failing test shares ANY token     : {audit['share_any_token']:.1%}")

    if save:
        out_path = config.ARTIFACTS / "bugsinpy_results.json"
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\n[bugsinpy] wrote {out_path.relative_to(config.WORKSPACE)}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        default="report",
        choices=["audit", "semif", "report"],
        help=(
            "audit: the T0 gate only; semif: score the pairs (GPU, resumable); "
            "report: run the arm and write the artifact"
        ),
    )
    parser.add_argument("--projects", nargs="*", default=None)
    args = parser.parse_args()

    if args.stage == "semif":
        print("\n=== SemIf scoring ===")
        score_semif(load_datasets(args.projects))
        return
    if args.stage == "audit":
        project_datasets = load_datasets(args.projects)
        audit = audit_bridge(project_datasets)
        overall = audit["overall"]
        print("=== gate T0: textual bridge audit ===")
        print(f"  bugs                              : {overall['bugs']}")
        print(f"  failing test shares ANY token     : {overall['share_any_token']:.1%}")
        print(f"  mean shared tokens, failing test  : {overall['fail_overlap_mean']:.2f}")
        print(f"  mean shared tokens, other tests   : {overall['other_overlap_mean']:.2f}")
        for project, row in sorted(audit["per_project"].items()):
            print(
                f"    {project:14s} bugs={row['bugs']:3d} shares_any={row['share_any_token']:.1%} "
                f"fail={row['fail_overlap_mean']:.2f} others={row['other_overlap_mean']:.2f}"
            )
        return

    run(projects=args.projects)


if __name__ == "__main__":
    main()
