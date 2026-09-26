"""The traceability-loss ladder (W1 in ``plan_next_steps.md``).

Motivation
----------
The deployment of interest is a test suite driving an embedded system across a boundary:
Python tests send commands over serial/socket to firmware, so the changed code does not run in
the test process. Four feature families that carry this benchmark die at once --

* **coverage** cannot cross the boundary,
* **filename / path proximity** assumes co-located, conventionally named unit tests,
* **history** assumes a ``(file, test)`` pair recurs, which it does not at thousands of tests,
* **text overlap** is reduced to protocol and feature vocabulary.

This module removes those families one at a time and re-measures *every* selector, including
SemIf, at each rung. The deliverable is a **degradation curve**: the question is not what any
single number is, but whether the ordering changes as traceability is removed -- and in
particular whether SemIf crosses the classical baselines.

Mechanism
---------
Feature availability is modelled by *zeroing* the unavailable columns of ``X`` and zeroing
``bm25``, rather than by dropping columns from the tree. A zeroed column is a feature that
carries no information, which is exactly the target condition, and it keeps one code path for
both the learned and the hand-built selectors. ``structural_rule`` therefore degrades to
"shortest test first" once coverage and filename matching are gone, and ``coverage`` degrades to
a constant -- which is the honest behaviour of a method whose input no longer exists.

SemIf is a *text* model and is unaffected by the rung, because its scores are keyed on the
(change, test) text pair. That is the point: the ladder shows the classical side falling away
underneath a flat semantic line. Text is only removed at the last rung, where the tree is also
evaluated without BM25 to give the true floor.

Populations
-----------
``starved141``  141 held-out changes with a complete full-pool SemIf cache (see
                ``scripts/complete_semif_full_cache.py``). Paired comparisons happen here.
                NOTE: this is the population the *old* starved filter selected. Under corrected
                full-suite labels the starved filter collapses (11 held-out faults at
                ``failures <= 5``), because it was keyed on an under-counted failure history, so
                this population is no longer interpretable as "starved" -- it is simply a
                held-out subset, and is named for provenance only.
``heldout530``  every held-out change. Classical selectors only, since SemIf has no full-pool
                cache there.

Usage
-----
    python -m rts.ladder                 # both label sources
    python -m rts.ladder --labels full
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import config, dataset, datasets, evaluate, features, models
from .dataset import REQ_LABELS, Population, Unmeasured, is_unmeasured

BUDGETS = (0.01, 0.05, 0.1, 0.2)
PROBE = 0.05
N_BOOTSTRAP = 2000

SEMIF_LADDER_CACHE = config.ARTIFACTS / "semif_scores_ladder141_full.jsonl"
SEMIF_NAME = "semif_reranker"
# The rungs and the SemIf cache reproduce the documented numbers by explicitly asking
# for the features the harness would otherwise leave off (history on an imposed order)
# and by opening the cache the paired comparison rests on.
STRUCTURED_HISTORY = True

# Feature families removed cumulatively. The rung name states what is *unavailable*.
FAMILIES: dict[str, tuple[str, ...]] = {
    "history": models.HISTORY_FEATURES,
    "coverage": models.COVERAGE_FEATURES,
    "traceability": models.TRACEABILITY_FEATURES,
}

RUNGS: list[tuple[str, tuple[str, ...]]] = [
    ("L0_all", ()),
    ("L1_nohistory", ("history",)),
    ("L2_nocoverage", ("history", "coverage")),
    ("L3_notrace", ("history", "coverage", "traceability")),
]


def _mask_context(
    X: np.ndarray, names: list[str], bm25: np.ndarray, removed: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Zero the columns of the removed families, and bm25 when text is removed."""
    drop = {n for fam in removed for n in FAMILIES[fam]}
    X_masked = X.copy()
    for i, name in enumerate(names):
        if name in drop:
            X_masked[:, :, i] = 0.0
    bm25_masked = bm25 if "text" not in removed else np.zeros_like(bm25)
    return X_masked, bm25_masked


def _cached_change_ids() -> set[str] | None:
    """Change ids with a complete full-pool SemIf cache, or ``None`` if there is none."""
    if not SEMIF_LADDER_CACHE.exists():
        return None
    ids: set[str] = set()
    with SEMIF_LADDER_CACHE.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["change_id"])
    return ids


def build_populations(ds: dataset.Dataset) -> dict[str, "Population | Unmeasured"]:
    """The ladder's two averaging populations, declared rather than hard-coded.

    A population declares its requirements; where a dataset -- or, here, an external
    artifact -- cannot meet them, the population is *unavailable*, never silently
    empty. ``starved141`` needs the SemIf cache to exist at all, so a missing cache
    makes it unmeasured rather than making it smaller.

    NOTE: ``starved141`` is a provenance name. Under corrected full-suite labels the
    starved filter collapses (11 held-out faults at ``failures <= 5``), because it was
    keyed on an under-counted failure history, so this population is no longer
    interpretable as "starved" -- it is simply a held-out subset.
    """
    populations: dict[str, Population | Unmeasured] = {}

    cached = _cached_change_ids()
    if cached is None:
        populations["starved141"] = Unmeasured(
            requirement="artifact:semif_ladder_cache",
            note=(
                f"{SEMIF_LADDER_CACHE.name} is missing, so the starved141 population "
                "cannot exist; the paired comparison has no SemIf scores to pair against"
            ),
        )
    else:
        populations["starved141"] = Population(
            name="starved141",
            note=(
                "held-out changes with a complete full-pool SemIf cache; paired "
                "comparisons happen here"
            ),
            requires=frozenset({REQ_LABELS}),
            predicate=lambda d: np.array(
                [d.change_id(c) in cached for c in d.changes], dtype=bool
            ),
        )

    populations["heldout530"] = Population(
        name="heldout530",
        note="every held-out change; classical selectors only, since SemIf has no full-pool cache there",
        requires=frozenset({REQ_LABELS}),
        predicate=lambda d: d.fault_mask
        & np.isin(np.arange(d.n_changes), d.test_idx),
    )
    return populations


def _population_rows(
    ds: dataset.Dataset, populations: dict[str, "Population | Unmeasured"]
) -> dict[str, np.ndarray | Unmeasured]:
    out: dict[str, np.ndarray | Unmeasured] = {}
    for name, spec in populations.items():
        if isinstance(spec, Unmeasured):
            out[name] = spec
            continue
        out[name] = spec.rows(ds, ds.test_idx)
    return out


def _selectors(include_semif: bool) -> list[models.Selector]:
    return [
        models.RandomSelector(),
        models.RecencySelector(),
        models.FailureRateSelector(),
        models.CoverageSelector(),
        models.StructuralRuleSelector(),
        models.LexicalSelector(),
        models.XGBoostSelector(include_lexical=True),
        models.XGBoostSelector(include_lexical=False),
    ] + ([models.SemIfSelector(scores_file=SEMIF_LADDER_CACHE)] if include_semif else [])


def run_label_source(label_source: str, verbose: bool = True) -> dict:
    ds = datasets.marshmallow(labels=label_source)
    X, names = features.structured_features(ds, history=STRUCTURED_HISTORY)
    bm25 = features.build_bm25_scores(ds)
    populations = _population_rows(ds, build_populations(ds))
    candidates = dataset.candidate_mask(ds, "full")

    if verbose:
        print("=" * 78)
        print(f"TRACEABILITY LADDER -- labels={label_source}")
        print("=" * 78)
        print(f"  changes {ds.n_changes}  tests {ds.n_tests}  held-out faults {len(ds.test_fault_idx)}")
        for name, rows in populations.items():
            size = "unmeasured" if is_unmeasured(rows) else len(rows)
            print(f"  {name} {size}")

    report: dict = {
        "labels": label_source,
        "n_changes": ds.n_changes,
        "n_tests": ds.n_tests,
        "held_out_faults": int(len(ds.test_fault_idx)),
        "populations": {
            k: (None if is_unmeasured(v) else int(len(v))) for k, v in populations.items()
        },
        "populations_unmeasured": {
            k: v.to_dict() for k, v in populations.items() if is_unmeasured(v)
        },
        "dataset_declaration": {
            "name": ds.name,
            "ordering": ds.ordering().value,
            "test_unit": ds.test_unit().value,
            "capabilities": sorted(ds.capabilities()),
            "semantics": dict(ds.semantics()),
        },
        "warnings": ds.warnings.to_list(),
        "rungs": {},
    }

    for rung, removed in RUNGS:
        t0 = time.perf_counter()
        X_rung, bm25_rung = _mask_context(X, names, bm25, removed)
        ctx = models.Context(ds=ds, X=X_rung, names=names, bm25=bm25_rung)
        scores_by_name: dict[str, np.ndarray] = {}
        for selector in _selectors(include_semif=SEMIF_LADDER_CACHE.exists()):
            try:
                scores_by_name[selector.name] = selector.scores(ctx)
            except FileNotFoundError as exc:
                if verbose:
                    print(f"  [skip] {selector.name}: {str(exc).splitlines()[0]}")
        rung_report: dict = {"removed": list(removed), "selectors": {}, "comparisons": {}}

        for pop_name, rows in populations.items():
            if is_unmeasured(rows):
                # Unavailable, not empty: record why and evaluate nothing, rather than
                # reporting an average over a population that cannot exist.
                if verbose:
                    print(f"    {pop_name}: unmeasured -- {rows.note}")
                rung_report["selectors"][pop_name] = {"unmeasured": rows.to_dict()}
                continue
            has_semif = SEMIF_NAME in scores_by_name and pop_name == "starved141"
            table: dict[str, dict] = {}
            for name, scores in scores_by_name.items():
                if name == SEMIF_NAME and not has_semif:
                    continue
                res = evaluate.evaluate(
                    scores, ds, rows, budgets=BUDGETS, n_bootstrap=1000,
                    candidates=candidates,
                )
                table[name] = {
                    "recall": {f"{r.budget:.2f}": round(r.recall, 4) for r in res},
                    "k": {f"{r.budget:.2f}": r.k for r in res},
                    "n_faults": res[0].n_faults,
                }
            rung_report["selectors"][pop_name] = table

            if has_semif:
                ref_hits = evaluate.per_change_hits(
                    scores_by_name[SEMIF_NAME], ds, rows, PROBE, candidates
                )
                comps: dict[str, dict] = {}
                for name, scores in scores_by_name.items():
                    if name == SEMIF_NAME:
                        continue
                    hits = evaluate.per_change_hits(scores, ds, rows, PROBE, candidates)
                    # paired_bootstrap(a, b) returns recall(a) - recall(b); a is the baseline
                    # here, so a NEGATIVE delta means SemIf is ahead.
                    stat = evaluate.paired_bootstrap(hits, ref_hits, N_BOOTSTRAP, config.SEED)
                    comps[name] = {
                        "delta_baseline_minus_semif": round(stat["delta"], 4),
                        "lo": round(stat["lo"], 4),
                        "hi": round(stat["hi"], 4),
                        "p": round(stat["p_value"], 5),
                        "semif_ahead": bool(stat["delta"] < 0),
                    }
                rung_report["comparisons"]["starved141_vs_semif_b0.05"] = comps

                # The headline quantity: SemIf's margin over the best classical selector,
                # at every budget. A growing margin as rungs are removed is the hypothesis.
                classical = [n for n in table if n != SEMIF_NAME and n != "random"]
                margins: dict[str, dict] = {}
                for budget in BUDGETS:
                    key = f"{budget:.2f}"
                    best = max(classical, key=lambda n: table[n]["recall"][key])
                    margins[key] = {
                        "best_classical": best,
                        "best_recall": table[best]["recall"][key],
                        "semif_recall": table[SEMIF_NAME]["recall"][key],
                        "semif_margin": round(
                            table[SEMIF_NAME]["recall"][key] - table[best]["recall"][key], 4
                        ),
                    }
                rung_report["semif_margin"] = margins

        report["rungs"][rung] = rung_report
        if verbose:
            print(f"\n  [{rung}] removed={list(removed) or 'nothing'}  ({time.perf_counter()-t0:.1f}s)")
            for pop_name in rung_report["selectors"]:
                table = rung_report["selectors"][pop_name]
                if "unmeasured" in table:
                    continue
                print(f"    {pop_name}:")
                for name, row in table.items():
                    rec = row["recall"]
                    print(
                        f"      {name:26s} b0.01={rec['0.01']:.3f}  b0.05={rec['0.05']:.3f}"
                        f"  b0.10={rec['0.10']:.3f}  b0.20={rec['0.20']:.3f}"
                    )
            comps = rung_report["comparisons"].get("starved141_vs_semif_b0.05")
            if comps:
                print("    SemIf vs baseline @b0.05 (negative delta = SemIf ahead):")
                for name, c in comps.items():
                    flag = "*" if c["p"] < 0.05 else " "
                    print(
                        f"      {name:26s} {c['delta_baseline_minus_semif']:+.3f} "
                        f"[{c['lo']:+.3f},{c['hi']:+.3f}] p={c['p']:.4f}{flag}"
                    )
                margins = rung_report.get("semif_margin")
                if margins:
                    print("    SemIf margin over the best classical selector:")
                    for key, m in margins.items():
                        print(
                            f"      b{key}: SemIf {m['semif_recall']:.3f} vs "
                            f"{m['best_classical']} {m['best_recall']:.3f} "
                            f"-> {m['semif_margin']:+.3f}"
                        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", nargs="*", default=["mutmut", "full"],
                        choices=["mutmut", "full"])
    args = parser.parse_args()

    out_path = config.ARTIFACTS / "ladder.json"
    report: dict = {}
    if out_path.exists():
        report = json.loads(out_path.read_text())
    for label_source in args.labels:
        report[label_source] = run_label_source(label_source)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\n[report] wrote {out_path}")


if __name__ == "__main__":
    main()
