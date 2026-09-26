"""The traceability-loss ladder (W1 in ``plan_next_steps.md``).

Motivation
----------
The deployment of interest is a test suite driving an embedded system across a boundary:
Python tests send commands over serial/socket to firmware, so the changed code does not
run in the test process. Four feature families that carry this benchmark die at once --

* **coverage** cannot cross the boundary,
* **filename / path proximity** assumes co-located, conventionally named unit tests,
* **history** assumes a ``(file, test)`` pair recurs, which it does not at thousands of
  tests,
* **text overlap** is reduced to protocol and feature vocabulary.

This module removes those families one at a time and re-measures *every* selector,
including SemIf, at each rung. The deliverable is a **degradation curve**: the question is
not what any single number is, but whether the ordering changes as traceability is
removed -- and in particular whether SemIf crosses the classical baselines.

Mechanism
---------
A rung is expressed as **the block with a family withheld**:
``features.structured(..., block=STRUCTURED.without_families(*removed))``. The columns
stay, because a rung must keep a stable column list for selectors that index by name, and
they come out zeroed and *reported* as unmeasured.

That is strictly better than the first pass, which copied the matrix and zeroed columns by
comparing names against a list in this file. A name-based ablation is a silent-drift
hazard: rename a column and the rung quietly stops removing anything, with no error and a
plausible number. Withholding a family now goes through the same unmeasured path a
genuinely absent capability takes, and the rung's own warnings say what was withheld.

SemIf is a *text* model and is unaffected by the rung, because its scores are keyed on the
(change, test) text pair. That is the point: the ladder shows the classical side falling
away underneath a flat semantic line.

Populations
-----------
``heldout530``  every held-out fault-bearing change. Classical selectors only, since SemIf
                has no full-pool cache there.
``starved141``  held-out changes with a *complete* full-pool SemIf cache, so paired
                comparisons have something to pair against. Its availability depends on an
                artifact rather than on the dataset, so a missing cache makes it
                **unmeasured** rather than smaller -- reporting fewer pairs as if they were
                the population would be a different claim.

                NOTE: the name is provenance only. Under corrected full-suite labels the
                starved filter collapses (11 held-out faults at ``failures <= 5``), because
                it was keyed on an under-counted failure history, so this is no longer
                interpretable as "starved" -- it is simply a held-out subset.

Usage
-----
    python -m rts.ladder                 # both label sources
    python -m rts.ladder --labels full
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from . import (
    accessors,
    config,
    contract,
    datasets,
    evaluate,
    features,
    models,
    populations,
    reporting,
    splits,
)
from .contract import Policy, Unmeasured, Warning

BUDGETS = (0.01, 0.05, 0.1, 0.2)
PROBE = 0.05
N_BOOTSTRAP = 2000

SEMIF_LADDER_CACHE = config.ARTIFACTS / "semif_scores_ladder141_full.jsonl"
SEMIF_NAME = "semif_reranker"
STRUCTURED_HISTORY = True

#: Feature families removed cumulatively. The rung name states what is *unavailable*.
RUNGS: list[tuple[str, tuple[str, ...]]] = [
    ("L0_all", ()),
    ("L1_nohistory", ("history",)),
    ("L2_nocoverage", ("history", "coverage")),
    ("L3_notrace", ("history", "coverage", "traceability")),
]


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


def build_populations(
    ds: contract.Dataset, split: splits.Split
) -> dict[str, "populations.Population | Unmeasured"]:
    """The ladder's two averaging populations, declared rather than hard-coded."""
    out: dict[str, populations.Population | Unmeasured] = {}

    cached = _cached_change_ids()
    if cached is None:
        out["starved141"] = Unmeasured(
            requirement="artifact:semif_ladder_cache",
            note=(
                f"{SEMIF_LADDER_CACHE.name} is missing, so the starved141 population cannot "
                "exist; the paired comparison has no SemIf scores to pair against"
            ),
        )
    else:
        ids = [ds.change_id(c) for c in ds.changes]
        mask = np.array([cid in cached for cid in ids], dtype=bool)
        out["starved141"] = populations.Population(
            name="starved141",
            note=(
                "held-out changes with a complete full-pool SemIf cache; paired "
                "comparisons happen here"
            ),
            needs=("labels",),
            predicate=lambda _material, mask=mask: mask,
        )

    out["heldout530"] = populations.Population(
        name="heldout530",
        note="every held-out fault-bearing change",
        needs=("labels",),
        predicate=lambda material: material["faults"],
    )
    return out


def _rung_block(removed: tuple[str, ...]) -> features.FeatureBlock:
    """The structured block with the rung's families withheld."""
    if not removed:
        return features.STRUCTURED
    return features.STRUCTURED.without_families(*removed)


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
    split = splits.make_split(ds)
    declared = build_populations(ds, split)
    candidates = accessors.candidates(ds, "full")

    if verbose:
        print("=" * 78)
        print(f"TRACEABILITY LADDER -- labels={label_source}")
        print("=" * 78)
        print(
            f"  changes {ds.n_changes}  tests {ds.n_tests}  "
            f"held-out faults {len(accessors.test_fault_idx(ds, split.test_idx))}"
        )
        for name, spec in declared.items():
            if isinstance(spec, Unmeasured):
                print(f"  {name}: unmeasured -- {spec.note}")
            else:
                print(f"  {name} {len(spec.rows(ds, split.test_idx))}")

    report: dict = {
        "labels": label_source,
        "n_changes": ds.n_changes,
        "n_tests": ds.n_tests,
        "held_out_faults": int(len(accessors.test_fault_idx(ds, split.test_idx))),
        "populations": {
            k: (None if isinstance(v, Unmeasured) else len(v.rows(ds, split.test_idx)))
            for k, v in declared.items()
        },
        "populations_unmeasured": {
            k: v.to_dict() for k, v in declared.items() if isinstance(v, Unmeasured)
        },
        "dataset_declaration": ds.declaration(),
        "rungs": {},
    }

    for rung, removed in RUNGS:
        t0 = time.perf_counter()
        block = _rung_block(removed)
        matrix = features.structured(ds, history=STRUCTURED_HISTORY, block=block)
        ctx = models.Context(
            ds=ds,
            features=matrix,
            split=split,
            bm25=features.text.build_bm25_scores(ds),
        )
        scores_by_name: dict[str, np.ndarray] = {}
        for selector in _selectors(include_semif=SEMIF_LADDER_CACHE.exists()):
            try:
                scores_by_name[selector.name] = selector.scores(ctx)
            except FileNotFoundError as exc:
                if verbose:
                    print(f"  [skip] {selector.name}: {str(exc).splitlines()[0]}")

        rung_report: dict = {
            "removed": list(removed),
            "withheld": [c for c, _ in matrix.unmeasured],
            "warnings": [w.to_dict() for w in matrix.warnings],
            "selectors": {},
            "comparisons": {},
        }

        for pop_name, spec in declared.items():
            if isinstance(spec, Unmeasured):
                # Unavailable, not empty: record why and evaluate nothing, rather than
                # reporting an average over a population that cannot exist.
                rung_report["selectors"][pop_name] = {"unmeasured": spec.to_dict()}
                continue
            rows = spec.rows(ds, split.test_idx)
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

                # The headline quantity: SemIf's margin over the best classical selector, at
                # every budget. A growing margin as rungs are removed is the hypothesis.
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
            print(
                f"\n  [{rung}] removed={list(removed) or 'nothing'}  "
                f"({time.perf_counter()-t0:.1f}s)"
            )
            for pop_name in report["rungs"][rung]["selectors"]:
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
    parser.add_argument(
        "--labels", nargs="*", default=["mutmut", "full"], choices=list(config.LABEL_SOURCES)
    )
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
