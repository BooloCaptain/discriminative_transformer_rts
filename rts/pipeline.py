"""End-to-end pipeline: dataset -> features -> selectors -> evaluation -> ablations.

Usage::

    python -m rts.pipeline                    # full run, full candidate set
    python -m rts.pipeline --candidates covered
    python -m rts.pipeline --skip-ablations
    python -m rts.pipeline --labels full
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from . import (
    accessors,
    config,
    datasets,
    evaluate,
    features,
    models,
    populations,
    reporting,
    splits,
)

# The study's documented arm explicitly enables history on an imposed dataset and
# explicitly names the averaging population. Both are opt-ins: the default is history off
# for an imposed order, and a population that cannot exist is unmeasured rather than
# silently empty.
HISTORY = True
POPULATION = "fault_bearing"


def build_context(ds, split: splits.Split) -> models.Context:
    matrix = features.structured(ds, history=HISTORY)
    return models.Context(
        ds=ds,
        features=matrix,
        split=split,
        bm25=features.text.build_bm25_scores(ds),
    )


def run(
    candidates_mode: str = "full",
    budgets: tuple[float, ...] = config.DEFAULT_BUDGETS,
    n_bootstrap: int = config.DEFAULT_BOOTSTRAP,
    include_semif: bool = True,
    run_ablations: bool = True,
    seed: int = config.SEED,
    labels: str = "mutmut",
) -> dict:
    t_start = time.perf_counter()
    print("=" * 78)
    print("RTS feasibility pipeline")
    print("=" * 78)

    ds = datasets.marshmallow(labels=labels, order_seed=seed)
    split = splits.make_split(ds)
    for key, value in reporting.describe(ds, split).items():
        print(f"  {key:>32}: {value}")

    candidates = accessors.candidates(ds, candidates_mode)
    cand_counts = accessors.candidate_counts(ds, candidates)
    print(f"\ncandidate mode      : {candidates_mode}")
    print(
        f"candidates per change: mean {cand_counts.mean():.1f}  "
        f"min {cand_counts.min()}  max {cand_counts.max()}"
    )

    ctx = build_context(ds, split)
    audit = reporting.audit(ds, split)
    for warning in audit:
        print(f"  [audit] {warning.code}: {warning.note}")

    # --- selectors ---
    selectors = models.default_selectors(include_semif=include_semif)
    all_scores: dict[str, np.ndarray] = {}
    results: dict[str, list[evaluate.BudgetResult]] = {}
    skipped: dict[str, str] = {}

    for selector in selectors:
        t0 = time.perf_counter()
        try:
            scores = selector.scores(ctx)
        except FileNotFoundError as exc:
            skipped[selector.name] = str(exc).splitlines()[0]
            print(f"\n[skip] {selector.name}: {skipped[selector.name]}")
            continue
        all_scores[selector.name] = scores
        results[selector.name] = evaluate.evaluate(
            scores, ds, split.test_idx, budgets=budgets,
            n_bootstrap=n_bootstrap, seed=seed, candidates=candidates,
            population=POPULATION,
        )
        print(evaluate.format_table(selector.name, results[selector.name]))
        print(f"  ({time.perf_counter() - t0:.1f}s)")

    if not results:
        raise SystemExit("no selectors produced scores")

    # --- feature importances ---
    for name, selector in [(s.name, s) for s in selectors if isinstance(s, models.XGBoostSelector)]:
        if selector.importances_:
            print(f"\n{name} feature importances")
            for feature, importance in list(selector.importances_.items())[:15]:
                print(f"  {feature:>24}: {importance:.4f}")

    # --- ablations ---
    ablations: dict[str, list[evaluate.BudgetResult]] = {}
    if run_ablations:
        print("\n" + "=" * 78)
        print("Ablations (change-shuffle and test-shuffle, BM25 as the probe)")
        print("=" * 78)
        for label, kwargs in [
            ("bm25_change_shuffled", {"shuffle_changes": True}),
            ("bm25_test_shuffled", {"shuffle_tests": True}),
            ("bm25_both_shuffled", {"shuffle_changes": True, "shuffle_tests": True}),
        ]:
            shuffled = features.text.build_bm25_scores(ds, seed=seed, **kwargs)
            ablations[label] = evaluate.evaluate(
                shuffled, ds, split.test_idx, budgets=budgets,
                n_bootstrap=n_bootstrap, seed=seed, candidates=candidates,
                population=POPULATION,
            )
            print(evaluate.format_table(label, ablations[label]))

    # --- paired comparisons at a representative budget ---
    print("\n" + "=" * 78)
    print("Paired bootstrap: recall differences at budget 0.05")
    print("=" * 78)
    probe_budget = 0.05
    hits = {
        name: evaluate.per_change_hits(scores, ds, split.test_idx, probe_budget, candidates)
        for name, scores in all_scores.items()
    }
    for label, kwargs in [
        ("bm25_change_shuffled", {"shuffle_changes": True}),
        ("bm25_test_shuffled", {"shuffle_tests": True}),
    ]:
        shuffled = features.text.build_bm25_scores(ds, seed=seed, **kwargs)
        hits[label] = evaluate.per_change_hits(
            shuffled, ds, split.test_idx, probe_budget, candidates
        )

    comparisons: dict[str, dict] = {}
    reference = "coverage" if "coverage" in hits else sorted(hits)[0]
    print(f"  reference: {reference}")
    for name in sorted(hits):
        if name == reference:
            continue
        stat = evaluate.paired_bootstrap(hits[name], hits[reference], n_bootstrap, seed)
        comparisons[name] = stat
        print(
            f"  {name:>24} vs {reference:<10} delta {stat['delta']:+.3f} "
            f"[{stat['lo']:+.3f}, {stat['hi']:+.3f}] p={stat['p_value']:.4f} n={stat['n']}"
        )

    # --- sparse arm ---
    sparse_report: dict[str, list[dict]] = {}
    recurrence: dict[str, float] = {}
    if run_ablations:
        print("\n" + "=" * 78)
        print("Sparse arm: changes whose (file, test) pairs recur least")
        print("=" * 78)
        counts = accessors.pair_counts(ds)
        faults = accessors.fault_idx(ds)
        covered_by = accessors.covered(ds)
        paths = accessors.change_paths(ds)
        fault_pairs = np.array(
            [
                counts.get((paths[i], sorted(ds.killing_tests(ds.changes[i]))[0]), 0)
                for i in faults
            ]
        )
        maxpc = np.array(
            [
                max((counts[(paths[i], t)] for t in covered_by[i]), default=0)
                for i in range(ds.n_changes)
            ]
        )
        recurrence = {
            "killing_pair_count_median": float(np.median(fault_pairs)),
            "killing_pair_count_mean": float(fault_pairs.mean()),
            "killing_pair_count_max": float(fault_pairs.max()),
            "per_change_max_pair_count_p05": float(np.percentile(maxpc, 5)),
            "per_change_max_pair_count_p50": float(np.percentile(maxpc, 50)),
        }
        for key, value in recurrence.items():
            print(f"  {key:>32}: {value:.1f}")
        print(
            "\n  Note: a (file, killing-test) pair recurs a median of "
            f"{np.median(fault_pairs):.0f} times, because mutation testing revisits\n"
            "  the same function many times (median 8 mutants per function). Real\n"
            "  evolution would revisit it far less, so history features are flattered\n"
            "  by this setup. The sparse arm below is the mitigation."
        )

        for threshold in (80, 160):
            mask = populations.sparse_mask(ds, max_pair_count=threshold)
            sub = split.test_idx[mask[split.test_idx]]
            n_faults = int(sum(1 for i in sub if ds.killing_tests(ds.changes[i])))
            if n_faults < 10:
                print(f"\n  threshold {threshold}: only {n_faults} held-out faults, skipped")
                continue
            print(
                f"\n  --- max_pair_count <= {threshold}: "
                f"{len(sub)} held-out changes, {n_faults} faults ---"
            )
            for name, scores in all_scores.items():
                res = evaluate.evaluate(
                    scores, ds, sub, budgets=(0.05, 0.1, 0.2),
                    n_bootstrap=n_bootstrap, seed=seed, candidates=candidates,
                    population=POPULATION,
                )
                sparse_report[f"{name}@{threshold}"] = evaluate.results_to_dicts(res)
                cells = "  ".join(f"b{r.budget:.2f}={r.recall:.3f}" for r in res)
                print(f"    {name:>24}: {cells}")

    # --- persist ---
    out_dir = config.ensure_artifacts_dir()
    payload = {
        "candidate_mode": candidates_mode,
        "budgets": list(budgets),
        "seed": seed,
        "labels": labels,
        "population": POPULATION,
        "dataset": reporting.describe(ds, split),
        "dataset_declaration": ds.declaration(),
        "split": {
            "fraction": split.fraction,
            "shuffle": split.shuffle,
            "effective_ordering": split.effective_ordering.value,
        },
        "warnings": [w.to_dict() for w in ctx.warnings],
        "audit": [w.to_dict() for w in audit],
        "results": {name: evaluate.results_to_dicts(res) for name, res in results.items()},
        "ablations": {name: evaluate.results_to_dicts(res) for name, res in ablations.items()},
        "sparse_arm": sparse_report,
        "recurrence": recurrence,
        "paired_vs_reference": comparisons,
        "reference": reference,
        "skipped": skipped,
    }
    (out_dir / f"results_{candidates_mode}.json").write_text(json.dumps(payload, indent=2))
    print(f"\n[done] wrote {out_dir / f'results_{candidates_mode}.json'}")
    print(f"[done] total {time.perf_counter() - t_start:.1f}s")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="full", choices=["full", "covered"])
    parser.add_argument("--no-semif", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=config.DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--labels", default="mutmut", choices=list(config.LABEL_SOURCES))
    args = parser.parse_args()

    run(
        candidates_mode=args.candidates,
        n_bootstrap=args.bootstrap,
        include_semif=not args.no_semif,
        run_ablations=not args.skip_ablations,
        seed=args.seed,
        labels=args.labels,
    )


if __name__ == "__main__":
    main()
