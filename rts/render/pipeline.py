"""Run the study condition and render ``artifacts/results_{full,covered}.json``.

The sweep is *declared* in :mod:`rts.studies` (``run_study``): the datasets, the feature block,
the ranker set and its BM25 shuffle controls, the averaging subsets including the low_cooccurrence
thresholds, the split, the budgets and the resample counts are all values there. This module runs
that condition and renders the artifact whose shape the recorded numbers, ``docs/implementation.md`` and
``figures.py`` are written against. It contains no experiment logic.

Three structural facts the artifact records, each of which the layer made explicit:

* the BM25 shuffle controls are **model levels**, not a driver's second loop. ``results`` and
  ``ablations`` are therefore two slices of one grid rather than two computations, and the
  controls carry the same provenance as everything else;
* the low-co-occurrence condition is a **second experiment**, because it reports three budgets where the headline
  condition reports six and ``budgets`` is a control -- a control is by definition constant across a run. The
  two conditions share one score cache, so the low_cooccurrence corners pay only for their metric sweeps;
* ``paired_vs_reference`` includes the *both-shuffled* control, which the driver did not pair.
  That is additive: the layer pairs every model in the group, and dropping one to match the old
  artifact's key set would discard a computed number for cosmetic parity.

Usage::

    python -m rts.render.pipeline
    python -m rts.render.pipeline --candidate sets covered
    python -m rts.render.pipeline --labels full
    python -m rts.render.pipeline --skip-ablations --no-semif
"""

from __future__ import annotations

import argparse
import json
import time

from .. import config, studies
from ..experiment import FACTOR_MODEL, FACTOR_SUBSET

#: The subset the headline numbers are averaged over, and the ranker every other one is
#: paired against at the probe budget.
SUBSET = "detectable"
REFERENCE = "coverage"
PROBE_BUDGET = 0.05


def _at_budgets(result, budgets) -> list[dict]:
    """The design point's rows restricted to ``budgets``, in the order the design point produced them.

    The low-co-occurrence condition reports a subset of the headline condition's budget grid; reading the rows out of the
    one sweep rather than re-running is why the two conditions can share a score cache.
    """
    wanted = {round(float(b), 6) for b in budgets}
    return [row for row in result.results if round(float(row["budget"]), 6) in wanted]


def render(label: str, candidate_policy: str, report, low_cooccurrence_report) -> dict:
    """The recorded artifact's shape, built from the reports.

    Everything the artifact says about the data comes from the report: the run recorded the
    dataset's declaration, its description under the split it used, and the ``(file, test)``
    pair recurrence, so the renderer does not rebuild the dataset to describe one. That also
    removes the old check that the renderer's split matched the run's -- the description now
    *is* the run's, so the mismatch it guarded against cannot happen.
    """
    main = [result for result in report.design_points if result.subset == SUBSET]
    if not main:
        raise RuntimeError(f"no design point was measured on subset {SUBSET!r}")
    sample = main[0]

    ablation_names = {name for name, _ in studies.ABLATION_MODELS}
    results: dict[str, list[dict]] = {}
    ablations: dict[str, list[dict]] = {}
    for result in main:
        name = result.design_point.name(FACTOR_MODEL)
        (ablations if name in ablation_names else results)[name] = result.results

    low_cooccurrence_condition: dict[str, list[dict]] = {}
    if low_cooccurrence_report is not None:
        for threshold in studies.LOW_COOCCURRENCE_THRESHOLDS:
            subset = f"low_cooccurrence{threshold}"
            for result in low_cooccurrence_report.design_points:
                if result.subset != subset:
                    continue
                key = f"{result.design_point.name(FACTOR_MODEL)}@{threshold}"
                low_cooccurrence_condition[key] = _at_budgets(result, studies.LOW_COOCCURRENCE_BUDGETS)

    contrasts = {
        record["design_point"]: {
            "delta": record["delta"],
            "lo": record["lo"],
            "hi": record["hi"],
            "p_value": record["p_value"],
            "n": record["n"],
        }
        for record in report.contrasts
        if record.get("measured")
        and record["reference"] == REFERENCE
        and record["group"].get(FACTOR_SUBSET) == SUBSET
    }

    return {
        "candidate_policy": candidate_policy,
        "budgets": list(report.environment.controls.budgets),
        "seed": report.environment.controls.seed,
        "labels": label,
        "subset": SUBSET,
        "dataset": report.describe(),
        "dataset_metadata": report.metadata(),
        "split": {
            "fraction": sample.split["fraction"],
            "shuffle": sample.split["shuffle"],
            "effective_order": sample.split["effective_order"],
        },
        # Two vocabularies, kept apart: the derivation's caveats, and what the dataset and split
        # say about trusting a number at all.
        "diagnostics": list(sample.diagnostics),
        "audit": list(sample.audit),
        "results": results,
        "ablations": ablations,
        "low_cooccurrence_condition": low_cooccurrence_condition,
        "recurrence": report.recurrence(),
        "paired_vs_reference": dict(sorted(contrasts.items())),
        "reference": REFERENCE,
        # A design point whose requirement is unmet is *reported* rather than skipped, so this is a map
        # from ranker to why it could not be measured -- empty when everything was.
        "skipped": {
            entry["factors"][FACTOR_MODEL]: entry["note"] for entry in report.undefined
        },
    }


def run(
    candidate_policy: str = "full",
    labels: str = "mutmut",
    include_semif: bool = True,
    run_ablations: bool = True,
    n_bootstrap: int = config.DEFAULT_BOOTSTRAP,
    seed: int = config.SEED,
) -> dict:
    t_start = time.perf_counter()
    print("=" * 78)
    print("RTS feasibility pipeline")
    print("=" * 78)

    # Applied per condition, so the overrides do not disturb the budgets that make the two conditions
    # different. The seed now reaches every stochastic component -- the split, the ordering, the
    # random baseline, the BM25 shuffles and XGBoost -- because each reads it from the run.
    overrides: dict = {}
    if n_bootstrap != config.DEFAULT_BOOTSTRAP:
        overrides["bootstrap resamples"] = n_bootstrap
    if seed != config.SEED:
        overrides["seed"] = seed

    report, low_cooccurrence_report = studies.run_study(
        labels,
        candidate_policy=candidate_policy,
        ablations=run_ablations,
        include_semif=include_semif,
        knobs_overrides=overrides or None,
        save=False,
        verbose=True,
    )
    payload = render(labels, candidate_policy, report, low_cooccurrence_report)

    print("\nfeature importances")
    for result in report.design_points:
        if not result.importances or result.subset != SUBSET:
            continue
        top = list(result.importances.items())[:5]
        print(f"  {result.design_point.name(FACTOR_MODEL)}")
        for feature, importance in top:
            print(f"    {feature:>24}: {importance:.4f}")

    out_dir = config.ensure_artifacts_dir()
    path = out_dir / f"results_{candidate_policy}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\n[done] wrote {path}")
    print(f"[done] total {time.perf_counter() - t_start:.1f}s")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-policy", default="full", choices=["full", "coverage_restricted"])
    parser.add_argument("--no-semif", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=config.DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--labels", default="mutmut", choices=list(config.LABEL_SOURCES))
    args = parser.parse_args()

    run(
        candidate_policy=args.candidate_policy,
        labels=args.labels,
        include_semif=not args.no_semif,
        run_ablations=not args.skip_ablations,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
