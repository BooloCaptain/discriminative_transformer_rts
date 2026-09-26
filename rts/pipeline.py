"""Run the study arm and render ``artifacts/results_{full,covered}.json``.

The sweep is *declared* in :mod:`rts.studies` (``run_study``): the datasets, the feature block,
the selector set and its BM25 shuffle controls, the averaging populations including the sparse
thresholds, the split, the budgets and the resample counts are all values there. This module runs
that arm and renders the artifact whose shape the recorded numbers, ``implementation.md`` and
``figures.py`` are written against. It contains no experiment logic.

Three structural facts the artifact records, each of which the layer made explicit:

* the BM25 shuffle controls are **model elements**, not a driver's second loop. ``results`` and
  ``ablations`` are therefore two slices of one grid rather than two computations, and the
  controls carry the same provenance as everything else;
* the sparse arm is a **second experiment**, because it reports three budgets where the headline
  arm reports six and ``budgets`` is a knob -- a knob is by definition constant across a run. The
  two arms share one score cache, so the sparse corners pay only for their metric sweeps;
* ``paired_vs_reference`` includes the *both-shuffled* control, which the driver did not pair.
  That is additive: the layer pairs every model in the group, and dropping one to match the old
  artifact's key set would discard a computed number for cosmetic parity.

Usage::

    python -m rts.pipeline
    python -m rts.pipeline --candidates covered
    python -m rts.pipeline --labels full
    python -m rts.pipeline --skip-ablations --no-semif
"""

from __future__ import annotations

import argparse
import json
import time

from . import config, studies
from .experiment import ROLE_MODEL, ROLE_POPULATION

#: The population the headline numbers are averaged over, and the selector every other one is
#: paired against at the probe budget.
POPULATION = "fault_bearing"
REFERENCE = "coverage"
PROBE_BUDGET = 0.05


def _at_budgets(cell, budgets) -> list[dict]:
    """The cell's rows restricted to ``budgets``, in the order the cell produced them.

    The sparse arm reports a subset of the headline arm's budget grid; reading the rows out of the
    one sweep rather than re-running is why the two arms can share a score cache.
    """
    wanted = {round(float(b), 6) for b in budgets}
    return [row for row in cell.results if round(float(row["budget"]), 6) in wanted]


def render(label: str, candidates: str, report, sparse_report) -> dict:
    """The recorded artifact's shape, built from the reports.

    Everything the artifact says about the data comes from the report: the run recorded the
    dataset's declaration, its description under the split it used, and the ``(file, test)``
    pair recurrence, so the renderer does not rebuild the dataset to describe one. That also
    removes the old check that the renderer's split matched the run's -- the description now
    *is* the run's, so the mismatch it guarded against cannot happen.
    """
    main = [cell for cell in report.cells if cell.population == POPULATION]
    if not main:
        raise RuntimeError(f"no cell was measured on population {POPULATION!r}")
    sample = main[0]

    ablation_names = {name for name, _ in studies.ABLATION_MODELS}
    results: dict[str, list[dict]] = {}
    ablations: dict[str, list[dict]] = {}
    for cell in main:
        name = cell.cell.name(ROLE_MODEL)
        (ablations if name in ablation_names else results)[name] = cell.results

    sparse_arm: dict[str, list[dict]] = {}
    if sparse_report is not None:
        for threshold in studies.SPARSE_THRESHOLDS:
            population = f"sparse{threshold}"
            for cell in sparse_report.cells:
                if cell.population != population:
                    continue
                key = f"{cell.cell.name(ROLE_MODEL)}@{threshold}"
                sparse_arm[key] = _at_budgets(cell, studies.SPARSE_BUDGETS)

    comparisons = {
        record["cell"]: {
            "delta": record["delta"],
            "lo": record["lo"],
            "hi": record["hi"],
            "p_value": record["p_value"],
            "n": record["n"],
        }
        for record in report.comparisons
        if record.get("measured")
        and record["reference"] == REFERENCE
        and record["group"].get(ROLE_POPULATION) == POPULATION
    }

    return {
        "candidate_mode": candidates,
        "budgets": list(report.environment.knobs.budgets),
        "seed": report.environment.knobs.seed,
        "labels": label,
        "population": POPULATION,
        "dataset": report.describe(),
        "dataset_declaration": report.declaration(),
        "split": {
            "fraction": sample.split["fraction"],
            "shuffle": sample.split["shuffle"],
            "effective_ordering": sample.split["effective_ordering"],
        },
        # Two vocabularies, kept apart: the derivation's caveats, and what the dataset and split
        # say about trusting a number at all.
        "warnings": list(sample.warnings),
        "audit": list(sample.audit),
        "results": results,
        "ablations": ablations,
        "sparse_arm": sparse_arm,
        "recurrence": report.recurrence(),
        "paired_vs_reference": dict(sorted(comparisons.items())),
        "reference": REFERENCE,
        # A cell whose requirement is unmet is *reported* rather than skipped, so this is a map
        # from selector to why it could not be measured -- empty when everything was.
        "skipped": {
            entry["factors"][ROLE_MODEL]: entry["note"] for entry in report.unmeasured
        },
    }


def run(
    candidates_mode: str = "full",
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

    # Applied per arm, so the overrides do not disturb the budgets that make the two arms
    # different. The seed now reaches every stochastic component -- the split, the ordering, the
    # random baseline, the BM25 shuffles and XGBoost -- because each reads it from the run.
    overrides: dict = {}
    if n_bootstrap != config.DEFAULT_BOOTSTRAP:
        overrides["n_bootstrap"] = n_bootstrap
    if seed != config.SEED:
        overrides["seed"] = seed

    report, sparse_report = studies.run_study(
        labels,
        candidates=candidates_mode,
        ablations=run_ablations,
        include_semif=include_semif,
        knobs_overrides=overrides or None,
        save=False,
        verbose=True,
    )
    payload = render(labels, candidates_mode, report, sparse_report)

    print("\nfeature importances")
    for cell in report.cells:
        if not cell.importances or cell.population != POPULATION:
            continue
        top = list(cell.importances.items())[:5]
        print(f"  {cell.cell.name(ROLE_MODEL)}")
        for feature, importance in top:
            print(f"    {feature:>24}: {importance:.4f}")

    out_dir = config.ensure_artifacts_dir()
    path = out_dir / f"results_{candidates_mode}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\n[done] wrote {path}")
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
        labels=args.labels,
        include_semif=not args.no_semif,
        run_ablations=not args.skip_ablations,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
