"""Render the traceability-loss ladder (W1 in ``docs/plan_next_steps.md``).

Motivation
----------
The deployment of interest is a test suite driving an embedded system across a boundary:
Python tests send commands over serial/socket to firmware, so the changed code does not run in
the test process. Four feature families that carry this benchmark die at once --

* **coverage** cannot cross the boundary,
* **filename / path proximity** assumes co-located, conventionally named unit tests,
* **history** assumes a ``(file, test)`` pair recurs, which it does not at thousands of tests,
* **text overlap** is reduced to protocol and feature vocabulary.

The deliverable is a **degradation curve**: the question is not what any single number is, but
whether the ordering changes as traceability is removed -- and in particular whether SemIf
crosses the classical baselines.

What this module is, after the experiment layer
----------------------------------------------
The sweep is *declared* in :mod:`rts.studies` (``ladder_arm``): the rungs, the selector set, the
two averaging populations, the budget grid and the resample counts are all values there. This
module runs that arm and renders ``artifacts/ladder.json``, whose shape predates the layer and is
kept because the recorded numbers, ``docs/implementation.md`` §12.2 and the figures are written
against it. It contains no experiment logic.

Two things the layer expresses that the old driver had to hand-code:

* **a rung is a block with a family withheld**, so withholding takes the same unmeasured path a
  genuinely absent capability takes -- a typo in a family name raises instead of quietly
  ablating nothing, which is the bug that motivated ``docs/refactor.md``;
* **SemIf is inapplicable to one of the two populations.** Its cache covers only the changes
  ``starved141`` selects, and ``semif.load_scores`` fills every uncached pair with a sentinel
  rather than reporting that it has no score, so evaluating it on the other population would
  produce a number that looks like a measurement and is not. The old driver avoided that by not
  tabulating SemIf there; the arm declares it with ``Element.applies``, so the cell is reported
  unmeasured with a reason instead of vanishing.

Usage
-----
    python -m rts.ladder                 # both label sources
    python -m rts.ladder --labels full
"""

from __future__ import annotations

import argparse
import json

from . import config, studies
from .data.contract import Unmeasured
from .experiment import ROLE_FEATURES, ROLE_MODEL, ROLE_POPULATION, run

#: The key the recorded artifact files the paired comparison under. Its name states which
#: population the pairing happened on, which is what makes the delta interpretable.
COMPARISON_KEY = "starved141_vs_semif_b0.05"


def _rung_tables(report, rung: str, population: str) -> dict:
    """One population's selector table for one rung, in the recorded shape.

    Unmeasured cells are omitted rather than tabulated, because a table entry is a claim that
    the number exists; the population's own availability is reported separately, and the cells
    that could not be measured are in the report.
    """
    table: dict[str, dict] = {}
    for cell in report.cells:
        factors = cell.cell.factors_dict()
        if factors[ROLE_FEATURES] != rung or factors[ROLE_POPULATION] != population:
            continue
        results = cell.results
        table[cell.cell.name(ROLE_MODEL)] = {
            "recall": {f"{r['budget']:.2f}": round(r["recall"], 4) for r in results},
            "k": {f"{r['budget']:.2f}": r["k"] for r in results},
            "n_faults": results[0]["n_faults"] if results else 0,
        }
    return table


def _semif_comparisons(report, rung: str) -> dict:
    """The paired comparison against SemIf on the ``starved141`` population.

    ``evaluate.paired_bootstrap(a, b)`` returns ``recall(a) - recall(b)`` with ``a`` the
    baseline, so a **negative** delta means SemIf is ahead -- which the recorded key spells out.
    """
    out: dict[str, dict] = {}
    for record in report.comparisons:
        if not record.get("measured"):
            continue
        if record["group"].get(ROLE_FEATURES) != rung:
            continue
        if record["group"].get(ROLE_POPULATION) != "starved141":
            continue
        out[record["cell"]] = {
            "delta_baseline_minus_semif": round(record["delta"], 4),
            "lo": round(record["lo"], 4),
            "hi": round(record["hi"], 4),
            "p": round(record["p_value"], 5),
            "semif_ahead": bool(record["delta"] < 0),
        }
    return out


def run_label_source(label_source: str, verbose: bool = True) -> dict:
    """Run the ladder for one label source and return the recorded artifact's shape.

    The dataset is rebuilt for exactly one thing: ``ladder_populations`` builds the two
    populations *from* the dataset, and a population is a value rather than a run statistic
    (``docs/refactor.md`` §7 permits two datasets to coexist, so rebuilding is cheap). Everything
    else -- the dataset's shape, its declaration and the two population sizes -- comes from the
    report, so this cannot describe a dataset the run did not measure.
    """
    ds = studies.dataset(label_source)
    declared = studies.ladder_populations(ds)
    report = run(studies.ladder_arm(label_source), save=False, verbose=verbose)
    margins = studies.semif_margins(report)
    described = report.describe()

    if verbose:
        print("=" * 78)
        print(f"TRACEABILITY LADDER -- labels={label_source}")
        print("=" * 78)
        print(
            f"  changes {described['changes']}  tests {described['tests']}  "
            f"held-out faults {described['held_out_faults']}"
        )
        print(f"  measured {len(report.cells)} of {report.n_cells} cells")
        for entry in report.unmeasured:
            print(f"  [unmeasured] {entry['key']}: {entry['note']}")

    payload: dict = {
        "labels": label_source,
        "n_changes": described["changes"],
        "n_tests": described["tests"],
        "held_out_faults": described["held_out_faults"],
        "populations": {
            k: (None if isinstance(v, Unmeasured) else report.population_size(k)[0])
            for k, v in declared.items()
        },
        "populations_unmeasured": {
            k: v.to_dict() for k, v in declared.items() if isinstance(v, Unmeasured)
        },
        "dataset_declaration": report.declaration(),
        "rungs": {},
    }

    for rung, removed in studies.RUNGS:
        rung_cells = [
            c for c in report.cells if c.cell.factors_dict()[ROLE_FEATURES] == rung
        ]
        sample = rung_cells[0] if rung_cells else None
        rung_report: dict = {
            "removed": list(removed),
            # The withheld columns and the derivation's warnings come from the same block the
            # cells were built from, so they describe the run rather than being restated here.
            "withheld": [u["column"] for u in sample.features["unmeasured"]] if sample else [],
            "warnings": list(sample.warnings) if sample else [],
            "selectors": {},
            "comparisons": {},
        }

        for population, spec in declared.items():
            if isinstance(spec, Unmeasured):
                rung_report["selectors"][population] = {"unmeasured": spec.to_dict()}
                continue
            rung_report["selectors"][population] = _rung_tables(report, rung, population)

        comparisons = _semif_comparisons(report, rung)
        if comparisons:
            rung_report["comparisons"][COMPARISON_KEY] = comparisons
            rung_report["semif_margin"] = margins.get(rung, {})

        payload["rungs"][rung] = rung_report

        if verbose:
            print(f"\n  [{rung}] removed={list(removed) or 'nothing'}")
            for population, table in rung_report["selectors"].items():
                if "unmeasured" in table:
                    print(f"    {population}: unmeasured")
                    continue
                print(f"    {population}:")
                for name, row in table.items():
                    cells = "  ".join(f"b{k}={v:.3f}" for k, v in row["recall"].items())
                    print(f"      {name:26s} {cells}")
            if comparisons:
                print("    SemIf vs baseline @b0.05 (negative delta = SemIf ahead):")
                for name, c in comparisons.items():
                    flag = "*" if c["p"] < 0.05 else " "
                    print(
                        f"      {name:26s} {c['delta_baseline_minus_semif']:+.3f} "
                        f"[{c['lo']:+.3f},{c['hi']:+.3f}] p={c['p']:.4f}{flag}"
                    )
            for key, m in rung_report.get("semif_margin", {}).items():
                print(
                    f"      b{key}: SemIf {m['semif_recall']:.3f} vs "
                    f"{m['best_classical']} {m['best_recall']:.3f} "
                    f"-> {m['semif_margin']:+.3f}"
                )

    return payload


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
