"""Verify the experiment layer reproduces the recorded artifacts.

The documented numbers are the deliverable (``refactor.md`` §8), so the layer is checked
against them rather than trusted: ``studies.study_arm()`` against
``artifacts/results_full.json`` and ``studies.ladder_arm()`` against
``artifacts/ladder.json``, for both label sources.

Run it after any change to the experiment layer::

    python scripts/verify_experiment_layer.py            # both arms
    python scripts/verify_experiment_layer.py study      # one arm
    python scripts/verify_experiment_layer.py ladder --labels full

Comparison is field by field and *exact* where the recorded artifact is unrounded, and at the
artifact's own rounding where it is rounded (the ladder rounds recall to 4 dp, the study arm
records full precision). A non-zero exit means the layer moved a number.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rts import config, studies  # noqa: E402
from rts.experiment import ROLE_FEATURES, ROLE_MODEL, ROLE_POPULATION, run  # noqa: E402

FAILURES: list[str] = []
CHECKS = [0]


def check(label: str, got, want, ndigits: int | None = None) -> None:
    CHECKS[0] += 1
    if isinstance(got, float) and isinstance(want, float):
        if math.isnan(got) and math.isnan(want):
            return
        if ndigits is not None:
            got, want = round(got, ndigits), round(want, ndigits)
    if got != want:
        FAILURES.append(f"{label}: got {got!r}, recorded {want!r}")


def load(name: str) -> dict:
    return json.loads((config.ARTIFACTS / name).read_text())


def cells_by(report, role: str) -> dict:
    out: dict[tuple, object] = {}
    for cell in report.cells:
        key = tuple(
            (r, n) for r, n in cell.cell.factors if r != role
        ) + ((role, cell.cell.name(role)),)
        out[tuple(sorted(key))] = cell
    return out


def find(report, **factors):
    wanted = {r: n for r, n in factors.items()}
    for cell in report.cells:
        have = cell.cell.factors_dict()
        if all(have.get(r) == n for r, n in wanted.items()):
            return cell
    return None


# --- the study arm ----------------------------------------------------------


def verify_study(labels: str = "mutmut", candidates: str = "full") -> None:
    recorded = load(f"results_{candidates}.json")
    experiment = studies.study_arm(labels, candidates=candidates)
    report = run(experiment, save=False, verbose=True)

    print(f"\n--- study arm ({labels}, {candidates}) vs results_{candidates}.json ---")
    for name, rows in recorded["results"].items():
        cell = find(
            report,
            dataset=f"marshmallow_{labels}",
            features="structured",
            model=name,
            population="fault_bearing",
        )
        if cell is None:
            FAILURES.append(f"study: no measured cell for model {name!r}")
            continue
        for want, got in zip(rows, cell.results):
            for field in ("budget", "k", "recall", "precision", "f_measure",
                          "suite_reduction", "recall_lo", "recall_hi", "n_faults"):
                check(f"study[{name}] b{want['budget']:.2f} {field}", got[field], want[field])
    print(f"  {len(recorded['results'])} selectors checked, {CHECKS[0]} field comparisons so far")

    # The layer records the split per cell; the recorded artifact records it once.
    for name in ("fraction", "shuffle", "effective_ordering"):
        cell = report.cells[0]
        check(f"study split.{name}", cell.split[name], recorded["split"][name])


# --- the ladder arm ---------------------------------------------------------


def verify_ladder(labels: str = "mutmut") -> None:
    recorded = load("ladder.json")[labels]
    experiment = studies.ladder_arm(labels)
    report = run(experiment, save=False, verbose=True)

    print(f"\n--- ladder arm ({labels}) vs ladder.json[{labels!r}] ---")
    # ``ladder.json`` records the dataset's total change count; a cell records the size of the
    # evaluation window, which is a different quantity. Compare the one both have: the number of
    # held-out faults, which is the population size the metrics were averaged over.
    faults_cell = find(report, features="L0_all", population="heldout530", model="coverage")
    if faults_cell is None:
        FAILURES.append("ladder: no measured cell for the held-out fault population")
    else:
        check("ladder held_out_faults", faults_cell.n_rows, recorded["held_out_faults"])

    unmeasured = {
        (u["factors"][ROLE_FEATURES], u["factors"][ROLE_POPULATION], u["factors"][ROLE_MODEL])
        for u in report.unmeasured
    }

    for rung, want_rung in recorded["rungs"].items():
        sample = find(report, features=rung, population="starved141", model="semif_reranker")
        if sample is not None:
            withheld = sorted(u["column"] for u in sample.features["unmeasured"])
            check(f"ladder[{rung}] withheld", withheld, sorted(want_rung["withheld"]))

        for population, table in want_rung["selectors"].items():
            if "unmeasured" in table:
                continue
            for model, want in table.items():
                cell = find(report, features=rung, population=population, model=model)
                if cell is None:
                    if (rung, population, model) in unmeasured:
                        continue
                    FAILURES.append(
                        f"ladder[{rung}/{population}]: no cell for {model!r}, and it is "
                        "not recorded as unmeasured"
                    )
                    continue
                for row in cell.results:
                    key = f"{row['budget']:.2f}"
                    check(
                        f"ladder[{rung}/{population}/{model}] b{key} recall",
                        row["recall"],
                        want["recall"][key],
                        ndigits=4,
                    )
                    check(
                        f"ladder[{rung}/{population}/{model}] b{key} k",
                        row["k"],
                        want["k"][key],
                    )
                    check(
                        f"ladder[{rung}/{population}/{model}] b{key} n_faults",
                        row["n_faults"],
                        want["n_faults"],
                    )

        # Paired comparisons, recorded for starved141 only because a reference model with no
        # scores for a population is not a reference (see studies._ladder_semif_applies).
        want_comps = want_rung["comparisons"].get("starved141_vs_semif_b0.05", {})
        got_comps = {
            record["cell"]: record
            for record in report.comparisons
            if record.get("measured")
            and record["group"].get(ROLE_FEATURES) == rung
            and record["group"].get(ROLE_POPULATION) == "starved141"
        }
        for model, want in want_comps.items():
            got = got_comps.get(model)
            if got is None:
                FAILURES.append(
                    f"ladder[{rung}] comparison for {model!r} missing; it is recorded"
                )
                continue
            check(f"ladder[{rung}][{model}] delta", got["delta"],
                  want["delta_baseline_minus_semif"], ndigits=4)
            check(f"ladder[{rung}][{model}] lo", got["lo"], want["lo"], ndigits=4)
            check(f"ladder[{rung}][{model}] hi", got["hi"], want["hi"], ndigits=4)
            check(f"ladder[{rung}][{model}] p", got["p_value"], want["p"], ndigits=5)

    # The headline quantity, read back off the report rather than recorded by the kernel.
    margins = studies.semif_margins(report)
    for rung, want in recorded["rungs"].items():
        got_rung = margins.get(rung, {})
        for key, want_margin in want.get("semif_margin", {}).items():
            got = got_rung.get(key)
            if got is None:
                FAILURES.append(f"ladder[{rung}] margin b{key} missing")
                continue
            check(f"ladder[{rung}] margin b{key}", got["semif_margin"],
                  want_margin["semif_margin"], ndigits=4)
            check(f"ladder[{rung}] margin b{key} best", got["best_classical"],
                  want_margin["best_classical"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", nargs="*", default=["study", "ladder"],
                        choices=["study", "ladder"])
    parser.add_argument("--labels", nargs="*", default=["mutmut", "full"])
    parser.add_argument("--candidates", default="full", choices=["full", "covered"])
    args = parser.parse_args()

    if "study" in args.arms:
        verify_study("mutmut", args.candidates)
        print(f"  study arm total: {CHECKS[0]} comparisons")
    if "ladder" in args.arms:
        before = CHECKS[0]
        for labels in args.labels:
            verify_ladder(labels)
        print(f"  ladder arms total: {CHECKS[0] - before} comparisons")

    print("\n" + "=" * 78)
    print(f"{CHECKS[0]} comparisons, {len(FAILURES)} mismatch(es)")
    for failure in FAILURES[:40]:
        print(f"  MISMATCH {failure}")
    if len(FAILURES) > 40:
        print(f"  ... and {len(FAILURES) - 40} more")
    print("=" * 78)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
