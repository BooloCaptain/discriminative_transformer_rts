"""Summarise a `probe_full_suite.py` output directory into the Gap 1 validation artifact.

Reads the per-mutant JSON written by the probe and reports everything the plan's
validation gates and deliverables need:

* gate 2  -- is the old (mutmut-selected) killing set a strict subset of the new one?
* gate 5  -- does every non-zero-outcome mutant record 1190 outcomes?
* the killer-count distribution, which is what replaces the documented
  "exactly one killing test per fault" property;
* out-of-coverage faults, split into killed mutants and mutmut *survivors* (a survivor's
  killers are out-of-coverage by construction, so this is the cleanest measure of what the
  ``covered`` candidate mask was discarding);
* killers that are not in the candidate pool, which must be zero after node-id
  canonicalisation.

Writes ``artifacts/full_suite_probe_summary.json`` next to the probe directory.

Usage
-----
    python scripts/summarise_full_suite_probe.py [PROBE_DIR]
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_full_suite import (  # noqa: E402
    BASELINE,
    MUTANT_SUFFIX_RE,
    WORKSPACE,
    canonical_nodeid,
    load_coverage,
    load_old_killers,
    load_pool,
    load_verdicts,
)


def main() -> None:
    probe_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else WORKSPACE / "artifacts" / "probe_full_suite"
    verdicts = load_verdicts()
    coverage = load_coverage()
    pool = load_pool()
    old = load_old_killers()

    records = []
    for path in sorted(probe_dir.glob("*.json")):
        data = json.loads(path.read_text())
        if data.get("mutant") == BASELINE:
            continue
        records.append(data)

    zero_outcome = [d for d in records if not d["records"]]
    fault_bearing = [d for d in records if d["failures"]]

    # Gate 5: outcome count.
    wrong_counts = [d for d in records if d["records"] and d["records"] != 1190]

    # Gate 2: old killing set is a subset of the new one.
    subset_ok = 0
    subset_violations = []
    for d in records:
        o = {canonical_nodeid(t) for t in old.get(d["mutant"], set())}
        if o <= set(d["failures"]):
            subset_ok += 1
        else:
            subset_violations.append(
                {"mutant": d["mutant"], "missing": sorted(o - set(d["failures"]))[:5]}
            )

    killers = [len(d["failures"]) for d in fault_bearing]
    killers_sorted = sorted(killers)

    def pct(q: float) -> float:
        if not killers_sorted:
            return float("nan")
        i = min(int(round(q * (len(killers_sorted) - 1))), len(killers_sorted) - 1)
        return float(killers_sorted[i])

    # Out-of-coverage analysis, split by the old verdict.
    def analyse(rows: list[dict]) -> dict:
        out_of_cov = []
        off_pool = 0
        off_pool_mutants = 0
        for d in rows:
            key = MUTANT_SUFFIX_RE.sub("", d["mutant"])
            covering = set(coverage.get(key, []))
            failures = set(d["failures"])
            extra = failures - pool
            off_pool += len(extra)
            off_pool_mutants += bool(extra)
            if not failures <= covering:
                out_of_cov.append(
                    {
                        "mutant": d["mutant"],
                        "covering": len(covering),
                        "killers": len(failures),
                        "outside": sorted(failures - covering),
                    }
                )
        return {
            "fault_bearing": len(rows),
            "with_out_of_coverage_killer": len(out_of_cov),
            "share_out_of_coverage": (len(out_of_cov) / len(rows)) if rows else float("nan"),
            "killers_not_in_pool": off_pool,
            "mutants_with_off_pool_killers": off_pool_mutants,
            "examples": out_of_cov[:10],
        }

    killed_rows = [d for d in fault_bearing if verdicts.get(d["mutant"]) in (1, 3)]
    survivor_rows = [d for d in fault_bearing if verdicts.get(d["mutant"]) == 0]
    survivors_total = sum(1 for m, e in verdicts.items() if e == 0)
    survivors_sampled = sum(1 for m in (d["mutant"] for d in records) if verdicts.get(m) == 0)

    why = Counter()
    for d in fault_bearing:
        key = MUTANT_SUFFIX_RE.sub("", d["mutant"])
        covering = set(coverage.get(key, []))
        for t in set(d["failures"]) - covering:
            why[t.split("::")[-1]] += 1

    summary = {
        "probe_dir": str(probe_dir),
        "mutants_summarised": len(records),
        "population": len(verdicts),
        "zero_outcome": len(zero_outcome),
        "zero_outcome_mutants": [d["mutant"] for d in zero_outcome[:20]],
        "wrong_outcome_count": len(wrong_counts),
        "gate2_subset_ok": subset_ok,
        "gate2_violations": subset_violations[:20],
        "fault_bearing": len(fault_bearing),
        "killers": {
            "median": float(statistics.median(killers)) if killers else float("nan"),
            "mean": float(statistics.mean(killers)) if killers else float("nan"),
            "max": max(killers) if killers else 0,
            "q25": pct(0.25),
            "q75": pct(0.75),
            "q95": pct(0.95),
            "single_killer_faults": sum(1 for k in killers if k == 1),
            "total_killers": sum(killers),
        },
        "killed": analyse(killed_rows),
        "survivors": {
            "population": survivors_total,
            "sampled": survivors_sampled,
            "converted_to_fault": len(survivor_rows),
            **analyse(survivor_rows),
        },
        "top_out_of_coverage_killers": [
            {"test": t, "occurrences": n} for t, n in why.most_common(10)
        ],
    }

    out_path = probe_dir.parent / "full_suite_probe_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    # Compact label artifact: one canonical test list plus, per mutant, the indices of the
    # tests that FAILED. Everything else ran and passed, which is the whole point of the
    # full-suite design, so the complement is implicit. Storing indices rather than node ids
    # keeps this ~1 MB instead of ~9 MB.
    all_ids = sorted(
        {t for d in records for t in d["failures"]}
        | {t for v in coverage.values() for t in v}
        | set(pool)
    )
    idx_of = {t: i for i, t in enumerate(all_ids)}
    labels_path = probe_dir.parent / "full_suite_labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "test_ids": all_ids,
                "n_tests": len(all_ids),
                "outcomes": "all tests ran and passed unless listed in failures",
                "mutants": {
                    d["mutant"]: {
                        "rc": d["rc"],
                        "n_ran": d["records"],
                        "failures": [idx_of[t] for t in d["failures"]],
                    }
                    for d in records
                },
            }
        )
    )
    print(f"wrote {labels_path} ({labels_path.stat().st_size / 1e6:.1f} MB)")

    print(f"mutants summarised        : {summary['mutants_summarised']} of {summary['population']}")
    print(f"zero-outcome mutants      : {summary['zero_outcome']}")
    print(f"mutants w/ != 1190 records: {summary['wrong_outcome_count']}")
    print(f"gate 2 old subset of new  : {summary['gate2_subset_ok']}/{len(records)}")
    for v in summary["gate2_violations"][:5]:
        print(f"    VIOLATION {v['mutant']} missing={v['missing']}")
    k = summary["killers"]
    print(f"fault-bearing             : {summary['fault_bearing']}")
    print(f"killers median/q25/q75/max: {k['median']:.0f} / {k['q25']:.0f} / {k['q75']:.0f} / {k['max']}")
    print(f"killers mean / total      : {k['mean']:.1f} / {k['total_killers']}")
    print(f"single-killer faults      : {k['single_killer_faults']}")
    print(
        f"killed: out-of-coverage   : {summary['killed']['with_out_of_coverage_killer']}"
        f"/{summary['killed']['fault_bearing']} "
        f"({summary['killed']['share_out_of_coverage']:.1%})"
    )
    s = summary["survivors"]
    print(
        f"survivors: {s['sampled']} sampled of {s['population']}, "
        f"converted to fault: {s['converted_to_fault']}"
    )
    print(f"killers not in pool       : {summary['killed']['killers_not_in_pool']} (killed) "
          f"+ {s['killers_not_in_pool']} (survivors)")
    print("top out-of-coverage killers:")
    for row in summary["top_out_of_coverage_killers"][:5]:
        print(f"    {row['occurrences']:3d}  {row['test']}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
