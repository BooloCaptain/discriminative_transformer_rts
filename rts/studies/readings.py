"""Readings: quantities computed from a report rather than recorded by the kernel.

A sweep produces data; a *reading* is an interpretation of it. Keeping the two apart is what
lets a table be recomputed from the layer's own records instead of being written down beside
them. ``semif_margins`` is the ladder's headline quantity and the only reader here.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from ..experiment import FACTOR_FEATURES, FACTOR_MODEL, RunReport

# --- reading a report back --------------------------------------------------


def semif_margins(
    report: RunReport,
    *,
    subset: str = "cache_covered",
    reference: str = "semif_reranker",
    exclude: Sequence[str] = ("random",),
    ndigits: int = 4,
) -> dict:
    """SemIf's margin over the best classical ranker, per rung and per budget.

    The ladder's headline quantity, computed from the report rather than recorded by the
    kernel: it is a *reading* of the sweep, and the layer's job was to make the sweep data. It
    rounds as the recorded artifact does, so a tie in the argmax breaks the same way.
    """
    tables: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for result in report.design_points:
        if result.subset != subset:
            continue
        rung = result.design_point.name(FACTOR_FEATURES)
        tables[rung][result.design_point.name(FACTOR_MODEL)] = {
            f"{r['budget']:.2f}": round(r["recall"], ndigits) for r in result.results
        }

    out: dict[str, dict] = {}
    for rung, table in tables.items():
        if reference not in table:
            out[rung] = {
                "undefined": f"{reference} has no measured design_point in subset {subset!r}"
            }
            continue
        margins: dict[str, dict] = {}
        for key in sorted(table[reference]):
            classical = [n for n in table if n != reference and n not in exclude]
            best = max(classical, key=lambda n: table[n][key])
            margins[key] = {
                "best_classical": best,
                "best_recall": table[best][key],
                "semif_recall": table[reference][key],
                "semif_margin": round(table[reference][key] - table[best][key], ndigits),
            }
        out[rung] = margins
    return out
