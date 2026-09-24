"""Evaluation: per-change budget, metric sweep, and paired bootstrap CIs.

Design notes
------------
* **Per-change budget.** For every change, ``k = ceil(budget * n_tests)`` tests are
  selected from that change's own suite. A single global budget would let a
  constant-selection strategy score well; a per-change budget does not.
* **Tie-breaking is deterministic.** Tests are ordered by score descending, with
  the test's index (i.e. its node id, sorted) breaking ties. This matters: with
  1187 candidates and coarse integer-ish features, ties are common.
* **A fault is caught** if any killing test is selected for that change.
* **Metrics are computed over fault-bearing changes only.** Survived mutants have
  no failing test, so recall and precision are undefined for them. They stay in
  the dataset to give the models negative signal during training.
* **Cumulative trick.** One descending argsort per change yields every budget at
  once via a cumulative max (hit at k) and cumulative sum (precision at k).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

from . import config, dataset


@dataclass
class BudgetResult:
    budget: float
    k: int
    recall: float
    precision: float
    f_measure: float
    suite_reduction: float
    recall_lo: float
    recall_hi: float
    n_faults: int


def _top_k_order(scores: np.ndarray, candidates: np.ndarray | None = None) -> np.ndarray:
    """Descending order of test indices per change, ties broken by index.

    Non-candidates are pushed to the end so that the first ``k`` entries of each
    row are always the selected set.
    """
    if candidates is not None:
        scores = np.where(candidates, scores, -np.inf)
    return np.argsort(-scores, axis=1, kind="stable")


def _curve(
    scores: np.ndarray,
    ds: dataset.Dataset,
    eval_idx: np.ndarray,
    candidates: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (labels_ordered, cum_hit, cum_count) for the evaluated changes.

    ``cum_hit[:, k-1]`` is 1 when a killing test appears in the top k.
    ``cum_count[:, k-1]`` is the number of killing tests in the top k.
    """
    order = _top_k_order(
        scores[eval_idx], None if candidates is None else candidates[eval_idx]
    )
    labels = ds.labels[eval_idx][np.arange(len(eval_idx))[:, None], order]
    cum_hit = np.maximum.accumulate(labels, axis=1)
    cum_count = np.cumsum(labels, axis=1)
    return labels, cum_hit, cum_count


def budget_k(budget: float, n_candidates: int) -> int:
    """Selected tests for a budget, against that change's own candidate count."""
    return max(1, math.ceil(budget * n_candidates))


def _bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if values.size == 0 or n_bootstrap <= 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate(
    scores: np.ndarray,
    ds: dataset.Dataset,
    eval_idx: np.ndarray,
    budgets: tuple[float, ...] = config.DEFAULT_BUDGETS,
    n_bootstrap: int = config.DEFAULT_BOOTSTRAP,
    seed: int = config.SEED,
    candidates: np.ndarray | None = None,
) -> list[BudgetResult]:
    """Metric sweep for one selector.

    When ``candidates`` is given, the budget is a fraction of each change's own
    candidate set, so the selected count varies per change.
    """
    rng = np.random.default_rng(seed)
    _, cum_hit, cum_count = _curve(scores, ds, eval_idx, candidates)
    cand_counts = (
        ds.candidate_counts(candidates)[eval_idx]
        if candidates is not None
        else np.full(len(eval_idx), ds.n_tests, dtype=np.int64)
    )

    # Restrict to changes that have a fault, within the evaluated window.
    held = set(eval_idx.tolist())
    fault_rows = np.array(
        [r for r, i in enumerate(eval_idx) if i in held and ds.changes[i].killing_tests],
        dtype=np.int64,
    )

    results: list[BudgetResult] = []
    for budget in budgets:
        k_per_change = np.array([budget_k(budget, int(c)) for c in cand_counts], dtype=np.int64)
        idx = k_per_change[fault_rows] - 1
        hits = cum_hit[fault_rows, idx].astype(np.float64)
        precision_per_change = cum_count[fault_rows, idx] / k_per_change[fault_rows]

        recall = float(hits.mean()) if hits.size else float("nan")
        precision = float(precision_per_change.mean()) if hits.size else float("nan")
        f_measure = (
            2 * recall * precision / (recall + precision)
            if (recall + precision) > 0
            else 0.0
        )
        lo, hi = _bootstrap_ci(hits, n_bootstrap, rng)
        mean_k = float(k_per_change[fault_rows].mean()) if hits.size else float("nan")

        results.append(
            BudgetResult(
                budget=budget,
                k=int(round(mean_k)),
                recall=recall,
                precision=precision,
                f_measure=f_measure,
                suite_reduction=1.0 - mean_k / ds.n_tests,
                recall_lo=lo,
                recall_hi=hi,
                n_faults=int(hits.size),
            )
        )
    return results


def per_change_hits(
    scores: np.ndarray,
    ds: dataset.Dataset,
    eval_idx: np.ndarray,
    budget: float,
    candidates: np.ndarray | None = None,
) -> dict[int, bool]:
    """Map change index -> caught, at one budget. Used for paired comparisons."""
    _, cum_hit, _ = _curve(scores, ds, eval_idx, candidates)
    cand_counts = (
        ds.candidate_counts(candidates)[eval_idx]
        if candidates is not None
        else np.full(len(eval_idx), ds.n_tests, dtype=np.int64)
    )
    held = set(eval_idx.tolist())
    fault_rows = [r for r, i in enumerate(eval_idx) if i in held and ds.changes[i].killing_tests]
    return {
        int(eval_idx[r]): bool(cum_hit[r, budget_k(budget, int(cand_counts[r])) - 1])
        for r in fault_rows
    }


def paired_bootstrap(
    hits_a: dict[int, bool],
    hits_b: dict[int, bool],
    n_bootstrap: int = config.DEFAULT_BOOTSTRAP,
    seed: int = config.SEED,
) -> dict:
    """Bootstrap the recall difference (a - b) over the shared set of changes."""
    rng = np.random.default_rng(seed)
    shared = sorted(set(hits_a) & set(hits_b))
    if not shared or n_bootstrap <= 0:
        return {
            "delta": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "p_value": float("nan"),
            "n": len(shared),
        }
    a = np.array([hits_a[i] for i in shared], dtype=np.float64)
    b = np.array([hits_b[i] for i in shared], dtype=np.float64)
    delta = a - b
    idx = rng.integers(0, len(shared), size=(n_bootstrap, len(shared)))
    means = delta[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    # Two-sided bootstrap p-value for delta == 0.
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {
        "delta": float(delta.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "p_value": float(min(p, 1.0)),
        "n": len(shared),
    }


def format_table(name: str, results: list[BudgetResult]) -> str:
    lines = [f"\n{name}"]
    lines.append(
        f"  {'budget':>7} {'k':>5} {'recall':>7} {'95% CI':>17} {'prec':>7} {'F1':>7} {'reduct':>7}"
    )
    for r in results:
        ci = f"[{r.recall_lo:.3f}, {r.recall_hi:.3f}]"
        lines.append(
            f"  {r.budget:>7.2f} {r.k:>5} {r.recall:>7.3f} {ci:>17} "
            f"{r.precision:>7.3f} {r.f_measure:>7.3f} {r.suite_reduction:>7.3f}"
        )
    return "\n".join(lines)


def results_to_dicts(results: list[BudgetResult]) -> list[dict]:
    return [asdict(r) for r in results]
