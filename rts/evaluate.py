"""Evaluation: the contract's other half, and the metric sweep over it.

The evaluation contract needs from a dataset only a quadruple::

    (scores, labels, candidate sets, rows) -> recall / hits

A dataset provides ``labels``, ``candidate_sets`` and its ``rows``; a ranker provides
``scores``. Evaluation therefore never depends on the dataset *type*, which is what lets
a wrapped dataset, a pooled dataset, or a bundle carrying its own label matrix be
evaluated by exactly the same code.

Design notes
------------
* **Per-change budget.** For every change, ``k = ceil(budget * n_candidates)`` tests are
  selected from that change's own suite. A single global budget would let a
  constant-selection strategy score well; a per-change budget does not.
* **Tie-breaking is deterministic.** Tests are ordered by score descending, with the test's
  index (i.e. its node id, sorted) breaking ties. This matters: with 1187 candidate sets and
  coarse integer-ish features, ties are common.
* **A fault is caught** if any killing test is selected for that change.
* **The averaging subset is named**, and it declares the inputs it reads rather than
  asserting a requirement set. Recall over all changes and recall over fault-bearing
  changes are different quantities, so the subset is part of the result rather than an
  implicit choice buried inside it.
* **Cumulative trick.** One descending argsort per change yields every budget at once via a
  cumulative max (hit at k) and cumulative sum (precision at k).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from . import config
from .data import accessors, contract, splits, subsets


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


@dataclass
class Evaluation:
    """One metric sweep, with the subset it was averaged over and any caveat.

    ``results`` is ``None`` exactly when ``undefined`` is set: a subset that cannot
    exist on this dataset has no rows, which is a different statement from an average over
    no rows. Keeping them apart is the point -- reporting the second as the first is a
    false statement about the data, not a rounding error.
    """

    subset: str
    results: list[BudgetResult] | None
    undefined: contract.Undefined | None = None
    n_rows: int = 0
    n_changes: int = 0
    diagnostics: tuple[contract.Diagnostic, ...] = ()

    @property
    def measured(self) -> bool:
        return self.undefined is None

    def to_dict(self) -> dict:
        if not self.measured:
            return {
                "subset": self.subset,
                "measured": False,
                **self.undefined.to_dict(),
            }
        return {
            "subset": self.subset,
            "measured": True,
            "n_rows": self.n_rows,
            "n_changes": self.n_changes,
            "diagnostics": [w.to_dict() for w in self.diagnostics],
            "results": results_to_dicts(self.results or []),
        }


class UndefinedSubset(RuntimeError):
    """Raised when a measurement is requested from a subset that cannot exist."""

    def __init__(self, evaluation: Evaluation):
        super().__init__(evaluation.undefined.note if evaluation.undefined else "undefined")
        self.evaluation = evaluation


def budget_k(budget: float, n_candidates: int) -> int:
    """Selected tests for a budget, against that change's own candidate count."""
    return max(1, math.ceil(budget * n_candidates))


def _bootstrap_ci(
    values: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> tuple[float, float]:
    if values.size == 0 or n_bootstrap <= 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


# --- the sweep, over matrices ----------------------------------------------


def curve_from_arrays(
    scores: np.ndarray,
    labels: np.ndarray,
    candidate_sets: np.ndarray | None,
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``(cum_hit, cum_count)`` for ``rows``, from a label matrix.

    The metric sweep's whole operation, expressed over matrices rather than over a
    dataset, so a bundle condition carrying its own label matrix is evaluated by exactly the same
    code as a dataset -- which is what stops a second implementation drifting from the
    first. Non-candidate sets are pushed to the end, so the first ``k`` entries of each row are
    always the selected set, and ties break by column index.
    """
    masked = (
        scores[rows] if candidate_sets is None else np.where(candidate_sets[rows], scores[rows], -np.inf)
    )
    order = np.argsort(-masked, axis=1, kind="stable")
    ordered = labels[rows][np.arange(len(rows))[:, None], order]
    return np.maximum.accumulate(ordered, axis=1), np.cumsum(ordered, axis=1)


def sweep_matrices(
    scores: np.ndarray,
    labels: np.ndarray,
    candidate_sets: np.ndarray | None,
    rows: np.ndarray,
    n_tests: int,
    budgets: tuple[float, ...] = config.DEFAULT_BUDGETS,
    n_bootstrap: int = config.DEFAULT_BOOTSTRAP,
    seed: int = config.SEED,
    avg_rows: np.ndarray | None = None,
) -> list[BudgetResult]:
    """Budget sweep over a label matrix.

    ``avg_rows`` are indices *into* ``rows`` and select the averaging subset. Left
    unset, every row with at least one label is averaged -- the fault-bearing default.
    """
    rng = np.random.default_rng(seed)
    cum_hit, cum_count = curve_from_arrays(scores, labels, candidate_sets, rows)
    cand_counts = (
        candidate_sets[rows].sum(axis=1).astype(np.int64)
        if candidate_sets is not None
        else np.full(len(rows), n_tests, dtype=np.int64)
    )
    if avg_rows is None:
        avg_rows = np.flatnonzero(labels[rows].sum(axis=1) > 0).astype(np.int64)

    results: list[BudgetResult] = []
    for budget in budgets:
        k_per_change = np.array(
            [budget_k(budget, int(c)) for c in cand_counts], dtype=np.int64
        )
        idx = k_per_change[avg_rows] - 1
        hits = cum_hit[avg_rows, idx].astype(np.float64)
        precision_per_change = cum_count[avg_rows, idx] / k_per_change[avg_rows]

        recall = float(hits.mean()) if hits.size else float("nan")
        precision = float(precision_per_change.mean()) if hits.size else float("nan")
        f_measure = (
            2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
        )
        lo, hi = _bootstrap_ci(hits, n_bootstrap, rng)
        mean_k = float(k_per_change[avg_rows].mean()) if hits.size else float("nan")

        results.append(
            BudgetResult(
                budget=budget,
                k=int(round(mean_k)),
                recall=recall,
                precision=precision,
                f_measure=f_measure,
                suite_reduction=1.0 - mean_k / n_tests,
                recall_lo=lo,
                recall_hi=hi,
                n_faults=int(hits.size),
            )
        )
    return results


def per_change_hit_matrix(
    scores: np.ndarray,
    labels: np.ndarray,
    candidate_sets: np.ndarray | None,
    rows: np.ndarray,
    budget: float,
) -> np.ndarray:
    """Per-row caught/not at one budget, from a label matrix."""
    cum_hit, _ = curve_from_arrays(scores, labels, candidate_sets, rows)
    cand_counts = (
        candidate_sets[rows].sum(axis=1).astype(np.int64)
        if candidate_sets is not None
        else np.full(len(rows), labels.shape[1], dtype=np.int64)
    )
    k = np.array([budget_k(budget, int(c)) for c in cand_counts], dtype=np.int64)
    return cum_hit[np.arange(len(rows)), k - 1].astype(np.float64)


# --- the sweep, over a dataset ---------------------------------------------


def subset_rows(
    ds: contract.Dataset,
    eval_idx: np.ndarray,
    subset: subsets.Subset | str,
) -> tuple[subsets.Subset, np.ndarray | contract.Undefined]:
    """Row indices into ``eval_idx`` to average over, or why they cannot exist."""
    spec = subsets.resolve(subset)
    mask = spec.mask(ds)
    if isinstance(mask, contract.Undefined):
        return spec, mask
    # A subset restricts *which rows the metric is averaged over*; it does not add rows
    # outside the evaluation window. A change with no killing test still has no recall to
    # average, so it is excluded either way.
    faults = accessors.fault_mask(ds)
    rows = np.array(
        [r for r, i in enumerate(eval_idx) if mask[i] and faults[i]], dtype=np.int64
    )
    return spec, rows


def evaluate_rows(
    scores: np.ndarray,
    ds: contract.Dataset,
    eval_idx: np.ndarray,
    budgets: tuple[float, ...] = config.DEFAULT_BUDGETS,
    n_bootstrap: int = config.DEFAULT_BOOTSTRAP,
    seed: int = config.SEED,
    candidate_sets: np.ndarray | None = None,
    subset: subsets.Subset | str = "detectable",
    split: splits.Split | None = None,
) -> Evaluation:
    """Metric sweep for one ranker, over a named averaging subset.

    When ``candidate_sets`` is given, the budget is a fraction of each change's own candidate
    set, so the selected count varies per change.

    ``split``, when given, is the evaluation boundary: ``eval_idx`` must lie inside its window,
    because a metric may only be averaged over rows the split held out for evaluation. The
    layer always satisfies this, since it evaluates exactly ``split.test_idx``; the guard
    matters for a caller that passes rows *directly*, which is what evaluating inside the
    held-out tail does. A zero-shot condition over every change has no split and leaves it out.
    """
    if split is not None:
        splits.require_in_window(split, eval_idx, what="evaluation rows")
    spec, rows = subset_rows(ds, eval_idx, subset)
    if isinstance(rows, contract.Undefined):
        return Evaluation(subset=spec.name, results=None, undefined=rows)
    return Evaluation(
        subset=spec.name,
        results=sweep_matrices(
            scores,
            accessors.labels(ds),
            candidate_sets,
            eval_idx,
            ds.n_tests,
            budgets,
            n_bootstrap,
            seed,
            avg_rows=rows,
        ),
        n_rows=int(len(rows)),
        n_changes=int(len(eval_idx)),
    )


def evaluate(
    scores: np.ndarray,
    ds: contract.Dataset,
    eval_idx: np.ndarray,
    budgets: tuple[float, ...] = config.DEFAULT_BUDGETS,
    n_bootstrap: int = config.DEFAULT_BOOTSTRAP,
    seed: int = config.SEED,
    candidate_sets: np.ndarray | None = None,
    subset: subsets.Subset | str = "detectable",
    split: splits.Split | None = None,
) -> list[BudgetResult]:
    """Metric sweep returning only the results list, for the many call sites that want it.

    Use :func:`evaluate_rows` when the subset name and the measured/undefined
    distinction need to travel with the numbers, or when the evaluation boundary should be
    enforced (see its ``split`` parameter).
    """
    evaluation = evaluate_rows(
        scores, ds, eval_idx, budgets, n_bootstrap, seed, candidate_sets, subset, split
    )
    if not evaluation.measured:
        raise UndefinedSubset(evaluation)
    return evaluation.results or []


def per_change_hits(
    scores: np.ndarray,
    ds: contract.Dataset,
    eval_idx: np.ndarray,
    budget: float,
    candidate_sets: np.ndarray | None = None,
) -> dict[int, bool]:
    """Map change index -> caught, at one budget. Used for paired contrasts."""
    return {
        int(eval_idx[r]): bool(value)
        for r, value in enumerate(
            per_change_hit_matrix(scores, accessors.labels(ds), candidate_sets, eval_idx, budget)
        )
        if accessors.fault_mask(ds)[int(eval_idx[r])]
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


def subset_report(evaluations: dict[str, Evaluation]) -> dict:
    """Tabulate subsets by whether they could be measured, for an artifact payload."""
    return {name: ev.to_dict() for name, ev in evaluations.items()}


__all__ = [
    "BudgetResult",
    "Evaluation",
    "UndefinedSubset",
    "budget_k",
    "curve_from_arrays",
    "evaluate",
    "evaluate_rows",
    "format_table",
    "paired_bootstrap",
    "per_change_hit_matrix",
    "per_change_hits",
    "subset_report",
    "subset_rows",
    "results_to_dicts",
    "sweep_matrices",
]
