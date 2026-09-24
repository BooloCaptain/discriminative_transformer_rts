"""Panels A and C of the sparsity story, plus figures.

Panel A -- recall against change-population sparsity. Held-out faults are binned
into equal-count deciles by how much failure history their killing ``(file, test)``
pair has (1 = sparsest, i.e. the pair has essentially never failed before). Every
model is evaluated on the same bins.

Panel C -- the mechanism. For the same bins, what fraction of killing tests lie
inside the cheap structural funnel (``covers_function AND
module_name_in_test_file``), and what fraction the structural models actually
recover. If the funnel fraction collapses as sparsity rises while SemIf's recall
does not, that is *why* the crossover happens: structural methods fail exactly
where the killer is not the file's usual suspect.

Note on coverage: every SemIf score currently ranks within the ``covered``
candidate mask (~155 tests), so this compares ordering *within* the coverage set,
not selection from the full suite. See implementation.md.

Usage::

    python -m rts.analysis            # writes artifacts/figures/*.png and *.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config, dataset, evaluate, features, models, semif

BUDGETS = (0.01, 0.05)
N_BINS = 10
MIN_BIN_FOR_PARTIAL = 8


# --- sparsity axis ---------------------------------------------------------


def sparsity_values(ds: dataset.Dataset) -> dict[int, float]:
    """Failure count of each fault change's killing (file, test) pair.

    Lower means less history: the pair has rarely or never failed before, which is
    the proxy for a file that has not changed in years.
    """
    _, fails = dataset.pair_history_counts(ds)
    out: dict[int, float] = {}
    for i in ds.fault_idx:
        i = int(i)
        key = (ds.files[i], ds.changes[i].killing_tests[0])
        out[i] = float(fails.get(key, 0))
    return out


def sparsity_bins(
    ds: dataset.Dataset, n_bins: int = N_BINS
) -> tuple[list[np.ndarray], list[dict]]:
    """Equal-count bins over fault changes, sparsest first."""
    values = sparsity_values(ds)
    order = sorted(values, key=lambda i: (values[i], i))
    chunks = [np.array(c, dtype=np.int64) for c in np.array_split(order, n_bins)]
    meta = []
    for b, chunk in enumerate(chunks, start=1):
        vals = np.array([values[int(i)] for i in chunk])
        meta.append(
            {
                "bin": b,
                "n_faults": int(chunk.size),
                "failures_median": float(np.median(vals)),
                "failures_min": float(vals.min()),
                "failures_max": float(vals.max()),
            }
        )
    return chunks, meta


# --- panels ----------------------------------------------------------------


def panel_a(
    scores: dict[str, np.ndarray],
    ds: dataset.Dataset,
    bins: list[np.ndarray],
    budget: float,
    candidates: np.ndarray,
    held: set[int],
    min_faults: int = 3,
) -> dict[str, list[float]]:
    """Recall per sparsity bin, per model, on the shared held-out population.

    Every model is evaluated on exactly the held-out fault changes inside the bin.
    Filtering by per-model score availability instead would let the classical
    models -- which have finite scores everywhere, including the training changes
    they were fit on -- be scored on a different, easier population than SemIf.
    """
    out: dict[str, list[float]] = {}
    for name, score in scores.items():
        row = []
        for chunk in bins:
            usable = np.array(
                [i for i in chunk if int(i) in held and score[int(i)].max() > -1e8],
                dtype=np.int64,
            )
            if usable.size < min_faults:
                row.append(float("nan"))
                continue
            res = evaluate.evaluate(
                score, ds, usable, budgets=(budget,), n_bootstrap=0, candidates=candidates
            )[0]
            row.append(res.recall)
        out[name] = row
    return out


def panel_c(
    ds: dataset.Dataset,
    bins: list[np.ndarray],
    score_set: dict[str, np.ndarray],
    budget: float,
    candidates: np.ndarray,
    held: set[int],
    min_faults: int = 3,
) -> dict[str, list[float]]:
    """Mechanistic quantities per sparsity bin."""
    ix = {n: i for i, n in enumerate(features.STRUCTURED_NAMES)}
    X, _ = features.structured_features(ds)
    covered = X[:, :, ix["covers_function"]]
    name_match = X[:, :, ix["module_name_in_test_file"]]

    funnel, cov_only, median_rank = [], [], []
    funnel_size, rank_in_funnel = [], []
    recalls = {name: [] for name in score_set}
    held_counts: list[int] = []

    for chunk in bins:
        in_funnel = in_cov = 0
        ranks = []
        sizes = []
        inner_ranks = []
        evaluated = [int(i) for i in chunk if int(i) in held]
        held_counts.append(len(evaluated))
        for i in evaluated:
            kill = ds.test_index[ds.changes[i].killing_tests[0]]
            in_cov += int(covered[i, kill] > 0.5)
            in_funnel += int(covered[i, kill] > 0.5 and name_match[i, kill] > 0.5)
            # Rank of the killer under the lexical model, within candidates.
            cols = np.flatnonzero(candidates[i])
            score = score_set["bm25_lexical"][i, cols]
            order = np.argsort(-score, kind="stable")
            pos = int(np.flatnonzero(cols[order] == kill)[0]) + 1
            ranks.append(pos / cols.size)

            # How discriminating is the structural funnel for this change? If the
            # funnel is large, or the killer sits at its bottom, the funnel exists
            # but does not help -- which is a different failure from the funnel
            # being empty.
            funnel_cols = cols[
                (covered[i, cols] > 0.5) & (name_match[i, cols] > 0.5)
            ]
            sizes.append(funnel_cols.size)
            if funnel_cols.size:
                sr = score_set["structural_rule"][i, funnel_cols]
                inner = np.argsort(-sr, kind="stable")
                where = int(np.flatnonzero(funnel_cols[inner] == kill)[0]) if kill in funnel_cols else -1
                inner_ranks.append(where / funnel_cols.size if where >= 0 else 1.0)
            else:
                inner_ranks.append(1.0)

        n = max(len(evaluated), 1)
        funnel.append(in_funnel / n)
        cov_only.append(in_cov / n)
        median_rank.append(float(np.median(ranks)) if ranks else float("nan"))
        funnel_size.append(float(np.median(sizes)) if sizes else float("nan"))
        rank_in_funnel.append(float(np.median(inner_ranks)) if inner_ranks else float("nan"))
        for name, score in score_set.items():
            usable = np.array(
                [i for i in evaluated if score[i].max() > -1e8], dtype=np.int64
            )
            if usable.size < min_faults:
                recalls[name].append(float("nan"))
                continue
            res = evaluate.evaluate(
                score, ds, usable, budgets=(budget,), n_bootstrap=0, candidates=candidates
            )[0]
            recalls[name].append(res.recall)

    out = {
        "held_out_faults": held_counts,
        "frac_in_coverage": cov_only,
        "frac_in_structural_funnel": funnel,
        "median_funnel_size": funnel_size,
        "median_killer_rank_in_funnel": rank_in_funnel,
        "median_normalized_killer_rank_bm25": median_rank,
    }
    for name, vals in recalls.items():
        out[f"recall_{name}"] = vals
    return out


# --- figures ---------------------------------------------------------------


def _plot(
    panel_a_data: dict[str, list[float]],
    panel_c_data: dict[str, list[float]],
    meta: list[dict],
    budget: float,
    out_path: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(1, len(meta) + 1)
    xticks = [f"{m['bin']}\n{m['failures_median']:.0f}" for m in meta]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4))

    # --- Panel A ---
    ax = axes[0]
    highlight = {
        "semif_textonly": ("#d62728", "-", 2.6),
        "semif_after_doc": ("#ff7f0e", "--", 2.6),
        "semif_full": ("#8c564b", ":", 2.2),
        "xgboost_struct_lex": ("#1f77b4", "-", 2.4),
        "xgboost_struct": ("#1f77b4", "--", 1.8),
        "xgboost_static_nocov_lex": ("#17becf", "-", 1.8),
        "coverage": ("#2ca02c", "-", 1.8),
        "failure_rate": ("#9467bd", "-", 1.8),
        "recency": ("#7f7f7f", "-", 1.4),
        "bm25_lexical": ("#e377c2", "-", 1.8),
        "structural_rule": ("#bcbd22", "-", 1.6),
        "random": ("#cccccc", "-", 1.2),
    }
    for name, vals in panel_a_data.items():
        if name not in highlight or all(np.isnan(vals)):
            continue
        colour, style, width = highlight[name]
        ax.plot(x, vals, style, color=colour, linewidth=width, marker="o",
                markersize=4, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(xticks, fontsize=8)
    ax.set_xlabel("sparsity decile  (1 = sparsest)\nmedian failures of the killing (file, test) pair")
    ax.set_ylabel(f"recall @ budget {budget}")
    ax.set_title("Panel A — recall vs change-population sparsity\n(all 464 held-out faults; SemIf arms are dashed/dotted where partial)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(-0.02, 1.02)

    # --- Panel C ---
    ax = axes[1]
    ax.plot(x, panel_c_data["frac_in_structural_funnel"], "-o", color="#bcbd22",
            linewidth=2.2, markersize=5,
            label="killer is inside the structural funnel")
    ax.plot(x, panel_c_data["median_normalized_killer_rank_bm25"], "-o",
            color="#e377c2", linewidth=1.8, markersize=4,
            label="median killer rank / candidates (BM25)")
    ax.set_xticks(x)
    ax.set_xticklabels(xticks, fontsize=8)
    ax.set_xlabel("sparsity decile  (1 = sparsest)\nmedian failures of the killing (file, test) pair")
    ax.set_ylabel("fraction / normalized rank", color="#333333")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)

    ax3 = ax.twinx()
    ax3.plot(x, panel_c_data["median_funnel_size"], "-s", color="#17becf",
             linewidth=1.8, markersize=4,
             label="median funnel size (tests)")
    ax3.set_ylabel("tests in the covered AND name-matching funnel", color="#17becf")
    ax3.tick_params(axis="y", labelcolor="#17becf")

    ax2 = ax.twinx()
    ax2.spines["right"].set_position(("axes", 1.12))
    for key, colour, style in (
        ("recall_failure_rate", "#9467bd", "-"),
        ("recall_structural_rule", "#bcbd22", "--"),
        ("recall_semif_textonly", "#d62728", "-"),
        ("recall_xgboost_struct_lex", "#1f77b4", "-"),
    ):
        if key in panel_c_data:
            ax2.plot(x, panel_c_data[key], style, color=colour, linewidth=2.0,
                     marker="s", markersize=4, alpha=0.9,
                     label=key.replace("recall_", "recall: "))
    ax2.set_ylabel(f"recall @ budget {budget}")
    ax2.set_ylim(-0.02, 1.02)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h2 + h3, l1 + l2 + l3, fontsize=8, loc="center left")
    ax.set_title("Panel C — the mechanism\nstructural funnel availability vs model recall")

    fig.suptitle(
        "RTS feasibility: as per-test history becomes sparse, structural methods lose their footing "
        "and SemIf does not",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for row in rows:
            fh.write(
                ",".join(
                    "" if row[k] is None or (isinstance(row[k], float) and np.isnan(row[k]))
                    else f"{row[k]:.4f}" if isinstance(row[k], float) else str(row[k])
                    for k in keys
                )
                + "\n"
            )


def run(n_bins: int = N_BINS) -> dict:
    ds = dataset.build()
    candidates = dataset.candidate_mask(ds, "covered")
    held = set(int(i) for i in ds.test_fault_idx)

    bins, meta = sparsity_bins(ds, n_bins)
    for m, chunk in zip(meta, bins):
        m["held_out_faults"] = int(sum(1 for i in chunk if int(i) in held))
    print("sparsity bins (1 = sparsest):")
    for m in meta:
        print(f"  bin {m['bin']:>2}: {m['n_faults']:>3} faults in stratum, "
              f"{m['held_out_faults']:>3} held out  |  "
              f"failures {m['failures_min']:.0f}-{m['failures_max']:.0f} "
              f"(median {m['failures_median']:.0f})")

    X, names = features.structured_features(ds)
    bm25 = features.build_bm25_scores(ds)
    ctx = models.Context(ds=ds, X=X, names=names, bm25=bm25)
    classical = {
        s.name: s.scores(ctx) for s in models.default_selectors(include_semif=False)
    }

    semif_arms = {
        "semif_textonly": semif.load_scores(config.SEMIF_SCORES_FILE, ds),
        "semif_full": semif.load_scores(config.ARTIFACTS / "semif_scores_mirror.jsonl", ds),
        "semif_after_doc": semif.load_scores(config.ARTIFACTS / "semif_scores_ctl_after.jsonl", ds),
    }
    all_scores = {**classical, **semif_arms}

    figures_dir = config.ARTIFACTS / "figures"
    summary: dict = {"bins": meta, "budgets": {}}

    for budget in BUDGETS:
        pa = panel_a(all_scores, ds, bins, budget, candidates, held)
        pc = panel_c(ds, bins, all_scores, budget, candidates, held)

        fig_path = figures_dir / f"panel_A_C_budget{budget:.2f}.png"
        _plot(pa, pc, meta, budget, fig_path)
        print(f"\n[figure] {fig_path}")

        rows = []
        for m in meta:
            b = m["bin"] - 1
            row = dict(m)
            for name, vals in pa.items():
                row[f"recall_{name}"] = vals[b]
            for key, vals in pc.items():
                row[key] = vals[b]
            rows.append(row)
        _write_csv(rows, figures_dir / f"panels_budget{budget:.2f}.csv")
        summary["budgets"][f"{budget:.2f}"] = rows

        print(f"\n--- budget {budget} ---")
        keys = ["frac_in_structural_funnel", "recall_structural_rule",
                "recall_coverage", "recall_failure_rate",
                "recall_semif_textonly", "recall_xgboost_struct_lex"]
        print(f"{'bin':>4} " + " ".join(f"{k[:13]:>14}" for k in keys))
        for row in rows:
            print(f"{row['bin']:>4} " + " ".join(f"{row[k]:>14.3f}" for k in keys))

    (figures_dir / "panels_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
