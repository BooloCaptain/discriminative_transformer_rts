"""Figures for the SemIf variation study and the full-suite starved correction.

Everything is read from ``artifacts/variations.json``, which the experiment driver
writes incrementally, so the figures never re-run an experiment and can never drift
from the numbers in ``implementation.md``. Same conventions as ``rts.analysis``:
matplotlib only, 150 dpi, saved under ``artifacts/figures/``.

Three figures:

``fig_starved_full_suite``  The correction. Recall against budget for the starved
    population on the full 1187-test suite, at both population sizes (n=43 and
    n=141), with SemIf against the classical selectors. This is the figure that
    shows the one positive result failing to survive.
``fig_history_coverage``    The mechanism. The history x coverage decomposition at
    b0.05, which shows coverage is the whole effect and history is harmful.
``fig_variation_levers``    The forest plot. Every paired delta from P1, P2, P3 and
    P5, against the baseline each proposal was designed to beat, with 95% CIs. One
    picture of "all four text-side levers fail".
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config  # noqa: E402

BUDGETS = (0.01, 0.05, 0.1, 0.2)

# Consistent identity per model across every figure.
STYLE = {
    "xgboost_static_lex": ("#1f77b4", "-", "o", 2.0),
    "xgboost_struct_lex": ("#17becf", "-", "s", 1.6),
    "structural_rule": ("#bcbd22", "-", "^", 1.6),
    "coverage": ("#7f7f7f", "-", "v", 1.4),
    "xgboost_static_nocov_lex": ("#8c564b", "--", "D", 1.4),
    "bm25_lexical": ("#9467bd", "--", "x", 1.4),
    "semif_textonly_full": ("#d62728", "-", "*", 3.0),
    "random": ("#bbbbbb", ":", ".", 1.0),
    "failure_rate": ("#e377c2", ":", "1", 1.2),
}
SEMIF_LABEL = "semif_textonly (SemIf, 4B reranker)"


def _load() -> dict:
    path = config.ARTIFACTS / "variations.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run rts.variations first")
    return json.loads(path.read_text())


def _series(group: dict, name: str) -> tuple[list[float], list[float], list[float]]:
    """(budgets, recall, ci half-width) for one model in an eval_group block."""
    rows = group["results"][name]
    by_budget = {round(r["budget"], 4): r for r in rows}
    xs, ys, err = [], [], []
    for b in BUDGETS:
        r = by_budget.get(round(b, 4))
        if r is None:
            continue
        xs.append(b)
        ys.append(r["recall"])
        err.append((r["recall_hi"] - r["recall_lo"]) / 2.0)
    return xs, ys, err


def _draw_curves(ax, group: dict, names: list[str], title: str, subtitle: str) -> None:
    for name in names:
        if name not in group["results"]:
            continue
        colour, style, marker, size = STYLE.get(name, ("#333333", "-", "o", 1.4))
        xs, ys, err = _series(group, name)
        label = SEMIF_LABEL if name == "semif_textonly_full" else name
        ax.errorbar(
            xs, ys, yerr=err, color=colour, linestyle=style, marker=marker,
            markersize=size + 3, linewidth=2.2 if name == "semif_textonly_full" else 1.5,
            capsize=3, label=label, zorder=5 if name == "semif_textonly_full" else 3,
        )
    ax.set_xscale("log")
    ax.set_xticks(list(BUDGETS))
    ax.set_xticklabels([f"{b:g}" for b in BUDGETS])
    ax.set_xlabel("budget (fraction of that change's 1187 candidates)")
    ax.set_ylabel("recall of held-out faults")
    ax.set_ylim(-0.02, 1.04)
    ax.grid(alpha=0.25)
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")


def fig_starved_full_suite(data: dict) -> Path | None:
    """The correction: starved population, full candidate set, both population sizes."""
    panels = []
    if "full_starved" in data:
        panels.append(("full_starved", "n=43 held-out faults (failures <= 2)"))
    if "full_starved5" in data:
        panels.append(("full_starved5", "n=141 held-out faults (failures <= 5)"))
    if not panels:
        return None

    names = [
        "xgboost_static_lex", "xgboost_struct_lex", "structural_rule",
        "xgboost_static_nocov_lex", "bm25_lexical", "semif_textonly_full",
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.8 * len(panels), 6.0), squeeze=False)
    for ax, (key, subtitle) in zip(axes[0], panels):
        _draw_curves(
            ax, data[key], names,
            "Starved arm on the full 1187-test suite", subtitle,
        )
    fig.suptitle(
        "The one positive result does not survive the full candidate set\n"
        "the `covered` mask made `covers_function` constant, removing the tree's best feature",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = config.ARTIFACTS / "figures" / "fig_starved_full_suite.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[figures] {out}")
    return out


def fig_history_coverage(data: dict) -> Path | None:
    """The mechanism: history x coverage decomposition, BM25 always on."""
    group = data.get("full_starved") or data.get("full_starved5")
    if not group:
        return None
    cells = [
        ("xgboost_static_lex", "coverage + BM25\n(no history)", "#1f77b4"),
        ("xgboost_struct_lex", "coverage + history\n+ BM25", "#17becf"),
        ("xgboost_static_nocov_lex", "BM25 only\n(no coverage, no history)", "#8c564b"),
        ("xgboost_struct_nocov_lex", "history + BM25\n(no coverage)", "#e377c2"),
    ]
    labels, values, colours = [], [], []
    for name, label, colour in cells:
        if name not in group["results"]:
            continue
        xs, ys, _ = _series(group, name)
        labels.append(label)
        values.append(ys[1])  # b0.05
        colours.append(colour)
    if not values:
        return None

    semif = _series(group, "semif_textonly_full")[1][1] if "semif_textonly_full" in group["results"] else None
    rule = _series(group, "structural_rule")[1][1] if "structural_rule" in group["results"] else None

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    bars = ax.bar(range(len(values)), values, color=colours, width=0.62)
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    if semif is not None:
        ax.axhline(semif, color="#d62728", linestyle="--", linewidth=2,
                   label=f"SemIf, 4B reranker ({semif:.3f})")
    if rule is not None:
        ax.axhline(rule, color="#bcbd22", linestyle=":", linewidth=2,
                   label=f"three-line structural rule ({rule:.3f})")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("recall @ budget 0.05")
    ax.set_ylim(0, 1.08)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title(
        "Mechanism: coverage is the entire effect and history is harmful\n"
        f"starved population, full 1187-test suite (n={group.get('faults', '?')} faults)",
        fontsize=11,
    )
    fig.tight_layout()
    out = config.ARTIFACTS / "figures" / "fig_history_coverage.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[figures] {out}")
    return out


def fig_variation_levers(data: dict) -> Path | None:
    """Forest plot: every paired delta from P1, P2, P3, P5 at budget 0.05.

    Each row is a paired bootstrap of the intervention against the baseline that
    proposal was designed to beat, so the *baselines differ by row* and are printed
    in the label. Zero means "the intervention changed nothing"; positive would mean
    it helped.
    """
    rows: list[tuple[str, str, float, float, float, float, str]] = []
    # (group label, row label, delta, lo, hi, p, colour)

    def add(group: dict, key: str, glabel: str, rlabel: str, colour: str) -> None:
        stat = group.get("comparisons", {}).get(key)
        if stat is None:
            return
        rows.append((glabel, rlabel, stat["delta"], stat["lo"], stat["hi"],
                     stat["p_value"], colour))

    if "p5" in data:
        for key, label in [
            ("rankaverage_xgb_semif|0.05|vs|xgboost_static_nocov_lex",
             "rank-average with SemIf  (vs tree, no coverage/history)"),
            ("rankaverage_xgb_struct_semif|0.05|vs|xgboost_struct",
             "rank-average with SemIf  (vs full-feature tree)"),
        ]:
            add(data["p5"], key, "P5  add SemIf as a feature", label, "#1f77b4")
    if "p1" in data:
        add(data["p1"], "semif_direct_pairwise|0.05|vs|semif_reranker",
            "P1  direct mode", "direct mode pairwise  (vs reranker)", "#d62728")
    if "p2" in data:
        for variant in ("execution", "fault", "retrieval", "terse"):
            add(data["p2"], f"semif_{variant}|0.05|vs|semif_default",
                "P2  instruction wording", f"'{variant}' wording  (vs default wording)",
                "#2ca02c")
    if "p3" in data:
        for section, label in [
            ("covered", "codebert cosine  (vs BM25, 464 faults)"),
            ("starved_full", "codebert cosine  (vs BM25, starved/full)"),
        ]:
            grp = data["p3"].get(section)
            if grp:
                add(grp, "embed_codebert|0.05|vs|bm25_lexical",
                    "P3  code embeddings", label, "#9467bd")
    if not rows:
        return None

    # Reverse so the first row reads at the top.
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(11.0, 0.52 * len(rows) + 3.2))
    for i, (_, label, delta, lo, hi, p, colour) in enumerate(rows):
        sig = p < 0.05
        ax.plot([lo, hi], [i, i], color=colour, linewidth=2.0,
                alpha=1.0 if sig else 0.45)
        ax.plot([delta], [i], marker="o" if sig else "o", color=colour,
                markersize=7 if sig else 5,
                markerfacecolor=colour if sig else "white", markeredgewidth=1.6)
        ax.text(hi + 0.012, i, f"{delta:+.3f}" + ("  *" if sig else ""),
                va="center", fontsize=8.5, color="#333333")
    ax.axvline(0.0, color="#333333", linewidth=1.2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for _, label, *_ in rows], fontsize=8.5)
    ax.set_xlabel("paired recall difference at budget 0.05 (positive would mean the intervention helped)")
    ax.set_xlim(min(lo for _, _, _, lo, _, _, _ in rows) - 0.05,
                max(hi for _, _, _, _, hi, _, _ in rows) + 0.09)
    ax.grid(alpha=0.25, axis="x")

    # Legend by lever, matching the row colours.
    handles = [
        plt.Line2D([], [], color=c, marker="o", linestyle="-", linewidth=2,
                   label=g) for g, c in [
            ("P5  add SemIf as a feature", "#1f77b4"),
            ("P1  direct mode", "#d62728"),
            ("P2  instruction wording", "#2ca02c"),
            ("P3  code embeddings", "#9467bd"),
        ]
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower left", title="lever",
              title_fontsize=8)
    ax.set_title(
        "Every text-side lever fails at budget 0.05\n"
        "filled marker = significant at p<0.05; each row is paired against the baseline "
        "named in the label",
        fontsize=11,
    )
    fig.tight_layout()
    out = config.ARTIFACTS / "figures" / "fig_variation_levers.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[figures] {out}")
    return out


def run() -> list[Path]:
    data = _load()
    made = [
        fig_starved_full_suite(data),
        fig_history_coverage(data),
        fig_variation_levers(data),
    ]
    return [p for p in made if p is not None]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for path in run():
        print(path)
