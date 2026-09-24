"""Complexity ladder: bundle one killed mutant with survived-mutant distractors.

Motivation
----------
The current benchmark is 1-line mutants killed by exactly one unit test, so
coverage plus filename matching nearly solves it. Real commits touch several
places at once, are often batched from unrelated edits, and span files. This
module broadens the *change* while holding the *answer* fixed.

Why survived distractors
------------------------
A bundle is "caught" if a selected test is sensitive to any edit in it (union
semantics). Survived mutants have no killing tests by construction, so the union
of the bundle's kill set is exactly the signal mutant's kill set. **The label is
therefore exact, not approximated** -- no re-running the suite is needed. Killed
distractors would break exactness and, worse, inflate the number of killing tests,
which makes RTS *easier* rather than harder. Rung 4 includes them deliberately, as
a separate control for that effect.

Why the candidate pool is held fixed
------------------------------------
Candidates are the signal mutant's covered set. The killing test always covers the
signal's mutated function, so this is lossless, and it keeps the ranking pool
identical across rungs. Bundling then varies only the change *description*, which
is the manipulation of interest.

The length confound
-------------------
Bundling lengthens the change text, and the placement controls showed a long
prefix degrades this reranker regardless of content. Rung text is therefore
available both full and truncated to a token budget, so a drop can be attributed
to discrimination rather than to query length.

Usage::

    python -m rts.bundles --cpu          # feature-model ladder, no GPU
    python -m rts.bundles --build-pairs  # write SemIf pair files for each rung
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from . import config, dataset, evaluate, features, models


@dataclass(frozen=True)
class Bundle:
    rung: int
    signal: int
    distractors: tuple[int, ...]
    members: tuple[int, ...]
    files: tuple[str, ...]
    change_size: int
    killing_tests: tuple[str, ...]
    ran_tests: tuple[str, ...]


# rung -> (n_distractors, cross_file, distractor_kind)
RUNGS: dict[int, tuple[int, bool, str]] = {
    0: (0, False, "none"),
    1: (2, False, "survived"),
    2: (5, False, "survived"),
    3: (5, True, "survived"),
    4: (5, False, "killed"),
}
# rung 0 is the existing single-mutant benchmark; kept for reference only.
BUNDLE_FEATURES = [
    "n_mutations",
    "n_mutations_covered",
    "frac_mutations_covered",
    "name_match_any",
    "min_path_distance",
    "change_size",
    "test_duration",
    "test_n_lines",
    "test_n_tokens",
    "n_tests_in_test_file",
    "test_failure_rate_cum",
    "test_runs_cum",
    "test_last_failure_age",
]


def _survivor_pool(ds: dataset.Dataset) -> dict[str, list[int]]:
    pool: dict[str, list[int]] = {}
    for i, c in enumerate(ds.changes):
        if c.survived:
            pool.setdefault(c.file, []).append(i)
    return pool


def make_bundles(
    ds: dataset.Dataset,
    rung: int,
    seed: int = config.SEED,
) -> list[Bundle]:
    """One bundle per fault-bearing change, deterministic in (rung, seed)."""
    if rung not in RUNGS:
        raise ValueError(f"unknown rung {rung}; expected one of {sorted(RUNGS)}")
    k, cross_file, kind = RUNGS[rung]

    fault = [int(i) for i in ds.fault_idx]
    survivors_by_file = _survivor_pool(ds)
    all_survivors = [i for v in survivors_by_file.values() for i in v]

    bundles: list[Bundle] = []
    for i in fault:
        rng = np.random.default_rng(seed * 100_003 + rung * 1009 + i)
        if kind == "none":  # rung 0: the original single-mutant benchmark
            distractors = ()
        elif kind == "survived":
            if cross_file:
                pool = [j for j in all_survivors if ds.files[j] != ds.files[i]]
                if len(pool) < k:
                    pool = all_survivors
            else:
                pool = survivors_by_file.get(ds.files[i], [])
                if len(pool) < k:
                    pool = all_survivors
            replace = len(pool) < k
            picks = rng.choice(np.array(pool, dtype=np.int64), size=k, replace=replace)
            distractors = tuple(int(x) for x in picks)
        else:  # rung 4: killed distractors
            pool = [j for j in fault if j != i]
            picks = rng.choice(np.array(pool, dtype=np.int64), size=k, replace=False)
            distractors = tuple(int(x) for x in picks)

        members = (i,) + distractors
        killing = tuple(
            sorted({t for m in members for t in ds.changes[m].killing_tests})
        )
        ran = tuple(sorted({t for m in members for t in ds.changes[m].ran_tests}))
        bundles.append(
            Bundle(
                rung=rung,
                signal=i,
                distractors=distractors,
                members=members,
                files=tuple(sorted({ds.files[m] for m in members})),
                change_size=int(sum(ds.changes[m].change_size for m in members)),
                killing_tests=killing,
                ran_tests=ran,
            )
        )
    return bundles


def bundle_text(ds: dataset.Dataset, bundle: Bundle, token_budget: int | None = None) -> str:
    """Change text: the signal's diff first, then distractors'.

    ``token_budget`` truncates on whitespace tokens so a rung can be compared
    against the single-mutant arm at matched query length.
    """
    parts = [features.change_query_text(ds.changes[m]) for m in bundle.members]
    text = "\n".join(parts)
    if token_budget is None:
        return text
    tokens = text.split()
    return " ".join(tokens[:token_budget])


# --- features --------------------------------------------------------------


def bundle_arrays(
    ds: dataset.Dataset,
    bundles: list[Bundle],
    pool: str = "signal",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, labels, candidates, signals) for a rung.

    ``X`` is ``[n_bundles, n_tests, n_features]``; ``labels`` marks killing tests.

    ``pool`` selects the candidate set:

    * ``signal`` -- the signal mutant's covered set. Lossless (the killer is always
      in it) and identical across rungs, so bundling varies only the change
      description. But it also *holds the coverage feature's selectivity fixed*,
      which masks the main way bundling should degrade structure.
    * ``union`` -- every bundled member's covered set. This is the realistic pool,
      and it grows with the bundle, so ``covers_function`` becomes less selective.
    """
    X_base, names = features.structured_features(ds)
    ix = {n: i for i, n in enumerate(names)}
    cov = X_base[:, :, ix["covers_function"]] > 0.5
    nm = X_base[:, :, ix["module_name_in_test_file"]] > 0.5
    pd = X_base[:, :, ix["path_distance"]]
    dur = X_base[:, :, ix["test_duration"]]
    nlin = X_base[:, :, ix["test_n_lines"]]
    ntok = X_base[:, :, ix["test_n_tokens"]]
    nfile = X_base[:, :, ix["n_tests_in_test_file"]]
    frate = X_base[:, :, ix["test_failure_rate_cum"]]
    nruns = X_base[:, :, ix["test_runs_cum"]]
    fage = X_base[:, :, ix["test_last_failure_age"]]

    n_b, n_t = len(bundles), ds.n_tests
    X = np.zeros((n_b, n_t, len(BUNDLE_FEATURES)), dtype=np.float32)
    labels = np.zeros((n_b, n_t), dtype=np.uint8)
    candidates = np.zeros((n_b, n_t), dtype=bool)

    for b, bundle in enumerate(bundles):
        members = np.array(bundle.members, dtype=np.int64)
        n_mut = len(members)
        cov_count = cov[members].sum(axis=0).astype(np.float32)
        X[b, :, 0] = n_mut
        X[b, :, 1] = cov_count
        X[b, :, 2] = cov_count / n_mut
        X[b, :, 3] = nm[members].any(axis=0).astype(np.float32)
        X[b, :, 4] = pd[members].min(axis=0)
        X[b, :, 5] = bundle.change_size
        X[b, :, 6] = dur[bundle.signal]
        X[b, :, 7] = nlin[bundle.signal]
        X[b, :, 8] = ntok[bundle.signal]
        X[b, :, 9] = nfile[bundle.signal]
        X[b, :, 10] = frate[bundle.signal]
        X[b, :, 11] = nruns[bundle.signal]
        X[b, :, 12] = fage[bundle.signal]

        for test in bundle.killing_tests:
            j = ds.test_index.get(test)
            if j is not None:
                labels[b, j] = 1
        covered_members = members if pool == "union" else np.array([bundle.signal])
        for m in covered_members:
            for test in ds.covered[int(m)]:
                j = ds.test_index.get(test)
                if j is not None:
                    candidates[b, j] = True
        for test in bundle.killing_tests:  # never drop a killing test
            j = ds.test_index.get(test)
            if j is not None:
                candidates[b, j] = True

    return X, labels, candidates, np.array([bundle.signal for bundle in bundles])


def bundle_bm25(
    ds: dataset.Dataset,
    bundles: list[Bundle],
    token_budget: int | None = None,
) -> np.ndarray:
    """BM25 of the bundled change text against every test."""
    from . import source

    infos = source.load_all(ds.test_ids)
    docs = [infos[t].source if t in infos else "" for t in ds.test_ids]
    scorer = features.BM25Scorer().fit(docs)
    out = np.zeros((len(bundles), ds.n_tests), dtype=np.float32)
    for b, bundle in enumerate(bundles):
        out[b] = scorer.score(bundle_text(ds, bundle, token_budget))
    return out


# --- evaluation ------------------------------------------------------------


def bundle_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    candidates: np.ndarray,
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masked = np.where(candidates[rows], scores[rows], -np.inf)
    order = np.argsort(-masked, axis=1, kind="stable")
    lab = labels[rows][np.arange(len(rows))[:, None], order]
    return lab, np.maximum.accumulate(lab, axis=1), np.cumsum(lab, axis=1)


def evaluate_rung(
    scores: np.ndarray,
    labels: np.ndarray,
    candidates: np.ndarray,
    rows: np.ndarray,
    budgets: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2),
    n_bootstrap: int = 500,
    seed: int = config.SEED,
) -> list[evaluate.BudgetResult]:
    """Recall/precision for a rung, matching evaluate.evaluate's conventions."""
    rng = np.random.default_rng(seed)
    n_tests = labels.shape[1]
    _, cum_hit, cum_count = bundle_curve(scores, labels, candidates, rows)
    cand_counts = candidates[rows].sum(axis=1)
    rows_i = np.arange(len(rows))

    results: list[evaluate.BudgetResult] = []
    for budget in budgets:
        k = np.array([evaluate.budget_k(budget, int(c)) for c in cand_counts])
        hits = cum_hit[rows_i, k - 1].astype(np.float64)
        precision_per_change = cum_count[rows_i, k - 1] / k
        recall = float(hits.mean())
        precision = float(precision_per_change.mean())
        f_measure = (
            2 * recall * precision / (recall + precision) if recall + precision else 0.0
        )
        lo, hi = evaluate._bootstrap_ci(hits, n_bootstrap, rng)
        mean_k = float(k.mean())
        results.append(
            evaluate.BudgetResult(
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


def bundle_hits(
    scores: np.ndarray,
    labels: np.ndarray,
    candidates: np.ndarray,
    rows: np.ndarray,
    budget: float,
) -> np.ndarray:
    """Per-bundle caught/not at one budget, for paired comparisons."""
    _, cum_hit, _ = bundle_curve(scores, labels, candidates, rows)
    cand_counts = candidates[rows].sum(axis=1)
    k = np.array([evaluate.budget_k(budget, int(c)) for c in cand_counts])
    return cum_hit[np.arange(len(rows)), k - 1].astype(np.float64)


def _xgb_scores(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    seed: int = config.SEED,
):
    import xgboost as xgb

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.15,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_eval)[:, 1], dict(
        sorted(
            zip(BUNDLE_FEATURES, model.feature_importances_.tolist()),
            key=lambda kv: -kv[1],
        )
    )


# --- driver ----------------------------------------------------------------


def run_cpu(
    n_held_out: int = 200,
    seed: int = config.SEED,
    rungs: tuple[int, ...] | None = None,
    pool: str = "signal",
) -> dict:
    ds = dataset.build()
    summary: dict = {}

    # Bundle count and held-out sample are identical across rungs, so the ladder is
    # paired: the same changes are evaluated, only the distractor set differs.
    n_b = len(make_bundles(ds, 0, seed))
    split = int(round(n_b * config.DEFAULT_TRAIN_FRACTION))
    held_pool = np.arange(split, n_b)
    if len(held_pool) > n_held_out:
        picker = np.random.default_rng(seed)
        held = np.sort(picker.choice(held_pool, size=n_held_out, replace=False))
    else:
        held = held_pool

    for rung in sorted(rungs or RUNGS):
        bundles = make_bundles(ds, rung, seed)
        X, labels, candidates, signals = bundle_arrays(ds, bundles, pool=pool)
        bm25 = bundle_bm25(ds, bundles)

        # Train on candidate pairs only. Training over the whole suite lets the
        # model spend its capacity learning the candidate mask (which is constant
        # inside the pool at evaluation time) instead of learning to rank within it.
        train_mask = candidates[:split]
        X_train = X[:split][train_mask]
        y_train = labels[:split][train_mask]
        held_rows, held_cols = np.nonzero(candidates[held])
        X_eval = X[held][held_rows, held_cols]
        pred, importances = _xgb_scores(X_train, y_train, X_eval)
        xgb_scores = np.full((n_b, ds.n_tests), -1e9, dtype=np.float32)
        xgb_scores[held[held_rows], held_cols] = pred

        ix = {n: i for i, n in enumerate(BUNDLE_FEATURES)}
        structural = (
            X[:, :, ix["name_match_any"]] * 2.0
            + 1.0 / (1.0 + X[:, :, ix["test_n_lines"]])
        )
        model_scores = {
            "bundled_xgboost": xgb_scores,
            "bundled_bm25": bm25,
            "bundled_structural": structural,
            "packing_frac_covered": X[:, :, ix["frac_mutations_covered"]],
            "random": np.random.default_rng(seed).random((n_b, ds.n_tests)).astype(np.float32),
        }

        n_files = np.array([len(b.files) for b in bundles])
        print(f"\n{'=' * 78}")
        print(
            f"RUNG {rung}: {RUNGS[rung][0]} distractors, cross_file={RUNGS[rung][1]}, "
            f"kind={RUNGS[rung][2]}   ({n_b} bundles, {len(held)} held out, pool={pool})"
        )
        print(f"  files per bundle: mean {n_files.mean():.2f}, "
              f"all-single-file {np.mean(n_files == 1):.2f}, "
              f"mean candidate pool {candidates.sum(axis=1).mean():.0f}")
        print(f"{'=' * 78}")
        print(f"{'model':>24}  " + "  ".join(f"b{b:<5}" for b in (0.01, 0.05, 0.1, 0.2)))
        for name, s in model_scores.items():
            res = evaluate_rung(s, labels, candidates, held)
            print(f"{name:>24}  " + "  ".join(f"{r.recall:.3f}" for r in res))
        print("  xgboost top features: "
              + ", ".join(f"{k}={v:.3f}" for k, v in list(importances.items())[:5]))

        summary[str(rung)] = {
            "n_bundles": n_b,
            "pool": pool,
            "n_held_out": int(len(held)),
            "mean_candidates": float(candidates.sum(axis=1).mean()),
            "mean_files_per_bundle": float(n_files.mean()),
            "frac_single_file": float(np.mean(n_files == 1)),
            "results": {
                name: evaluate.results_to_dicts(evaluate_rung(s, labels, candidates, held))
                for name, s in model_scores.items()
            },
            "hits": {
                name: bundle_hits(s, labels, candidates, held, 0.05).astype(int).tolist()
                for name, s in model_scores.items()
            },
            "importances": importances,
        }

    config.ensure_artifacts_dir()
    out = config.ARTIFACTS / f"bundles_cpu_{pool}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] wrote {out}")
    return summary


def plot_ladder(
    rungs: tuple[int, ...] = (0, 2, 3),
    n_held_out: int = 200,
    budget: float = 0.05,
    seed: int = config.SEED,
    out_path=None,
) -> str:
    """Figure: recall vs change complexity, and degradation relative to rung 0."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = dataset.build()
    n_b = len(make_bundles(ds, 0, seed))
    split = int(round(n_b * config.DEFAULT_TRAIN_FRACTION))
    held = np.sort(
        np.random.default_rng(seed).choice(np.arange(split, n_b), size=n_held_out, replace=False)
    )

    curves: dict[str, list[float]] = {}
    hits: dict[int, dict[str, np.ndarray]] = {}
    ix_all = None
    for rung in rungs:
        bundle_list = make_bundles(ds, rung, seed)
        X, labels, candidates, _ = bundle_arrays(ds, bundle_list)
        if ix_all is None:
            ix_all = {n: i for i, n in enumerate(BUNDLE_FEATURES)}
        bm = bundle_bm25(ds, bundle_list)
        sf = load_bundle_scores(
            config.ARTIFACTS / f"semif_bundles_rung{rung}_signal.jsonl", n_b, ds.n_tests
        )
        train_mask = candidates[:split]
        pred, _ = _xgb_scores(X[:split][train_mask], labels[:split][train_mask],
                              X[held][candidates[held]])
        xg = np.full((n_b, ds.n_tests), -1e9, dtype=np.float32)
        r, c = np.nonzero(candidates[held])
        xg[held[r], c] = pred
        model_scores = {
            "SemIf (text)": sf,
            "BM25 (text)": bm,
            "XGBoost (structural)": xg,
            "structural rule": X[:, :, ix_all["name_match_any"]] * 2.0
            + 1.0 / (1.0 + X[:, :, ix_all["test_n_lines"]]),
            "random": np.random.default_rng(seed).random((n_b, ds.n_tests)).astype(np.float32),
        }
        hits[rung] = {name: bundle_hits(s, labels, candidates, held, budget)
                      for name, s in model_scores.items()}
        for name, h in hits[rung].items():
            curves.setdefault(name, []).append(float(h.mean()))

    styles = {
        "SemIf (text)": ("#d62728", "-", 2.6),
        "BM25 (text)": ("#e377c2", "--", 2.4),
        "XGBoost (structural)": ("#1f77b4", "-", 2.6),
        "structural rule": ("#bcbd22", "-", 1.8),
        "random": ("#cccccc", "-", 1.4),
    }
    labels_x = {0: "1 mutation\n1 file", 2: "6 mutations\n1 file", 3: "6 mutations\n3.8 files"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    ax = axes[0]
    for name, vals in curves.items():
        colour, style, width = styles[name]
        ax.plot(range(len(rungs)), vals, style, color=colour, linewidth=width,
                marker="o", markersize=6, label=name)
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels([labels_x[r] for r in rungs], fontsize=9)
    ax.set_xlabel("change complexity")
    ax.set_ylabel(f"recall @ budget {budget}")
    ax.set_title("Recall vs change complexity")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 0.65)

    ax = axes[1]
    rng = np.random.default_rng(seed)
    for name, vals in curves.items():
        if name == "random":
            continue
        deltas, los, his = [], [], []
        for k, rung in enumerate(rungs):
            if rung == rungs[0]:
                deltas.append(0.0); los.append(0.0); his.append(0.0); continue
            d = hits[rung][name] - hits[rungs[0]][name]
            idx = rng.integers(0, d.size, size=(2000, d.size))
            means = d[idx].mean(axis=1)
            deltas.append(float(d.mean()))
            los.append(float(np.percentile(means, 2.5)))
            his.append(float(np.percentile(means, 97.5)))
        colour, style, width = styles[name]
        err = [np.array(deltas) - np.array(los), np.array(his) - np.array(deltas)]
        ax.errorbar(range(len(rungs)), deltas, yerr=err, fmt="o", color=colour,
                    linestyle=style, linewidth=width, markersize=6, capsize=4, label=name)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels([labels_x[r] for r in rungs], fontsize=9)
    ax.set_xlabel("change complexity")
    ax.set_ylabel(f"recall change vs baseline (budget {budget})")
    ax.set_title("Degradation relative to the single-mutation baseline\n(95% paired bootstrap CI)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    fig.suptitle(
        "Bundling unrelated mutations into one change hurts text models and leaves "
        "structural models untouched",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = config.ARTIFACTS / "figures" / "complexity_ladder.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out_path:
        out = out_path
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[figure] {out}")
    return str(out)


def _test_source(ds: dataset.Dataset, col: int) -> str:
    """Source text of one test function, for the SemIf Document side."""
    from . import source

    nodeid = ds.test_ids[col]
    info = source.load_all([nodeid]).get(nodeid)
    return info.source if info else ""


def score_semif(
    rungs: tuple[int, ...] = (0, 3),
    n_held_out: int = 200,
    batch_size: int = 8,
    seed: int = config.SEED,
    pool: str = "signal",
) -> dict:
    """Score SemIf on bundled changes for the given rungs (held-out bundles only).

    SemIf is zero-shot, so only the held-out bundles need scoring. The candidate
    pool matches the CPU ladder so the arms are comparable.
    """
    from . import semif_runner as sr

    ds = dataset.build()
    n_b = len(make_bundles(ds, 0, seed))
    split = int(round(n_b * config.DEFAULT_TRAIN_FRACTION))
    held_pool = np.arange(split, n_b)
    picker = np.random.default_rng(seed)
    held = np.sort(
        picker.choice(held_pool, size=min(n_held_out, len(held_pool)), replace=False)
    )

    model = tokenizer = None
    out_stats: dict = {}
    for rung in rungs:
        bundles = make_bundles(ds, rung, seed)
        _, labels, candidates, _ = bundle_arrays(ds, bundles, pool=pool)

        pairs: list[tuple[str, str]] = []
        index: list[tuple[int, int]] = []
        for b in held:
            b = int(b)
            text = bundle_text(ds, bundles[b])
            for j in np.flatnonzero(candidates[b]):
                j = int(j)
                pairs.append((text, _test_source(ds, j)))
                index.append((b, j))

        cache = config.ARTIFACTS / f"semif_bundles_rung{rung}_{pool}.jsonl"
        print(f"\n=== rung {rung}: {len(pairs):,} pairs -> {cache} ===", flush=True)
        if model is None:
            model, tokenizer, meta = sr.load_model()
            print(f"loaded {meta['device']} {meta['dtype']}", flush=True)

        pair_set = sr.PairSet(pairs=pairs, index=index, feature_blocks=None)
        out_stats[rung] = sr.score_to_cache(
            model, tokenizer, ds, pair_set, cache, batch_size=batch_size
        )
    return out_stats


def load_bundle_scores(path, n_bundles: int, n_tests: int) -> np.ndarray:
    """Load a rung's SemIf cache into a [n_bundles, n_tests] score matrix."""
    out = np.full((n_bundles, n_tests), -1e9, dtype=np.float32)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            out[int(record["change_row"]), int(record["test_col"])] = record["score"]
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="run the feature-model ladder")
    parser.add_argument("--plot", action="store_true", help="plot the complexity ladder")
    parser.add_argument("--score-semif", action="store_true", help="score SemIf on rungs")
    parser.add_argument("--sample", type=int, default=200,
                        help="held-out bundles to evaluate")
    parser.add_argument("--rungs", default=None,
                        help="comma-separated rungs, e.g. 0,1,2,3")
    parser.add_argument("--pool", default="signal", choices=["signal", "union"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    rungs = tuple(int(x) for x in args.rungs.split(",")) if args.rungs else None
    if args.plot:
        plot_ladder(rungs=rungs or (0, 2, 3), n_held_out=args.sample)
    elif args.score_semif:
        score_semif(
            rungs=rungs or (0, 3),
            n_held_out=args.sample,
            batch_size=args.batch_size,
            pool=args.pool,
        )
    elif args.cpu:
        run_cpu(n_held_out=args.sample, rungs=rungs, pool=args.pool)
    else:
        parser.error("pass --cpu, --plot or --score-semif")
