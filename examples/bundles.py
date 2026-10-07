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
annotations). Survived mutants have no killing tests by construction, so the union
of the bundle's kill set is exactly the focal mutant's kill set. **The label is
therefore exact, not approximated** -- no re-running the suite is needed. Killed
distractors would break exactness and, worse, inflate the number of killing tests,
which makes RTS *easier* rather than harder. Rung 4 includes them deliberately, as
a separate control for that effect.

Why the candidate pool is held fixed
------------------------------------
Candidates are the focal mutant's coverage set. The killing test always covers the
focal's mutated function, so this is lossless, and it keeps the ranking pool
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

import numpy as np

from examples import bundle_features, config, datasets
from examples.datasets import Bundle
from rts import evaluate, features
from rts.data import accessors, contract

# Rung construction is a study choice, so the rung definitions stay here; the *shape* of
# a bundled dataset lives on the contract (datasets.BundleDataset), because bundling is
# composition rather than a special case (§3). The base dataset and the wrapper are
# built together so the bundle features and the evaluation see one pool.


# rung -> (n_distractors, cross_file, distractor_kind)
RUNGS: dict[int, tuple[int, bool, str]] = {
    0: (0, False, "none"),
    1: (2, False, "survived"),
    2: (5, False, "survived"),
    3: (5, True, "survived"),
    4: (5, False, "killed"),
    # 5: same shape as rung 3, but distractors are chosen for relatedness instead
    # of at random. This is the coherent-change condition: a real batch of edits
    # usually touches one concept, giving a reranker a theme it can read.
    5: (5, True, "coherent"),
}
# rung 0 is the existing single-mutant benchmark; kept for reference only.


def _survivor_pool(ds: contract.Dataset) -> dict[str, list[int]]:
    pool: dict[str, list[int]] = {}
    paths = accessors.change_paths(ds)
    for i, change in enumerate(ds.changes):
        if getattr(change, "survived", False):
            pool.setdefault(paths[i], []).append(i)
    return pool


def _coverage_matrix(ds: contract.Dataset) -> np.ndarray:
    """[n_changes, n_tests] float32 indicator of which tests cover each change."""
    if not ds.has_capability("coverage"):
        raise contract.CapabilityMissing(ds.name, "coverage")
    C = np.zeros((ds.n_changes, ds.n_tests), dtype=np.float32)
    for i, covered in enumerate(accessors.coverage_sets(ds)):
        for test in covered:
            j = ds.test_index.get(test)
            if j is not None:
                C[i, j] = 1.0
    return C


def _coherent_ranking(
    ds: contract.Dataset,
    fault: list[int],
    survivors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Rank survivors by coverage-profile Jaccard similarity to each signal.

    Structural relatedness stands in for semantic coherence: two mutants covered by
    a similar set of tests sit on similar execution paths, so bundling them reads as
    one themed change rather than several unrelated edits. One matmul over the
    boolean coverage matrix, so this is cheap.
    """
    C = _coverage_matrix(ds)
    survivors_arr = np.array(survivors, dtype=np.int64)
    signal_arr = np.array(fault, dtype=np.int64)
    Cs, Cg = C[survivors_arr], C[signal_arr]
    inter = Cg @ Cs.T
    union = Cg.sum(1)[:, None] + Cs.sum(1)[None, :] - inter
    jaccard = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    return jaccard, survivors_arr


def make_bundles(
    ds: contract.Dataset,
    rung: int,
    seed: int = config.SEED,
) -> list[Bundle]:
    """One bundle per fault-bearing change, deterministic in (rung, seed)."""
    if rung not in RUNGS:
        raise ValueError(f"unknown rung {rung}; expected one of {sorted(RUNGS)}")
    k, cross_file, kind = RUNGS[rung]

    fault = [int(i) for i in accessors.fault_idx(ds)]
    paths = accessors.change_paths(ds)
    survivors_by_file = _survivor_pool(ds)
    all_survivors = [i for v in survivors_by_file.values() for i in v]

    jaccard = survivors_arr = None
    signal_pos: dict[int, int] = {}
    if kind == "coherent":
        jaccard, survivors_arr = _coherent_ranking(ds, fault, all_survivors)
        signal_pos = {int(s): p for p, s in enumerate(fault)}

    bundles: list[Bundle] = []
    for i in fault:
        rng = np.random.default_rng(seed * 100_003 + rung * 1009 + i)
        if kind == "none":  # rung 0: the original single-mutant benchmark
            distractors = ()
        elif kind == "coherent":
            p = signal_pos[i]
            external = np.array(
                [paths[int(j)] != paths[i] for j in survivors_arr],
                dtype=bool,
            )
            top = np.argsort(-np.where(external, jaccard[p], -1.0), kind="stable")[:k]
            distractors = tuple(int(survivors_arr[t]) for t in top)
        elif kind == "survived":
            if cross_file:
                pool = [j for j in all_survivors if paths[j] != paths[i]]
                if len(pool) < k:
                    pool = all_survivors
            else:
                pool = survivors_by_file.get(paths[i], [])
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
        bundles.append(Bundle(rung=rung, focal=i, members=members))
    return bundles


def bundle_text(ds: contract.Dataset, bundle: Bundle, token_budget: int | None = None) -> str:
    """Change text: the focal change's diff first, then distractors'.

    ``token_budget`` truncates on whitespace tokens so a rung can be compared
    against the single-mutant condition at matched query length.
    """
    parts = [features.derived.change_query_text(ds, ds.changes[m]) for m in bundle.members]
    text = "\n".join(parts)
    if token_budget is None:
        return text
    tokens = text.split()
    return " ".join(tokens[:token_budget])


# --- features --------------------------------------------------------------


def bundle_matrix(bundle_ds, *, temporal: bool = True):
    """The bundle feature matrix, from the declared block in :mod:`examples.bundle_features`."""
    return bundle_features.bundle(bundle_ds, temporal=temporal)


def bundle_arrays(base, bundle_ds):
    """Return (matrix, labels, candidate sets, focals) for a rung."""
    matrix = bundle_matrix(bundle_ds, temporal=True)
    labels = accessors.labels(bundle_ds)
    candidate_sets = accessors.candidate_sets(bundle_ds, "coverage_restricted")
    focals = np.array([b.focal for b in bundle_ds.changes])
    return matrix, labels, candidate_sets, focals


def bundle_bm25(
    ds: contract.Dataset,
    bundles: list[Bundle],
    token_budget: int | None = None,
) -> np.ndarray:
    """BM25 of the bundled change text against every test."""
    docs = [ds.test_source(t) or "" for t in ds.test_ids]
    scorer = features.text.BM25Scorer().fit(docs)
    out = np.zeros((len(bundles), ds.n_tests), dtype=np.float32)
    for b, bundle in enumerate(bundles):
        out[b] = scorer.score(bundle_text(ds, bundle, token_budget))
    return out


# --- evaluation ------------------------------------------------------------


def bundle_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    candidate_sets: np.ndarray,
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masked = np.where(candidate_sets[rows], scores[rows], -np.inf)
    order = np.argsort(-masked, axis=1, kind="stable")
    lab = labels[rows][np.arange(len(rows))[:, None], order]
    return lab, np.maximum.accumulate(lab, axis=1), np.cumsum(lab, axis=1)


def evaluate_rung(
    scores: np.ndarray,
    labels: np.ndarray,
    candidate_sets: np.ndarray,
    rows: np.ndarray,
    budgets: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2),
    n_bootstrap: int = 500,
    seed: int = config.SEED,
) -> list[evaluate.BudgetResult]:
    """Recall/precision for a rung, through the harness's one metric sweep.

    A rung carries its own label and candidate matrices rather than a dataset, so it
    calls the matrix form of the sweep. That is the same code path a dataset uses --
    the point of the evaluation contract is that it cannot drift from it.
    """
    return evaluate.sweep_matrices(
        scores, labels, candidate_sets, rows, labels.shape[1],
        budgets=budgets, n_bootstrap=n_bootstrap, seed=seed,
    )


def bundle_hits(
    scores: np.ndarray,
    labels: np.ndarray,
    candidate_sets: np.ndarray,
    rows: np.ndarray,
    budget: float,
) -> np.ndarray:
    """Per-bundle caught/not at one budget, for paired contrasts."""
    return evaluate.per_change_hit_matrix(scores, labels, candidate_sets, rows, budget)


def _xgb_scores(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    columns,
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
            zip(columns, model.feature_importances_.tolist()),
            key=lambda kv: -kv[1],
        )
    )


# --- driver ----------------------------------------------------------------


def run_cpu(
    n_held_out: int = 200,
    seed: int = config.SEED,
    rungs: tuple[int, ...] | None = None,
    pool: str = "focal",
) -> dict:
    ds = datasets.marshmallow()
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
        bundle_ds = datasets.bundles(ds, rung, seed, pool=pool)
        matrix, labels, candidate_sets, signals = bundle_arrays(ds, bundle_ds)
        bm25 = bundle_bm25(ds, list(bundle_ds.changes))

        # Train on candidate pairs only. Training over the whole suite lets the
        # model spend its capacity learning the candidate mask (which is constant
        # inside the pool at evaluation time) instead of learning to rank within it.
        train_mask = candidate_sets[:split]
        X_train = matrix.X[:split][train_mask]
        y_train = labels[:split][train_mask]
        held_rows, held_cols = np.nonzero(candidate_sets[held])
        X_eval = matrix.X[held][held_rows, held_cols]
        pred, importances = _xgb_scores(X_train, y_train, X_eval, matrix.columns)
        xgb_scores = np.full((n_b, ds.n_tests), -1e9, dtype=np.float32)
        xgb_scores[held[held_rows], held_cols] = pred

        structural = (
            matrix.column("filename_match_any") * 2.0
            + 1.0 / (1.0 + matrix.column("test_lines"))
        )
        model_scores = {
            "bundled_xgboost": xgb_scores,
            "bundled_bm25": bm25,
            "bundled_structural": structural,
            "packing_frac_covered": matrix.column("frac_mutations_covered"),
            "random": np.random.default_rng(seed).random((n_b, ds.n_tests)).astype(np.float32),
        }

        n_files = np.array([len(bundle_ds.files(b)) for b in bundle_ds.changes])
        print(f"\n{'=' * 78}")
        print(
            f"RUNG {rung}: {RUNGS[rung][0]} distractors, cross_file={RUNGS[rung][1]}, "
            f"kind={RUNGS[rung][2]}   ({n_b} bundles, {len(held)} held out, pool={pool})"
        )
        print(f"  files per bundle: mean {n_files.mean():.2f}, "
              f"all-single-file {np.mean(n_files == 1):.2f}, "
              f"mean candidate pool {candidate_sets.sum(axis=1).mean():.0f}")
        print(f"{'=' * 78}")
        print(f"{'model':>24}  " + "  ".join(f"b{b:<5}" for b in (0.01, 0.05, 0.1, 0.2)))
        for name, s in model_scores.items():
            res = evaluate_rung(s, labels, candidate_sets, held)
            print(f"{name:>24}  " + "  ".join(f"{r.recall:.3f}" for r in res))
        print("  xgboost top features: "
              + ", ".join(f"{k}={v:.3f}" for k, v in list(importances.items())[:5]))

        summary[str(rung)] = {
            "n_bundles": n_b,
            "pool": pool,
            "n_held_out": int(len(held)),
            "mean_candidate_count": float(candidate_sets.sum(axis=1).mean()),
            "mean_files_per_bundle": float(n_files.mean()),
            "frac_single_file": float(np.mean(n_files == 1)),
            "results": {
                name: evaluate.results_to_dicts(evaluate_rung(s, labels, candidate_sets, held))
                for name, s in model_scores.items()
            },
            "hits": {
                name: bundle_hits(s, labels, candidate_sets, held, 0.05).astype(int).tolist()
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
    rungs: tuple[int, ...] = (0, 2, 3, 5),
    n_held_out: int = 200,
    budget: float = 0.05,
    seed: int = config.SEED,
    out_path=None,
) -> str:
    """Figure: recall vs change complexity, and degradation relative to rung 0."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = datasets.marshmallow()
    n_b = len(make_bundles(ds, 0, seed))
    split = int(round(n_b * config.DEFAULT_TRAIN_FRACTION))
    held = np.sort(
        np.random.default_rng(seed).choice(np.arange(split, n_b), size=n_held_out, replace=False)
    )

    curves: dict[str, list[float]] = {}
    hits: dict[int, dict[str, np.ndarray]] = {}
    for rung in rungs:
        bundle_ds = datasets.bundles(ds, rung, seed)
        matrix, labels, candidate_sets, _ = bundle_arrays(ds, bundle_ds)
        bm = bundle_bm25(ds, list(bundle_ds.changes))
        sf = load_bundle_scores(
            config.ARTIFACTS / f"semif_bundles_rung{rung}_focal.jsonl", n_b, ds.n_tests
        )
        train_mask = candidate_sets[:split]
        pred, _ = _xgb_scores(matrix.X[:split][train_mask], labels[:split][train_mask],
                              matrix.X[held][candidate_sets[held]], matrix.columns)
        xg = np.full((n_b, ds.n_tests), -1e9, dtype=np.float32)
        r, c = np.nonzero(candidate_sets[held])
        xg[held[r], c] = pred
        model_scores = {
            "SemIf (text)": sf,
            "BM25 (text)": bm,
            "XGBoost (structural)": xg,
            "structural rule": matrix.column("filename_match_any") * 2.0
            + 1.0 / (1.0 + matrix.column("test_lines")),
            "random": np.random.default_rng(seed).random((n_b, ds.n_tests)).astype(np.float32),
        }
        hits[rung] = {name: bundle_hits(s, labels, candidate_sets, held, budget)
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
    labels_x = {
        0: "1 mutation\n1 file",
        2: "6 mutations\n1 file",
        3: "6 mutations\n3.8 files\n(unrelated)",
        5: "6 mutations\n2.6 files\n(coherent)",
    }

    fig, factors = plt.subplots(1, 2, figsize=(14, 5.6))
    ax = factors[0]
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

    ax = factors[1]
    rng = np.random.default_rng(seed)
    for name, _vals in curves.items():
        if name == "random":
            continue
        deltas, los, his = [], [], []
        for rung in rungs:
            if rung == rungs[0]:
                deltas.append(0.0)
                los.append(0.0)
                his.append(0.0)
                continue
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
        "Broadening a change hurts text models and leaves structural models untouched "
        "-- coherence does not rescue the reranker",
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


def _test_source(ds: contract.Dataset, col: int) -> str:
    """Source text of one test function, for the SemIf Document side."""
    return ds.test_source(ds.test_ids[col]) or ""


def score_semif(
    rungs: tuple[int, ...] = (0, 3),
    n_held_out: int = 200,
    batch_size: int = 8,
    seed: int = config.SEED,
    pool: str = "focal",
) -> dict:
    """Score SemIf on bundled changes for the given rungs (held-out bundles only).

    SemIf is zero-shot, so only the held-out bundles need scoring. The candidate
    pool matches the CPU ladder so the conditions are comparable.
    """
    from examples import semif_runner as sr

    ds = datasets.marshmallow()
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
        bundle_ds = datasets.bundles(ds, rung, seed, pool=pool)
        _, labels, candidate_sets, _ = bundle_arrays(ds, bundle_ds)

        pairs: list[tuple[str, str]] = []
        index: list[tuple[int, int]] = []
        for b in held:
            b = int(b)
            text = bundle_text(ds, bundle_ds.changes[b])
            for j in np.flatnonzero(candidate_sets[b]):
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
    parser.add_argument("--pool", default="focal", choices=["focal", "union"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    rungs = tuple(int(x) for x in args.rungs.split(",")) if args.rungs else None
    if args.plot:
        plot_ladder(rungs=rungs or (0, 2, 3, 5), n_held_out=args.sample)
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
