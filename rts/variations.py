"""Driver for the SemIf variation experiments (P1-P5) and the full-suite starved arm.

Each experiment is independent and writes its section into
``artifacts/variations.json`` so results survive an interrupted session. Run with
``--only`` to execute one, or with no arguments to run every experiment whose
inputs are present.

What each experiment is for
---------------------------
``full_starved``  Outstanding handoff item 1. Re-evaluates the one positive result
                  (starved failure history) with ``full`` candidates, so the
                  ``coverage`` baseline is no longer degenerate. This is the test
                  of whether the starved win survives contact with how RTS is
                  actually deployed. It also carries a history x coverage
                  decomposition, which is what identifies coverage as the entire
                  effect.
``full_starved_seeds``
                  Is the correction a lucky tree fit? Refits the two strongest trees
                  under four seeds on the same arm.
``p5``            Is the transformer redundant? Adds SemIf to the strongest cheap
                  structured model. Two forms: a trained XGBoost column is the
                  proposal as written, and a fitted-free rank average is the
                  leakage-safe version that needs no scoring of the training
                  window (SemIf scores exist only for held-out changes -- see the
                  note in ``p5_redundancy``).
``p5_trained``    The proposal as written, using a temporal split *inside* the
                  held-out window so training rows have SemIf scores.
``p2``            Instruction and prompt sweep, the one major lever never tested.
                  Evaluated at two starvation thresholds for free.
``p3``            Code-specialised embedding baseline: is there a semantic signal
                  that a 4B natural-language reranker is simply the wrong model for?
``p1``            Direct mode, pairwise: changes the task formulation rather than
                  the model. Throughput-bound, so scoped to the starved population.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config, dataset, evaluate, features, models, semif

BUDGETS = (0.01, 0.05, 0.1, 0.2)
PROBE = 0.05
N_BOOTSTRAP = 2000


# --- shared helpers --------------------------------------------------------


def build_context(ds: dataset.Dataset) -> models.Context:
    X, names = features.structured_features(ds)
    bm25 = features.build_bm25_scores(ds)
    return models.Context(ds=ds, X=X, names=names, bm25=bm25)


def rank_normalize(scores: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Per-change descending rank, normalised to [0, 1]; 0 is best.

    Non-candidates get ``inf`` so they can never win. Used by the fitted-free
    combination below: because each change's candidates are ranked independently,
    two selectors on different scales can be averaged without fitting a weight on
    the evaluation data.
    """
    masked = np.where(candidates, scores, -np.inf)
    order = np.argsort(-masked, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(scores.shape[1])[None, :]
    denom = max(scores.shape[1] - 1, 1)
    return np.where(candidates, ranks / denom, np.inf)


def rank_average(*score_matrices: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Mean normalised rank; higher is better. No parameters are fitted."""
    norms = [rank_normalize(s, candidates) for s in score_matrices]
    return -np.mean(norms, axis=0)


def eval_group(
    ds: dataset.Dataset,
    rows: np.ndarray,
    candidates: np.ndarray,
    scores_by_name: dict[str, np.ndarray],
    budgets: tuple[float, ...] = BUDGETS,
    reference: str | list[str] | None = None,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = config.SEED,
) -> dict:
    """Recall sweep plus paired bootstrap against each reference.

    ``reference`` may be a list: with more than one strong baseline it is not enough
    to beat the model you happened to pick as the reference, so every comparison
    that the write-up needs is computed here.
    """
    results: dict[str, list[dict]] = {}
    hits: dict[str, dict[float, dict[int, bool]]] = {}
    for name, scores in scores_by_name.items():
        results[name] = evaluate.results_to_dicts(
            evaluate.evaluate(
                scores, ds, rows, budgets=budgets, n_bootstrap=1000, seed=seed,
                candidates=candidates,
            )
        )
        hits[name] = {
            b: evaluate.per_change_hits(scores, ds, rows, b, candidates) for b in budgets
        }

    references = [reference] if isinstance(reference, str) else list(reference or [])
    references = [r for r in references if r in scores_by_name]
    comparisons: dict[str, dict] = {"references": references}
    for ref in references:
        for name in scores_by_name:
            if name == ref:
                continue
            for b in budgets:
                stat = evaluate.paired_bootstrap(hits[name][b], hits[ref][b], n_bootstrap, seed)
                stat["budget"] = b
                stat["reference"] = ref
                comparisons[f"{name}|{b}|vs|{ref}"] = stat
    return {"results": results, "comparisons": comparisons}


def _table(group: dict, title: str, name_width: int = 34) -> str:
    results = group["results"]
    lines = [f"\n{title}"]
    header = f"  {'model':<{name_width}}" + "".join(f"{b:>9.2f}" for b in BUDGETS)
    lines.append(header)
    for name, res in results.items():
        cells = "".join(f"{r['recall']:>9.3f}" for r in res)
        lines.append(f"  {name:<{name_width}}{cells}")
    if group.get("comparisons"):
        for key, stat in group["comparisons"].items():
            if key == "references":
                continue
            name, budget, _, ref = key.split("|")
            lines.append(
                f"    {name:<{name_width - 2}} b{float(budget):<5.2f} vs {ref:<26} "
                f"{stat['delta']:+.3f} [{stat['lo']:+.3f}, {stat['hi']:+.3f}] "
                f"p={stat['p_value']:.4f} n={stat['n']}"
            )
    return "\n".join(lines)


def _selectors(ds: dataset.Dataset, ctx: models.Context, candidates_mode: str) -> dict[str, np.ndarray]:
    """The classical reference arms, all trained on the same candidate pairs."""
    return {
        "random": models.RandomSelector().scores(ctx),
        "recency": models.RecencySelector().scores(ctx),
        "failure_rate": models.FailureRateSelector().scores(ctx),
        "coverage": models.CoverageSelector().scores(ctx),
        "structural_rule": models.StructuralRuleSelector().scores(ctx),
        "bm25_lexical": ctx.bm25,
        "xgboost_static_nocov_lex": models.XGBoostSelector(
            exclude_history=True, exclude_coverage=True, include_lexical=True,
            candidates_mode=candidates_mode,
        ).scores(ctx),
    }


# --- experiment 1: full-suite starved arm ----------------------------------


def merge_caches(paths: list[Path], out_path: Path) -> Path:
    """Concatenate score caches, de-duplicating by (change_row, test_col).

    Used to assemble a superset arm incrementally: `failures <= 2` is a subset of
    `failures <= 5`, so the 141-change arm is built from the 43-change cache plus a
    98-change delta rather than re-scoring 51k pairs that already exist.
    """
    merged: dict[tuple[int, int], dict] = {}
    for path in paths:
        with Path(path).open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "change_row" in record and "test_col" in record:
                    merged[(int(record["change_row"]), int(record["test_col"]))] = record
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for record in merged.values():
            fh.write(json.dumps(record) + "\n")
    print(f"[merge] {out_path.name}: {len(merged):,} unique pairs from {len(paths)} caches")
    return out_path


def full_starved(max_failures: int = 2) -> dict:
    """Re-evaluate the starved arm with the full 1187-test candidate set.

    ``max_failures=2`` is the 43-fault arm that produced the original positive
    result; ``max_failures=5`` is the 141-fault decision-grade confirmation. The
    second is assembled from the first plus a scored delta (see ``merge_caches``).
    """
    if max_failures == 2:
        cache = config.ARTIFACTS / "semif_scores_starved2_full.jsonl"
    else:
        cache = config.ARTIFACTS / f"semif_scores_starved{max_failures}_full.jsonl"
        if not cache.exists():
            merge_caches(
                [
                    config.ARTIFACTS / "semif_scores_starved2_full.jsonl",
                    config.ARTIFACTS / f"semif_scores_starved{max_failures}_extra_full.jsonl",
                ],
                cache,
            )
    if not cache.exists():
        raise FileNotFoundError(f"missing {cache}; run the starved full-suite scoring arm first")

    ds = dataset.build()
    candidates = dataset.candidate_mask(ds, "full")
    mask = dataset.starved_mask(ds, max_failures=max_failures)
    rows = ds.test_idx[mask[ds.test_idx]]
    ctx = build_context(ds)

    scores = _selectors(ds, ctx, "full")
    # The full candidate set is the only regime where `covers_function` separates
    # candidates from non-candidates, so the strongest tree has to be in the
    # comparison. Reporting only the cheap static model would understate the
    # classical side on exactly the arm being used to qualify the starved result.
    #
    # A history x coverage decomposition with lexical always on, so the mechanism
    # ("coverage stops being a constant") is measured rather than asserted:
    #   struct_lex       history + coverage + BM25
    #   static_lex       coverage + BM25
    #   struct_nocov_lex history + BM25
    #   static_nocov_lex BM25 only
    scores["xgboost_static_lex"] = models.XGBoostSelector(
        exclude_history=True, include_lexical=True, candidates_mode="full"
    ).scores(ctx)
    scores["xgboost_struct_nocov_lex"] = models.XGBoostSelector(
        exclude_coverage=True, include_lexical=True, candidates_mode="full"
    ).scores(ctx)
    scores["xgboost_struct"] = models.XGBoostSelector(candidates_mode="full").scores(ctx)
    scores["xgboost_struct_lex"] = models.XGBoostSelector(
        include_lexical=True, candidates_mode="full"
    ).scores(ctx)
    scores["semif_textonly_full"] = semif.load_scores(cache, ds)
    # A fitted-free combination, on the same footing as P5's rank average.
    scores["rankaverage_xgb_semif"] = rank_average(
        scores["xgboost_static_nocov_lex"], scores["semif_textonly_full"],
        candidates=candidates,
    )

    group = eval_group(ds, rows, candidates, scores,
                       reference=["xgboost_static_nocov_lex", "structural_rule",
                                  "xgboost_struct_lex", "xgboost_static_lex"])
    print(_table(group, f"Full-candidate starved arm (failures<={max_failures}, "
                        f"{len(rows)} held-out changes, budget = fraction of 1187 tests)"))
    return {
        "experiment": "full_starved",
        "max_failures": max_failures,
        "changes": int(len(rows)),
        "faults": int(sum(1 for i in rows if ds.changes[i].killing_tests)),
        "candidates": "full",
        **group,
    }

# --- experiment 5: is the transformer redundant? ---------------------------


def p5_redundancy() -> dict:
    """Does SemIf carry information the cheap structured model does not already have?

    Two arms, because the proposal as written has a hidden cost. Adding the SemIf
    score as an XGBoost *column* requires SemIf scores for the training window, and
    the cache only covers held-out changes (329k extra pairs, ~3.3 h, which the
    iteration-cost analysis rules out). Options:

    * ``trained`` -- restrict training to a SemIf-scored prefix of the training
      window and train the baseline on the same reduced rows, so the delta between
      with and without the column is attributable to the column. Weaker overall
      models, valid comparison.
    * ``rank_average`` -- fit nothing. Average the two selectors' per-change rank
      positions. If SemIf adds independent signal the average beats both parents;
      if it is redundant the average sits between them. Needs no training data and
      no extra scoring.

    The rank average is reported here; ``trained`` is run by ``--p5-trained`` when
    the reduced-window SemIf scores exist.
    """
    ds = dataset.build()
    candidates = dataset.candidate_mask(ds, "covered")
    rows = ds.test_idx
    ctx = build_context(ds)

    base = _selectors(ds, ctx, "covered")
    base["xgboost_struct"] = models.XGBoostSelector(candidates_mode="covered").scores(ctx)
    # Only caches that cover *every* held-out change may enter this comparison.
    # `semif_scores_ctl_after.jsonl` was scored on the starved subset only, so
    # including it here would score 323 of 464 changes as unscored and make it look
    # far worse than it is; it is evaluated separately on its own subset.
    base["semif_textonly"] = semif.load_scores(config.SEMIF_SCORES_FILE, ds)

    combos = {
        "rankaverage_xgb_semif": rank_average(
            base["xgboost_static_nocov_lex"], base["semif_textonly"], candidates=candidates
        ),
        "rankaverage_xgb_struct_semif": rank_average(
            base["xgboost_struct"], base["semif_textonly"], candidates=candidates
        ),
    }
    scores = dict(base)
    scores.update(combos)
    group = eval_group(
        ds, rows, candidates, scores,
        reference=["xgboost_static_nocov_lex", "semif_textonly"],
    )
    print(_table(group, "P5 redundancy (covered candidates, all 464 held-out faults)"))

    # Parents-vs-child deltas are the actual test, so report them explicitly.
    child_tests = {}
    for child, parent in [
        ("rankaverage_xgb_semif", "xgboost_static_nocov_lex"),
        ("rankaverage_xgb_struct_semif", "xgboost_struct"),
    ]:
        for other in (parent, "semif_textonly"):
            for b in BUDGETS:
                key = f"{child}|{b}|vs|{other}"
                a = evaluate.per_change_hits(scores[child], ds, rows, b, candidates)
                c = evaluate.per_change_hits(scores[other], ds, rows, b, candidates)
                child_tests[key] = evaluate.paired_bootstrap(a, c, N_BOOTSTRAP, config.SEED)
    print("\n  rank-average vs each parent (the redundancy test):")
    for key, stat in child_tests.items():
        print(f"    {key:<58} {stat['delta']:+.3f} "
              f"[{stat['lo']:+.3f}, {stat['hi']:+.3f}] p={stat['p_value']:.4f}")

    return {
        "experiment": "p5_redundancy",
        "candidates": "covered",
        "changes": int(len(rows)),
        "note": (
            "rank averages are fitted-free and leakage-safe; the trained-column "
            "variant needs SemIf scores over the training window"
        ),
        "parent_tests": child_tests,
        **group,
    }


def _fit_predict(
    ctx: models.Context,
    train_rows: np.ndarray,
    candidates: np.ndarray,
    feat_idx: list[int],
    extra: np.ndarray | None = None,
    seed: int = config.SEED,
):
    """Fit XGBoost on candidate pairs of ``train_rows``; predict the full grid.

    ``extra`` is an optional per-(change, test) score matrix. Unscored pairs are
    ``-1e9`` in the SemIf caches; they become NaN so XGBoost treats them as missing
    rather than as an extreme low value.
    """
    import xgboost as xgb

    mask = candidates[train_rows]

    def design(rows: np.ndarray, cols_mask: np.ndarray | None) -> np.ndarray:
        block = ctx.X[rows][:, :, feat_idx]
        extras = [ctx.bm25[rows]]
        if extra is not None:
            values = extra[rows].astype(np.float64).copy()
            values[np.abs(values) >= 1e8] = np.nan
            extras.append(values)
        if cols_mask is not None:
            block = block[cols_mask]
            extras = [values[cols_mask] for values in extras]
        flat = block.reshape(-1, len(feat_idx))
        return np.hstack([flat] + [values.reshape(-1, 1) for values in extras])

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.15, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, tree_method="hist", n_jobs=-1,
        random_state=seed, eval_metric="logloss",
    )
    model.fit(design(train_rows, mask), ctx.ds.labels[train_rows][mask].reshape(-1))
    X_all = design(np.arange(ctx.ds.n_changes), None)
    matrix = model.predict_proba(X_all)[:, 1].reshape(ctx.ds.n_changes, ctx.ds.n_tests)
    return matrix, model


def p5_trained(eval_fraction: float = 0.3) -> dict:
    """The proposal as written: SemIf as an extra XGBoost column.

    Why the split is inside the held-out window
    -------------------------------------------
    SemIf scores exist only for held-out changes (scoring the training window would
    cost 329k extra pairs, ~3.3 h). Rather than fit a blend weight on the evaluation
    rows, this uses a *temporal* split inside the held-out window: train on the
    first 70% of held-out changes, evaluate on the last 30%. The split respects the
    imposed change order and no evaluation row is ever trained on, so the "with
     column" vs "without column" delta is attributable to the column alone.

    Two feature families are tested because redundancy depends on what the tree
    already has: ``static_nocov_lex`` is the cheapest strong model (no coverage, no
    history, plus BM25), and ``struct_lex`` is the full feature set.
    """
    ds = dataset.build()
    candidates = dataset.candidate_mask(ds, "covered")
    ctx = build_context(ds)
    held = ds.test_idx
    split = int(round(len(held) * (1.0 - eval_fraction)))
    train_rows, eval_rows = held[:split], held[split:]
    semif_scores = semif.load_scores(config.SEMIF_SCORES_FILE, ds)

    families = {
        "static_nocov_lex": [
            i for i, name in enumerate(ctx.names)
            if name not in set(models.HISTORY_FEATURES) | set(models.COVERAGE_FEATURES)
        ],
        "struct_lex": list(range(len(ctx.names))),
    }

    scores: dict[str, np.ndarray] = {}
    importances: dict[str, dict[str, float]] = {}
    for family, feat_idx in families.items():
        for tag, extra in (("", None), ("_semif", semif_scores)):
            matrix, model = _fit_predict(ctx, train_rows, candidates, feat_idx, extra)
            key = f"{family}{tag}"
            scores[key] = matrix
            names = [ctx.names[i] for i in feat_idx] + ["bm25"]
            if extra is not None:
                names.append("semif")
            importances[key] = dict(
                sorted(
                    zip(names, model.feature_importances_.tolist()), key=lambda kv: -kv[1]
                )
            )

    group = eval_group(ds, eval_rows, candidates, scores, reference="static_nocov_lex")
    print(_table(
        group,
        f"P5 trained column (held-out-internal temporal split: "
        f"{len(train_rows)} train / {len(eval_rows)} eval changes)",
    ))

    family_tests: dict[str, dict] = {}
    for family in families:
        for b in BUDGETS:
            a = evaluate.per_change_hits(scores[f"{family}_semif"], ds, eval_rows, b, candidates)
            c = evaluate.per_change_hits(scores[family], ds, eval_rows, b, candidates)
            stat = evaluate.paired_bootstrap(a, c, N_BOOTSTRAP, config.SEED)
            stat["budget"] = b
            family_tests[f"{family}_semif|{b}"] = stat
    print("\n  adding the SemIf column, paired on the evaluation window:")
    for key, stat in family_tests.items():
        print(f"    {key:<40} {stat['delta']:+.3f} "
              f"[{stat['lo']:+.3f}, {stat['hi']:+.3f}] p={stat['p_value']:.4f}")
    for key, imp in importances.items():
        print(f"    importance {key:<28} {list(imp.items())[:3]}")

    return {
        "experiment": "p5_trained",
        "candidates": "covered",
        "train_changes": int(len(train_rows)),
        "eval_changes": int(len(eval_rows)),
        "eval_faults": int(sum(1 for i in eval_rows if ds.changes[i].killing_tests)),
        "family_tests": family_tests,
        "importances": importances,
        **group,
    }


def full_starved_seeds(seeds: tuple[int, ...] = (1, 2, 3, 4)) -> dict:
    """Is the full-suite starved correction a lucky XGBoost fit?

    The correction rests on only 43 held-out faults, so a single fit could in
    principle be favourable by chance. Refit the strongest tree under several seeds and
    report the paired delta against SemIf at each budget. Only the *model* seed varies:
    the dataset seed fixes the imposed change order and the temporal split, and the
    SemIf cache rows are keyed to that order, so varying it would invalidate the paired
    comparison rather than test it.

    SemIf itself is deterministic (greedy forward passes, no sampling), so there is no
    SemIf-side seed to vary.
    """
    cache = config.ARTIFACTS / "semif_scores_starved2_full.jsonl"
    ds = dataset.build()
    candidates = dataset.candidate_mask(ds, "full")
    mask = dataset.starved_mask(ds, max_failures=2)
    rows = ds.test_idx[mask[ds.test_idx]]
    ctx = build_context(ds)
    semif_scores = semif.load_scores(cache, ds)
    semif_hits = {
        b: evaluate.per_change_hits(semif_scores, ds, rows, b, candidates) for b in BUDGETS
    }

    out: dict[str, dict] = {}
    for seed in seeds:
        for model_name, kwargs in (
            ("xgboost_struct_lex", {"include_lexical": True}),
            ("xgboost_static_lex", {"include_lexical": True, "exclude_history": True}),
        ):
            tree = models.XGBoostSelector(
                candidates_mode="full", seed=seed, **kwargs
            ).scores(ctx)
            entry: dict[str, dict] = {}
            for b in BUDGETS:
                hits = evaluate.per_change_hits(tree, ds, rows, b, candidates)
                stat = evaluate.paired_bootstrap(semif_hits[b], hits, N_BOOTSTRAP, config.SEED)
                stat["budget"] = b
                stat["tree_recall"] = float(np.mean(list(hits.values())))
                stat["semif_recall"] = float(np.mean(list(semif_hits[b].values())))
                entry[f"b{b}"] = stat
            out[f"{model_name}|seed{seed}"] = entry
            cells = "  ".join(f"b{b}={entry[f'b{b}']['delta']:+.3f}" for b in BUDGETS)
            print(f"  {model_name} seed {seed}: SemIf minus tree  {cells}", flush=True)

    return {
        "experiment": "full_starved_seeds",
        "seeds": list(seeds),
        "changes": int(len(rows)),
        "note": (
            "delta is SemIf minus the tree; negative means the tree wins. Two trees are "
            "tested because the coverage+BM25 model without history is the strongest "
            "classical selector on this arm."
        ),
        "results": out,
    }


# --- experiment 2: instruction sweep --------------------------------------


def p2_instruction(max_failures: int = 5) -> dict:
    """Instruction sweep, evaluated at two starvation thresholds.

    The instruction caches are scored on the ``failures <= 5`` population, which is a
    superset of ``failures <= 2``, so the sparser (43-change) subset can be evaluated
    for free. Reporting both guards against the null being an artifact of the
    population: a wording that helped only in the densest part of the starved range
    would show up as a difference between the two thresholds.
    """
    ds = dataset.build()
    candidates = dataset.candidate_mask(ds, "covered")
    ctx = build_context(ds)

    base_scores = {
        "bm25_lexical": ctx.bm25,
        "xgboost_static_nocov_lex": models.XGBoostSelector(
            exclude_history=True, exclude_coverage=True, include_lexical=True,
            candidates_mode="covered",
        ).scores(ctx),
    }
    found = []
    scores = dict(base_scores)
    for name in ("default", *sorted(k for k in _instruction_names() if k != "default")):
        path = _instruction_cache(name, max_failures)
        if path.exists():
            scores[f"semif_{name}"] = semif.load_scores(path, ds)
            found.append(name)
    if not found:
        raise FileNotFoundError("no instruction caches present")
    reference = "semif_default" if "semif_default" in scores else f"semif_{found[0]}"

    groups: dict[str, dict] = {}
    for threshold in (max_failures, 2):
        mask = dataset.starved_mask(ds, max_failures=threshold)
        rows = ds.test_idx[mask[ds.test_idx]]
        key = f"starved{threshold}"
        groups[key] = eval_group(ds, rows, candidates, scores, reference=reference)
        groups[key]["changes"] = int(len(rows))
        groups[key]["faults"] = int(sum(1 for i in rows if ds.changes[i].killing_tests))
        print(_table(
            groups[key],
            f"P2 instruction sweep (starved failures<={threshold}, {len(rows)} held-out "
            f"changes, covered candidates)",
        ))
    return {
        "experiment": "p2_instruction",
        "max_failures": max_failures,
        "variants": found,
        **groups[f"starved{max_failures}"],
        "secondary_threshold": groups["starved2"],
    }


def _instruction_names() -> list[str]:
    from .semif_runner import INSTRUCTION_VARIANTS

    return list(INSTRUCTION_VARIANTS)


def _instruction_cache(name: str, max_failures: int) -> Path:
    if name == "default":
        # The existing text-only cache already covers every held-out change with
        # the default wording; the starved rows are a subset of it.
        return Path(config.SEMIF_SCORES_FILE)
    return config.ARTIFACTS / f"semif_scores_instr_{name}_starved{max_failures}.jsonl"


# --- experiment 3: code-embedding baseline --------------------------------


def p3_embed(device: str = "cpu", force: bool = False) -> dict:
    from . import embed

    ds = dataset.build()
    candidates_covered = dataset.candidate_mask(ds, "covered")
    candidates_full = dataset.candidate_mask(ds, "full")
    cache = config.ARTIFACTS / "embed_scores.npy"

    if cache.exists() and not force:
        scores_matrix = np.load(cache)
        print(f"[p3] loaded {cache}")
    else:
        scores_matrix = embed.build_scores(ds, device=device)
        np.save(cache, scores_matrix)
        print(f"[p3] saved {cache}")

    ctx = build_context(ds)
    covered = {
        "bm25_lexical": ctx.bm25,
        "semif_textonly": semif.load_scores(config.SEMIF_SCORES_FILE, ds),
        "embed_codebert": scores_matrix,
        "xgboost_static_nocov_lex": models.XGBoostSelector(
            exclude_history=True, exclude_coverage=True, include_lexical=True,
            candidates_mode="covered",
        ).scores(ctx),
    }
    group_covered = eval_group(
        ds, ds.test_idx, candidates_covered, covered, reference="bm25_lexical"
    )
    print(_table(group_covered, "P3 embedding baseline (covered candidates, 464 held-out faults)"))

    starved = dataset.starved_mask(ds, max_failures=5)
    rows = ds.test_idx[starved[ds.test_idx]]
    group_starved = eval_group(
        ds, rows, candidates_covered, covered, reference="bm25_lexical"
    )
    print(_table(group_starved, f"P3 embedding baseline (starved failures<=5, {len(rows)} changes)"))

    # The full-suite arm is where a semantic ranker could actually pay off, because
    # the covered mask is the crutch that lets structure win cheaply.
    group_full = eval_group(
        ds, rows, candidates_full,
        {
            "bm25_lexical": ctx.bm25,
            "embed_codebert": scores_matrix,
            "structural_rule": models.StructuralRuleSelector().scores(ctx),
            "coverage": models.CoverageSelector().scores(ctx),
        },
        reference="bm25_lexical",
    )
    print(_table(group_full, f"P3 embedding baseline (starved failures<=5, full 1187 candidates)"))

    return {
        "experiment": "p3_embed",
        "model": config.EMBED_MODEL,
        "revision": config.EMBED_MODEL_REVISION,
        "covered": group_covered,
        "starved_covered": group_starved,
        "starved_full": group_full,
    }


# --- experiment 1': direct mode pairwise ----------------------------------


def p1_direct(max_failures: int | None = 2) -> dict:
    """P1: does direct mode change the verdict?

    Defaults to the starved `failures <= 2` population because direct mode is *slow*:
    Qwen3.5-4B is a hybrid gated-delta-net model whose custom kernels fall back to
    reference PyTorch (`causal_conv1d` / `flash-linear-attention` absent here), giving
    ~1.3 pairs/s against 30 pairs/s for the reranker -- a 23x penalty that makes the
    full held-out grid a ~16 h job. The 43-change starved population is 8,329 pairs
    (104 min measured) and is the same population as the documented covered-candidate
    starved arm, so the comparison is paired and directly interpretable.

    The cache is resumable, so a partial run is still usable: rows whose candidates
    are not all scored are dropped, because ``load_scores`` fills missing pairs with
    -1e9 and a partially scored change would rank as if nothing matched.
    """
    ds = dataset.build()
    candidates = dataset.candidate_mask(ds, "covered")
    rows = ds.test_idx
    if max_failures is not None:
        mask = dataset.starved_mask(ds, max_failures=max_failures)
        rows = ds.test_idx[mask[ds.test_idx]]
    cache = config.ARTIFACTS / (
        "semif_direct_heldout_covered.jsonl" if max_failures is None
        else f"semif_direct_starved{max_failures}_covered.jsonl"
    )
    if not cache.exists():
        raise FileNotFoundError(f"missing {cache}; run rts.direct_runner first")

    from .semif_runner import load_done_keys

    done = load_done_keys(cache)
    complete = [
        int(i) for i in rows
        if all((int(i), int(j)) in done for j in np.flatnonzero(candidates[i]))
    ]
    dropped = len(rows) - len(complete)
    rows = np.array(complete, dtype=np.int64)
    print(f"[p1] {len(rows)} of {len(rows) + dropped} changes fully scored in the cache")

    ctx = build_context(ds)
    scores = _selectors(ds, ctx, "covered")
    scores["semif_reranker"] = semif.load_scores(config.SEMIF_SCORES_FILE, ds)
    scores["semif_direct_pairwise"] = semif.load_scores(cache, ds)
    group = eval_group(
        ds, rows, candidates, scores,
        reference=["semif_reranker", "xgboost_static_nocov_lex"],
    )
    print(_table(group, f"P1 direct mode pairwise (Qwen3.5-4B), {len(rows)} held-out changes"))
    return {
        "experiment": "p1_direct",
        "max_failures": max_failures,
        "changes": int(len(rows)),
        "changes_unscored": int(dropped),
        "faults": int(sum(1 for i in rows if ds.changes[i].killing_tests)),
        "model": config.DIRECT_MODEL,
        "revision": config.DIRECT_MODEL_REVISION,
        "note": "direct mode measured at ~1.25 pairs/s; scope limited by throughput, not design",
        **group,
    }


# --- driver ---------------------------------------------------------------


def main(only: list[str] | None = None) -> dict:
    report_path = config.ARTIFACTS / "variations.json"
    report: dict = {}
    if report_path.exists():
        report = json.loads(report_path.read_text())

    todo = only or ["full_starved", "full_starved5", "full_starved_seeds", "p5", "p5_trained",
                    "p2", "p3", "p1"]
    for name in todo:
        print("\n" + "=" * 78, flush=True)
        print(f"EXPERIMENT: {name}", flush=True)
        print("=" * 78, flush=True)
        try:
            if name == "full_starved":
                report["full_starved"] = full_starved()
            elif name == "full_starved5":
                report["full_starved5"] = full_starved(max_failures=5)
            elif name == "full_starved_seeds":
                report["full_starved_seeds"] = full_starved_seeds()
            elif name == "p5":
                report["p5"] = p5_redundancy()
            elif name == "p5_trained":
                report["p5_trained"] = p5_trained()
            elif name == "p2":
                report["p2"] = p2_instruction()
            elif name == "p3":
                report["p3"] = p3_embed()
            elif name == "p1":
                report["p1"] = p1_direct()
            else:
                raise ValueError(f"unknown experiment: {name}")
        except FileNotFoundError as exc:
            print(f"[skip] {name}: {exc}", flush=True)
            report.setdefault("skipped", {})[name] = str(exc)
        report_path.write_text(json.dumps(report, indent=2))
        print(f"[report] wrote {report_path}", flush=True)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None,
                        choices=["full_starved", "full_starved5", "full_starved_seeds", "p5",
                                 "p5_trained", "p2", "p3", "p1"])
    args = parser.parse_args()
    main(args.only)
