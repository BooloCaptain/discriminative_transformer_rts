"""Render the variation experiments into ``artifacts/variations.json``.

Six of the seven sections are declared in :mod:`rts.studies` and rendered here from the reports
the layer produces: the full-candidate cold start condition at both starvation thresholds, the XGBoost
seed refit, the instruction-wording sweep, the code-embedding baseline, and the redundancy test.
Each section is its own experiment rather than one grid, because they differ in the controls a control
is allowed to differ in -- ``candidate_sets`` (the cold start conditions use the full pool, the instruction and
redundancy conditions the coverage set) and ``budgets`` -- while sharing the dataset, the split, the
feature block and therefore one score cache.

Two sections are **not** migrated, and both keep their original implementation below:

* ``p5_trained`` needs an evaluation window *inside* the held-out tail and a NaN convention for
  unscored pairs. Both are expressible -- a ``Split`` value can carry any train/test pair, and
  the NaN rule belongs in the extra-column design -- but each is a change to what an existing
  concept means, so they are a deliberate next step rather than a migration detail.
* ``p1_direct`` cannot be reproduced at all: its cache (``semif_direct_cold_start2_coverage_restricted.jsonl``)
  is absent from the artifacts, and the original raises ``FileNotFoundError`` without it. The
  code is kept because it is the record of what was run and would work again if the cache were
  regenerated; the layer would report the same case as an *undefined* design point rather than a crash.

Usage::

    python -m rts.render.variations                  # every section whose inputs are present
    python -m rts.render.variations --only p2
    python -m rts.render.variations --no-save
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .. import config, evaluate, features, studies
from ..data import accessors, contract, datasets, splits, subsets
from ..experiment import FACTOR_MODEL, FACTOR_SUBSET, run
from ..model import rankers, semif

BUDGETS = studies.VARIATION_BUDGETS
PROBE = studies.VARIATION_PROBE
N_BOOTSTRAP = studies.VARIATION_RESAMPLES

# The variation experiments reproduce the documented condition, so they explicitly ask for the features
# the harness leaves off for a synthetic-order dataset (§2.4).
STRUCTURED_TEMPORAL = True


# --- rendering a report into the recorded shape ----------------------------


def _group(report, subset: str, references) -> dict:
    """One ``eval_group``-shaped block: the recall table plus the paired contrasts.

    The key format ``{model}|{budget}|vs|{reference}`` is the artifact's, not the layer's: the
    layer records one contrast per (reference, probe budget) as records with a group, and this
    is where they become the flat map a reader and ``figures.py`` expect.
    """
    results: dict[str, list[dict]] = {}
    for result in report.design_points:
        if result.subset == subset:
            results[result.design_point.name(FACTOR_MODEL)] = result.results

    refs = list(references)
    contrasts: dict[str, dict] = {"references": refs}
    for record in report.contrasts:
        if not record.get("measured") or record["reference"] not in refs:
            continue
        if record["group"].get(FACTOR_SUBSET) != subset:
            continue
        budget = record["probe_budget"]
        contrasts[f"{record['design_point']}|{budget}|vs|{record['reference']}"] = {
            "delta": record["delta"],
            "lo": record["lo"],
            "hi": record["hi"],
            "p_value": record["p_value"],
            "n": record["n"],
            "budget": budget,
            "reference": record["reference"],
        }
    return {"results": results, "contrasts": contrasts}


def _table(group: dict, title: str, name_width: int = 34) -> str:
    results = group["results"]
    lines = [f"\n{title}"]
    lines.append(f"  {'model':<{name_width}}" + "".join(f"{b:>9.2f}" for b in BUDGETS))
    for name, res in results.items():
        design_points = "".join(f"{r['recall']:>9.3f}" for r in res)
        lines.append(f"  {name:<{name_width}}{design_points}")
    for key, stat in group.get("contrasts", {}).items():
        if key == "references":
            continue
        name, budget, _, ref = key.split("|")
        lines.append(
            f"    {name:<{name_width - 2}} b{float(budget):<5.2f} vs {ref:<26} "
            f"{stat['delta']:+.3f} [{stat['lo']:+.3f}, {stat['hi']:+.3f}] "
            f"p={stat['p_value']:.4f} n={stat['n']}"
        )
    return "\n".join(lines)


# --- experiment 1: the full-candidate cold start condition ---------------------------


def cold_start(max_failures: int = 2) -> dict:
    """Re-evaluate the cold start condition over the full 1187-test candidate set.

    ``max_failures=2`` is the 43-fault condition that produced the original positive result;
    ``max_failures=5`` is the 141-fault decision-grade confirmation. The second is a superset of
    the first, which is why one cache scored for it covers both.
    """
    subset = subsets.cold_start(max_failures)
    report = run(studies.cold_start_condition(max_failures), save=False, verbose=False)

    references = (
        "xgboost_static_nocov_lex",
        "structural_rule",
        "xgboost_struct_lex",
        "xgboost_static_lex",
    )
    # The two counts come from the report, so they are the ones the run itself used: ``changes``
    # is the subset's rows in the window and ``faults`` is what every recall was averaged
    # over. They used to be re-derived from the dataset, and this function used to run the whole
    # condition twice -- the second `report = run(...)` was a duplicated line, so the section cost
    # double for an identical report.
    changes, faults = report.subset_size(subset.name)
    group = _group(report, subset.name, references)
    print(_table(
        group,
        f"Full-candidate cold_start condition (failures<={max_failures}, {changes} held-out changes, "
        "budget = fraction of 1187 tests)",
    ))
    return {
        "experiment": "cold_start",
        "max_failures": max_failures,
        "changes": changes,
        "faults": faults,
        "candidate_policy": "full",
        **group,
    }


# --- experiment 1'': is the correction a lucky fit? -------------------------


def cold_start_seeds(seeds: tuple[int, ...] = studies.SEED_SWEEP) -> dict:
    """Refit the cold start condition's two strongest trees under several *model* seeds.

    The correction rests on only 43 held-out faults, so a single fit could in principle be
    favourable by chance. Only the model seed varies: the run seed fixes the synthetic change order
    and the temporal split, and the caches are keyed to that order, so varying it would invalidate
    the paired contrast rather than test it. SemIf is deterministic -- greedy forward passes, no
    sampling -- so there is no SemIf-side seed to vary.
    """
    tree_names = ("xgboost_struct_lex", "xgboost_static_lex")
    shared: dict = {}
    out: dict[str, dict] = {}
    spec = subsets.cold_start(2)
    subset = spec.name
    for seed in seeds:
        experiment = studies.cold_start_seeds_condition(2, model_seed=seed)
        # A model seed changes what the model computes, so the score cache is keyed by it and
        # each refit trains afresh; only the SemIf matrix is shared.
        report = run(experiment, scores=shared, save=False, verbose=False)
        by_name = {c.design_point.name(FACTOR_MODEL): c for c in report.design_points if c.subset == subset}
        semif = by_name["semif_textonly_full"]
        records = {
            (r["design_point"], r["reference"], r["probe_budget"]): r
            for r in report.contrasts
            if r.get("measured") and r["group"].get(FACTOR_SUBSET) == subset
        }
        for tree_name in tree_names:
            tree = by_name[tree_name]
            entry: dict[str, dict] = {}
            for budget in BUDGETS:
                # The condition declares the *trees* as references, so the SemIf design point is the one
                # paired and the record's delta is already `semif - tree`.
                record = records[("semif_textonly_full", tree_name, budget)]
                tree_row = next(r for r in tree.results if r["budget"] == budget)
                semif_row = next(r for r in semif.results if r["budget"] == budget)
                entry[f"b{budget}"] = {
                    "delta": record["delta"],
                    "lo": record["lo"],
                    "hi": record["hi"],
                    "p_value": record["p_value"],
                    "n": record["n"],
                    "budget": budget,
                    "tree_recall": tree_row["recall"],
                    "semif_recall": semif_row["recall"],
                }
            # The recorded convention ("SemIf minus the tree") therefore falls out of the
            # declaration rather than being applied by the renderer.
            out[f"{tree_name}|seed{seed}"] = entry
            design_points = "  ".join(f"b{b}={entry[f'b{b}']['delta']:+.3f}" for b in BUDGETS)
            print(f"  {tree_name} seed {seed}: SemIf minus tree  {design_points}", flush=True)

    changes, _ = report.subset_size(subset)
    return {
        "experiment": "cold_start_seeds",
        "seeds": list(seeds),
        "changes": changes,
        "note": (            "delta is SemIf minus the tree; negative means the tree wins. Two trees are "
            "tested because the coverage+BM25 model without temporal features is the strongest "
            "classical ranker on this condition."
        ),
        "results": out,
    }


# --- experiment 2: the instruction sweep -----------------------------------


def p2_instruction(max_failures: int = 5, *, secondary: int = 2) -> dict:
    """Instruction sweep, evaluated at two starvation thresholds.

    The instructions are scored on the ``failures <= 5`` subset, which is a superset of
    ``failures <= 2``, so the sparser subset is evaluated for free. Reporting both guards against
    the null being an artifact of the subset: a wording that helped only in the densest part
    of the cold start range would show up as a difference between the two thresholds.
    """
    report = run(studies.instruction_condition(max_failures, secondary=secondary), save=False, verbose=False)

    groups: dict[str, dict] = {}
    for threshold in (max_failures, secondary):
        spec = subsets.cold_start(threshold)
        changes, faults = report.subset_size(spec.name)
        group = _group(report, spec.name, ("semif_default",))
        group["changes"] = changes
        group["faults"] = faults
        groups[spec.name] = group
        print(_table(
            group,
            f"P2 instruction sweep (cold_start failures<={threshold}, {changes} held-out "
            "changes, coverage-restricted candidate sets)",
        ))

    primary_name = subsets.cold_start(max_failures).name
    secondary_name = subsets.cold_start(secondary).name
    primary = groups[primary_name]
    # A variant is reported when its cache was present, which the layer reports as a *measured*
    # design point rather than as the driver's FileNotFoundError-and-skip.
    variants = [
        level.name.removeprefix("semif_")
        for level in studies.instruction_model_factor(max_failures).levels
        if level.name.startswith("semif_") and level.name in primary["results"]
    ]
    return {
        "experiment": "p2_instruction",
        "max_failures": max_failures,
        "variants": variants,
        **primary,
        "secondary_threshold": groups[secondary_name],
    }


# --- experiment 3: the code-embedding baseline -----------------------------


def p3_embed(device: str = "cpu", force: bool = False) -> dict:
    """The code-encoder baseline, over the coverage set and over the full pool.

    Two conditions, because ``candidate_sets`` is a control and the full-pool contrast is the one where a
    semantic ranker could pay off -- the coverage-restricted mask is the crutch that lets structure win
    cheaply. The matrix itself is produced by ``rts.model.embed``; if it is absent the layer reports the
    design points that read it as undefined, so this only needs to build it when asked.
    """
    from ..model import embed

    ds = studies.dataset("mutmut")
    cache = studies.embed_cache()
    if force or not cache.exists():
        np.save(cache, embed.build_scores(ds, device=device))
        print(f"[p3] wrote {cache}")

    shared: dict = {}
    covered = run(studies.embedding_condition("coverage_restricted"), scores=shared, save=False, verbose=False)
    full = run(studies.embedding_condition("full"), scores=shared, save=False, verbose=False)

    group_covered = _group(covered, "detectable", ("bm25_lexical",))
    section_cold_start_coverage_restricted = _group(
        covered, subsets.cold_start(5).name, ("bm25_lexical",)
    )
    section_cold_start_full = _group(full, subsets.cold_start(5).name, ("bm25_lexical",))
    print(_table(group_covered, "P3 embedding baseline (coverage_restricted candidate_sets, 464 held-out faults)"))
    print(_table(section_cold_start_coverage_restricted, "P3 embedding baseline (cold start failures<=5, coverage-restricted)"))
    print(_table(section_cold_start_full, "P3 embedding baseline (cold_start failures<=5, full 1187)"))

    return {
        "experiment": "p3_embed",
        "model": config.EMBED_MODEL,
        "revision": config.EMBED_MODEL_REVISION,
        "coverage_restricted": group_covered,
        "cold_start_coverage_restricted": section_cold_start_coverage_restricted,
        "cold_start_full": section_cold_start_full,
    }


# --- experiment 5: is the transformer redundant? ---------------------------


def p5_redundancy() -> dict:
    """Does SemIf carry information the cheap structured model does not already have?

    Two routes were considered. Adding the SemIf score as an XGBoost *column* needs scores for
    the training window, and the cache covers only held-out changes (329k extra pairs, ~3.3 h),
    so that route is the separate ``p5_trained`` section. This one fits nothing: it averages the
    two rankers' per-change rank positions, so if SemIf adds independent signal the average
    beats both parents and if it is redundant the average sits between them.

    Only caches covering *every* held-out change may enter: ``semif_scores_ctl_after.jsonl`` was
    scored on the cold start subset, so including it here would score 323 of 464 changes as unscored
    and make it look far worse than it is.
    """
    report = run(studies.redundancy_condition(), save=False, verbose=False)

    compared = ("xgboost_static_nocov_lex", "semif_textonly")
    group = _group(report, "detectable", compared)
    changes = int(report.describe()["test_changes"])
    print(_table(group, "P5 redundancy (coverage_restricted candidate_sets, all 464 held-out faults)"))

    # Parents-versus-child is the actual test, so report those pairs explicitly rather than
    # leaving a reader to find them among the eighty contrasts. The third reference is
    # declared for exactly these pairs and is not part of the group's contrast map.
    pairs = (
        ("rankaverage_xgb_semif", ("xgboost_static_nocov_lex", "semif_textonly")),
        ("rankaverage_xgb_struct_semif", ("xgboost_struct", "semif_textonly")),
    )
    records = {
        (r["design_point"], r["reference"], r["probe_budget"]): r
        for r in report.contrasts
        if r.get("measured") and r["group"].get(FACTOR_SUBSET) == "detectable"
    }
    parent_tests: dict[str, dict] = {}
    for child, others in pairs:
        for other in others:
            for budget in BUDGETS:
                record = records[(child, other, budget)]
                parent_tests[f"{child}|{budget}|vs|{other}"] = {
                    "delta": record["delta"],
                    "lo": record["lo"],
                    "hi": record["hi"],
                    "p_value": record["p_value"],
                    "n": record["n"],
                }
    print("\n  rank-average vs each parent (the redundancy test):")
    for key, stat in parent_tests.items():
        print(f"    {key:<58} {stat['delta']:+.3f} "
              f"[{stat['lo']:+.3f}, {stat['hi']:+.3f}] p={stat['p_value']:.4f}")

    return {
        "experiment": "p5_redundancy",
        "candidate_policy": "coverage_restricted",
        "changes": changes,
        "note": (
            "rank averages are fitted-free and leakage-safe; the trained-column "
            "variant needs SemIf scores over the training window"
        ),
        "parent_tests": parent_tests,
        **group,
    }


# ===========================================================================
# NOT MIGRATED. The two sections below keep their original implementation, for
# the reasons in the module docstring. They are the only users of the helpers
# that follow.
# ===========================================================================


def build_context(ds: contract.Dataset) -> rankers.Context:
    matrix = features.structured(ds, history=STRUCTURED_TEMPORAL)
    split = splits.make_split(ds)
    bm25 = features.text.build_bm25_scores(ds)
    return rankers.Context(ds=ds, features=matrix, split=split, bm25=bm25)


def eval_group(
    ds: contract.Dataset,
    rows: np.ndarray,
    candidate_sets: np.ndarray,
    scores_by_name: dict[str, np.ndarray],
    budgets: tuple[float, ...] = BUDGETS,
    reference: str | list[str] | None = None,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = config.SEED,
) -> dict:
    """Recall sweep plus paired bootstrap against each reference. Retained for the two sections

    that are not migrated; everything else gets this shape from the layer, which is the point of
    the migration.
    """
    results: dict[str, list[dict]] = {}
    hits: dict[str, dict[float, dict[int, bool]]] = {}
    for name, scores in scores_by_name.items():
        results[name] = evaluate.results_to_dicts(
            evaluate.evaluate(
                scores, ds, rows, budgets=budgets, n_bootstrap=1000, seed=seed,
                candidate_sets=candidate_sets,
            )
        )
        hits[name] = {
            b: evaluate.per_change_hits(scores, ds, rows, b, candidate_sets) for b in budgets
        }

    references = [reference] if isinstance(reference, str) else list(reference or [])
    references = [r for r in references if r in scores_by_name]
    contrasts: dict[str, dict] = {"references": references}
    for ref in references:
        for name in scores_by_name:
            if name == ref:
                continue
            for b in budgets:
                stat = evaluate.paired_bootstrap(hits[name][b], hits[ref][b], n_bootstrap, seed)
                stat["budget"] = b
                stat["reference"] = ref
                contrasts[f"{name}|{b}|vs|{ref}"] = stat
    return {"results": results, "contrasts": contrasts}


def _selectors(ds: contract.Dataset, ctx: rankers.Context, candidate_policy: str) -> dict[str, np.ndarray]:
    """The classical reference conditions, all trained on the same candidate pairs."""
    return {
        "random": rankers.RandomRanker().scores(ctx),
        "recency": rankers.RecencyRanker().scores(ctx),
        "failure_rate": rankers.FailureRateRanker().scores(ctx),
        "coverage": rankers.CoverageRanker().scores(ctx),
        "structural_rule": rankers.StructuralRuleRanker().scores(ctx),
        "bm25_lexical": ctx.bm25,
        "xgboost_static_nocov_lex": rankers.XGBoostRanker(
            exclude_temporal=True, exclude_coverage=True, include_lexical=True,
            candidate_policy=candidate_policy,
        ).scores(ctx),
    }


def _fit_predict(
    ctx: rankers.Context,
    train_rows: np.ndarray,
    candidate_sets: np.ndarray,
    feat_idx: list[int],
    extra: np.ndarray | None = None,
    seed: int = config.SEED,
):
    """Fit XGBoost on candidate pairs of ``train_rows``; predict the full grid.

    ``extra`` is an optional per-(change, test) score matrix. Unscored pairs are ``-1e9`` in the
    SemIf caches; they become NaN so XGBoost treats them as missing rather than as an extreme low
    value. That convention is one of the two reasons this section is not yet migrated.
    """
    import xgboost as xgb

    mask = candidate_sets[train_rows]

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
    model.fit(design(train_rows, mask), accessors.labels(ctx.ds)[train_rows][mask].reshape(-1))
    X_all = design(np.arange(ctx.ds.n_changes), None)
    matrix = model.predict_proba(X_all)[:, 1].reshape(ctx.ds.n_changes, ctx.ds.n_tests)
    return matrix, model


def p5_trained(eval_fraction: float = 0.3) -> dict:
    """The proposal as written: SemIf as an extra XGBoost column.

    Why the split is inside the held-out window
    -------------------------------------------
    SemIf scores exist only for held-out changes (scoring the training window would cost 329k
    extra pairs, ~3.3 h). Rather than fit a blend weight on the evaluation rows, this uses a
    *temporal* split inside the held-out window: train on the first 70% of held-out changes,
    evaluate on the last 30%. The split respects the synthetic change order and no evaluation row is
    ever trained on, so the "with column" versus "without column" delta is attributable to the
    column alone.

    Two feature families are tested because redundancy depends on what the tree already has:
    ``static_nocov_lex`` is the cheapest strong model (no coverage, no history, plus BM25), and
    ``struct_lex`` is the full feature set.
    """
    ds = datasets.marshmallow()
    held_split = splits.make_split(ds)
    candidate_sets = accessors.candidate_sets(ds, "coverage_restricted")
    ctx = build_context(ds)
    held = held_split.test_idx
    split = int(round(len(held) * (1.0 - eval_fraction)))
    train_rows, eval_rows = held[:split], held[split:]
    semif_scores = semif.load_scores(config.SEMIF_SCORES_FILE, ds)

    families = {
        "static_nocov_lex": [
            i for i, name in enumerate(ctx.names)
            if name not in set(rankers.TEMPORAL_FEATURES) | set(rankers.COVERAGE_FEATURES)
        ],
        "struct_lex": list(range(len(ctx.names))),
    }

    scores: dict[str, np.ndarray] = {}
    importances: dict[str, dict[str, float]] = {}
    for family, feat_idx in families.items():
        for tag, extra in (("", None), ("_semif", semif_scores)):
            matrix, model = _fit_predict(ctx, train_rows, candidate_sets, feat_idx, extra)
            key = f"{family}{tag}"
            scores[key] = matrix
            names = [ctx.names[i] for i in feat_idx] + ["bm25"]
            if extra is not None:
                names.append("semif")
            importances[key] = dict(
                sorted(zip(names, model.feature_importances_.tolist()), key=lambda kv: -kv[1])
            )

    group = eval_group(ds, eval_rows, candidate_sets, scores, reference="static_nocov_lex")
    print(_table(
        group,
        "P5 trained column (held-out-internal temporal split: "
        f"{len(train_rows)} train / {len(eval_rows)} eval changes)",
    ))

    family_tests: dict[str, dict] = {}
    for family in families:
        for b in BUDGETS:
            a = evaluate.per_change_hits(scores[f"{family}_semif"], ds, eval_rows, b, candidate_sets)
            c = evaluate.per_change_hits(scores[family], ds, eval_rows, b, candidate_sets)
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
        "candidate_policy": "coverage_restricted",
        "train_changes": int(len(train_rows)),
        "eval_changes": int(len(eval_rows)),
        "eval_faults": int(sum(1 for i in eval_rows if ds.killing_tests(ds.changes[i]))),
        "family_tests": family_tests,
        "importances": importances,
        **group,
    }


def p1_direct(max_failures: int | None = 2) -> dict:
    """P1: does direct mode change the verdict? -- NOT REPRODUCIBLE HERE.

    Defaults to the cold start ``failures <= 2`` subset because direct mode is *slow*: Qwen3.5-4B
    is a hybrid gated-delta-net model whose custom kernels fall back to reference PyTorch
    (``causal_conv1d`` / ``flash-linear-attention`` absent here), giving ~1.3 pairs/s against 30
    pairs/s for the reranker -- a 23x penalty that makes the full held-out grid a ~16 h job.

    The cache it reads (``semif_direct_cold_start2_coverage_restricted.jsonl``) is **absent** from the
    artifacts, so this raises. It is kept as the record of what was run, and because it would work
    again if the cache were regenerated.
    """
    ds = datasets.marshmallow()
    split = splits.make_split(ds)
    candidate_sets = accessors.candidate_sets(ds, "coverage_restricted")
    rows = split.test_idx
    if max_failures is not None:
        mask = subsets.cold_start_mask(ds, max_failures=max_failures)
        rows = split.test_idx[mask[split.test_idx]]
    cache = config.ARTIFACTS / (
        "semif_direct_heldout_coverage_restricted.jsonl" if max_failures is None
        else f"semif_direct_cold_start{max_failures}_coverage_restricted.jsonl"
    )
    if not cache.exists():
        raise FileNotFoundError(f"missing {cache}; run rts.model.direct_runner first")

    from ..model.semif_runner import load_done_keys

    done = load_done_keys(cache)
    complete = [
        int(i) for i in rows
        if all((int(i), int(j)) in done for j in np.flatnonzero(candidate_sets[i]))
    ]
    dropped = len(rows) - len(complete)
    rows = np.array(complete, dtype=np.int64)
    print(f"[p1] {len(rows)} of {len(rows) + dropped} changes fully scored in the cache")

    ctx = build_context(ds)
    scores = _selectors(ds, ctx, "coverage_restricted")
    scores["semif_reranker"] = semif.load_scores(config.SEMIF_SCORES_FILE, ds)
    scores["semif_direct_pairwise"] = semif.load_scores(cache, ds)
    group = eval_group(
        ds, rows, candidate_sets, scores,
        reference=["semif_reranker", "xgboost_static_nocov_lex"],
    )
    print(_table(group, f"P1 direct mode pairwise (Qwen3.5-4B), {len(rows)} held-out changes"))
    return {
        "experiment": "p1_direct",
        "max_failures": max_failures,
        "changes": int(len(rows)),
        "changes_unscored": int(dropped),
        "faults": int(sum(1 for i in rows if ds.killing_tests(ds.changes[i]))),
        "model": config.DIRECT_MODEL,
        "revision": config.DIRECT_MODEL_REVISION,
        "note": "direct mode measured at ~1.25 pairs/s; scope limited by throughput, not design",
        **group,
    }


# --- driver ---------------------------------------------------------------

SECTIONS = {
    "cold_start": lambda: cold_start(2),
    "cold_start5": lambda: cold_start(5),
    "cold_start_seeds": cold_start_seeds,
    "p2": p2_instruction,
    "p3": p3_embed,
    "p5": p5_redundancy,
    "p5_trained": p5_trained,
    "p1": p1_direct,
}


def main(only: list[str] | None = None, save: bool = True) -> dict:
    out_path = config.ARTIFACTS / "variations.json"
    report: dict = {}
    if out_path.exists():
        report = json.loads(out_path.read_text())

    names = [n for n in SECTIONS if only is None or n in only]
    # Skip a section whose inputs are absent rather than crashing the whole run: a undefined
    # design point is a finding, but a section that cannot run at all should not cost the others.
    if only is None:
        usable = []
        for name in names:
            if name in ("p1", "p5_trained"):
                print(f"EXPERIMENT SKIPPED (not migrated): {name}", flush=True)
                continue
            usable.append(name)
        names = usable

    for name in names:
        print(f"EXPERIMENT: {name}", flush=True)
        try:
            report[name] = SECTIONS[name]()
        except FileNotFoundError as exc:
            print(f"  skipped: {exc}", flush=True)
            continue
        if save:
            out_path.write_text(json.dumps(report, indent=2))
            print(f"[report] wrote {out_path}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None, choices=sorted(SECTIONS))
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    main(only=args.only, save=not args.no_save)
