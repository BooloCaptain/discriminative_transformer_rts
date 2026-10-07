"""The core experiment: every baseline and model the harness ships, on real data.

Where ``examples/example.py`` is the minimal template over the fixture dataset, this is the
worked sweep over a real one. It declares, purely:

* the baselines -- random, recency, failure rate, coverage, the structural rule, and BM25;
* the lexical shuffle controls (change-shuffled, test-shuffled), which test whether BM25 is
  conditioning on the change or on a change-independent test prior;
* the gradient-boosted trees over the structured features, including the no-coverage and
  with-lexical ablations;
* the SemIf reranker, which reads a cache if one exists and otherwise scores live, and which
  is skipped on a CPU-only run because it is a GPU design point;
* two averaging subsets -- the fault-bearing default and the cold-start proxy.

Run it with the checkout present (it is gitignored):

    python -m examples.core_experiment          # CPU: baselines and trees
    python -m examples.core_experiment --gpu     # also run SemIf (needs a GPU)

For a sweep that needs no checkout at all, see ``examples/example.py``.
"""

from __future__ import annotations

import argparse

from examples import config, datasets, rankers, subsets
from rts import features
from rts.data import subsets as kernel_subsets
from rts.experiment import (
    FACTOR_BUDGET,
    FACTOR_DATASET,
    FACTOR_FEATURES,
    FACTOR_MODEL,
    FACTOR_SPLIT,
    FACTOR_SUBSET,
    Contrast,
    Experiment,
    Factor,
    budget_level,
    constant,
    run,
    split_level,
)
from rts.model.rankers import RandomRanker

#: The cheap rankers: no fitting, one score per (change, test) pair.
BASELINES = (
    ("random", RandomRanker()),
    ("recency", rankers.RecencyRanker()),
    ("failure_rate", rankers.FailureRateRanker()),
    ("coverage", rankers.CoverageRanker()),
    ("structural_rule", rankers.StructuralRuleRanker()),
    ("bm25", rankers.LexicalRanker()),
    ("bm25_change_shuffled", rankers.LexicalRanker(shuffle_changes=True)),
    ("bm25_test_shuffled", rankers.LexicalRanker(shuffle_tests=True)),
)

#: The trees, and the ablations that say which feature family carries their recall.
TREES = (
    ("xgboost_struct", rankers.XGBoostRanker()),
    ("xgboost_struct_lex", rankers.XGBoostRanker(include_lexical=True)),
    ("xgboost_static_lex", rankers.XGBoostRanker(exclude_temporal=True, include_lexical=True)),
    (
        "xgboost_static_nocov_lex",
        rankers.XGBoostRanker(
            exclude_temporal=True, exclude_coverage=True, include_lexical=True
        ),
    ),
)


def model_factor() -> Factor:
    """Every model as a level. The reranker is a GPU level; a CPU run reports it undefined."""
    levels = [constant(name, ranker) for name, ranker in BASELINES]
    levels += [constant(name, ranker) for name, ranker in TREES]
    levels.append(constant("semif_reranker", rankers.SemIfRanker(), tier="gpu"))
    return Factor(FACTOR_MODEL, tuple(levels), note="baselines, trees, and the reranker")


def experiment() -> Experiment:
    return Experiment(
        name="core.marshmallow",
        datasets=Factor(FACTOR_DATASET, (constant("marshmallow", datasets.marshmallow()),)),
        features=Factor(FACTOR_FEATURES, (constant("structured", features.STRUCTURED),)),
        models=model_factor(),
        subsets=Factor(
            FACTOR_SUBSET,
            (
                constant("detectable", kernel_subsets.DETECTABLE),
                constant("cold_start5", subsets.cold_start(5)),
            ),
        ),
        splits=Factor(FACTOR_SPLIT, (split_level(train_fraction=0.8),)),
        budgets=Factor(
            FACTOR_BUDGET, (budget_level(0.05), budget_level(0.1), budget_level(0.2))
        ),
        contrasts=(
            Contrast(FACTOR_MODEL, "random", 0.05),
            Contrast(FACTOR_MODEL, "bm25", 0.05),
        ),
        note="every baseline and model, on the mutation dataset",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="also spend the gpu tier (SemIf)")
    args = parser.parse_args()

    if not config.SUT.exists():
        raise SystemExit(
            f"the marshmallow checkout is missing at {config.SUT}; this example needs it "
            "(it is gitignored). Use examples/example.py for a checkout-free sweep."
        )

    report = run(
        experiment(), tiers=("cpu", "gpu") if args.gpu else ("cpu",), verbose=False
    )
    print(report.format_table())
    print(f"\nmeasured={len(report.design_points)}  undefined={len(report.undefined)}")
    if not args.gpu:
        skipped = [u for u in report.undefined if u.get("requirement") == "tier:gpu"]
        if skipped:
            print(f"{len(skipped)} SemIf design point(s) skipped (gpu tier); pass --gpu to run")
    for contrast in report.contrasts:
        if contrast.get("measured"):
            print(
                f"  vs {contrast['reference']:>14}  {contrast['design_point']:>24}  "
                f"delta={contrast['delta']:+.3f}  p={contrast['p_value']:.4f}  n={contrast['n']}"
            )


if __name__ == "__main__":
    main()
