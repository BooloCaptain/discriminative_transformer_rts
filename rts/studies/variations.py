"""The variation conditions: the cold start, seed, instruction, embedding and redundancy sweeps.

Each is its own experiment rather than one grid, because they differ in the controls a control is
allowed to differ in -- ``candidate_sets`` (the cold start conditions use the full pool, the instruction and
redundancy conditions the coverage set) and ``budgets`` -- while sharing the dataset, the split and
the feature block, and therefore one score cache.

Two sections of the recorded artifact are NOT here: ``p5_trained`` needs an evaluation window
inside the held-out tail plus a NaN convention for unscored pairs, and ``p1_direct``'s cache is
absent from the artifacts, so its numbers cannot be reproduced at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .. import config
from ..data import subsets
from ..experiment import (
    FACTOR_MODEL,
    FACTOR_SUBSET,
    Contrast,
    Controls,
    Experiment,
    Factor,
    Level,
    constant,
)
from ..model import rankers
from .factors import (
    _model_element,
    dataset_factor,
    split_factor,
    structured_feature_factor,
    subset_factor,
)

# --- the variation conditions -----------------------------------------------------
#
# The ``variations`` driver's seven sections. Each is its own experiment rather than one grid,
# because they differ in the controls a control is allowed to differ in: ``candidate_sets`` (the cold start
# conditions use the full pool, the instruction and redundancy conditions the coverage set) and ``budgets``.
# What they share is the dataset, the split and the feature block, which is what lets one score
# cache serve them all.
#
# Two sections are NOT here, and ``docs/experiment.md`` §13 says why: ``p5_trained`` needs an
# evaluation window *inside* the held-out tail plus a NaN convention for unscored pairs, and
# ``p1_direct``'s cache is absent from the artifacts, so its numbers cannot be reproduced at all.

#: The variation conditions' budget grid, probe budget and resample counts: four points, not the study
#: condition's six, and the table intervals and the paired tests use different counts.
VARIATION_BUDGETS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2)
VARIATION_PROBE = 0.05
VARIATION_TABLE_RESAMPLES = 1000
VARIATION_RESAMPLES = 2000

#: The starvation thresholds the cold start conditions are evaluated at. They are *nested* -- a killing
#: pair with at most 2 failures also has at most 5 -- which is why one cache scored for the wider
#: threshold covers the narrower subset, and why the instruction sweep can report both for
#: the price of one.
COLD_START_THRESHOLDS: tuple[int, ...] = (2, 5)

#: The model seeds the XGBoost refit is repeated under. Only the *model* seed varies: the run
#: seed fixes the synthetic change order and the split, and the caches are keyed to that order.
SEED_SWEEP: tuple[int, ...] = (1, 2, 3, 4)


def cold_start_cache(max_failures: int) -> Path:
    """The SemIf cache for a starvation threshold.

    The ``<=5`` cache is assembled from the ``<=2`` one plus a scored delta rather than
    re-scoring pairs that already exist, which is sound because the subsets nest.
    """
    if max_failures == 2:
        return config.ARTIFACTS / "semif_scores_cold_start2_full.jsonl"
    return config.ARTIFACTS / f"semif_scores_cold_start{max_failures}_full.jsonl"


def instruction_names() -> tuple[str, ...]:
    """The wording variants, in the recorded order, with the control first.

    The control is the study's own text-only cache: the point of the sweep is whether a
    *different* wording beats the question the other conditions were measured against, so the control
    has to be that one rather than a re-scored copy of it.
    """
    from ..model.semif_runner import INSTRUCTION_VARIANTS

    return ("default", *(n for n in sorted(INSTRUCTION_VARIANTS) if n != "default"))


def instruction_cache(name: str, max_failures: int) -> Path:
    if name == "default":
        return Path(config.SEMIF_SCORES_FILE)
    return config.ARTIFACTS / f"semif_scores_instr_{name}_cold_start{max_failures}.jsonl"


def embed_cache() -> Path:
    return config.ARTIFACTS / "embed_scores.npy"


def _cached(
    name: str,
    path: Path,
    *,
    tier: str = "cpu",
    note: str = "",
    loader=None,
) -> Level:
    """A level for a precomputed score matrix.

    Tier ``cpu``, not ``cache``: the *model* is as cheap as reading a file, but the design point still
    assembles features and sweeps metrics, so the design point's cost is not a file read.

    ``loader`` matters because a cache is not always a scored-pair log: the embedding baseline is
    a whole ``[n_changes, n_tests]`` matrix saved with ``numpy.save``, so reading it with the
    default loader is a decode error rather than a wrong number.
    """
    return Level(
        name,
        lambda binding, name=name, path=path, loader=loader: rankers.CachedScores(
            name, path, loader=loader
        ),
        tier=tier,
        note=note or f"precomputed scores from {path.name}",
    )


def _rank_average(name: str, parents: Sequence[rankers.Ranker], candidate_policy: str) -> rankers.Ranker:
    """A fitted-free rank average of parents that are already declared."""
    return rankers.RankAverageRanker(name, parents, candidate_policy=candidate_policy)


def cold_start_model_factor(max_failures: int, *, seeds: Sequence[int] = ()) -> Factor:
    """The cold start condition's ranker set, or the two-tree subset the seed sweep refits.

    The four XGBoost conditions are a history x coverage decomposition with lexical always on, so the
    mechanism -- that the full candidate set is the only regime where ``function_coverage``
    separates candidate sets from non-candidate sets -- is measured rather than asserted:

        struct_lex        history + coverage + BM25
        static_lex        coverage + BM25
        struct_nocov_lex  history + BM25
        static_nocov_lex  BM25 only

    ``seeds`` switches to the refit subset: the two strongest trees at each seed, plus the SemIf
    cache they are paired against. The seed is part of the level's identity because it is part
    of what the level computes.

    The two levels that appear twice -- ``static_nocov_lex`` once standalone and once as a
    rank-average parent, and the SemIf cache likewise -- are built by a factory so that every
    level has its *own* instance. It matters most for the tree, which records state
    (``importances_``, the extra-column cache): one instance shared by two levels would let
    one design point's state describe another's, and would fit the same model twice under one name.
    ``tests/test_studies.py`` asserts the invariant for every condition.
    """

    def static_nocov_lex() -> rankers.XGBoostRanker:
        return rankers.XGBoostRanker(
            exclude_temporal=True, exclude_coverage=True, include_lexical=True,
            candidate_policy="full",
        )

    def semif_scores() -> rankers.CachedScores:
        return rankers.CachedScores("semif_textonly_full", cold_start_cache(max_failures))

    if seeds:
        levels: list[Level] = [
            _model_element(
                rankers.XGBoostRanker(
                    include_lexical=True, candidate_policy="full", seed=seed
                ),
                note=f"xgboost_struct_lex refit under model seed {seed}",
            )
            for seed in seeds
        ]
        levels.extend(
            _model_element(
                rankers.XGBoostRanker(
                    exclude_temporal=True,
                    include_lexical=True,
                    candidate_policy="full",
                    seed=seed,
                ),
                note=f"xgboost_static_lex refit under model seed {seed}",
            )
            for seed in seeds
        )
        levels.append(_model_element(semif_scores()))
        return Factor(FACTOR_MODEL, tuple(levels), note="the seed sweep's refit subset")

    struct = rankers.XGBoostRanker(candidate_policy="full")
    return Factor(
        FACTOR_MODEL,
        (
            _model_element(rankers.RandomRanker()),
            _model_element(rankers.RecencyRanker()),
            _model_element(rankers.FailureRateRanker()),
            _model_element(rankers.CoverageRanker()),
            _model_element(rankers.StructuralRuleRanker()),
            _model_element(rankers.LexicalRanker()),
            _model_element(static_nocov_lex()),
            _model_element(
                rankers.XGBoostRanker(
                    exclude_temporal=True, include_lexical=True, candidate_policy="full"
                )
            ),
            _model_element(
                rankers.XGBoostRanker(
                    exclude_coverage=True, include_lexical=True, candidate_policy="full"
                )
            ),
            _model_element(struct),
            _model_element(rankers.XGBoostRanker(include_lexical=True, candidate_policy="full")),
            _model_element(semif_scores()),
            _model_element(
                _rank_average(
                    "rankaverage_xgb_semif",
                    (static_nocov_lex(), semif_scores()),
                    "full",
                )
            ),
        ),
        note="the cold_start condition's rankers; the two caches are the same model under one wording",
    )


def instruction_model_factor(max_failures: int) -> Factor:
    """BM25, the strongest cheap tree, and one SemIf level per wording variant."""
    levels = [
        _model_element(rankers.LexicalRanker()),
        _model_element(
            rankers.XGBoostRanker(
                exclude_temporal=True, exclude_coverage=True, include_lexical=True,
                candidate_policy="coverage_restricted",
            )
        ),
    ]
    levels.extend(
        _cached(f"semif_{name}", instruction_cache(name, max_failures))
        for name in instruction_names()
    )
    return Factor(FACTOR_MODEL, tuple(levels), note="the instruction-wording sweep")


def embed_model_factor(candidate_sets: str) -> Factor:
    """The embedding baseline's parents: the code encoder against BM25 and two cheap rules."""
    levels = [
        _model_element(rankers.LexicalRanker()),
        _model_element(rankers.CoverageRanker()),
        _model_element(rankers.StructuralRuleRanker()),
        _cached("embed_codebert", embed_cache(), loader=rankers.load_matrix),
    ]
    if candidate_sets == "coverage_restricted":
        levels.insert(1, _cached("semif_textonly", Path(config.SEMIF_SCORES_FILE)))
        levels.append(
            _model_element(
                rankers.XGBoostRanker(
                    exclude_temporal=True, exclude_coverage=True, include_lexical=True,
                    candidate_policy="coverage_restricted",
                )
            )
        )
    order = (
        ("bm25_lexical", "semif_textonly", "embed_codebert", "xgboost_static_nocov_lex")
        if candidate_sets == "coverage_restricted"
        else ("bm25_lexical", "embed_codebert", "structural_rule", "coverage")
    )
    by_name = {e.name: e for e in levels}
    return Factor(FACTOR_MODEL, tuple(by_name[name] for name in order), note="the embedding baseline")


def redundancy_model_factor() -> Factor:
    """P5's parents plus the two rank averages that test whether SemIf is redundant.

    A parent that is also tabulated standalone gets a fresh instance per level, so two
    levels never share one -- see :func:`cold_start_model_factor`, and it is the stateful trees
    that make it matter.
    """

    def static_nocov_lex() -> rankers.XGBoostRanker:
        return rankers.XGBoostRanker(
            exclude_temporal=True, exclude_coverage=True, include_lexical=True,
            candidate_policy="coverage_restricted",
        )

    def struct() -> rankers.XGBoostRanker:
        return rankers.XGBoostRanker(candidate_policy="coverage_restricted")

    def semif() -> rankers.CachedScores:
        return rankers.CachedScores("semif_textonly", Path(config.SEMIF_SCORES_FILE))

    return Factor(
        FACTOR_MODEL,
        (
            _model_element(rankers.RandomRanker()),
            _model_element(rankers.RecencyRanker()),
            _model_element(rankers.FailureRateRanker()),
            _model_element(rankers.CoverageRanker()),
            _model_element(rankers.StructuralRuleRanker()),
            _model_element(rankers.LexicalRanker()),
            _model_element(static_nocov_lex()),
            _model_element(struct()),
            _model_element(semif()),
            _model_element(
                _rank_average(
                    "rankaverage_xgb_semif", (static_nocov_lex(), semif()), "coverage_restricted"
                )
            ),
            _model_element(
                _rank_average(
                    "rankaverage_xgb_struct_semif", (struct(), semif()), "coverage_restricted"
                )
            ),
        ),
        note="the redundancy test: two trees, the text model, and both rank averages",
    )


def _variation_knobs(candidate_policy: str) -> Controls:
    return Controls(
        seed=config.SEED,
        budgets=VARIATION_BUDGETS,
        n_bootstrap=VARIATION_TABLE_RESAMPLES,
        n_bootstrap_paired=VARIATION_RESAMPLES,
        candidate_policy=candidate_policy,
    )


def variation_contrasts(references: Sequence[str]) -> tuple[Contrast, ...]:
    """Every reference paired at every budget.

    A contrast is defined by one probe budget, and the recorded artifacts pair at all four of
    the variation grid -- because a lever that helps only at a low budget is a different finding
    from one that helps throughout. So the pairs are declared rather than the renderer computing
    a second family of statistics outside the report.
    """
    return tuple(
        Contrast(FACTOR_MODEL, reference, budget)
        for reference in references
        for budget in VARIATION_BUDGETS
    )


def cold_start_condition(max_failures: int = 2, *, name: str | None = None) -> Experiment:
    """The full-candidate cold start condition: the positive result, re-measured without the crutch."""
    return Experiment(
        name=name or f"cold_start{max_failures}",
        datasets=dataset_factor(("mutmut",)),
        features=structured_feature_factor(),
        models=cold_start_model_factor(max_failures),
        subsets=subset_factor((subsets.cold_start(max_failures),)),
        splits=split_factor(),
        controls=_variation_knobs("full"),
        contrasts=variation_contrasts(
            (
                "xgboost_static_nocov_lex",
                "structural_rule",
                "xgboost_struct_lex",
                "xgboost_static_lex",
            )
        ),
        temporal=True,
        note="the cold_start condition over the full candidate pool",
    )


def cold_start_seeds_condition(
    max_failures: int = 2, *, model_seed: int, name: str | None = None
) -> Experiment:
    """One refit of the cold start condition's two strongest trees, under its own model seed.

    A *model* seed, not a run seed: varying ``seed`` would move the synthetic change order and the
    split, and the score caches are keyed to that order, so it would invalidate the paired
    contrast instead of testing it.
    """
    threshold = subsets.cold_start(max_failures)
    controls = _variation_knobs("full")
    return Experiment(
        name=name or f"cold_start{max_failures}.seed{model_seed}",
        datasets=dataset_factor(("mutmut",)),
        features=structured_feature_factor(),
        models=cold_start_model_factor(max_failures, seeds=(model_seed,)),
        subsets=subset_factor((threshold,)),
        splits=split_factor(),
        controls=replace(controls, model_seed=model_seed),
        # The trees are the references, so the SemIf design point is the one paired -- and the layer
        # computes ``design_point - reference``, which is the recorded convention ("SemIf minus the
        # tree") without the renderer having to flip a sign.
        contrasts=variation_contrasts(("xgboost_struct_lex", "xgboost_static_lex")),
        temporal=True,
        note=f"the cold_start condition refit under model seed {model_seed}",
    )


def instruction_condition(
    max_failures: int = 5,
    *,
    secondary: int = 2,
    name: str | None = None,
) -> Experiment:
    """The instruction-wording sweep, over both starvation thresholds at once.

    Both thresholds are subset levels in one run, which is free: the caches are scored for
    the wider threshold and the narrower subset is a subset of it.
    """
    thresholds = [subsets.cold_start(max_failures)]
    if secondary != max_failures:
        thresholds.append(subsets.cold_start(secondary))
    return Experiment(
        name=name or f"instruction{max_failures}",
        datasets=dataset_factor(("mutmut",)),
        features=structured_feature_factor(),
        models=instruction_model_factor(max_failures),
        subsets=subset_factor(tuple(thresholds)),
        splits=split_factor(),
        controls=_variation_knobs("coverage_restricted"),
        contrasts=variation_contrasts(("semif_default",)),
        temporal=True,
        note="the instruction sweep: five wordings against the one the study used",
    )


def embedding_condition(candidate_policy: str = "coverage_restricted", *, name: str | None = None) -> Experiment:
    """The code-embedding baseline, at one candidate mode.

    Two conditions rather than one because the full-pool contrast is the one where a semantic ranker
    could pay off -- the coverage-restricted mask is the crutch that lets structure win cheaply -- and
    ``candidate_policy`` is a control.
    """
    chosen: list[subsets.Subset] = []
    if candidate_policy == "coverage_restricted":
        chosen.append(subsets.DETECTABLE)
    chosen.append(subsets.cold_start(5))
    return Experiment(
        name=name or f"embed.{candidate_policy}",
        datasets=dataset_factor(("mutmut",)),
        features=structured_feature_factor(),
        models=embed_model_factor(candidate_policy),
        subsets=Factor(
            FACTOR_SUBSET,
            tuple(constant(p.name, p) for p in chosen),
            note="all held-out faults, and the cold_start subset",
        ),
        splits=split_factor(),
        controls=_variation_knobs(candidate_policy),
        contrasts=variation_contrasts(("bm25_lexical",)),
        temporal=True,
        note="the code-embedding baseline against BM25 and two cheap structural rules",
    )


def redundancy_condition(*, name: str | None = None) -> Experiment:
    """P5: does SemIf carry signal the cheap structured model does not already have?

    The test is not a correlation but a *parent-versus-child* contrast: if the second model
    adds independent signal the rank average beats both parents, and if it is redundant the
    average sits between them. Three references are declared because the pairs the write-up
    needs span two of them -- the averages are compared against their own structural parent as
    well as against each other's -- and a contrast is defined by its reference.
    """
    return Experiment(
        name=name or "redundancy",
        datasets=dataset_factor(("mutmut",)),
        features=structured_feature_factor(),
        models=redundancy_model_factor(),
        subsets=subset_factor(("detectable",)),
        splits=split_factor(),
        controls=_variation_knobs("coverage_restricted"),
        contrasts=variation_contrasts(
            ("xgboost_static_nocov_lex", "semif_textonly", "xgboost_struct")
        ),
        temporal=True,
        note="is the transformer redundant given cheap structure?",
    )
