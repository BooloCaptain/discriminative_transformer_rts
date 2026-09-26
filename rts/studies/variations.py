"""The variation arms: the starved, seed, instruction, embedding and redundancy sweeps.

Each is its own experiment rather than one grid, because they differ in the knobs a knob is
allowed to differ in -- ``candidates`` (the starved arms use the full pool, the instruction and
redundancy arms the covered set) and ``budgets`` -- while sharing the dataset, the split and
the feature block, and therefore one score cache.

Two sections of the recorded artifact are NOT here: ``p5_trained`` needs an evaluation window
inside the held-out tail plus a NaN convention for unscored pairs, and ``p1_direct``'s cache is
absent from the artifacts, so its numbers cannot be reproduced at all.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .. import config, models, populations
from ..experiment import (
    ROLE_MODEL,
    ROLE_POPULATION,
    Axis,
    Comparison,
    Element,
    Experiment,
    Knobs,
    constant,
)
from .axes import (
    _model_element,
    dataset_axis,
    population_axis,
    split_axis,
    structured_feature_axis,
)

# --- the variation arms -----------------------------------------------------
#
# The ``variations`` driver's seven sections. Each is its own experiment rather than one grid,
# because they differ in the knobs a knob is allowed to differ in: ``candidates`` (the starved
# arms use the full pool, the instruction and redundancy arms the covered set) and ``budgets``.
# What they share is the dataset, the split and the feature block, which is what lets one score
# cache serve them all.
#
# Two sections are NOT here, and ``docs/experiment.md`` §13 says why: ``p5_trained`` needs an
# evaluation window *inside* the held-out tail plus a NaN convention for unscored pairs, and
# ``p1_direct``'s cache is absent from the artifacts, so its numbers cannot be reproduced at all.

#: The variation arms' budget grid, probe budget and resample counts: four points, not the study
#: arm's six, and the table intervals and the paired tests use different counts.
VARIATION_BUDGETS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2)
VARIATION_PROBE = 0.05
VARIATION_TABLE_RESAMPLES = 1000
VARIATION_RESAMPLES = 2000

#: The starvation thresholds the starved arms are evaluated at. They are *nested* -- a killing
#: pair with at most 2 failures also has at most 5 -- which is why one cache scored for the wider
#: threshold covers the narrower population, and why the instruction sweep can report both for
#: the price of one.
STARVED_THRESHOLDS: tuple[int, ...] = (2, 5)

#: The model seeds the XGBoost refit is repeated under. Only the *model* seed varies: the run
#: seed fixes the imposed change order and the split, and the caches are keyed to that order.
SEED_SWEEP: tuple[int, ...] = (1, 2, 3, 4)


def starved_cache(max_failures: int) -> Path:
    """The SemIf cache for a starvation threshold.

    The ``<=5`` cache is assembled from the ``<=2`` one plus a scored delta rather than
    re-scoring pairs that already exist, which is sound because the populations nest.
    """
    if max_failures == 2:
        return config.ARTIFACTS / "semif_scores_starved2_full.jsonl"
    return config.ARTIFACTS / f"semif_scores_starved{max_failures}_full.jsonl"


def instruction_names() -> tuple[str, ...]:
    """The wording variants, in the recorded order, with the control first.

    The control is the study's own text-only cache: the point of the sweep is whether a
    *different* wording beats the question the other arms were measured against, so the control
    has to be that one rather than a re-scored copy of it.
    """
    from ..semif_runner import INSTRUCTION_VARIANTS

    return ("default", *(n for n in sorted(INSTRUCTION_VARIANTS) if n != "default"))


def instruction_cache(name: str, max_failures: int) -> Path:
    if name == "default":
        return Path(config.SEMIF_SCORES_FILE)
    return config.ARTIFACTS / f"semif_scores_instr_{name}_starved{max_failures}.jsonl"


def embed_cache() -> Path:
    return config.ARTIFACTS / "embed_scores.npy"


def _cached(
    name: str,
    path: Path,
    *,
    tier: str = "cpu",
    note: str = "",
    loader=None,
) -> Element:
    """An element for a precomputed score matrix.

    Tier ``cpu``, not ``cache``: the *model* is as cheap as reading a file, but the cell still
    assembles features and sweeps metrics, so the cell's cost is not a file read.

    ``loader`` matters because a cache is not always a scored-pair log: the embedding baseline is
    a whole ``[n_changes, n_tests]`` matrix saved with ``numpy.save``, so reading it with the
    default loader is a decode error rather than a wrong number.
    """
    return Element(
        name,
        lambda binding, name=name, path=path, loader=loader: models.CachedScores(
            name, path, loader=loader
        ),
        tier=tier,
        note=note or f"precomputed scores from {path.name}",
    )


def _rank_average(name: str, parents: Sequence[models.Selector], candidates_mode: str) -> models.Selector:
    """A fitted-free rank average of parents that are already declared."""
    return models.RankAverageSelector(name, parents, candidates_mode=candidates_mode)


def starved_model_axis(max_failures: int, *, seeds: Sequence[int] = ()) -> Axis:
    """The starved arm's selector set, or the two-tree subset the seed sweep refits.

    The four XGBoost arms are a history x coverage decomposition with lexical always on, so the
    mechanism -- that the full candidate set is the only regime where ``covers_function``
    separates candidates from non-candidates -- is measured rather than asserted:

        struct_lex        history + coverage + BM25
        static_lex        coverage + BM25
        struct_nocov_lex  history + BM25
        static_nocov_lex  BM25 only

    ``seeds`` switches to the refit subset: the two strongest trees at each seed, plus the SemIf
    cache they are paired against. The seed is part of the element's identity because it is part
    of what the element computes.
    """
    if seeds:
        semif = models.CachedScores("semif_textonly_full", starved_cache(max_failures))
        elements: list[Element] = [
            _model_element(
                models.XGBoostSelector(
                    include_lexical=True, candidates_mode="full", seed=seed
                ),
                note=f"xgboost_struct_lex refit under model seed {seed}",
            )
            for seed in seeds
        ]
        elements.extend(
            _model_element(
                models.XGBoostSelector(
                    exclude_history=True,
                    include_lexical=True,
                    candidates_mode="full",
                    seed=seed,
                ),
                note=f"xgboost_static_lex refit under model seed {seed}",
            )
            for seed in seeds
        )
        elements.append(_model_element(semif))
        return Axis(ROLE_MODEL, tuple(elements), note="the seed sweep's refit subset")

    static_nocov_lex = models.XGBoostSelector(
        exclude_history=True, exclude_coverage=True, include_lexical=True, candidates_mode="full"
    )
    struct = models.XGBoostSelector(candidates_mode="full")
    semif = models.CachedScores("semif_textonly_full", starved_cache(max_failures))
    return Axis(
        ROLE_MODEL,
        (
            _model_element(models.RandomSelector()),
            _model_element(models.RecencySelector()),
            _model_element(models.FailureRateSelector()),
            _model_element(models.CoverageSelector()),
            _model_element(models.StructuralRuleSelector()),
            _model_element(models.LexicalSelector()),
            _model_element(static_nocov_lex),
            _model_element(
                models.XGBoostSelector(
                    exclude_history=True, include_lexical=True, candidates_mode="full"
                )
            ),
            _model_element(
                models.XGBoostSelector(
                    exclude_coverage=True, include_lexical=True, candidates_mode="full"
                )
            ),
            _model_element(struct),
            _model_element(models.XGBoostSelector(include_lexical=True, candidates_mode="full")),
            _model_element(semif),
            _model_element(
                _rank_average("rankaverage_xgb_semif", (static_nocov_lex, semif), "full")
            ),
        ),
        note="the starved arm's selectors; the two caches are the same model under one wording",
    )


def instruction_model_axis(max_failures: int) -> Axis:
    """BM25, the strongest cheap tree, and one SemIf element per wording variant."""
    elements = [
        _model_element(models.LexicalSelector()),
        _model_element(
            models.XGBoostSelector(
                exclude_history=True, exclude_coverage=True, include_lexical=True,
                candidates_mode="covered",
            )
        ),
    ]
    elements.extend(
        _cached(f"semif_{name}", instruction_cache(name, max_failures))
        for name in instruction_names()
    )
    return Axis(ROLE_MODEL, tuple(elements), note="the instruction-wording sweep")


def embed_model_axis(candidates: str) -> Axis:
    """The embedding baseline's parents: the code encoder against BM25 and two cheap rules."""
    elements = [
        _model_element(models.LexicalSelector()),
        _model_element(models.CoverageSelector()),
        _model_element(models.StructuralRuleSelector()),
        _cached("embed_codebert", embed_cache(), loader=models.load_matrix),
    ]
    if candidates == "covered":
        elements.insert(1, _cached("semif_textonly", Path(config.SEMIF_SCORES_FILE)))
        elements.append(
            _model_element(
                models.XGBoostSelector(
                    exclude_history=True, exclude_coverage=True, include_lexical=True,
                    candidates_mode="covered",
                )
            )
        )
    order = (
        ("bm25_lexical", "semif_textonly", "embed_codebert", "xgboost_static_nocov_lex")
        if candidates == "covered"
        else ("bm25_lexical", "embed_codebert", "structural_rule", "coverage")
    )
    by_name = {e.name: e for e in elements}
    return Axis(ROLE_MODEL, tuple(by_name[name] for name in order), note="the embedding baseline")


def redundancy_model_axis() -> Axis:
    """P5's parents plus the two rank averages that test whether SemIf is redundant."""
    static_nocov_lex = models.XGBoostSelector(
        exclude_history=True, exclude_coverage=True, include_lexical=True, candidates_mode="covered"
    )
    struct = models.XGBoostSelector(candidates_mode="covered")
    semif = models.CachedScores("semif_textonly", Path(config.SEMIF_SCORES_FILE))
    return Axis(
        ROLE_MODEL,
        (
            _model_element(models.RandomSelector()),
            _model_element(models.RecencySelector()),
            _model_element(models.FailureRateSelector()),
            _model_element(models.CoverageSelector()),
            _model_element(models.StructuralRuleSelector()),
            _model_element(models.LexicalSelector()),
            _model_element(static_nocov_lex),
            _model_element(struct),
            _model_element(semif),
            _model_element(
                _rank_average("rankaverage_xgb_semif", (static_nocov_lex, semif), "covered")
            ),
            _model_element(
                _rank_average("rankaverage_xgb_struct_semif", (struct, semif), "covered")
            ),
        ),
        note="the redundancy test: two trees, the text model, and both rank averages",
    )


def _variation_knobs(candidates: str) -> Knobs:
    return Knobs(
        seed=config.SEED,
        budgets=VARIATION_BUDGETS,
        n_bootstrap=VARIATION_TABLE_RESAMPLES,
        n_bootstrap_paired=VARIATION_RESAMPLES,
        candidates=candidates,
    )


def variation_comparisons(references: Sequence[str]) -> tuple[Comparison, ...]:
    """Every reference paired at every budget.

    A comparison is defined by one probe budget, and the recorded artifacts pair at all four of
    the variation grid -- because a lever that helps only at a low budget is a different finding
    from one that helps throughout. So the pairs are declared rather than the renderer computing
    a second family of statistics outside the report.
    """
    return tuple(
        Comparison(ROLE_MODEL, reference, budget)
        for reference in references
        for budget in VARIATION_BUDGETS
    )


def starved_arm(max_failures: int = 2, *, name: str | None = None) -> Experiment:
    """The full-candidate starved arm: the positive result, re-measured without the crutch."""
    return Experiment(
        name=name or f"starved{max_failures}",
        datasets=dataset_axis(("mutmut",)),
        features=structured_feature_axis(),
        models=starved_model_axis(max_failures),
        populations=population_axis((populations.starved(max_failures),)),
        splits=split_axis(),
        knobs=_variation_knobs("full"),
        comparisons=variation_comparisons(
            (
                "xgboost_static_nocov_lex",
                "structural_rule",
                "xgboost_struct_lex",
                "xgboost_static_lex",
            )
        ),
        history=True,
        note="the starved arm over the full candidate pool",
    )


def starved_seeds_arm(
    max_failures: int = 2, *, model_seed: int, name: str | None = None
) -> Experiment:
    """One refit of the starved arm's two strongest trees, under its own model seed.

    A *model* seed, not a run seed: varying ``seed`` would move the imposed change order and the
    split, and the score caches are keyed to that order, so it would invalidate the paired
    comparison instead of testing it.
    """
    threshold = populations.starved(max_failures)
    knobs = _variation_knobs("full")
    return Experiment(
        name=name or f"starved{max_failures}.seed{model_seed}",
        datasets=dataset_axis(("mutmut",)),
        features=structured_feature_axis(),
        models=starved_model_axis(max_failures, seeds=(model_seed,)),
        populations=population_axis((threshold,)),
        splits=split_axis(),
        knobs=replace(knobs, model_seed=model_seed),
        # The trees are the references, so the SemIf cell is the one paired -- and the layer
        # computes ``cell - reference``, which is the recorded convention ("SemIf minus the
        # tree") without the renderer having to flip a sign.
        comparisons=variation_comparisons(("xgboost_struct_lex", "xgboost_static_lex")),
        history=True,
        note=f"the starved arm refit under model seed {model_seed}",
    )


def instruction_arm(
    max_failures: int = 5,
    *,
    secondary: int = 2,
    name: str | None = None,
) -> Experiment:
    """The instruction-wording sweep, over both starvation thresholds at once.

    Both thresholds are population elements in one run, which is free: the caches are scored for
    the wider threshold and the narrower population is a subset of it.
    """
    thresholds = [populations.starved(max_failures)]
    if secondary != max_failures:
        thresholds.append(populations.starved(secondary))
    return Experiment(
        name=name or f"instruction{max_failures}",
        datasets=dataset_axis(("mutmut",)),
        features=structured_feature_axis(),
        models=instruction_model_axis(max_failures),
        populations=population_axis(tuple(thresholds)),
        splits=split_axis(),
        knobs=_variation_knobs("covered"),
        comparisons=variation_comparisons(("semif_default",)),
        history=True,
        note="the instruction sweep: five wordings against the one the study used",
    )


def embed_arm(candidates: str = "covered", *, name: str | None = None) -> Experiment:
    """The code-embedding baseline, at one candidate mode.

    Two arms rather than one because the full-pool comparison is the one where a semantic ranker
    could pay off -- the covered mask is the crutch that lets structure win cheaply -- and
    ``candidates`` is a knob.
    """
    chosen: list[populations.Population] = []
    if candidates == "covered":
        chosen.append(populations.FAULT_BEARING)
    chosen.append(populations.starved(5))
    return Experiment(
        name=name or f"embed.{candidates}",
        datasets=dataset_axis(("mutmut",)),
        features=structured_feature_axis(),
        models=embed_model_axis(candidates),
        populations=Axis(
            ROLE_POPULATION,
            tuple(constant(p.name, p) for p in chosen),
            note="all held-out faults, and the starved subset",
        ),
        splits=split_axis(),
        knobs=_variation_knobs(candidates),
        comparisons=variation_comparisons(("bm25_lexical",)),
        history=True,
        note="the code-embedding baseline against BM25 and two cheap structural rules",
    )


def redundancy_arm(*, name: str | None = None) -> Experiment:
    """P5: does SemIf carry signal the cheap structured model does not already have?

    The test is not a correlation but a *parent-versus-child* comparison: if the second model
    adds independent signal the rank average beats both parents, and if it is redundant the
    average sits between them. Three references are declared because the pairs the write-up
    needs span two of them -- the averages are compared against their own structural parent as
    well as against each other's -- and a comparison is defined by its reference.
    """
    return Experiment(
        name=name or "redundancy",
        datasets=dataset_axis(("mutmut",)),
        features=structured_feature_axis(),
        models=redundancy_model_axis(),
        populations=population_axis(("fault_bearing",)),
        splits=split_axis(),
        knobs=_variation_knobs("covered"),
        comparisons=variation_comparisons(
            ("xgboost_static_nocov_lex", "semif_textonly", "xgboost_struct")
        ),
        history=True,
        note="is the transformer redundant given cheap structure?",
    )
