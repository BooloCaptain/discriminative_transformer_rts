"""The named conditions: the headline condition, the low-co-occurrence condition, and the runner that shares one cache.

A condition is a complete ``Experiment`` value. Two of these are verification vehicles: with the
right controls :func:`study_condition` reproduces ``artifacts/results_{full,covered}.json`` and
:func:`ladder_condition` reproduces ``artifacts/ladder.json``, both through
``scripts/verify_experiment_layer.py``.

The low-co-occurrence condition is a second experiment rather than a subset of the headline condition, because it
reports three budgets where the headline condition reports six and ``budgets`` is a control -- a control is
by definition constant across a run. The two conditions share one score cache, so the low_cooccurrence corners
pay only for their metric sweeps.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .. import config
from ..data.contract import Dataset
from ..experiment import (
    FACTOR_MODEL,
    Binding,
    Contrast,
    Controls,
    Environment,
    Experiment,
    RunReport,
    run,
)
from ..model import rankers
from .factors import (
    dataset_factor,
    model_factor,
    semif_scoring_model_factor,
    split_factor,
    structured_feature_factor,
    subset_factor,
)

#: The low_cooccurrence-condition thresholds. ``max_pair_count`` is a *subset* parameter, so a threshold is a
#: subset level rather than a control. The *budgets* are a control, though, and the low-co-occurrence condition
#: reports three where the headline condition reports six -- so the low-co-occurrence condition is a second experiment
#: rather than a corner of the first grid. That is a real consequence of the design and it is
#: recorded here rather than hidden: a metric grid cannot be a factor while the sweep is
#: vectorised across budgets.
LOW_COOCCURRENCE_THRESHOLDS: tuple[int, ...] = (80, 160)
LOW_COOCCURRENCE_BUDGETS: tuple[float, ...] = (0.05, 0.1, 0.2)


# --- the conditions ---------------------------------------------------------------


def study_condition(
    label: str = "mutmut",
    *,
    candidate_policy: str = "full",
    ablations: bool = True,
    include_semif: bool = True,
    name: str | None = None,
) -> Experiment:
    """The study's headline condition: every ranker, one dataset, the BM25 shuffle controls.

    With ``candidate_policy="full"`` and ``label="mutmut"`` this reproduces
    ``artifacts/results_full.json``. The shuffle controls are part of the same grid because they
    share the dataset, the split, the feature block, the candidate pool and the budgets -- only
    the model differs, and that is the role they are.
    """
    return Experiment(
        name=name or f"study.{label}.{candidate_policy}",
        datasets=dataset_factor((label,)),
        features=structured_feature_factor(),
        models=model_factor(ablations=ablations, include_semif=include_semif),
        subsets=subset_factor(("detectable",)),
        splits=split_factor(),
        controls=Controls(
            seed=config.SEED,
            budgets=config.DEFAULT_BUDGETS,
            n_bootstrap=config.DEFAULT_BOOTSTRAP,
            candidate_policy=candidate_policy,
        ),
        contrasts=(
            Contrast(
                FACTOR_MODEL,
                "coverage",
                0.05,
                note="every ranker against the coverage baseline at the study's probe_budget",
            ),
        ),
        # The recorded condition explicitly enables history on a synthetic order. That is an
        # override, not an option, and it produces the diagnostic the layer carries.
        temporal=True,
        note="the headline condition: the full ranker set, full candidate set",
    )


def low_cooccurrence_condition(
    label: str = "mutmut",
    *,
    thresholds: Sequence[int] = LOW_COOCCURRENCE_THRESHOLDS,
    budgets: Sequence[float] = LOW_COOCCURRENCE_BUDGETS,
    candidate_policy: str = "full",
    include_semif: bool = True,
    name: str | None = None,
) -> Experiment:
    """The low-co-occurrence condition: the same rankers over the rarely-recurring corners of the data.

    Its own experiment rather than a subset of :func:`study_condition`, because it reports three
    budgets where the headline condition reports six and ``budgets`` is a control. The rankers are the
    headline set without the shuffle controls, matching what the recorded artifact tabulates.
    """
    return Experiment(
        name=name or f"low_cooccurrence.{label}.{candidate_policy}",
        datasets=dataset_factor((label,)),
        features=structured_feature_factor(),
        models=model_factor(rankers.default_rankers(include_semif=include_semif)),
        subsets=subset_factor((), low_cooccurrence=thresholds),
        splits=split_factor(),
        controls=Controls(
            seed=config.SEED,
            budgets=tuple(budgets),
            n_bootstrap=config.DEFAULT_BOOTSTRAP,
            candidate_policy=candidate_policy,
        ),
        temporal=True,
        note="the low-co-occurrence condition: (file, test) pairs that rarely recur",
    )


def run_study(
    label: str = "mutmut",
    *,
    candidate_policy: str = "full",
    ablations: bool = True,
    include_semif: bool = True,
    low_cooccurrence: Sequence[int] = LOW_COOCCURRENCE_THRESHOLDS,
    out_dir: Path | str | None = None,
    knobs_overrides: dict | None = None,
    save: bool = True,
    verbose: bool = True,
) -> tuple[RunReport, RunReport | None]:
    """Run the headline condition and, unless ``low_cooccurrence`` is empty, the low-co-occurrence condition.

    Two experiments, one score cache: the second run's contexts differ from the first's only in
    which rows the metric averages, so it reuses every score matrix and only re-sweeps the
    metrics. Returns ``(headline, sparse_or_None)``.

    ``knobs_overrides`` applies to both conditions' declared controls *individually*, so an override can
    change the seed or the resample count without disturbing the budgets that make the two conditions
    different.
    """
    scores: dict = {}

    def effective(experiment: Experiment) -> Controls:
        if not knobs_overrides:
            return experiment.controls
        return replace(experiment.controls, **knobs_overrides)

    headline_experiment = study_condition(
        label, candidate_policy=candidate_policy, ablations=ablations, include_semif=include_semif
    )
    headline = run(
        headline_experiment,
        out_dir=out_dir,
        controls=effective(headline_experiment),
        scores=scores,
        save=save,
        verbose=verbose,
    )
    if not low_cooccurrence:
        return headline, None
    low_cooccurrence_experiment = low_cooccurrence_condition(
        label, thresholds=low_cooccurrence, candidate_policy=candidate_policy, include_semif=include_semif
    )
    secondary = run(
        low_cooccurrence_experiment,
        out_dir=out_dir,
        controls=effective(low_cooccurrence_experiment),
        scores=scores,
        save=False,
        verbose=verbose,
    )
    return headline, secondary


def semif_production_condition(
    cache: Path | None = None,
    *,
    candidate_policy: str = "full",
    instruction: str | None = None,
    feature_mode: str | None = None,
    placement: str = "instruct",
    name: str | None = None,
) -> Experiment:
    """Produce a SemIf score cache **as a design point**, rather than as a precondition of one.

    The study's most expensive step used to live outside the layer: a driver ran the model and
    wrote a file, and a model level then read it. So there was no tier, no cost and no design point
    key for the one thing that costs GPU hours, and -- worse -- a stale cache was
    indistinguishable from a fresh one.

    Run it with the GPU tier enabled to score and write the cache::

        python -m rts.studies semif.produce --tiers gpu

    or without, to have the design point reported *undefined* with the path instead of scored. The
    default target is a scratch artifact rather than any recorded cache, so running this can
    never overwrite a number the study is written against; pass ``cache=`` to aim it at a
    specific one. Cost is pairs/30 s at batch 8 over the covered pool and roughly a third of
    that rate over the full pool, so the full held-out grid is hours -- which is the reason it
    is an explicitly named condition rather than part of ``study``.
    """
    target = Path(cache) if cache is not None else config.ARTIFACTS / "semif_scores_scored.jsonl"
    return Experiment(
        name=name or "semif.produce",
        datasets=dataset_factor(("mutmut",)),
        features=structured_feature_factor(),
        models=semif_scoring_model_factor(
            target,
            candidate_policy=candidate_policy,
            instruction=instruction,
            feature_mode=feature_mode,
            placement=placement,
        ),
        subsets=subset_factor(("detectable",)),
        splits=split_factor(),
        controls=Controls(
            seed=config.SEED,
            budgets=(0.05,),
            n_bootstrap=0,
            candidate_policy=candidate_policy,
        ),
        temporal=True,
        note="SemIf score production, declared as a design point (GPU tier)",
    )


def dataset(label: str = "mutmut", *, controls: Controls | None = None, **shared) -> Dataset:
    """The built dataset for a label source, for a caller that needs the data itself.

    For an adapter rendering a dataset description, or a check computing a statistic, rather
    than for a sweep over it. Rebuilding is cheap and explicitly permitted: two datasets may
    coexist in one process (``docs/refactor.md`` §7), so a consumer does not have to reach into a
    run's internals to get at what it measured.
    """
    level = dataset_factor((label,)).get(f"marshmallow_{label}")
    value = level.build(Binding(env=Environment(controls=controls or Controls(), shared=dict(shared))))
    if not isinstance(value, Dataset):
        raise TypeError(f"dataset level {level.name!r} did not build a Dataset: {value!r}")
    return value
