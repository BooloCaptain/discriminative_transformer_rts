"""The named arms: the headline arm, the sparse arm, and the runner that shares one cache.

An arm is a complete ``Experiment`` value. Two of these are verification vehicles: with the
right knobs :func:`study_arm` reproduces ``artifacts/results_{full,covered}.json`` and
:func:`ladder_arm` reproduces ``artifacts/ladder.json``, both through
``scripts/verify_experiment_layer.py``.

The sparse arm is a second experiment rather than a population of the headline arm, because it
reports three budgets where the headline arm reports six and ``budgets`` is a knob -- a knob is
by definition constant across a run. The two arms share one score cache, so the sparse corners
pay only for their metric sweeps.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .. import config, models
from ..contract import Dataset
from ..experiment import (
    ROLE_MODEL,
    Binding,
    Comparison,
    Environment,
    Experiment,
    Knobs,
    RunReport,
    run,
)
from .axes import (
    dataset_axis,
    model_axis,
    population_axis,
    semif_scoring_model_axis,
    split_axis,
    structured_feature_axis,
)

#: The sparse-arm thresholds. ``max_pair_count`` is a *population* parameter, so a threshold is a
#: population element rather than a knob. The *budgets* are a knob, though, and the sparse arm
#: reports three where the headline arm reports six -- so the sparse arm is a second experiment
#: rather than a corner of the first grid. That is a real consequence of the design and it is
#: recorded here rather than hidden: a metric grid cannot be a factor while the sweep is
#: vectorised across budgets.
SPARSE_THRESHOLDS: tuple[int, ...] = (80, 160)
SPARSE_BUDGETS: tuple[float, ...] = (0.05, 0.1, 0.2)


# --- the arms ---------------------------------------------------------------


def study_arm(
    label: str = "mutmut",
    *,
    candidates: str = "full",
    ablations: bool = True,
    include_semif: bool = True,
    name: str | None = None,
) -> Experiment:
    """The study's headline arm: every selector, one dataset, the BM25 shuffle controls.

    With ``candidates="full"`` and ``label="mutmut"`` this reproduces
    ``artifacts/results_full.json``. The shuffle controls are part of the same grid because they
    share the dataset, the split, the feature block, the candidate pool and the budgets -- only
    the model differs, and that is the role they are.
    """
    return Experiment(
        name=name or f"study.{label}.{candidates}",
        datasets=dataset_axis((label,)),
        features=structured_feature_axis(),
        models=model_axis(ablations=ablations, include_semif=include_semif),
        populations=population_axis(("fault_bearing",)),
        splits=split_axis(),
        knobs=Knobs(
            seed=config.SEED,
            budgets=config.DEFAULT_BUDGETS,
            n_bootstrap=config.DEFAULT_BOOTSTRAP,
            candidates=candidates,
        ),
        comparisons=(
            Comparison(
                ROLE_MODEL,
                "coverage",
                0.05,
                note="every selector against the coverage baseline at the study's probe budget",
            ),
        ),
        # The recorded arm explicitly enables history on an imposed order. That is an
        # override, not an option, and it produces the warning the layer carries.
        history=True,
        note="the headline arm: the full selector set, full candidate set",
    )


def sparse_arm(
    label: str = "mutmut",
    *,
    thresholds: Sequence[int] = SPARSE_THRESHOLDS,
    budgets: Sequence[float] = SPARSE_BUDGETS,
    candidates: str = "full",
    include_semif: bool = True,
    name: str | None = None,
) -> Experiment:
    """The sparse arm: the same selectors over the rarely-recurring corners of the data.

    Its own experiment rather than a population of :func:`study_arm`, because it reports three
    budgets where the headline arm reports six and ``budgets`` is a knob. The selectors are the
    headline set without the shuffle controls, matching what the recorded artifact tabulates.
    """
    return Experiment(
        name=name or f"sparse.{label}.{candidates}",
        datasets=dataset_axis((label,)),
        features=structured_feature_axis(),
        models=model_axis(models.default_selectors(include_semif=include_semif)),
        populations=population_axis((), sparse=thresholds),
        splits=split_axis(),
        knobs=Knobs(
            seed=config.SEED,
            budgets=tuple(budgets),
            n_bootstrap=config.DEFAULT_BOOTSTRAP,
            candidates=candidates,
        ),
        history=True,
        note="the sparse arm: (file, test) pairs that rarely recur",
    )


def run_study(
    label: str = "mutmut",
    *,
    candidates: str = "full",
    ablations: bool = True,
    include_semif: bool = True,
    sparse: Sequence[int] = SPARSE_THRESHOLDS,
    out_dir: Path | str | None = None,
    knobs_overrides: dict | None = None,
    save: bool = True,
    verbose: bool = True,
) -> tuple[RunReport, RunReport | None]:
    """Run the headline arm and, unless ``sparse`` is empty, the sparse arm.

    Two experiments, one score cache: the second run's contexts differ from the first's only in
    which rows the metric averages, so it reuses every score matrix and only re-sweeps the
    metrics. Returns ``(headline, sparse_or_None)``.

    ``knobs_overrides`` applies to both arms' declared knobs *individually*, so an override can
    change the seed or the resample count without disturbing the budgets that make the two arms
    different.
    """
    scores: dict = {}

    def effective(experiment: Experiment) -> Knobs:
        if not knobs_overrides:
            return experiment.knobs
        return replace(experiment.knobs, **knobs_overrides)

    headline_experiment = study_arm(
        label, candidates=candidates, ablations=ablations, include_semif=include_semif
    )
    headline = run(
        headline_experiment,
        out_dir=out_dir,
        knobs=effective(headline_experiment),
        scores=scores,
        save=save,
        verbose=verbose,
    )
    if not sparse:
        return headline, None
    sparse_experiment = sparse_arm(
        label, thresholds=sparse, candidates=candidates, include_semif=include_semif
    )
    secondary = run(
        sparse_experiment,
        out_dir=out_dir,
        knobs=effective(sparse_experiment),
        scores=scores,
        save=False,
        verbose=verbose,
    )
    return headline, secondary


def semif_production_arm(
    cache: Path | None = None,
    *,
    candidates: str = "full",
    instruction: str | None = None,
    feature_mode: str | None = None,
    placement: str = "instruct",
    name: str | None = None,
) -> Experiment:
    """Produce a SemIf score cache **as a cell**, rather than as a precondition of one.

    The study's most expensive step used to live outside the layer: a driver ran the model and
    wrote a file, and a model element then read it. So there was no tier, no cost and no cell
    key for the one thing that costs GPU hours, and -- worse -- a stale cache was
    indistinguishable from a fresh one.

    Run it with the GPU tier enabled to score and write the cache::

        python -m rts.studies semif.produce --tiers gpu

    or without, to have the cell reported *unmeasured* with the path instead of scored. The
    default target is a scratch artifact rather than any recorded cache, so running this can
    never overwrite a number the study is written against; pass ``cache=`` to aim it at a
    specific one. Cost is pairs/30 s at batch 8 over the covered pool and roughly a third of
    that rate over the full pool, so the full held-out grid is hours -- which is the reason it
    is an explicitly named arm rather than part of ``study``.
    """
    target = Path(cache) if cache is not None else config.ARTIFACTS / "semif_scores_scored.jsonl"
    return Experiment(
        name=name or "semif.produce",
        datasets=dataset_axis(("mutmut",)),
        features=structured_feature_axis(),
        models=semif_scoring_model_axis(
            target,
            candidates_mode=candidates,
            instruction=instruction,
            feature_mode=feature_mode,
            placement=placement,
        ),
        populations=population_axis(("fault_bearing",)),
        splits=split_axis(),
        knobs=Knobs(
            seed=config.SEED,
            budgets=(0.05,),
            n_bootstrap=0,
            candidates=candidates,
        ),
        history=True,
        note="SemIf score production, declared as a cell (GPU tier)",
    )


def dataset(label: str = "mutmut", *, knobs: Knobs | None = None, **shared) -> Dataset:
    """The built dataset for a label source, for a caller that needs the data itself.

    For an adapter rendering a dataset description, or a check computing a statistic, rather
    than for a sweep over it. Rebuilding is cheap and explicitly permitted: two datasets may
    coexist in one process (``docs/refactor.md`` §7), so a consumer does not have to reach into a
    run's internals to get at what it measured.
    """
    element = dataset_axis((label,)).get(f"marshmallow_{label}")
    value = element.build(Binding(env=Environment(knobs=knobs or Knobs(), shared=dict(shared))))
    if not isinstance(value, Dataset):
        raise TypeError(f"dataset element {element.name!r} did not build a Dataset: {value!r}")
    return value
