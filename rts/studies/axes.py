"""The study's axes, by role.

A *role* is a slot in a cell and an *axis* is what that slot can range over, so everything
here is a value the layer turns into elements: the datasets, the feature blocks, the selector
sets, the averaging populations and the split. The builders are functions rather than module
constants because several of them hold selector instances that train on use -- a fresh axis
per arm keeps two runs from sharing one model object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .. import bundles, config, features
from ..data import datasets, populations, splits
from ..experiment import (
    ROLE_DATASET,
    ROLE_FEATURES,
    ROLE_MODEL,
    ROLE_POPULATION,
    ROLE_SPLIT,
    Axis,
    Element,
    constant,
)

# Imported by name: ``model_axis`` has a parameter called ``selectors``, so a module binding
# of the same name would be shadowed by it inside that function.
from ..model.selectors import LexicalSelector, ProducedScores, Selector, default_selectors

# --- datasets ---------------------------------------------------------------


def _marshmallow(labels: str) -> Element:
    return Element(
        f"marshmallow_{labels}",
        lambda b, labels=labels: datasets.marshmallow(labels=labels, order_seed=b.knobs.seed),
        note=f"the mutation-testing dataset, {labels} labels",
    )


def _bundle_dataset(base_labels: str, rung: int, pool: str) -> Element:
    """A dataset derived from another dataset -- the pattern that needs no special support.

    ``make`` is arbitrary Python, so a derived dataset is just an element whose builder
    constructs (and here shares) its base.
    """

    def make(binding, rung=rung, pool=pool, base_labels=base_labels):
        base = datasets.marshmallow(labels=base_labels, order_seed=binding.knobs.seed)
        return datasets.bundles(base, rung, seed=binding.knobs.seed, pool=pool)

    kind = bundles.RUNGS[rung][2]
    return Element(
        f"bundles_r{rung}_{pool}_{base_labels}",
        make,
        note=f"rung {rung} ({kind} distractors), pool={pool}",
    )


def dataset_axis(
    label_sources: Sequence[str] = ("mutmut", "full"),
    *,
    bundle_rungs: Sequence[int] = (),
    pool: str = "signal",
) -> Axis:
    """One element per label source, plus one per bundle rung when asked for.

    A label source is a dataset, not a knob, because a source is what the labels come from:
    the refactor made it a constructor argument precisely so two label sources can coexist in
    one process, and an axis is how that gets expressed.
    """
    elements = [_marshmallow(labels) for labels in label_sources]
    elements.extend(
        _bundle_dataset(label_sources[0], rung, pool) for rung in bundle_rungs
    )
    return Axis(ROLE_DATASET, tuple(elements), note="one dataset per label source")


# --- feature sets -----------------------------------------------------------


def structured_feature_axis() -> Axis:
    return Axis(
        ROLE_FEATURES,
        (constant("structured", features.STRUCTURED, tier="cpu", note="the 15 declared columns"),),
        note="the one block the study's headline arm uses",
    )


# --- models -----------------------------------------------------------------


#: The BM25 ablation probes from ``docs/plan.md``: shuffle the change text, the test text, or both,
#: and re-score the same pairs. They are *controls* rather than competitors, but they are
#: produced by the same machinery from the same context, so they are model elements instead of a
#: driver's second loop with its own bookkeeping.
ABLATION_MODELS: tuple[tuple[str, dict], ...] = (
    ("bm25_change_shuffled", {"shuffle_changes": True}),
    ("bm25_test_shuffled", {"shuffle_tests": True}),
    ("bm25_both_shuffled", {"shuffle_changes": True, "shuffle_tests": True}),
)

def _model_element(selector: Selector, tier: str = "cpu", note: str = "") -> Element:
    """An element for a selector.

    ``tier`` is the caller's, not the selector's: it says whether *measuring* this element
    needs a GPU, which no selector in the study does -- SemIf reads a precomputed cache, which
    is why it declares an artifact requirement instead.
    """
    return Element(
        selector.name,
        lambda _binding, selector=selector: selector,
        tier=tier,
        note=note or type(selector).__name__,
    )


def model_axis(
    selectors: Sequence[Selector] | None = None,
    *,
    ablations: bool = False,
    include_semif: bool = True,
) -> Axis:
    chosen = (
        list(selectors)
        if selectors is not None
        else default_selectors(include_semif=include_semif)
    )
    if ablations:
        chosen.extend(
            LexicalSelector(**kwargs) for _, kwargs in ABLATION_MODELS
        )
    return Axis(
        ROLE_MODEL,
        tuple(_model_element(s) for s in chosen),
        note="the study's selector set" + (", plus the BM25 shuffle controls" if ablations else ""),
    )


def semif_scoring_model_axis(
    cache: Path,
    *,
    candidates_mode: str = "full",
    instruction: str | None = None,
    feature_mode: str | None = None,
    placement: str = "instruct",
    shuffle: bool = False,
) -> Axis:
    """The one model whose cache the cell **produces** instead of reading.

    Every other SemIf element in the study declares a cache as an artifact requirement, so an
    absent one is an unmeasured cell. This one declares none: the cell scores the held-out
    changes against the candidate pool and writes the cache. That makes the study's most
    expensive step visible to the layer -- with a tier, a cost estimate and a cell key -- rather
    than a precondition no one can see.

    The prompt configuration is part of the element's **name**, exactly as the cache filename
    encodes it today, because the score key is keyed on the name: two wordings must not share a
    matrix.
    """
    parts = ["semif_scored", candidates_mode]
    if instruction:
        parts.append(f"instr_{instruction}")
    if feature_mode:
        parts.append(feature_mode)
    if placement != "instruct":
        parts.append(placement)
    if shuffle:
        parts.append("shuffled")
    name = "_".join(parts)

    def produce(ctx, path):
        from ..data import accessors
        from ..model import semif_runner

        return semif_runner.score_context(
            ctx.ds,
            ctx.split.test_idx,
            accessors.candidates(ctx.ds, candidates_mode),
            path,
            instruction=instruction,
            feature_mode=feature_mode,
            placement=placement,
            shuffle=shuffle,
        )

    def verify(ctx, path):
        """Refuse a cache that does not cover the pairs this cell needs.

        The cache is identified by its path, so an existing file may have been produced for
        other rows, another candidate pool or another wording -- and it would be read as this
        cell's. The pair set is the same one ``score_context`` builds, so a complete cache for
        this context passes and any other raises instead of yielding a number.
        """
        from ..data import accessors
        from ..model import semif_runner

        missing = semif_runner.missing_pairs(
            ctx.ds,
            ctx.split.test_idx,
            accessors.candidates(ctx.ds, candidates_mode),
            path,
            instruction=instruction,
            feature_mode=feature_mode,
            placement=placement,
            shuffle=shuffle,
        )
        if missing:
            raise RuntimeError(
                f"{Path(path).name} is missing {len(missing)} pair(s) this cell needs; it "
                "was produced for a different context (rows, candidate pool or prompt) or "
                "is torn -- delete it and re-run with the GPU tier"
            )

    return Axis(
        ROLE_MODEL,
        (
            Element(
                name,
                lambda _binding: ProducedScores(
                    name, cache, produce, verifier=verify
                ),
                tier="gpu",
                note=(
                    "SemIf scores for the held-out changes over the "
                    f"{candidates_mode} candidate pool, produced by the cell when "
                    f"{cache.name} is absent"
                ),
            ),
        ),
        note="score production declared as a cell",
    )


# --- populations ------------------------------------------------------------


def population_axis(
    names: Sequence["populations.Population | str"] = (),
    *,
    sparse: Sequence[int] = (),
) -> Axis:
    """Populations, by registry name or as values, plus parameterised sparse thresholds.

    A parameterised population is passed as a value, because it is not in the registry: the
    registry holds the study's named vocabulary, which is a different thing from the set of
    populations a particular sweep uses. A sparse threshold and a starvation threshold are both
    of the second kind -- the number is part of the population's identity, so it belongs to the
    element rather than to a global name table.
    """
    elements: list[Element] = []
    for spec in names:
        population = populations.resolve(spec)
        elements.append(
            constant(
                population.name,
                population,
                note="from populations.STUDY" if isinstance(spec, str) else population.note,
            )
        )
    for threshold in sparse:
        population = populations.low_pair_recurrence(threshold)
        elements.append(constant(population.name, population, note=population.note))
    return Axis(
        ROLE_POPULATION,
        tuple(elements),
        note="the averaging populations a metric is reported over",
    )


# --- splits -----------------------------------------------------------------


def split_axis(
    train_fraction: float = config.DEFAULT_TRAIN_FRACTION,
    shuffle: bool = False,
) -> Axis:
    label = f"split{int(round(train_fraction * 100))}" + ("_shuffled" if shuffle else "")
    return Axis(
        ROLE_SPLIT,
        (
            Element(
                label,
                lambda binding: splits.make_split(
                    binding.dataset,
                    train_fraction=train_fraction,
                    shuffle=shuffle,
                    seed=binding.knobs.seed,
                ),
                tier="cpu",
                note=f"{train_fraction:.0%} train prefix" + (", shuffled" if shuffle else ""),
            ),
        ),
        note="evaluation configuration, not a property of the data",
    )
