"""The study's factors, by role.

A *role* is a slot in a design point and a *factor* is what that slot can range over, so everything
here is a value the layer turns into levels: the datasets, the feature blocks, the ranker
sets, the averaging subsets and the split. The builders are functions rather than module
constants because several of them hold ranker instances that train on use -- a fresh factor
per condition keeps two runs from sharing one model object.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .. import bundles, config, features
from ..data import datasets, splits, subsets
from ..experiment import (
    FACTOR_DATASET,
    FACTOR_FEATURES,
    FACTOR_MODEL,
    FACTOR_SPLIT,
    FACTOR_SUBSET,
    Factor,
    Level,
    constant,
)

# Imported by name: ``model_factor`` has a parameter called ``rankers``, so a module binding
# of the same name would be shadowed by it inside that function.
from ..model.rankers import (
    LexicalRanker,
    ProducedScores,
    Ranker,
    default_rankers,
)

# --- datasets ---------------------------------------------------------------


def _marshmallow(labels: str) -> Level:
    return Level(
        f"marshmallow_{labels}",
        lambda b, labels=labels: datasets.marshmallow(labels=labels, order_seed=b.controls.seed),
        note=f"the mutation-testing dataset, {labels} labels",
    )


def _bundle_dataset(base_labels: str, rung: int, pool: str) -> Level:
    """A dataset derived from another dataset -- the pattern that needs no special support.

    ``make`` is arbitrary Python, so a derived dataset is just a level whose builder
    constructs (and here shares) its base.
    """

    def make(binding, rung=rung, pool=pool, base_labels=base_labels):
        base = datasets.marshmallow(labels=base_labels, order_seed=binding.controls.seed)
        return datasets.bundles(base, rung, seed=binding.controls.seed, pool=pool)

    kind = bundles.RUNGS[rung][2]
    return Level(
        f"bundles_r{rung}_{pool}_{base_labels}",
        make,
        note=f"rung {rung} ({kind} distractors), pool={pool}",
    )


def dataset_factor(
    label_sources: Sequence[str] = ("mutmut", "full"),
    *,
    bundle_rungs: Sequence[int] = (),
    pool: str = "focal",
) -> Factor:
    """One level per label source, plus one per bundle rung when asked for.

    A label source is a dataset, not a control, because a source is what the labels come from:
    the refactor made it a constructor argument precisely so two label sources can coexist in
    one process, and a factor is how that gets expressed.
    """
    levels = [_marshmallow(labels) for labels in label_sources]
    levels.extend(
        _bundle_dataset(label_sources[0], rung, pool) for rung in bundle_rungs
    )
    return Factor(FACTOR_DATASET, tuple(levels), note="one dataset per label source")


# --- feature sets -----------------------------------------------------------


def structured_feature_factor() -> Factor:
    return Factor(
        FACTOR_FEATURES,
        (constant("structured", features.STRUCTURED, tier="cpu", note="the 15 declared columns"),),
        note="the one block the study's headline condition uses",
    )


# --- models -----------------------------------------------------------------


#: The BM25 ablation probes from ``docs/plan.md``: shuffle the change text, the test text, or both,
#: and re-score the same pairs. They are *controls* rather than competitors, but they are
#: produced by the same machinery from the same context, so they are model levels instead of a
#: driver's second loop with its own bookkeeping.
ABLATION_MODELS: tuple[tuple[str, dict], ...] = (
    ("bm25_change_shuffled", {"shuffle_changes": True}),
    ("bm25_test_shuffled", {"shuffle_tests": True}),
    ("bm25_both_shuffled", {"shuffle_changes": True, "shuffle_tests": True}),
)

def _model_element(ranker: Ranker, tier: str = "cpu", note: str = "") -> Level:
    """A level for a ranker.

    ``tier`` is the caller's, not the ranker's: it says whether *measuring* this level
    needs a GPU, which no ranker in the study does -- SemIf reads a precomputed cache, which
    is why it declares an artifact requirement instead.
    """
    return Level(
        ranker.name,
        lambda _binding, ranker=ranker: ranker,
        tier=tier,
        note=note or type(ranker).__name__,
    )


def model_factor(
    rankers: Sequence[Ranker] | None = None,
    *,
    ablations: bool = False,
    include_semif: bool = True,
) -> Factor:
    chosen = (
        list(rankers)
        if rankers is not None
        else default_rankers(include_semif=include_semif)
    )
    if ablations:
        chosen.extend(
            LexicalRanker(**kwargs) for _, kwargs in ABLATION_MODELS
        )
    return Factor(
        FACTOR_MODEL,
        tuple(_model_element(s) for s in chosen),
        note="the study's ranker set" + (", plus the BM25 shuffle controls" if ablations else ""),
    )


def semif_scoring_model_factor(
    cache: Path,
    *,
    candidate_policy: str = "full",
    instruction: str | None = None,
    feature_mode: str | None = None,
    placement: str = "instruct",
    shuffle: bool = False,
) -> Factor:
    """The one model whose cache the design point **produces** instead of reading.

    Every other SemIf level in the study declares a cache as an artifact requirement, so an
    absent one is a undefined design point. This one declares none: the design point scores the held-out
    changes against the candidate pool and writes the cache. That makes the study's most
    expensive step visible to the layer -- with a tier, a cost estimate and a design point key -- rather
    than a precondition no one can see.

    The prompt configuration is part of the level's **name**, exactly as the cache filename
    encodes it today, because the score key is keyed on the name: two wordings must not share a
    matrix.
    """
    parts = ["semif_scored", candidate_policy]
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
            accessors.candidate_sets(ctx.ds, candidate_policy),
            path,
            instruction=instruction,
            feature_mode=feature_mode,
            placement=placement,
            shuffle=shuffle,
        )

    def verify(ctx, path):
        """Refuse a cache that does not cover the pairs this design point needs.

        The cache is identified by its path, so an existing file may have been produced for
        other rows, another candidate pool or another wording -- and it would be read as this
        design point's. The pair set is the same one ``score_context`` builds, so a complete cache for
        this context passes and any other raises instead of yielding a number.
        """
        from ..data import accessors
        from ..model import semif_runner

        missing = semif_runner.missing_pairs(
            ctx.ds,
            ctx.split.test_idx,
            accessors.candidate_sets(ctx.ds, candidate_policy),
            path,
            instruction=instruction,
            feature_mode=feature_mode,
            placement=placement,
            shuffle=shuffle,
        )
        if missing:
            raise RuntimeError(
                f"{Path(path).name} is missing {len(missing)} pair(s) this design point needs; it "
                "was produced for a different context (rows, candidate pool or prompt) or "
                "is torn -- delete it and re-run with the GPU tier"
            )

    return Factor(
        FACTOR_MODEL,
        (
            Level(
                name,
                lambda _binding: ProducedScores(
                    name, cache, produce, verifier=verify
                ),
                tier="gpu",
                note=(
                    "SemIf scores for the held-out changes over the "
                    f"{candidate_policy} candidate pool, produced by the design point when "
                    f"{cache.name} is absent"
                ),
            ),
        ),
        note="score production declared as a design point",
    )


# --- subsets ------------------------------------------------------------


def subset_factor(
    names: Sequence[subsets.Subset | str] = (),
    *,
    low_cooccurrence: Sequence[int] = (),
) -> Factor:
    """Populations, by registry name or as values, plus parameterised low_cooccurrence thresholds.

    A parameterised subset is passed as a value, because it is not in the registry: the
    registry holds the study's named vocabulary, which is a different thing from the set of
    subsets a particular sweep uses. A low_cooccurrence threshold and a starvation threshold are both
    of the second kind -- the number is part of the subset's identity, so it belongs to the
    level rather than to a global name table.
    """
    levels: list[Level] = []
    for spec in names:
        subset = subsets.resolve(spec)
        levels.append(
            constant(
                subset.name,
                subset,
                note="from subsets.STUDY" if isinstance(spec, str) else subset.note,
            )
        )
    for threshold in low_cooccurrence:
        subset = subsets.low_cooccurrence(threshold)
        levels.append(constant(subset.name, subset, note=subset.note))
    return Factor(
        FACTOR_SUBSET,
        tuple(levels),
        note="the averaging subsets a metric is reported over",
    )


# --- splits -----------------------------------------------------------------


def split_factor(
    train_fraction: float = config.DEFAULT_TRAIN_FRACTION,
    shuffle: bool = False,
) -> Factor:
    label = f"split{int(round(train_fraction * 100))}" + ("_shuffled" if shuffle else "")
    return Factor(
        FACTOR_SPLIT,
        (
            Level(
                label,
                lambda binding: splits.make_split(
                    binding.dataset,
                    train_fraction=train_fraction,
                    shuffle=shuffle,
                    seed=binding.controls.seed,
                ),
                tier="cpu",
                note=f"{train_fraction:.0%} train prefix" + (", shuffled" if shuffle else ""),
            ),
        ),
        note="evaluation configuration, not a property of the data",
    )
