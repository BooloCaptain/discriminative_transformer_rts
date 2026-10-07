"""RTS rankers: the interface, the context, and the generic score mechanisms.

Every ranker implements ``scores(ctx) -> [n_changes, n_tests]``. Higher is better;
evaluation takes the top ``k`` per change. A ranker never sees labels for the changes it is
being evaluated on: the split travels in the context, and fitting is the ranker's own
responsibility.

This module holds what is **not** specific to any one study -- the context a ranker reads,
the interface itself, the trivial random baseline, the fitted-free ranked-average
combination, and the two cache forms: a score matrix read from a file
(:class:`CachedScores`) and one produced on first use (:class:`ProducedScores`). The
concrete models a study measures are plugins and live beside that study, so the kernel
carries no model of its own beyond the trivial one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import config, features
from ..data import accessors, splits
from ..data.contract import Dataset, Diagnostic


@dataclass
class Context:
    """Everything a ranker may read.

    The split is here rather than derived inside a ranker because it is evaluation
    configuration: the experiment chooses it, every ranker in a run must see the same
    one, and a ranker that invented its own would train and be evaluated on different
    partitions without saying so.
    """

    ds: Dataset
    features: features.FeatureMatrix
    split: splits.Split
    bm25: np.ndarray  # [n_changes, n_tests]
    extras: dict = field(default_factory=dict)
    #: The run's seed. A ranker that needs randomness reads it here rather than from
    #: ``config``, so that a run-level seed override reaches the model instead of stopping at
    #: the experiment layer.
    seed: int = config.SEED

    @property
    def X(self) -> np.ndarray:
        return self.features.X

    @property
    def names(self) -> list[str]:
        return list(self.features.columns)

    def feature(self, name: str) -> np.ndarray:
        return self.features.column(name)

    @property
    def diagnostics(self) -> tuple[Diagnostic, ...]:
        """Diagnostics raised while materialising the features for this context."""
        return self.features.diagnostics


class Ranker:
    name = "ranker"

    def scores(self, ctx: Context) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def requirements(self) -> tuple[str, ...]:
        """Inputs beyond the dataset contract that this ranker reads.

        The ranker half of the declaration the dataset contract already has: the entity
        that reads the inputs says what it needs, so the need cannot drift from the code.
        The experiment layer resolves these at the design point boundary, and an unresolved
        requirement makes the design point *undefined* rather than raising from inside ``scores``.

        Two spellings are understood: ``"artifact:<path>"`` for a file the ranker reads
        (a score cache), and a :class:`rts.data.contract.Requirement` value (``"coverage"``,
        ``"durations"``, ...) for inputs that come from the dataset. An unrecognised
        spelling raises where it is resolved, for the same reason
        :meth:`rts.data.contract.Dataset.has_capability` does.
        """
        return ()


class RandomRanker(Ranker):
    """A uniform random score per (change, test) pair.

    The floor every other model must clear, and small enough to serve as the worked example
    of the ranker plugin point.
    """

    name = "random"

    def __init__(self, seed: int | None = None):
        # ``None`` means "the run's seed", so an experiment's seed control reaches the model.
        self.seed = seed

    def scores(self, ctx: Context) -> np.ndarray:
        rng = np.random.default_rng(ctx.seed if self.seed is None else self.seed)
        return rng.random((ctx.ds.n_changes, ctx.ds.n_tests)).astype(np.float32)


def normalised_rank(scores: np.ndarray, candidate_sets: np.ndarray) -> np.ndarray:
    """Per-change descending rank, normalised to ``[0, 1]``; 0 is best.

    Non-candidate sets get ``inf`` so they can never win. This is what lets two rankers on
    different scales be averaged without fitting a weight: each change's candidate sets are ranked
    independently, so the result is invariant to either input's units -- which is what makes the
    combination below leakage-safe.
    """
    masked = np.where(candidate_sets, scores, -np.inf)
    order = np.argsort(-masked, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(scores.shape[1])[None, :]
    denom = max(scores.shape[1] - 1, 1)
    return np.where(candidate_sets, ranks / denom, np.inf)


#: A score loader: ``(cache path, dataset) -> [n_changes, n_tests]``. The dataset is a
#: parameter because a cache may need canonicalising against the pool it is read into.
ScoreLoader = Callable[[Path, Dataset], np.ndarray]

#: A score *producer*: ``(context, cache path) -> (matrix, stats)``. The context supplies the
#: dataset, the split and the candidate mask, so a producer reads nothing from a driver.
ScoreProducer = Callable[["Context", Path], tuple[np.ndarray, dict]]


def load_matrix(path: Path, _ds: Dataset) -> np.ndarray:
    """The trivial loader: a cache that is already a ``[n_changes, n_tests]`` matrix."""
    return np.load(path)


class CachedScores(Ranker):
    """Scores read from a precomputed ``[n_changes, n_tests]`` artifact.

    This is the general form of every cached model: reranker caches, instruction-wording
    caches and an embedding baseline all differ only in which file they read and how it is
    parsed. Declaring the cache as a *requirement* is what matters: an absent one makes the
    design point undefined with the path rather than raising from inside ``scores``, which is
    the difference between a hole in the report and a dead run when a study has a dozen caches
    of which any may be missing.

    ``loader`` parses the file; the default reads a NumPy ``.npy`` matrix, and a study whose
    cache is a scored-pair log supplies its own.
    """

    def __init__(self, name: str, path: Path | str, loader: ScoreLoader | None = None):
        self.name = name
        self.path = Path(path)
        self._loader = loader or load_matrix

    def requirements(self) -> tuple[str, ...]:
        """The precomputed cache. Producing it is a precondition, not something a design point does."""
        return (f"artifact:{self.path}",)

    def scores(self, ctx: Context) -> np.ndarray:
        return self._loader(self.path, ctx.ds)


class RankAverageRanker(Ranker):
    """The fitted-free combination of several rankers: mean normalised rank position.

    Nothing is fitted, so it cannot leak across the train/evaluation boundary the way a blend
    weight fitted on the rows being evaluated would. The reason to include it is the
    *redundancy* test: if the second model adds independent signal then the average beats both
    parents, and if it is redundant the average sits between them. That is a stronger statement
    than a correlation, and it needs no extra data.

    Ranks are taken over each change's own candidate set (see :func:`normalised_rank`), which is
    why a candidate mode is required: the average is only defined relative to a pool.
    """

    def __init__(
        self,
        name: str,
        rankers: Sequence[Ranker],
        candidate_policy: str = "full",
    ):
        if len(rankers) < 2:
            raise ValueError(f"{name!r} averages rank positions, so it needs two or more parents")
        self.name = name
        self.rankers = tuple(rankers)
        self.candidate_policy = candidate_policy

    def requirements(self) -> tuple[str, ...]:
        """Whatever the parents read, in first-seen order and without repeats."""
        seen: list[str] = []
        for ranker in self.rankers:
            for requirement in ranker.requirements():
                if requirement not in seen:
                    seen.append(requirement)
        return tuple(seen)

    def scores(self, ctx: Context) -> np.ndarray:
        candidate_sets = accessors.candidate_sets(ctx.ds, self.candidate_policy)
        norms = [normalised_rank(s.scores(ctx), candidate_sets) for s in self.rankers]
        return -np.mean(norms, axis=0)


class ProducedScores(Ranker):
    """Scores read from a cache, **produced on first use** if the cache is absent.

    :class:`CachedScores` treats its artifact as a precondition: absent means the design point is
    undefined with the path. This treats it as an *output*. That is the right shape for an
    expensive score production -- scoring (change, test) pairs with a large model -- where the
    artifact is exactly what the design point would compute, and where leaving production outside the
    layer makes a stale cache indistinguishable from a fresh one and the expensive condition invisible.

    It is a *general* form: the scorer is a callable, so the pinned model, its configuration and
    the rows to score are supplied by whichever level declares the capability. That also makes
    the produce path testable without the model.

    ``requirements()`` is deliberately empty. Declaring the cache would make the design point
    undefined *before* it could produce it, which is the trap this class exists to avoid; the
    level that wraps it declares ``tier="gpu"`` instead, so a run that is not spending the
    expensive tier reports the design point as undefined rather than silently reading a half-built cache.

    A cache is identified by its **path**, so "it exists" is not the same claim as "it is the
    one this design point needs": the file may have been produced for other rows, another candidate
    pool or another configuration, or torn mid-write. ``verifier`` is the level's chance to
    refuse such a file -- it is handed the context and the path and raises if the cache cannot
    serve it. Without one the file is trusted, which is the documented contract rather than a
    guarantee.

    A verifier *raises* rather than producing a undefined design point, and that is deliberate: the
    level declared that it can produce this cache, so a file at the path that cannot serve
    the context is a broken invocation with one remedy, not a hole in the grid.
    """

    def __init__(
        self,
        name: str,
        cache: Path | str,
        scorer: ScoreProducer,
        loader: ScoreLoader | None = None,
        verifier: Callable[[Context, Path], None] | None = None,
    ):
        self.name = name
        self.path = Path(cache)
        self._produce = scorer
        self._loader = loader or load_matrix
        self._verify = verifier
        #: What the last production did -- pairs scored, throughput, whether it resumed.
        #: Recorded on the design point so a produced number carries its cost, the same way a
        #: measured design point carries its seconds.
        self.last_stats: dict = {}

    def requirements(self) -> tuple[str, ...]:
        """None: the cache is this ranker's output, not its precondition."""
        return ()

    def scores(self, ctx: Context) -> np.ndarray:
        if self.path.exists():
            if self._verify is not None:
                self._verify(ctx, self.path)
            self.last_stats = {"read_from": str(self.path), "produced": False}
        else:
            _, self.last_stats = self._produce(ctx, self.path)
            self.last_stats["produced"] = True
        return self._loader(self.path, ctx.ds)


__all__ = [
    "CachedScores",
    "Context",
    "ProducedScores",
    "RandomRanker",
    "RankAverageRanker",
    "Ranker",
    "ScoreLoader",
    "ScoreProducer",
    "load_matrix",
    "normalised_rank",
]
