"""RTS rankers: statistical baselines, XGBoost, and the SemIf reranker.

Every ranker implements ``scores(ctx) -> [n_changes, n_tests]``. Higher is better;
evaluation takes the top ``k`` per change. Selectors never see labels for changes they
are being evaluated on -- the structured features are already cumulative, and XGBoost is
fit on the training window only.

The context carries a :class:`~rts.features.block.FeatureMatrix` rather than a bare
``(X, names)`` pair. That is what lets a ranker ask for a column by name and fail
loudly if it was renamed, instead of doing ``names.index(...)`` and silently reading
whatever moved into that slot. It also carries the matrix's diagnostics, so a model-input
coercion -- a column that could not be measured and was zeroed -- travels to the layer
that produced the number rather than being printed and forgotten.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import config, features
from ..data import accessors, splits
from ..data.contract import Dataset, Diagnostic
from . import semif

# Feature families, taken from the block's own declaration so there is one definition of
# what "the temporal family" is rather than a second list here that can drift from it.
TEMPORAL_FEATURES = features.STRUCTURED.family("temporal")
COVERAGE_FEATURES = features.STRUCTURED.family("coverage")
PROXIMITY_FEATURES = features.STRUCTURED.family("proximity")


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

    # --- helpers ---
    @staticmethod
    def _prefer_small_coverage(covered: np.ndarray, n_covering: np.ndarray) -> np.ndarray:
        """Rank covered tests first, then by ascending coverage-set size.

        Ties inside a coverage set are otherwise arbitrary, and a test whose function is
        covered by 3 tests is far more likely to be the killer than one in a set of 761.
        """
        return np.where(covered > 0, 1e6, 0.0) - n_covering


class RandomRanker(Ranker):
    name = "random"

    def __init__(self, seed: int | None = None):
        # ``None`` means "the run's seed", so an experiment's seed control reaches the model.
        self.seed = seed

    def scores(self, ctx: Context) -> np.ndarray:
        rng = np.random.default_rng(ctx.seed if self.seed is None else self.seed)
        return rng.random((ctx.ds.n_changes, ctx.ds.n_tests)).astype(np.float32)


class RecencyRanker(Ranker):
    """Select tests that failed most recently. Expected to be uninformative.

    The synthetic history has no real temporal structure, so this baseline exists to be
    reported as a limitation of the setup, not as a finding about recency.
    """

    name = "recency"

    def scores(self, ctx: Context) -> np.ndarray:
        return -ctx.feature("failure_recency")


class FailureRateRanker(Ranker):
    name = "failure_rate"

    def scores(self, ctx: Context) -> np.ndarray:
        return ctx.feature("cumulative_failure_rate")


class CoverageRanker(Ranker):
    """Select tests that cover the mutated function (mutmut's own association)."""

    name = "coverage"

    def scores(self, ctx: Context) -> np.ndarray:
        return self._prefer_small_coverage(
            ctx.feature("function_coverage"), ctx.feature("coverage_set_size")
        )


class StructuralRuleRanker(Ranker):
    """Hand-built rule: covered tests whose file name matches the changed module.

    This exists because it turns out to explain most of the achievable recall. The
    conjunction ``function_coverage AND filename_match`` narrows the suite to a median
    of 9 candidate tests, so most of the task is solved by cheap structural funneling
    rather than by anything semantic. Any model claiming to work must be measured against
    this, not just against random.
    """

    name = "structural_rule"

    def scores(self, ctx: Context) -> np.ndarray:
        covered = ctx.feature("function_coverage")
        name_match = ctx.feature("filename_match")
        n_lines = ctx.feature("test_lines")
        # Covered first, then name-matching, then shortest test first.
        return covered * 2.0 + name_match + 1.0 / (1.0 + n_lines)


class LexicalRanker(Ranker):
    """BM25 between the changed lines and the test source.

    The two shuffle flags produce a *shuffled* BM25 instead of reading the context's canonical
    one, which is the change-shuffle ablation ``docs/plan.md`` specifies: if recall barely drops when
    the change text is shuffled, the lexical signal is a change-independent test prior rather
    than a match between the change and the test. Expressing the probe as a ranker keeps it in
    the same grid as everything else, with the same provenance, instead of in a driver's second
    loop with its own bookkeeping.
    """

    name = "bm25_lexical"

    def __init__(self, shuffle_changes: bool = False, shuffle_tests: bool = False):
        self.shuffle_changes = shuffle_changes
        self.shuffle_tests = shuffle_tests
        if shuffle_changes and shuffle_tests:
            self.name = "bm25_both_shuffled"
        elif shuffle_changes:
            self.name = "bm25_change_shuffled"
        elif shuffle_tests:
            self.name = "bm25_test_shuffled"

    def scores(self, ctx: Context) -> np.ndarray:
        if not (self.shuffle_changes or self.shuffle_tests):
            return ctx.bm25
        return features.text.build_bm25_scores(
            ctx.ds,
            shuffle_changes=self.shuffle_changes,
            shuffle_tests=self.shuffle_tests,
            seed=ctx.seed,
        )


class PerPoolRandomRanker(Ranker):
    """Uniform scores drawn per change over that change's **own** candidate pool.

    :class:`RandomRanker` draws over the whole matrix, which is the right baseline when every
    change is ranked against one suite. When each change has its own pool, drawing over the
    union gives each change a different set of ranks, so the baseline would move for a reason
    that has nothing to do with the method under test.

    One generator for the whole matrix, consumed change by change in canonical order. The draw
    *sequence* is part of a recorded baseline, so it is deliberately not re-derived per row.
    """

    name = "random"

    def __init__(self, candidate_policy: str = "full", seed: int | None = None):
        self.candidate_policy = candidate_policy
        # ``None`` means "the run's seed", the same rule the other rankers follow.
        self.seed = seed

    def scores(self, ctx: Context) -> np.ndarray:
        candidate_sets = accessors.candidate_sets(ctx.ds, self.candidate_policy)
        rng = np.random.default_rng(ctx.seed if self.seed is None else self.seed)
        out = np.full((ctx.ds.n_changes, ctx.ds.n_tests), -1e9, dtype=np.float32)
        for row in range(ctx.ds.n_changes):
            cols = np.flatnonzero(candidate_sets[row])
            if cols.size == 0:
                continue
            out[row, cols] = rng.random(cols.size).astype(np.float32)
        return out


class PerPoolLexicalRanker(Ranker):
    """BM25 fitted once per change, over that change's own candidate documents.

    :class:`LexicalRanker` scores every pair against one index fitted over the whole suite,
    which makes a term's idf depend on every other change's tests. That is right when one suite
    serves every change. With per-change pools it is a different quantity -- and for a pooled
    multi-project corpus it would let one project's vocabulary move another project's scores.

    ``query`` selects which side of the change is the query:

    * ``"change"`` -- the added-and-removed-lines extraction the study's own BM25 uses;
    * ``"diff"`` -- the raw unified diff, ``+``/``-`` markers and hunk headers included, which
      is what the BugsInPy condition's recorded numbers used.

    The two are recorded as a choice rather than reconciled: they differ, and which one a model
    is entitled to see is a decision about the condition, not a spelling of one idea.
    """

    name = "bm25_lexical"

    def __init__(self, candidate_policy: str = "full", query: str = "change"):
        if query not in ("diff", "change"):
            raise ValueError(f"query must be 'diff' or 'change', not {query!r}")
        self.candidate_policy = candidate_policy
        self.query = query

    def scores(self, ctx: Context) -> np.ndarray:
        candidate_sets = accessors.candidate_sets(ctx.ds, self.candidate_policy)
        out = np.full((ctx.ds.n_changes, ctx.ds.n_tests), -1e9, dtype=np.float32)
        for row, change in enumerate(ctx.ds.changes):
            cols = np.flatnonzero(candidate_sets[row])
            if cols.size == 0:
                continue
            docs = [ctx.ds.test_source(ctx.ds.test_ids[int(c)]) or "" for c in cols]
            scorer = features.text.BM25Scorer().fit(docs)
            query = (
                ctx.ds.diff_text(change)
                if self.query == "diff"
                else features.derived.change_query_text(ctx.ds, change)
            )
            out[row, cols] = scorer.score(query)
        return out


class XGBoostRanker(Ranker):
    """Gradient-boosted trees on the structured features.

    Three independent switches:

    * ``include_lexical`` adds the BM25 score as a input column. This is the design point that
      separates "the features matter" from "the model matters": if trees given the lexical
      signal match the transformer, the transformer's semantic machinery is not doing the
      work.
    * ``exclude_temporal`` drops the cumulative temporal features. This isolates how much of
      the score is failure-history memorisation rather than coverage or static structure
      -- important here because mutation testing revisits the same function many times, so
      a (function, killing-test) pair recurs far more often than it would in real
      evolution.
    * ``exclude_coverage`` drops the coverage features, which is the target regime: where
      the changed code does not run in the test process there is no coverage to have.
    * ``exclude`` drops arbitrary columns by name, so an ablation that is not one of the
      named families does not need a new flag.
    """

    def __init__(
        self,
        include_lexical: bool = False,
        exclude_temporal: bool = False,
        exclude_coverage: bool = False,
        exclude: tuple[str, ...] = (),
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.15,
        seed: int | None = None,
        extra_score_files: dict[str, Path] | None = None,
        candidate_policy: str | None = None,
    ):
        self.include_lexical = include_lexical
        self.exclude_temporal = exclude_temporal
        self.exclude_coverage = exclude_coverage
        self.exclude = tuple(exclude)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        # ``None`` means "the run's seed", so an experiment's seed control reaches the model
        # rather than stopping at ``config``. The same rule as ``RandomRanker``.
        self.seed = seed
        # P5: extra per-(change, test) score columns supplied by another model. The
        # canonical use is adding the SemIf reranker score to the structured features to
        # test whether the transformer is redundant given BM25 plus cheap structure.
        self.extra_score_files = dict(extra_score_files or {})
        # When set, XGBoost trains only on the candidate pairs it will actually be asked to
        # rank. Training over the whole suite (the historical default) lets it spend its
        # capacity learning the candidate mask, which is constant at evaluation time -- that
        # is what produced the once-reported 0.71 importance on ``function_coverage``. None
        # preserves the old behaviour so existing documented numbers stay reproducible.
        self.candidate_policy = candidate_policy
        parts = ["xgboost"]
        parts.append("struct" if not exclude_temporal else "static")
        if exclude_coverage:
            parts.append("nocov")
        if include_lexical:
            parts.append("lex")
        for tag in self.extra_score_files:
            parts.append(tag)
        self.name = "_".join(parts)
        self._model = None
        self.importances_: dict[str, float] = {}
        self._extra_cache: dict[str, np.ndarray] = {}

    def requirements(self) -> tuple[str, ...]:
        """One score-cache artifact per extra column, since each is read from a file."""
        return tuple(f"artifact:{path}" for path in self.extra_score_files.values())

    def _dropped(self) -> set[str]:
        dropped: set[str] = set(self.exclude)
        if self.exclude_temporal:
            dropped |= set(TEMPORAL_FEATURES)
        if self.exclude_coverage:
            dropped |= set(COVERAGE_FEATURES)
        return dropped

    def _kept_columns(self, ctx: Context) -> list[int]:
        dropped = self._dropped()
        return [i for i, n in enumerate(ctx.names) if n not in dropped]

    def _extras(self, ctx: Context) -> list[tuple[str, np.ndarray]]:
        """Extra score matrices, loaded from cache on first use."""
        for tag, path in self.extra_score_files.items():
            if tag not in self._extra_cache:
                self._extra_cache[tag] = semif.load_scores(path, ctx.ds)
        return [(tag, self._extra_cache[tag]) for tag in self.extra_score_files]

    def _design(
        self,
        ctx: Context,
        rows: np.ndarray,
        cols_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Flattened design matrix over (change, test) pairs.

        ``cols_mask`` (shape ``[len(rows), n_tests]``) restricts the output to a subset of
        the pairs, which is how candidate-only training is implemented.
        """
        keep = self._kept_columns(ctx)
        block = ctx.X[rows][:, :, keep]
        extras = [ctx.bm25[rows]] if self.include_lexical else []
        extras.extend(matrix[rows] for _, matrix in self._extras(ctx))
        if cols_mask is not None:
            block = block[cols_mask]
            extras = [extra[cols_mask] for extra in extras]
        flat = block.reshape(-1, len(keep))
        if extras:
            flat = np.hstack([flat] + [extra.reshape(-1, 1) for extra in extras])
        return flat

    def scores(self, ctx: Context) -> np.ndarray:
        import xgboost as xgb

        train_rows = ctx.split.train_idx
        train_mask = None
        if self.candidate_policy is not None:
            train_mask = accessors.candidate_sets(ctx.ds, self.candidate_policy)[train_rows]
        X_train = self._design(ctx, train_rows, train_mask)
        labels = accessors.labels(ctx.ds)[train_rows]
        y_train = (labels[train_mask] if train_mask is not None else labels).reshape(-1)

        model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            tree_method="hist",
            n_jobs=-1,
            random_state=ctx.seed if self.seed is None else self.seed,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)
        self._model = model

        keep = self._kept_columns(ctx)
        feature_names = [ctx.names[i] for i in keep]
        if self.include_lexical:
            feature_names.append("bm25")
        for tag, _ in self._extras(ctx):
            feature_names.append(tag)
        self.importances_ = dict(
            sorted(
                zip(feature_names, model.feature_importances_.tolist()),
                key=lambda kv: -kv[1],
            )
        )

        X_all = self._design(ctx, np.arange(ctx.ds.n_changes))
        return model.predict_proba(X_all)[:, 1].reshape(ctx.ds.n_changes, ctx.ds.n_tests)


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

    This is the general form of every cached model in the study: the reranker caches, the five
    instruction-wording caches, the direct-mode cache and the embedding baseline all differ only
    in which file they read and how it is parsed. Declaring the cache as a *requirement* is what
    matters: an absent one makes the design point undefined with the path rather than raising from
    inside ``scores``, which is the difference between a hole in the report and a dead run when
    a study has a dozen caches of which any may be missing.
    """

    def __init__(self, name: str, path: Path | str, loader: ScoreLoader | None = None):
        self.name = name
        self.path = Path(path)
        self._loader = loader or semif.load_scores

    def requirements(self) -> tuple[str, ...]:
        """The precomputed cache. Producing it is a precondition, not something a design point does."""
        return (f"artifact:{self.path}",)

    def scores(self, ctx: Context) -> np.ndarray:
        return self._loader(self.path, ctx.ds)


class SemIfRanker(CachedScores):
    """Frozen SemIf + Qwen reranker scores, read from a precomputed cache.

    Scoring is expensive and needs the SemIf checkout plus the checkpoint, so it is computed
    once by ``rts.model.semif`` and cached. See docs/implementation.md for the pinned configuration.
    """

    def __init__(self, scores_file=None):
        super().__init__("semif_reranker", scores_file or config.SEMIF_SCORES_FILE)

    @property
    def scores_file(self) -> Path:
        """The cache this reads, i.e. :attr:`CachedScores.path` under its older name."""
        return self.path


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
    undefined with the path. This treats it as an *output*. That is the right shape for the
    study's most expensive step -- scoring (change, test) pairs with a 4B reranker -- where the
    artifact is exactly what the design point would compute, and where leaving production outside the
    layer made a stale cache indistinguishable from a fresh one and the GPU condition invisible.

    It is a *general* form rather than a SemIf one: the scorer is a callable, so the pinned
    model, its prompt configuration and the rows to score are supplied by whichever level
    declares the capability. That also makes the produce path testable without a GPU.

    ``requirements()`` is deliberately empty. Declaring the cache would make the design point
    undefined *before* it could produce it, which is the trap this class exists to avoid; the
    level that wraps it declares ``tier="gpu"`` instead, so a run that is not spending the
    GPU tier reports the design point as undefined rather than silently reading a half-built cache.

    A cache is identified by its **path**, so "it exists" is not the same claim as "it is the
    one this design point needs": the file may have been produced for other rows, another candidate
    pool or another prompt wording, or torn mid-write. ``verifier`` is the level's chance to
    refuse such a file -- it is handed the context and the path and raises if the cache cannot
    serve it. Without one the file is trusted, which is the documented contract rather than a
    guarantee, and it is why the SemIf production level supplies
    :func:`rts.model.semif_runner.missing_pairs`.

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
        self._loader = loader or semif.load_scores
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


def default_rankers(include_semif: bool = True) -> list[Ranker]:
    rankers: list[Ranker] = [
        RandomRanker(),
        RecencyRanker(),
        FailureRateRanker(),
        CoverageRanker(),
        StructuralRuleRanker(),
        LexicalRanker(),
        XGBoostRanker(include_lexical=False),
        XGBoostRanker(include_lexical=True),
        XGBoostRanker(exclude_temporal=True, include_lexical=False),
        XGBoostRanker(exclude_temporal=True, exclude_coverage=True),
        XGBoostRanker(exclude_temporal=True, exclude_coverage=True, include_lexical=True),
    ]
    if include_semif:
        rankers.append(SemIfRanker())
    return rankers
