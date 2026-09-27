"""RTS selectors: statistical baselines, XGBoost, and the SemIf reranker.

Every selector implements ``scores(ctx) -> [n_changes, n_tests]``. Higher is better;
evaluation takes the top ``k`` per change. Selectors never see labels for changes they
are being evaluated on -- the structured features are already cumulative, and XGBoost is
fit on the training window only.

The context carries a :class:`~rts.features.block.FeatureMatrix` rather than a bare
``(X, names)`` pair. That is what lets a selector ask for a column by name and fail
loudly if it was renamed, instead of doing ``names.index(...)`` and silently reading
whatever moved into that slot. It also carries the matrix's warnings, so a model-input
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
from ..data.contract import Dataset, Warning
from . import semif

# Feature families, taken from the block's own declaration so there is one definition of
# what "the history family" is rather than a second list here that can drift from it.
HISTORY_FEATURES = features.STRUCTURED.family("history")
COVERAGE_FEATURES = features.STRUCTURED.family("coverage")
TRACEABILITY_FEATURES = features.STRUCTURED.family("traceability")


@dataclass
class Context:
    """Everything a selector may read.

    The split is here rather than derived inside a selector because it is evaluation
    configuration: the experiment chooses it, every selector in a run must see the same
    one, and a selector that invented its own would train and be evaluated on different
    partitions without saying so.
    """

    ds: Dataset
    features: features.FeatureMatrix
    split: splits.Split
    bm25: np.ndarray  # [n_changes, n_tests]
    extras: dict = field(default_factory=dict)
    #: The run's seed. A selector that needs randomness reads it here rather than from
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
    def warnings(self) -> tuple[Warning, ...]:
        """Warnings raised while materialising the features for this context."""
        return self.features.warnings


class Selector:
    name = "selector"

    def scores(self, ctx: Context) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def requirements(self) -> tuple[str, ...]:
        """Material beyond the dataset contract that this selector reads.

        The selector half of the declaration the dataset contract already has: the entity
        that reads the material says what it needs, so the need cannot drift from the code.
        The experiment layer resolves these at the cell boundary, and an unresolved
        requirement makes the cell *unmeasured* rather than raising from inside ``scores``.

        Two spellings are understood: ``"artifact:<path>"`` for a file the selector reads
        (a score cache), and a :class:`rts.data.contract.Requirement` value (``"coverage"``,
        ``"durations"``, ...) for material that comes from the dataset. An unrecognised
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


class RandomSelector(Selector):
    name = "random"

    def __init__(self, seed: int | None = None):
        # ``None`` means "the run's seed", so an experiment's seed knob reaches the model.
        self.seed = seed

    def scores(self, ctx: Context) -> np.ndarray:
        rng = np.random.default_rng(ctx.seed if self.seed is None else self.seed)
        return rng.random((ctx.ds.n_changes, ctx.ds.n_tests)).astype(np.float32)


class RecencySelector(Selector):
    """Select tests that failed most recently. Expected to be uninformative.

    The synthetic history has no real temporal structure, so this baseline exists to be
    reported as a limitation of the setup, not as a finding about recency.
    """

    name = "recency"

    def scores(self, ctx: Context) -> np.ndarray:
        return -ctx.feature("test_last_failure_age")


class FailureRateSelector(Selector):
    name = "failure_rate"

    def scores(self, ctx: Context) -> np.ndarray:
        return ctx.feature("test_failure_rate_cum")


class CoverageSelector(Selector):
    """Select tests that cover the mutated function (mutmut's own association)."""

    name = "coverage"

    def scores(self, ctx: Context) -> np.ndarray:
        return self._prefer_small_coverage(
            ctx.feature("covers_function"), ctx.feature("n_covering_tests")
        )


class StructuralRuleSelector(Selector):
    """Hand-built rule: covered tests whose file name matches the changed module.

    This exists because it turns out to explain most of the achievable recall. The
    conjunction ``covers_function AND filename_stem_match`` narrows the suite to a median
    of 9 candidate tests, so most of the task is solved by cheap structural funneling
    rather than by anything semantic. Any model claiming to work must be measured against
    this, not just against random.
    """

    name = "structural_rule"

    def scores(self, ctx: Context) -> np.ndarray:
        covered = ctx.feature("covers_function")
        name_match = ctx.feature("filename_stem_match")
        n_lines = ctx.feature("test_n_lines")
        # Covered first, then name-matching, then shortest test first.
        return covered * 2.0 + name_match + 1.0 / (1.0 + n_lines)


class LexicalSelector(Selector):
    """BM25 between the changed lines and the test source.

    The two shuffle flags produce a *shuffled* BM25 instead of reading the context's canonical
    one, which is the change-shuffle ablation ``docs/plan.md`` specifies: if recall barely drops when
    the change text is shuffled, the lexical signal is a change-independent test prior rather
    than a match between the change and the test. Expressing the probe as a selector keeps it in
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


class PerPoolRandomSelector(Selector):
    """Uniform scores drawn per change over that change's **own** candidate pool.

    :class:`RandomSelector` draws over the whole matrix, which is the right baseline when every
    change is ranked against one suite. When each change has its own pool, drawing over the
    union gives each change a different set of ranks, so the baseline would move for a reason
    that has nothing to do with the method under test.

    One generator for the whole matrix, consumed change by change in canonical order. The draw
    *sequence* is part of a recorded baseline, so it is deliberately not re-derived per row.
    """

    name = "random"

    def __init__(self, candidates_mode: str = "full", seed: int | None = None):
        self.candidates_mode = candidates_mode
        # ``None`` means "the run's seed", the same rule the other selectors follow.
        self.seed = seed

    def scores(self, ctx: Context) -> np.ndarray:
        candidates = accessors.candidates(ctx.ds, self.candidates_mode)
        rng = np.random.default_rng(ctx.seed if self.seed is None else self.seed)
        out = np.full((ctx.ds.n_changes, ctx.ds.n_tests), -1e9, dtype=np.float32)
        for row in range(ctx.ds.n_changes):
            cols = np.flatnonzero(candidates[row])
            if cols.size == 0:
                continue
            out[row, cols] = rng.random(cols.size).astype(np.float32)
        return out


class PerPoolLexicalSelector(Selector):
    """BM25 fitted once per change, over that change's own candidate documents.

    :class:`LexicalSelector` scores every pair against one index fitted over the whole suite,
    which makes a term's idf depend on every other change's tests. That is right when one suite
    serves every change. With per-change pools it is a different quantity -- and for a pooled
    multi-project corpus it would let one project's vocabulary move another project's scores.

    ``query`` selects which side of the change is the query:

    * ``"change"`` -- the added-and-removed-lines extraction the study's own BM25 uses;
    * ``"diff"`` -- the raw unified diff, ``+``/``-`` markers and hunk headers included, which
      is what the BugsInPy arm's recorded numbers used.

    The two are recorded as a choice rather than reconciled: they differ, and which one a model
    is entitled to see is a decision about the arm, not a spelling of one idea.
    """

    name = "bm25_lexical"

    def __init__(self, candidates_mode: str = "full", query: str = "change"):
        if query not in ("diff", "change"):
            raise ValueError(f"query must be 'diff' or 'change', not {query!r}")
        self.candidates_mode = candidates_mode
        self.query = query

    def scores(self, ctx: Context) -> np.ndarray:
        candidates = accessors.candidates(ctx.ds, self.candidates_mode)
        out = np.full((ctx.ds.n_changes, ctx.ds.n_tests), -1e9, dtype=np.float32)
        for row, change in enumerate(ctx.ds.changes):
            cols = np.flatnonzero(candidates[row])
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


class XGBoostSelector(Selector):
    """Gradient-boosted trees on the structured features.

    Three independent switches:

    * ``include_lexical`` adds the BM25 score as an input column. This is the cell that
      separates "the features matter" from "the model matters": if trees given the lexical
      signal match the transformer, the transformer's semantic machinery is not doing the
      work.
    * ``exclude_history`` drops the cumulative history features. This isolates how much of
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
        exclude_history: bool = False,
        exclude_coverage: bool = False,
        exclude: tuple[str, ...] = (),
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.15,
        seed: int | None = None,
        extra_score_files: dict[str, Path] | None = None,
        candidates_mode: str | None = None,
    ):
        self.include_lexical = include_lexical
        self.exclude_history = exclude_history
        self.exclude_coverage = exclude_coverage
        self.exclude = tuple(exclude)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        # ``None`` means "the run's seed", so an experiment's seed knob reaches the model
        # rather than stopping at ``config``. The same rule as ``RandomSelector``.
        self.seed = seed
        # P5: extra per-(change, test) score columns supplied by another model. The
        # canonical use is adding the SemIf reranker score to the structured features to
        # test whether the transformer is redundant given BM25 plus cheap structure.
        self.extra_score_files = dict(extra_score_files or {})
        # When set, XGBoost trains only on the candidate pairs it will actually be asked to
        # rank. Training over the whole suite (the historical default) lets it spend its
        # capacity learning the candidate mask, which is constant at evaluation time -- that
        # is what produced the once-reported 0.71 importance on ``covers_function``. None
        # preserves the old behaviour so existing documented numbers stay reproducible.
        self.candidates_mode = candidates_mode
        parts = ["xgboost"]
        parts.append("struct" if not exclude_history else "static")
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
        if self.exclude_history:
            dropped |= set(HISTORY_FEATURES)
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
        if self.candidates_mode is not None:
            train_mask = accessors.candidates(ctx.ds, self.candidates_mode)[train_rows]
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


def normalised_rank(scores: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Per-change descending rank, normalised to ``[0, 1]``; 0 is best.

    Non-candidates get ``inf`` so they can never win. This is what lets two selectors on
    different scales be averaged without fitting a weight: each change's candidates are ranked
    independently, so the result is invariant to either input's units -- which is what makes the
    combination below leakage-safe.
    """
    masked = np.where(candidates, scores, -np.inf)
    order = np.argsort(-masked, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(scores.shape[1])[None, :]
    denom = max(scores.shape[1] - 1, 1)
    return np.where(candidates, ranks / denom, np.inf)


#: A score loader: ``(cache path, dataset) -> [n_changes, n_tests]``. The dataset is a
#: parameter because a cache may need canonicalising against the pool it is read into.
ScoreLoader = Callable[[Path, Dataset], np.ndarray]

#: A score *producer*: ``(context, cache path) -> (matrix, stats)``. The context supplies the
#: dataset, the split and the candidate mask, so a producer reads nothing from a driver.
ScoreProducer = Callable[["Context", Path], tuple[np.ndarray, dict]]


def load_matrix(path: Path, _ds: Dataset) -> np.ndarray:
    """The trivial loader: a cache that is already a ``[n_changes, n_tests]`` matrix."""
    return np.load(path)


class CachedScores(Selector):
    """Scores read from a precomputed ``[n_changes, n_tests]`` artifact.

    This is the general form of every cached model in the study: the reranker caches, the five
    instruction-wording caches, the direct-mode cache and the embedding baseline all differ only
    in which file they read and how it is parsed. Declaring the cache as a *requirement* is what
    matters: an absent one makes the cell unmeasured with the path rather than raising from
    inside ``scores``, which is the difference between a hole in the report and a dead run when
    a study has a dozen caches of which any may be missing.
    """

    def __init__(self, name: str, path: Path | str, loader: ScoreLoader | None = None):
        self.name = name
        self.path = Path(path)
        self._loader = loader or semif.load_scores

    def requirements(self) -> tuple[str, ...]:
        """The precomputed cache. Producing it is a precondition, not something a cell does."""
        return (f"artifact:{self.path}",)

    def scores(self, ctx: Context) -> np.ndarray:
        return self._loader(self.path, ctx.ds)


class SemIfSelector(CachedScores):
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


class RankAverageSelector(Selector):
    """The fitted-free combination of several selectors: mean normalised rank position.

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
        selectors: Sequence[Selector],
        candidates_mode: str = "full",
    ):
        if len(selectors) < 2:
            raise ValueError(f"{name!r} averages rank positions, so it needs two or more parents")
        self.name = name
        self.selectors = tuple(selectors)
        self.candidates_mode = candidates_mode

    def requirements(self) -> tuple[str, ...]:
        """Whatever the parents read, in first-seen order and without repeats."""
        seen: list[str] = []
        for selector in self.selectors:
            for requirement in selector.requirements():
                if requirement not in seen:
                    seen.append(requirement)
        return tuple(seen)

    def scores(self, ctx: Context) -> np.ndarray:
        candidates = accessors.candidates(ctx.ds, self.candidates_mode)
        norms = [normalised_rank(s.scores(ctx), candidates) for s in self.selectors]
        return -np.mean(norms, axis=0)


class ProducedScores(Selector):
    """Scores read from a cache, **produced on first use** if the cache is absent.

    :class:`CachedScores` treats its artifact as a precondition: absent means the cell is
    unmeasured with the path. This treats it as an *output*. That is the right shape for the
    study's most expensive step -- scoring (change, test) pairs with a 4B reranker -- where the
    artifact is exactly what the cell would compute, and where leaving production outside the
    layer made a stale cache indistinguishable from a fresh one and the GPU arm invisible.

    It is a *general* form rather than a SemIf one: the scorer is a callable, so the pinned
    model, its prompt configuration and the rows to score are supplied by whichever element
    declares the capability. That also makes the produce path testable without a GPU.

    ``requirements()`` is deliberately empty. Declaring the cache would make the cell
    unmeasured *before* it could produce it, which is the trap this class exists to avoid; the
    element that wraps it declares ``tier="gpu"`` instead, so a run that is not spending the
    GPU tier reports the cell as unmeasured rather than silently reading a half-built cache.

    A cache is identified by its **path**, so "it exists" is not the same claim as "it is the
    one this cell needs": the file may have been produced for other rows, another candidate
    pool or another prompt wording, or torn mid-write. ``verifier`` is the element's chance to
    refuse such a file -- it is handed the context and the path and raises if the cache cannot
    serve it. Without one the file is trusted, which is the documented contract rather than a
    guarantee, and it is why the SemIf production element supplies
    :func:`rts.model.semif_runner.missing_pairs`.

    A verifier *raises* rather than producing an unmeasured cell, and that is deliberate: the
    element declared that it can produce this cache, so a file at the path that cannot serve
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
        #: Recorded on the cell so a produced number carries its cost, the same way a
        #: measured cell carries its seconds.
        self.last_stats: dict = {}

    def requirements(self) -> tuple[str, ...]:
        """None: the cache is this selector's output, not its precondition."""
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


def default_selectors(include_semif: bool = True) -> list[Selector]:
    selectors: list[Selector] = [
        RandomSelector(),
        RecencySelector(),
        FailureRateSelector(),
        CoverageSelector(),
        StructuralRuleSelector(),
        LexicalSelector(),
        XGBoostSelector(include_lexical=False),
        XGBoostSelector(include_lexical=True),
        XGBoostSelector(exclude_history=True, include_lexical=False),
        XGBoostSelector(exclude_history=True, exclude_coverage=True),
        XGBoostSelector(exclude_history=True, exclude_coverage=True, include_lexical=True),
    ]
    if include_semif:
        selectors.append(SemIfSelector())
    return selectors
