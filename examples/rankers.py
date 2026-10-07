"""The study's concrete rankers: baselines, trees, and the cached-score models.

Each implements :class:`rts.model.rankers.Ranker` and reads only the kernel contract, so this
module is a worked example of the harness's model plugin point. The generic mechanisms -- the
context, the interface, the random baseline, the ranked average, and the cached/produced score
forms -- stay in :mod:`rts.model.rankers`; what is here is the set of models a study measured.
"""

from __future__ import annotations

import numpy as np

from examples import config, semif
from rts import features
from rts.data import accessors
from rts.model.rankers import CachedScores, Context, RandomRanker, Ranker

# Feature families, taken from the block's own declaration so there is one definition of what
# "the temporal family" is rather than a second list here that can drift from it.
TEMPORAL_FEATURES = features.STRUCTURED.family("temporal")
COVERAGE_FEATURES = features.STRUCTURED.family("coverage")
PROXIMITY_FEATURES = features.STRUCTURED.family("proximity")


def _prefer_small_coverage(covered: np.ndarray, n_covering: np.ndarray) -> np.ndarray:
    """Rank covered tests first, then by ascending coverage-set size.

    Ties inside a coverage set are otherwise arbitrary, and a test whose function is covered
    by 3 tests is far more likely to be the killer than one in a set of 761.
    """
    return np.where(covered > 0, 1e6, 0.0) - n_covering


class RecencyRanker(Ranker):
    """Select tests that failed most recently. Expected to be uninformative."""

    name = "recency"

    def scores(self, ctx: Context) -> np.ndarray:
        return -ctx.feature("failure_recency")


class FailureRateRanker(Ranker):
    name = "failure_rate"

    def scores(self, ctx: Context) -> np.ndarray:
        return ctx.feature("cumulative_failure_rate")


class CoverageRanker(Ranker):
    """Select tests that cover the mutated function (the mutation tool's own association)."""

    name = "coverage"

    def scores(self, ctx: Context) -> np.ndarray:
        return _prefer_small_coverage(
            ctx.feature("function_coverage"), ctx.feature("coverage_set_size")
        )


class StructuralRuleRanker(Ranker):
    """A hand-built rule: covered tests whose file name matches the changed module."""

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
    one, which is the change-shuffle ablation: if recall barely drops when the change text is
    shuffled, the lexical signal is a change-independent test prior rather than a match between
    the change and the test.
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

    :class:`rts.model.rankers.RandomRanker` draws over the whole matrix, which is the right
    baseline when every change is ranked against one suite. When each change has its own pool,
    drawing over the union gives each change a different set of ranks, so the baseline would
    move for a reason that has nothing to do with the method under test.
    """

    name = "random"

    def __init__(self, candidate_policy: str = "full", seed: int | None = None):
        self.candidate_policy = candidate_policy
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
    serves every change; with per-change pools it is a different quantity, and for a pooled
    multi-project corpus it would let one project's vocabulary move another's scores.
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

    Switches:

    * ``include_lexical`` adds the BM25 score as an input column;
    * ``exclude_temporal`` drops the cumulative temporal features;
    * ``exclude_coverage`` drops the coverage features;
    * ``exclude`` drops arbitrary columns by name.
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
        extra_score_files: dict | None = None,
        candidate_policy: str | None = None,
    ):
        self.include_lexical = include_lexical
        self.exclude_temporal = exclude_temporal
        self.exclude_coverage = exclude_coverage
        self.exclude = tuple(exclude)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        # ``None`` means "the run's seed", so an experiment's seed control reaches the model.
        self.seed = seed
        # Extra per-(change, test) score columns supplied by another model.
        self.extra_score_files = dict(extra_score_files or {})
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
        """Flattened design matrix over (change, test) pairs."""
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


class SemIfRanker(CachedScores):
    """Frozen SemIf + Qwen reranker scores, read from a precomputed cache."""

    def __init__(self, scores_file=None):
        super().__init__(
            "semif_reranker",
            scores_file or config.SEMIF_SCORES_FILE,
            loader=semif.load_scores,
        )

    @property
    def scores_file(self):
        """The cache this reads, i.e. :attr:`CachedScores.path` under its older name."""
        return self.path


def default_rankers(include_semif: bool = True) -> list[Ranker]:
    """The study's ranker set: baselines, the structural rule, BM25, and the trees."""
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


__all__ = [
    "CoverageRanker",
    "FailureRateRanker",
    "LexicalRanker",
    "PerPoolLexicalRanker",
    "PerPoolRandomRanker",
    "RecencyRanker",
    "SemIfRanker",
    "StructuralRuleRanker",
    "XGBoostRanker",
    "default_rankers",
]
