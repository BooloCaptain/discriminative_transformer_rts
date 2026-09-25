"""RTS selectors: statistical baselines, XGBoost, and the SemIf reranker.

Every selector implements ``scores(ctx) -> [n_changes, n_tests]``. Higher is
better; evaluation takes the top ``k`` per change. Selectors never see labels for
changes they are being evaluated on -- the structured features are already
cumulative, and XGBoost is fit on the training window only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config, dataset, semif

# Cumulative history features, i.e. everything derived from prior change outcomes.
HISTORY_FEATURES = (
    "test_failure_rate_cum",
    "test_runs_cum",
    "test_last_failure_age",
)

# Per-test coverage features. In a huge codebase with long-running integration
# tests, obtaining these means running the suite you are trying to avoid, so a
# deployment may genuinely not have them.
COVERAGE_FEATURES = (
    "covers_function",
    "n_covering_tests",
    "coverage_rank_prior",
)


@dataclass
class Context:
    """Everything a selector may read."""

    ds: dataset.Dataset
    X: np.ndarray  # structured features [n_changes, n_tests, n_features]
    names: list[str]
    bm25: np.ndarray  # [n_changes, n_tests]
    extras: dict = field(default_factory=dict)

    def feature(self, name: str) -> np.ndarray:
        return self.X[:, :, self.names.index(name)]


class Selector:
    name = "selector"

    def scores(self, ctx: Context) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    # --- helpers ---
    @staticmethod
    def _prefer_small_coverage(covered: np.ndarray, n_covering: np.ndarray) -> np.ndarray:
        """Rank covered tests first, then by ascending coverage-set size.

        Ties inside a coverage set are otherwise arbitrary, and a test whose
        function is covered by 3 tests is far more likely to be the killer than
        one in a set of 761.
        """
        return np.where(covered > 0, 1e6, 0.0) - n_covering


class RandomSelector(Selector):
    name = "random"

    def __init__(self, seed: int = config.SEED):
        self.seed = seed

    def scores(self, ctx: Context) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.random((ctx.ds.n_changes, ctx.ds.n_tests)).astype(np.float32)


class RecencySelector(Selector):
    """Select tests that failed most recently. Expected to be uninformative.

    The synthetic history has no real temporal structure, so this baseline exists
    to be reported as a limitation of the setup, not as a finding about recency.
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
    conjunction ``covers_function AND module_name_in_test_file`` narrows the suite
    to a median of 9 candidate tests, so most of the task is solved by cheap
    structural funneling rather than by anything semantic. Any model claiming to
    work must be measured against this, not just against random.
    """

    name = "structural_rule"

    def scores(self, ctx: Context) -> np.ndarray:
        covered = ctx.feature("covers_function")
        name_match = ctx.feature("module_name_in_test_file")
        n_lines = ctx.feature("test_n_lines")
        # Covered first, then name-matching, then shortest test first.
        return covered * 2.0 + name_match + 1.0 / (1.0 + n_lines)


class LexicalSelector(Selector):
    """BM25 between the changed lines and the test source."""

    name = "bm25_lexical"

    def scores(self, ctx: Context) -> np.ndarray:
        return ctx.bm25


class XGBoostSelector(Selector):
    """Gradient-boosted trees on the structured features.

    Two independent switches:

    * ``include_lexical`` adds the BM25 score as an input column. This is the cell
      that separates "the features matter" from "the model matters": if trees
      given the lexical signal match the transformer, the transformer's semantic
      machinery is not doing the work.
    * ``exclude_history`` drops the cumulative history features. This isolates how
      much of the score is failure-history memorisation rather than coverage or
      static structure -- important here because mutation testing revisits the
      same function many times, so a (function, killing-test) pair recurs far
      more often than it would in real evolution.
    """

    def __init__(
        self,
        include_lexical: bool = False,
        exclude_history: bool = False,
        exclude_coverage: bool = False,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.15,
        seed: int = config.SEED,
        extra_score_files: dict[str, Path] | None = None,
        candidates_mode: str | None = None,
    ):
        self.include_lexical = include_lexical
        self.exclude_history = exclude_history
        self.exclude_coverage = exclude_coverage
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        # P5: extra per-(change, test) score columns supplied by another model.
        # The canonical use is adding the SemIf reranker score to the structured
        # features to test whether the transformer is redundant given BM25 plus
        # cheap structure.
        self.extra_score_files = dict(extra_score_files or {})
        # When set, XGBoost trains only on the candidate pairs it will actually be
        # asked to rank. Training over the whole suite (the historical default)
        # lets it spend its capacity learning the candidate mask, which is
        # constant at evaluation time -- that is what produced the once-reported
        # 0.71 importance on ``covers_function``. None preserves the old behaviour
        # so existing documented numbers stay reproducible.
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

    def _dropped(self) -> set[str]:
        dropped: set[str] = set()
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

        ``cols_mask`` (shape ``[len(rows), n_tests]``) restricts the output to a
        subset of the pairs, which is how candidate-only training is implemented.
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

        train_rows = ctx.ds.train_idx
        train_mask = None
        if self.candidates_mode is not None:
            train_mask = dataset.candidate_mask(ctx.ds, self.candidates_mode)[train_rows]
        X_train = self._design(ctx, train_rows, train_mask)
        labels = ctx.ds.labels[train_rows]
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
            random_state=self.seed,
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


class SemIfSelector(Selector):
    """Frozen SemIf + Qwen reranker scores, read from a precomputed cache.

    Scoring is expensive (~1.9 decisions/s) and needs the SemIf checkout plus the
    checkpoint, so it is computed once by ``rts.semif`` and cached. See
    implementation.md for the exact pinned configuration.
    """

    name = "semif_reranker"

    def __init__(self, scores_file=None):
        self.scores_file = scores_file or config.SEMIF_SCORES_FILE

    def scores(self, ctx: Context) -> np.ndarray:
        return semif.load_scores(self.scores_file, ctx.ds)


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
