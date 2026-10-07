"""A namespace of ranker plugins for the kernel tests.

The kernel ships the interface and the generic mechanisms; the concrete models are plugins that
live in :mod:`examples`. Tests need both, so they import them from here under one name.
"""

from __future__ import annotations

from examples.rankers import (  # noqa: F401
    CoverageRanker,
    FailureRateRanker,
    LexicalRanker,
    PerPoolLexicalRanker,
    PerPoolRandomRanker,
    RecencyRanker,
    SemIfRanker,
    StructuralRuleRanker,
    XGBoostRanker,
    default_rankers,
)
from rts.model.rankers import (  # noqa: F401
    CachedScores,
    Context,
    ProducedScores,
    RandomRanker,
    RankAverageRanker,
    Ranker,
    ScoreLoader,
    ScoreProducer,
    load_matrix,
    normalised_rank,
)

__all__ = [
    "CachedScores",
    "Context",
    "CoverageRanker",
    "FailureRateRanker",
    "LexicalRanker",
    "PerPoolLexicalRanker",
    "PerPoolRandomRanker",
    "ProducedScores",
    "RandomRanker",
    "RankAverageRanker",
    "Ranker",
    "RecencyRanker",
    "ScoreLoader",
    "ScoreProducer",
    "SemIfRanker",
    "StructuralRuleRanker",
    "XGBoostRanker",
    "default_rankers",
    "load_matrix",
    "normalised_rank",
]
