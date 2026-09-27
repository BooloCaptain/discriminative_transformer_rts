"""Lexical features: BM25 between the changed lines and the test source.

The cheapest competing explanation for a transformer win, and therefore a required
baseline rather than an optional extra. It is also a harness function over the
contract's ``change_query_text`` and ``test_source`` primitives, so it works on any
dataset -- which is what lets the same BM25 numbers be compared across the mutation SUT
and the real-bug corpus.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Sequence

import numpy as np

from ..data import accessors
from ..data.contract import Dataset

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
KEYWORDS = {
    "self", "def", "return", "if", "else", "elif", "for", "while", "in", "not",
    "and", "or", "None", "True", "False", "import", "from", "class", "try",
    "except", "finally", "with", "as", "lambda", "yield", "assert", "raise",
    "pass", "break", "continue", "is", "global", "nonlocal", "del", "await",
    "async", "type", "list", "dict", "set", "tuple", "str", "int", "float",
    "bool", "len", "range", "print", "the", "a", "an", "of", "to",
}


def tokenize(text: str) -> list[str]:
    """Lowercase identifier/number tokens, minus language keywords.

    Stopwords are dropped because in this corpus they are dominated by Python
    boilerplate that appears in every test and would swamp the lexical signal.
    """
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in KEYWORDS]


class BM25Scorer:
    """Okapi BM25 with an inverted index. Change text is the query, tests are documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = 0
        self.doc_len = np.zeros(0, dtype=np.float32)
        self.avg_len = 1.0
        self.idf: dict[str, float] = {}
        self.postings: dict[str, list[tuple[int, float]]] = {}

    def fit(self, docs: list[str]) -> "BM25Scorer":
        self.n_docs = len(docs)
        self.postings = defaultdict(list)
        lengths = np.zeros(self.n_docs, dtype=np.float32)
        df: dict[str, int] = defaultdict(int)

        for i, doc in enumerate(docs):
            tokens = tokenize(doc)
            lengths[i] = len(tokens)
            counts: dict[str, int] = defaultdict(int)
            for token in tokens:
                counts[token] += 1
            for token, count in counts.items():
                df[token] += 1
                self.postings[token].append((i, float(count)))

        self.doc_len = lengths
        self.avg_len = float(lengths.mean()) if self.n_docs else 1.0
        # BM25 idf with the +0.5 smoothing, floored at a small positive value so
        # that very common terms do not go negative.
        for token, n in df.items():
            self.idf[token] = max(
                math.log(1.0 + (self.n_docs - n + 0.5) / (n + 0.5)), 1e-6
            )
        return self

    def score(self, query: str) -> np.ndarray:
        """Score every document against the query. Returns an array of length n_docs."""
        out = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs:
            return out
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / (self.avg_len or 1.0))
        seen: set[str] = set()
        for token in tokenize(query):
            if token in seen:
                continue
            seen.add(token)
            idf = self.idf.get(token)
            if idf is None:
                continue
            for doc, tf in self.postings[token]:
                out[doc] += idf * (tf * (self.k1 + 1.0)) / (tf + norm[doc])
        return out


def test_documents(ds: Dataset) -> list[str]:
    """The test side of every pair: one document per test in the pool.

    ``test_source`` returning ``None`` and returning ``""`` are different states, and
    both become an empty document here -- the difference is preserved for the size
    features, not for a lexical score, which has nothing to read either way.
    """
    return [accessors.test_source(ds, t) or "" for t in ds.test_ids]


def change_documents(ds: Dataset) -> list[str]:
    """The change side of every pair: added plus removed lines."""
    from .derived import change_query_text

    return [change_query_text(ds, c) for c in ds.changes]


def build_bm25_scores(
    ds: Dataset,
    shuffle_changes: bool = False,
    shuffle_tests: bool = False,
    seed: int = 20260924,
) -> np.ndarray:
    """``[n_changes, n_tests]`` BM25 score of the change text against each test.

    Two ablation switches, both of which destroy one half of the pair while preserving
    its distribution:

    * ``shuffle_changes`` --- permute the change text across changes. If recall barely
      drops, the model was exploiting a change-independent test prior.
    * ``shuffle_tests`` --- permute the test documents. The symmetric check: if recall
      barely drops, the change side is not doing any work.
    """
    docs = test_documents(ds)
    if shuffle_tests:
        rng = np.random.default_rng(seed + 1)
        test_perm = rng.permutation(len(docs))
        docs = [docs[i] for i in test_perm]

    scorer = BM25Scorer().fit(docs)

    texts = change_documents(ds)
    if shuffle_changes:
        rng = np.random.default_rng(seed)
        change_perm = rng.permutation(len(texts))
        texts = [texts[i] for i in change_perm]

    raw = np.zeros((ds.n_changes, ds.n_tests), dtype=np.float32)
    for i, text in enumerate(texts):
        raw[i] = scorer.score(text)
    # With shuffled documents, position p holds the text of a different test, so keeping
    # the score at position p is precisely the ablation: test p is scored against the
    # wrong test's source. Mapping the scores back would undo it.
    return raw


def bm25_over(pairs: Sequence[tuple[str, str]], query: str) -> np.ndarray:
    """BM25 of one query against an explicit document list.

    Used by the BugsInPy arm, which fits the scorer per bug on that bug's own pool: a
    global index over eight projects would make a term's idf depend on the other seven,
    which is a different quantity from the one the marshmallow arm reports.
    """
    return BM25Scorer().fit(list(pairs)).score(query)


__all__ = [
    "BM25Scorer",
    "KEYWORDS",
    "bm25_over",
    "build_bm25_scores",
    "change_documents",
    "test_documents",
    "tokenize",
]
