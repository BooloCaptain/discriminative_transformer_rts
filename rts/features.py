"""Lexical features: BM25 between the changed lines and the test source.

The structured, cumulative features used to live here. They are now harness functions
over the dataset contract (see :mod:`rts.dataset`), because they are defined for
*every* dataset rather than being computed once for this SUT -- and keeping one
implementation is what makes an identical statistic mean an identical quantity across
datasets.

What remains here is the **unstructured / lexical** family, which is also the cheapest
competing explanation for a transformer win: token overlap between the changed lines
and the test source. It is a harness function too, built on the contract's
``change_query_text`` and ``test_source`` primitives, so it works on any dataset.

Also here: the tokenizer the lexical baseline and the BugsInPy bridge audit share.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np

from . import dataset as contract

# Re-exported so callers that ask the feature layer what the columns of ``X`` are
# keep working, and so there is one list rather than two that can drift.
from .dataset import (  # noqa: F401
    COVERAGE_COLUMNS,
    DURATION_COLUMNS,
    HISTORY_COLUMNS,
    STRUCTURED_NAMES,
    change_query_text,
    structured_features,
)

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


# --- BM25 -----------------------------------------------------------------


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


def test_documents(ds: contract.Dataset) -> list[str]:
    """The test side of every pair: one document per test in the pool.

    ``test_source`` returning ``None`` and returning ``""`` are different states, and
    both become an empty document here -- the difference is preserved for the size
    features, not for a lexical score, which has nothing to read either way.
    """
    return [ds.test_source(t) or "" for t in ds.test_ids]


def change_documents(ds: contract.Dataset) -> list[str]:
    """The change side of every pair: added plus removed lines."""
    return [contract.change_query_text(ds, c) for c in ds.changes]


def build_bm25_scores(
    ds: contract.Dataset,
    shuffle_changes: bool = False,
    shuffle_tests: bool = False,
    seed: int = 20260924,
) -> np.ndarray:
    """``[n_changes, n_tests]`` BM25 score of the change text against each test.

    Two ablation switches, both of which destroy one half of the pair while
    preserving its distribution:

    * ``shuffle_changes`` --- permute the change text across changes. If recall
      barely drops, the model was exploiting a change-independent test prior.
    * ``shuffle_tests`` --- permute the test documents. The symmetric check: if
      recall barely drops, the change side is not doing any work.
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
    # With shuffled documents, position p holds the text of a different test, so
    # keeping the score at position p is precisely the ablation: test p is scored
    # against the wrong test's source. Mapping the scores back would undo it.
    return raw


if __name__ == "__main__":
    import time

    from . import datasets

    ds = datasets.marshmallow()
    t0 = time.perf_counter()
    X, names = structured_features(ds, history=True)
    t1 = time.perf_counter()
    print(f"structured X: {X.shape} {X.nbytes / 1e6:.1f} MB in {t1 - t0:.1f}s")
    for k, name in enumerate(names):
        col = X[:, :, k]
        print(f"  {name:>24}: min {col.min():.3f}  max {col.max():.3f}  mean {col.mean():.3f}")

    t0 = time.perf_counter()
    bm = build_bm25_scores(ds)
    t1 = time.perf_counter()
    print(f"bm25: {bm.shape} in {t1 - t0:.1f}s")
    faults = ds.fault_idx
    hit = [bm[i, ds.test_index[t]] for i in faults for t in ds.killing_tests(ds.changes[i])]
    print(f"  mean bm25 on killing tests: {np.mean(hit):.3f}")
    print(f"  mean bm25 overall        : {bm.mean():.3f}")
