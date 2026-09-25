"""Feature extraction: cumulative structured features and lexical text features.

Two families, matching ``plan.md``:

**Structured** (``structured_features``) --- diff metadata, coverage, directory
proximity, and per-test history. Every history-derived feature is computed
*cumulatively*: at change ``i`` it uses only outcomes from changes strictly
before ``i``. The temporal split does not provide this on its own; the feature
computation has to.

**Unstructured / lexical** (``BM25Scorer``) --- token overlap between the changed
lines and the test source. This doubles as the standalone lexical baseline, which
is the cheapest competing explanation for a transformer win.

Also here: the shuffle transforms used by the ablation in ``ablations.py``.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np

from . import artifacts, dataset, source

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


def change_query_text(change) -> str:
    """The change side of a pair: added and removed lines, weighted by repetition.

    Including the removed lines matters because a mutation's meaning often comes
    from what it replaced.
    """
    added = "\n".join(change.changed_lines)
    removed = "\n".join(change.removed_lines)
    return f"{added}\n{removed}"


def build_bm25_scores(
    ds: dataset.Dataset,
    shuffle_changes: bool = False,
    shuffle_tests: bool = False,
    seed: int = 20260924,
) -> np.ndarray:
    """[n_changes, n_tests] BM25 score of the change text against each test.

    Two ablation switches, both of which destroy one half of the pair while
    preserving its distribution:

    * ``shuffle_changes`` --- permute the change text across changes. If recall
      barely drops, the model was exploiting a change-independent test prior.
    * ``shuffle_tests`` --- permute the test documents. The symmetric check: if
      recall barely drops, the change side is not doing any work.
    """
    infos = source.load_all(ds.test_ids)
    docs = [infos[t].source if t in infos else "" for t in ds.test_ids]
    test_perm = None
    if shuffle_tests:
        rng = np.random.default_rng(seed + 1)
        test_perm = rng.permutation(len(docs))
        docs = [docs[i] for i in test_perm]

    scorer = BM25Scorer().fit(docs)

    texts = [change_query_text(c) for c in ds.changes]
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


# --- Structured ------------------------------------------------------------


STRUCTURED_NAMES = [
    "covers_function",
    "n_covering_tests",
    "coverage_rank_prior",
    "path_distance",
    "n_tests_in_test_file",
    "filename_stem_match",
    "test_duration",
    "test_n_lines",
    "test_n_tokens",
    "change_size",
    "change_added_lines",
    "change_removed_lines",
    "test_failure_rate_cum",
    "test_runs_cum",
    "test_last_failure_age",
]


def _path_parts(directory: str) -> list[str]:
    """Components of a directory path. Pass a directory, not a file path."""
    return [part for part in directory.split("/") if part]


def _dir_distance(a: list[str], b: list[str]) -> int:
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return len(a) + len(b) - 2 * common


def structured_features(
    ds: dataset.Dataset,
) -> tuple[np.ndarray, list[str]]:
    """Return ``(X, names)`` with ``X`` of shape ``[n_changes, n_tests, n_features]``.

    Every feature is either static per pair, or cumulative over strictly earlier
    changes. Nothing peeks at the current change's own outcome.
    """
    n_c, n_t = ds.n_changes, ds.n_tests
    n_f = len(STRUCTURED_NAMES)
    X = np.zeros((n_c, n_t, n_f), dtype=np.float32)

    # --- static per-test ---
    durations_map = artifacts.duration_by_test()
    durations = np.array(
        [durations_map.get(t, 0.0) for t in ds.test_ids], dtype=np.float32
    )
    infos = source.load_all(ds.test_ids)
    n_lines = np.array([infos[t].n_lines if t in infos else 0 for t in ds.test_ids], dtype=np.float32)
    n_tokens = np.array([infos[t].n_tokens if t in infos else 0 for t in ds.test_ids], dtype=np.float32)

    test_dirs = [str(ds.test_ids[i].split("::")[0]).rsplit("/", 1)[0] for i in range(n_t)]
    test_names = [ds.test_ids[i].split("::")[0].rsplit("/", 1)[-1] for i in range(n_t)]

    # --- static per change ---
    change_dirs = [c.file.rsplit("/", 1)[0] for c in ds.changes]
    change_stems = [c.file.rsplit("/", 1)[-1].removesuffix(".py") for c in ds.changes]

    # --- coverage / proximity matrices ---
    coverage_mask = np.zeros((n_c, n_t), dtype=np.float32)
    for i, cov in enumerate(ds.covered):
        for test in cov:
            coverage_mask[i, ds.test_index[test]] = 1.0

    n_covering = coverage_mask.sum(axis=1, keepdims=True)
    # Prior that is 1 for covered tests and 0 otherwise, scaled by how many tests
    # cover the function: a widely covered function makes any single test less
    # likely to be the killer.
    with np.errstate(divide="ignore", invalid="ignore"):
        coverage_rank_prior = np.where(n_covering > 0, 1.0 / np.maximum(n_covering, 1.0), 0.0)

    path_distance = np.zeros((n_c, n_t), dtype=np.float32)
    name_match = np.zeros((n_c, n_t), dtype=np.float32)
    dir_cache: dict[tuple[str, str], int] = {}
    for i in range(n_c):
        for j in range(n_t):
            key = (change_dirs[i], test_dirs[j])
            d = dir_cache.get(key)
            if d is None:
                d = _dir_distance(_path_parts(change_dirs[i]), _path_parts(test_dirs[j]))
                dir_cache[key] = d
            path_distance[i, j] = d
            # "test_utils.py" for "utils.py": a strong, cheap naming signal.
            name_match[i, j] = 1.0 if change_stems[i] and change_stems[i] in test_names[j] else 0.0

    # Test-file size: how many tests share a file with this one. Cheap context for
    # whether a test is a focused unit test or one of many in a large module suite.
    file_counts: dict[str, int] = defaultdict(int)
    for nodeid in ds.test_ids:
        file_counts[nodeid.split("::")[0]] += 1
    n_tests_in_file = np.array(
        [file_counts[t.split("::")[0]] for t in ds.test_ids], dtype=np.float32
    )

    change_size = np.array([c.change_size for c in ds.changes], dtype=np.float32)
    change_added = np.array([len(c.changed_lines) for c in ds.changes], dtype=np.float32)
    change_removed = np.array([len(c.removed_lines) for c in ds.changes], dtype=np.float32)

    # --- cumulative history, in temporal order ---
    alpha = 1.0  # Laplace prior so an unseen test starts at 0.5
    running_fails = np.zeros(n_t, dtype=np.float64)
    running_runs = np.zeros(n_t, dtype=np.float64)
    last_failure = np.full(n_t, -1, dtype=np.int64)

    failure_rate = np.zeros((n_c, n_t), dtype=np.float32)
    runs_cum = np.zeros((n_c, n_t), dtype=np.float32)
    last_failure_age = np.zeros((n_c, n_t), dtype=np.float32)

    for i in range(n_c):
        failure_rate[i] = (running_fails + alpha) / (running_runs + 2 * alpha)
        runs_cum[i] = running_runs
        age = np.where(last_failure >= 0, i - last_failure, n_c)
        last_failure_age[i] = age
        # Update only after emitting features for change i.
        running_fails += ds.labels[i]
        running_runs += ds.ran[i]
        last_failure = np.where(ds.labels[i] == 1, i, last_failure)

    X[:, :, 0] = coverage_mask
    X[:, :, 1] = n_covering
    X[:, :, 2] = coverage_rank_prior
    X[:, :, 3] = path_distance
    X[:, :, 4] = n_tests_in_file[None, :]
    X[:, :, 5] = name_match
    X[:, :, 6] = durations[None, :]
    X[:, :, 7] = n_lines[None, :]
    X[:, :, 8] = n_tokens[None, :]
    X[:, :, 9] = change_size[:, None]
    X[:, :, 10] = change_added[:, None]
    X[:, :, 11] = change_removed[:, None]
    X[:, :, 12] = failure_rate
    X[:, :, 13] = runs_cum
    X[:, :, 14] = last_failure_age

    return X, list(STRUCTURED_NAMES)


if __name__ == "__main__":
    import time

    ds = dataset.build()
    t0 = time.perf_counter()
    X, names = structured_features(ds)
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
    hit = [bm[i, ds.test_index[t]] for i in faults for t in ds.changes[i].killing_tests]
    print(f"  mean bm25 on killing tests: {np.mean(hit):.3f}")
    print(f"  mean bm25 overall        : {bm.mean():.3f}")
