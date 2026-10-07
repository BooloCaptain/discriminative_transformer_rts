"""Pairwise scorer for SemIf + Qwen3-Reranker-4B, and a smoke test.

Why a native-template scorer rather than SemIf's ``--mode reranker``
--------------------------------------------------------------------
SemIf's reranker wrapper is shaped for its own decision interface: a row carries
2-16 *options*, the question plus candidate answer go into ``<Query>``, and the
state goes into ``<Document>``. For RTS we want one score per (change, test) pair
over the whole suite, which does not fit that shape.

The reranker model is natively a yes/no judge over (Query, Document), so we build
the prompt directly with our own Instruct/Query/Document mapping and read the
yes-vs-no log-odds at the assistant position. We import SemIf's ``PREFIX``,
``SUFFIX``, and ``_answer_ids`` so the contract stays demonstrably identical to
the reference implementation, and we can validate against ``semif-score`` on a
sample.

Global comparability: every pair is scored independently with the same prompt
skeleton, and the readout is a log-odds ratio between two fixed answer tokens, so
scores are on one scale across all pairs and changes.

Performance: pairs are batched (left-padded) in one forward pass each. Measured
throughput is reported so the cost estimate can be replaced with a real number.
"""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import config, features
from ..data import accessors, contract, datasets, splits, subsets
from . import semif

INSTRUCTION = (
    "Given a code change, judge whether the test below exercises the changed "
    "behaviour, such that this change could make the test fail. Answer yes only "
    "if the test would plausibly need to run for this change."
)

# P2 instruction sweep. The diagnosis in docs/implementation.md is that the pinned
# reranker was trained for *topical relevance over natural language*, while RTS
# asks an *executional and causal* question, so the instruction is a real lever
# rather than boilerplate. Each variant changes exactly one thing: the framing of
# the question. The rest of the prompt skeleton is identical, so any difference is
# attributable to the wording.
#
#   default   -- the reference wording used by every existing cache.
#   execution -- the diagnosis made explicit: ask about observable runtime
#                behaviour rather than topical relevance.
#   fault     -- ask for the prediction directly ("will it fail").
#   retrieval -- *negative control*: lean into the native reranker prior
#                (topical relevance). If leaning in helps, the prior-mismatch
#                diagnosis is wrong.
#   terse     -- shortest possible question, testing whether the instruction block
#                is mostly dilution. The placement controls showed ~200 tokens of
#                uninformative prefix text cost 0.21 recall, and this instruction
#                is ~35 tokens instead of ~40, so the effect should be small unless
#                the content is actively misleading.
INSTRUCTION_VARIANTS: dict[str, str] = {
    "default": INSTRUCTION,
    "execution": (
        "Would running the test below against the changed code produce a different "
        "result? Answer yes only if executing this test would observe the change, "
        "for example because it calls the changed code or asserts on behaviour the "
        "change alters."
    ),
    "fault": (
        "Will the test below fail if this code change is applied? Answer yes only "
        "if the change would make this test fail."
    ),
    "retrieval": (
        "Is the test below relevant to the code change? Answer yes if the test "
        "concerns the part of the code that this change modifies."
    ),
    "terse": "Does this test depend on the behaviour that this change modifies?",
}

# Structured features that never vary across a change's candidate sets under the
# `covered` mask, and so cannot help rank within that change. Verified empirically:
# path proximity is included because every test lives in the same directory tree, so
# it is constant per change rather than merely low-variance.
PER_CHANGE_CONSTANT = frozenset(
    {
        "function_coverage",
        "coverage_set_size",
        "coverage_set_size_prior",
        "path_proximity",
        "code_churn",
        "added_lines",
        "removed_lines",
    }
)

# Diagnostic conditions for the mirror-degradation question. Each holds everything else
# fixed and varies one property of the injected block:
#   full          -- all 15 features (the original mirror treatment)
#   informative   -- only features that actually vary across candidate sets
#   placebo       -- same field names and length, every value replaced by a constant
#   shuffled      -- real values and distribution, decorrelated from the candidate
#   after_document -- full features, but placed after <Document> instead of <Instruct>
CONTROL_MODES = ("full", "informative", "placebo", "shuffled")


def _prefix_suffix():
    """Reuse SemIf's exact prompt scaffolding and yes/no contract."""
    from semif_phase1.reranker import PREFIX, SUFFIX, _answer_ids

    return PREFIX, SUFFIX, _answer_ids


def build_prompt(
    change_text: str,
    test_text: str,
    prefix: str,
    suffix: str,
    orientation: str = "change_query",
    features_block: str = "",
    placement: str = "instruct",
    instruction: str | None = None,
) -> str:
    """Render one pair.

    ``orientation`` decides which side is the Query and which the Document.
    Rerankers are asymmetric, so this is a real design choice, not a detail:
    ``change_query`` treats the change as the information need and the test as the
    candidate document; ``test_query`` inverts it.

    ``instruction`` overrides the question text (see ``INSTRUCTION_VARIANTS``).

    ``features_block`` carries the structured features for the fairness condition.
    ``placement`` controls where it goes: inside ``<Instruct>`` (before the
    Query/Document, which also pushes the content further from the final position
    the reranker reads) or appended after ``<Document>``. The placement switch
    isolates a position/length effect from a content effect.
    """
    if instruction is None:
        instruction = INSTRUCTION
    if orientation == "change_query":
        query, document = change_text, test_text
    elif orientation == "test_query":
        query, document = test_text, change_text
    else:
        raise ValueError(f"unknown orientation: {orientation!r}")
    if placement == "instruct":
        head = instruction + (f"\n\n{features_block}" if features_block else "")
        body = (
            f"<Instruct>: {head}\n"
            f"<Query>: {query.strip()}\n"
            f"<Document>: {document.strip()}"
        )
    elif placement == "after_document":
        body = (
            f"<Instruct>: {instruction}\n"
            f"<Query>: {query.strip()}\n"
            f"<Document>: {document.strip()}"
        )
        if features_block:
            body += f"\n\n{features_block}"
    else:
        raise ValueError(f"unknown placement: {placement!r}")
    return prefix + body + suffix


def load_model(device: str = "auto", dtype: str = "bfloat16"):
    """Load the pinned reranker through SemIf's own loader."""
    from semif_phase1.core import load_causal_model

    return load_causal_model(
        config.SEMIF_MODEL, config.SEMIF_MODEL_REVISION, device=device, dtype=dtype
    )


def _answer_ids_cached(tokenizer):
    _, _, answer_ids = _prefix_suffix()
    return answer_ids(tokenizer)


def format_features(
    ds: contract.Dataset,
    matrix: features.FeatureMatrix,
    row: int,
    col: int,
    mode: str = "full",
    value_col: int | None = None,
) -> str:
    """Serialize structured features for one pair into prompt text.

    ``mode`` produces the diagnostic variants:

    * ``full`` -- every feature, as XGBoost sees them (the mirror treatment).
    * ``informative`` -- only features that vary across a change's candidate sets, so
      the block stops carrying per-change constants that cannot discriminate.
    * ``placebo`` -- identical field names and length, every value replaced by a
      constant. Same distraction, zero information: this isolates a length/format
      effect from a content effect.
    * ``shuffled`` -- real values with the real marginal distribution, but read from
      a different candidate in the same change (``value_col``). Format and
      distribution preserved, association with the candidate destroyed.

    In every mode the identifier lines describe the actual candidate, so only the
    feature values are manipulated.
    """
    source = col if value_col is None else value_col
    test_file, _, test_name = ds.test_ids[col].partition("::")
    change = ds.changes[row]
    names = matrix.columns
    lines = [
        f"- changed file: {accessors.change_paths(ds)[row]}",
        f"- changed function: {change.func_name}",
        f"- test file: {test_file}",
        f"- test name: {test_name}",
    ]
    if mode == "informative":
        selected = [i for i, n in enumerate(names) if n not in PER_CHANGE_CONSTANT]
    else:
        selected = list(range(len(names)))

    boolean = {"function_coverage", "filename_match"}
    integer = {
        "coverage_set_size", "tests_per_file", "test_lines", "test_tokens",
        "code_churn", "added_lines", "removed_lines", "cumulative_runs",
        "path_proximity",
    }
    for k in selected:
        name = names[k]
        if mode == "placebo":
            lines.append(f"- {name}: n/a")
            continue
        value = float(matrix.X[row, source, k])
        if name == "failure_recency":
            lines.append(
                f"- {name}: never failed in prior changes"
                if value >= ds.n_changes
                else f"- {name}: {int(round(value))}"
            )
        elif name in boolean:
            lines.append(f"- {name}: {'yes' if value > 0.5 else 'no'}")
        elif name in integer:
            lines.append(f"- {name}: {int(round(value))}")
        else:
            lines.append(f"- {name}: {value:.4f}")
    return "Precomputed candidate metadata (from the test_suite):\n" + "\n".join(lines)


def score_batch(model, tokenizer, prompts: list[str], max_tokens: int) -> tuple[list[float], dict]:
    """Score prompts in one padded forward pass; return yes-vs-no log-odds."""
    import torch

    prefix, suffix, answer_ids = _prefix_suffix()
    no_id, yes_id = answer_ids(tokenizer)

    encoded = []
    for text in prompts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError("empty prompt encoding")
        if len(ids) > max_tokens:
            ids = ids[:max_tokens]
        encoded.append(ids)

    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    width = max(len(ids) for ids in encoded)
    device = next(model.parameters()).device
    input_ids = torch.tensor(
        [[pad] * (width - len(ids)) + ids for ids in encoded], device=device
    )
    attention_mask = torch.tensor(
        [[0] * (width - len(ids)) + [1] * len(ids) for ids in encoded], device=device
    )

    kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    if "logits_to_keep" in inspect.signature(model.forward).parameters:
        kwargs["logits_to_keep"] = 1

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(**kwargs).logits[:, -1, :].float()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    selected = logits[:, [no_id, yes_id]]
    log_odds = (selected[:, 1] - selected[:, 0]).tolist()
    timing = {
        "pairs": len(prompts),
        "forward_seconds": elapsed,
        "padded_tokens": int(input_ids.numel()),
        "max_prompt_tokens": width,
        "pairs_per_second": len(prompts) / elapsed if elapsed > 0 else float("nan"),
    }
    return log_odds, timing


def score_pairs(
    model,
    tokenizer,
    pairs: list[tuple[str, str]],
    batch_size: int = 8,
    max_tokens: int | None = None,
    progress_every: int = 20,
    orientation: str = "change_query",
    feature_blocks: list[str] | None = None,
    batch_callback=None,
    empty_cache_every: int = 50,
    placement: str = "instruct",
    bucket_by_length: bool = False,
    instruction: str | None = None,
) -> tuple[list[float], dict]:
    """Score (change_text, test_text) pairs in batches. Returns scores and stats.

    ``bucket_by_length`` sorts pairs by approximate prompt length before batching,
    on the theory that padding wastes compute. **Measured: it does not help.**
    Bucketing reduced padding only 483 -> 442 tokens/pair (8%) while costing ~18%
    throughput (21.3 -> 17.6 pairs/s at batch 16), because the run is compute-bound
    on the forward pass rather than padding-bound. Off by default; kept because the
    reasoning was plausible and the negative result is worth recording.

    ``batch_callback(indices, scores)`` fires after each batch with the *original*
    positions, so callers can persist progress incrementally even though batches are
    no longer contiguous. ``empty_cache_every`` periodically releases the ROCm /
    CUDA allocator cache; long runs on this device otherwise fragment and fail with
    an HSA allocation error partway through.
    """
    max_tokens = max_tokens or config.SEMIF_MAX_TOKENS
    prefix, suffix, _ = _prefix_suffix()
    scores: list[float] = [0.0] * len(pairs)
    total_tokens = 0
    total_seconds = 0.0
    started = time.perf_counter()

    if bucket_by_length:
        # Character length is a cheap proxy for token length and ordering only
        # needs to be approximate, so the tokenizer is not run twice.
        keys = np.array([len(c) + len(t) for c, t in pairs], dtype=np.int64)
        sequence = np.argsort(keys, kind="stable")
    else:
        sequence = np.arange(len(pairs))

    for start in range(0, len(pairs), batch_size):
        idx = sequence[start : start + batch_size]
        chunk = [pairs[int(i)] for i in idx]
        blocks = (
            [feature_blocks[int(i)] for i in idx] if feature_blocks else None
        )
        prompts = [
            build_prompt(
                c, t, prefix, suffix,
                orientation=orientation,
                features_block=blocks[i] if blocks else "",
                placement=placement,
                instruction=instruction,
            )
            for i, (c, t) in enumerate(chunk)
        ]
        chunk_scores, timing = score_batch(model, tokenizer, prompts, max_tokens)
        for i, value in zip(idx, chunk_scores):
            scores[int(i)] = value
        total_tokens += timing["padded_tokens"]
        total_seconds += timing["forward_seconds"]

        if batch_callback is not None:
            batch_callback([int(i) for i in idx], chunk_scores)
        if empty_cache_every and (start // batch_size) % empty_cache_every == 0:
            import torch as _torch

            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()

        if progress_every and (start // batch_size) % progress_every == 0:
            done = min(start + batch_size, len(pairs))
            rate = done / max(time.perf_counter() - started, 1e-9)
            print(f"  [{done}/{len(pairs)}] {rate:.1f} pairs/s", flush=True)

    wall = time.perf_counter() - started
    stats = {
        "pairs": len(pairs),
        "batch_size": batch_size,
        "orientation": orientation,
        "wall_seconds": wall,
        "forward_seconds": total_seconds,
        "padded_tokens": total_tokens,
        "mean_padded_tokens_per_pair": total_tokens / max(len(pairs), 1),
        "pairs_per_second": len(pairs) / max(wall, 1e-9),
        "mirror": bool(feature_blocks),
        "bucketed": bool(bucket_by_length),
        "instruction": instruction or INSTRUCTION,
    }
    return scores, stats


# --- dataset scoring -------------------------------------------------------


@dataclass
class PairSet:
    pairs: list[tuple[str, str]]
    index: list[tuple[int, int]]
    feature_blocks: list[str] | None = None
    placement: str = "instruct"
    instruction: str | None = None


def build_pair_set(
    ds: contract.Dataset,
    rows: np.ndarray,
    candidate_sets: np.ndarray,
    shuffle: bool = False,
    seed: int = config.SEED,
    include_features: bool = False,
    feature_mode: str | None = None,
    placement: str = "instruct",
    instruction: str | None = None,
) -> PairSet:
    """Build pairs, optionally with a prompt metadata block.

    ``feature_mode`` selects the block variant, or ``None`` for text-only prompts.
    ``include_features=True`` is the older spelling of ``feature_mode="full"``.
    """
    if include_features and feature_mode is None:
        feature_mode = "full"

    texts = [features.derived.change_query_text(ds, c) for c in ds.changes]
    if shuffle:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(texts))
        texts = [texts[i] for i in perm]

    matrix = None
    want_blocks = feature_mode is not None
    if want_blocks:
        # The mirror/informative conditions exist to compare a model's input against the
        # classical model's, so they ask for the same feature block explicitly.
        matrix = features.structured(ds, temporal=True)

    pairs: list[tuple[str, str]] = []
    index: list[tuple[int, int]] = []
    blocks: list[str] | None = [] if want_blocks else None
    for r in rows:
        r = int(r)
        cols = [int(j) for j in np.flatnonzero(candidate_sets[r])]
        if feature_mode == "shuffled":
            # Values come from a different candidate in the same change, so the
            # block keeps its format and value distribution but loses any
            # association with the candidate it is describing.
            rng_row = np.random.default_rng(seed + r)
            value_sources = [cols[p] for p in rng_row.permutation(len(cols))]
        else:
            value_sources = cols
        for pos, j in enumerate(cols):
            pairs.append((texts[r], ds.test_source(ds.test_ids[j]) or ""))
            index.append((r, j))
            if blocks is not None:
                blocks.append(
                    format_features(
                        ds, matrix, r, j,
                        mode=feature_mode, value_col=value_sources[pos],
                    )
                )
    return PairSet(pairs=pairs, index=index, feature_blocks=blocks, placement=placement,
                   instruction=instruction)


def load_done_keys(path: Path) -> set[tuple[int, int]]:
    """(change_row, test_col) pairs already written to a cache file.

    Tolerates a torn final line, which is what a crash mid-write leaves behind.
    """
    done: set[tuple[int, int]] = set()
    path = Path(path)
    if not path.exists():
        return done
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "change_row" in record and "test_col" in record:
                done.add((int(record["change_row"]), int(record["test_col"])))
    return done


def _missing_pairs(pair_set: PairSet, path: Path) -> set[tuple[int, int]]:
    """The pairs of ``pair_set`` that ``path`` does not hold. One implementation."""
    return set(pair_set.index) - load_done_keys(Path(path))


def missing_pairs(
    ds: contract.Dataset,
    rows: np.ndarray,
    candidate_sets: np.ndarray,
    path: Path,
    *,
    shuffle: bool = False,
    seed: int = config.SEED,
    feature_mode: str | None = None,
    placement: str = "instruct",
    instruction: str | None = None,
) -> set[tuple[int, int]]:
    """The pairs a cache is missing for this context; empty means it can be read as its own.

    A cache is identified by its *path*, so the file at a path may have been produced for
    different rows, a different candidate pool or a different prompt wording -- and reading
    it anyway yields a matrix that looks like a measurement and is not. Completeness against
    the pair set the context actually needs is the check the record format supports, and it
    catches both that case and a torn write.

    The parameterised entry point onto :func:`_missing_pairs`, which :func:`score_context`
    calls with the pair set it already built, so the two cannot disagree about what
    "complete" means. The check reads the ``(change_row, test_col)`` indices, which
    :func:`score_to_cache` writes beside the stable identifiers -- so it is a check on caches
    this module wrote, and a cache carrying only identifiers would read as empty here.
    """
    pair_set = build_pair_set(
        ds,
        rows,
        candidate_sets,
        shuffle=shuffle,
        seed=seed,
        feature_mode=feature_mode,
        placement=placement,
        instruction=instruction,
    )
    return _missing_pairs(pair_set, path)


def score_to_cache(
    model,
    tokenizer,
    ds: contract.Dataset,
    pair_set: PairSet,
    out_path: Path,
    batch_size: int = 8,
    max_tokens: int | None = None,
    orientation: str = "change_query",
    resume: bool = True,
) -> dict:
    """Score pairs, appending each batch so the run is resumable.

    Scores used to be written only at the very end, which meant a GPU failure at
    90% discarded hours of work. They are now flushed per batch and already-present
    pairs are skipped on restart.
    """
    done = load_done_keys(out_path) if resume else set()
    keep = [i for i, (r, c) in enumerate(pair_set.index) if (r, c) not in done]
    if not keep:
        print(f"[semif] {out_path} already complete ({len(done)} pairs)")
        return {"pairs": 0, "already_done": len(done), "complete": True}
    if done:
        print(f"[semif] resuming {out_path.name}: {len(keep)} of "
              f"{len(pair_set.index)} pairs remain")

    pairs = [pair_set.pairs[i] for i in keep]
    index = [pair_set.index[i] for i in keep]
    blocks = (
        [pair_set.feature_blocks[i] for i in keep] if pair_set.feature_blocks else None
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = out_path.open("a")

    def write_batch(indices, chunk_scores: list[float]) -> None:
        for i, score in zip(indices, chunk_scores):
            row, col = index[i]
            handle.write(
                json.dumps(
                    {
                        "change_row": row,
                        "test_col": col,
                        "change_id": ds.change_id(ds.changes[row]),
                        "test_nodeid": ds.test_ids[col],
                        "score": float(score),
                    }
                )
                + "\n"
            )
        handle.flush()

    try:
        scores, stats = score_pairs(
            model, tokenizer, pairs,
            batch_size=batch_size, max_tokens=max_tokens, orientation=orientation,
            feature_blocks=blocks, batch_callback=write_batch,
            placement=pair_set.placement, instruction=pair_set.instruction,
        )
    finally:
        handle.close()

    print(f"[semif] wrote {len(scores)} new pairs to {out_path}")
    print(f"[semif] {stats}")
    return stats


# --- context-driven scoring (the experiment layer's entry point) -----------


def score_context(
    ds: contract.Dataset,
    rows: np.ndarray,
    candidate_sets: np.ndarray,
    out_path: Path,
    *,
    batch_size: int = 8,
    max_tokens: int | None = None,
    orientation: str = "change_query",
    instruction: str | None = None,
    feature_mode: str | None = None,
    placement: str = "instruct",
    shuffle: bool = False,
    seed: int = config.SEED,
    resume: bool = True,
    model=None,
    tokenizer=None,
) -> tuple[np.ndarray, dict]:
    """Score ``rows`` against their candidate sets: write the cache, return the matrix.

    The context-driven entry point, and the reason score *production* can be a design point rather
    than a driver's invisible precondition. Everything the pair set needs comes from the
    dataset, the rows and the candidate mask, so a
    :class:`~rts.model.rankers.ProducedScores` ranker calls this from ``scores(ctx)`` and the
    layer sees the study's most expensive step with a tier, a cost and a design point key.

    ``model``/``tokenizer`` may be injected -- a test supplies a fake, so the produce path
    needs no GPU -- and are loaded from the pinned checkpoint otherwise.

    The matrix is read **back from the cache** rather than assembled from the in-memory
    scores, which makes "produced" and "read later" the same object by construction. The
    completeness check is the other half of that: a partially written cache is otherwise
    indistinguishable from a finished one, and only the sentinel values would say so.
    """
    if model is None or tokenizer is None:
        model, tokenizer, _metadata = load_model()

    pair_set = build_pair_set(
        ds, rows, candidate_sets,
        shuffle=shuffle, seed=seed,
        feature_mode=feature_mode, placement=placement, instruction=instruction,
    )
    stats = score_to_cache(
        model, tokenizer, ds, pair_set, out_path,
        batch_size=batch_size, max_tokens=max_tokens,
        orientation=orientation, resume=resume,
    )

    expected = set(pair_set.index)
    missing = _missing_pairs(pair_set, out_path)
    if missing:
        raise RuntimeError(
            f"{out_path} is missing {len(missing)} of {len(expected)} pair(s) after scoring; "
            "a partial cache would be read as if it were complete"
        )

    matrix = semif.load_scores(Path(out_path), ds)
    stats["pairs_expected"] = len(expected)
    stats["pairs_in_cache"] = len(expected) - len(missing)
    return matrix, stats


# --- smoke test ------------------------------------------------------------


def pilot(
    n_changes: int = 50,
    batch_size: int = 16,
    orientation: str = "change_query",
    max_tokens: int | None = None,
    seed: int = config.SEED,
) -> dict:
    """Score N held-out fault changes on their covered candidate sets.

    Produces recall numbers directly comparable to the other rankers, plus real
    throughput so the full-run cost can be extrapolated instead of guessed.
    """
    from .. import evaluate

    ds = datasets.marshmallow(order_seed=seed)
    split = splits.make_split(ds)
    candidate_sets = accessors.candidate_sets(ds, "coverage_restricted")
    fault_held = [int(i) for i in accessors.test_fault_idx(ds, split.test_idx)]
    rows = np.array(fault_held[:n_changes], dtype=np.int64)

    pair_set = build_pair_set(ds, rows, candidate_sets)
    print(f"changes         : {len(rows)}")
    print(f"pairs           : {len(pair_set.pairs)}")

    model, tokenizer, metadata = load_model()
    print(f"loaded          : {metadata['device']} {metadata['dtype']}")
    scores, stats = score_pairs(
        model, tokenizer, pair_set.pairs,
        batch_size=batch_size, max_tokens=max_tokens, orientation=orientation,
    )
    print(f"throughput      : {stats['pairs_per_second']:.1f} pairs/s "
          f"(batch {batch_size}, {stats['mean_padded_tokens_per_pair']:.0f} tokens/pair)")

    matrix = np.full((ds.n_changes, ds.n_tests), -1e9, dtype=np.float32)
    for (row, col), score in zip(pair_set.index, scores):
        matrix[row, col] = score

    results = evaluate.evaluate(
        matrix, ds, rows, budgets=(0.01, 0.05, 0.1, 0.2),
        n_bootstrap=1000, seed=seed, candidate_sets=candidate_sets,
    )
    print(evaluate.format_table(f"semif_reranker ({orientation}, pilot n={len(rows)})", results))

    full_pairs = int(candidate_sets[split.test_idx].sum())
    print(f"\nfull held-out cost at this throughput: {full_pairs:,} pairs "
          f"-> {full_pairs / stats['pairs_per_second'] / 3600:.1f} h")
    return {"stats": stats, "results": evaluate.results_to_dicts(results)}


def smoke_test(n_changes: int = 10, n_distractors: int = 9, batch_size: int = 8, seed: int = config.SEED):
    """Rank a known killing test against distractors, for changes with a fault.

    This is the cheapest possible go/no-go: real data, known answer, no scoring of
    the full grid.
    """
    ds = datasets.marshmallow(order_seed=seed)
    split = splits.make_split(ds)
    rng = np.random.default_rng(seed)
    faults = [i for i in accessors.test_fault_idx(ds, split.test_idx)]
    picked = rng.choice(faults, size=min(n_changes, len(faults)), replace=False)

    print(f"loading {config.SEMIF_MODEL} @ {config.SEMIF_MODEL_REVISION[:12]} ...")
    model, tokenizer, metadata = load_model()
    print(f"loaded on {metadata['device']} dtype={metadata['dtype']} "
          f"torch={metadata['torch_version']} transformers={metadata['transformers_version']}")

    hits = 0
    for n, i in enumerate(picked, 1):
        killing = sorted(ds.killing_tests(ds.changes[i]))[0]
        kill_col = ds.test_index[killing]
        pool = [c for c in accessors.coverage_sets(ds)[i] if c != killing]
        distractors = rng.choice(pool, size=min(n_distractors, len(pool)), replace=False)
        cols = [kill_col] + [ds.test_index[ds.test_ids[c]] if isinstance(c, (int, np.integer)) else ds.test_index[c] for c in distractors]
        cols = [int(c) for c in cols]

        texts = features.derived.change_query_text(ds, ds.changes[i])
        pairs = [(texts, ds.test_source(ds.test_ids[c]) or "") for c in cols]
        scores, _ = score_pairs(model, tokenizer, pairs, batch_size=batch_size, progress_every=0)

        order = np.argsort(-np.array(scores))
        rank = int(np.flatnonzero(order == 0)[0]) + 1
        hits += rank == 1
        print(
            f"  {n:>2}. {ds.changes[i].func_name:<34} killing test rank {rank}/{len(cols)} "
            f"(margin {scores[0] - max(scores[1:]):+.2f})"
        )

    print(f"\nkilling test ranked #1 in {hits}/{len(picked)} cases "
          f"(chance = {1 / (n_distractors + 1):.2f})")
    return hits, len(picked)


def score_heldout(
    batch_size: int = 16,
    orientation: str = "change_query",
    max_tokens: int | None = None,
    shuffle: bool = False,
    include_features: bool = False,
    out_path: Path | None = None,
    seed: int = config.SEED,
    candidate_policy: str = "coverage_restricted",
    cold_start_max_failures: int | None = None,
    instruction: str | None = None,
    feature_mode: str | None = None,
    placement: str = "instruct",
    train_prefix: int | None = None,
    exclude_scored: Path | None = None,
) -> dict:
    """Score every held-out change (or a cold start subset) against its candidate sets.

    SemIf is zero-shot, so the training window is never needed: scoring only the
    held-out changes keeps all 464 faults and costs a fifth of the full grid.
    Writes the cache in the format ``rts.model.semif.load_scores`` consumes.

    ``include_features`` selects the fairness condition: with it, the prompt carries the
    same 15 structured features XGBoost receives. Without it the prompt is
    text-only, which is the control.

    Three extensions added for the variation experiments:

    * ``candidate_policy="full"`` scores the whole 1187-test suite. This is the
      outstanding "full-suite cold start condition": with ``covered`` candidate sets the
      ``coverage`` baseline is degenerate (every candidate already covers the
      mutated function), so this is what makes the one positive result comparable
      to how RTS is actually deployed.
    * ``cold_start_max_failures`` restricts rows to the cold start subset, so the
      full suite only has to be scored for those changes.
    * ``instruction`` swaps the question text (see ``INSTRUCTION_VARIANTS``); the
      prompt skeleton is otherwise identical.
    * ``train_prefix`` scores the last N changes of the *training* window instead of
      the held-out window. The P5 "SemIf as an XGBoost column" variant needs the
      feature on training rows, and the cache only covers held-out changes. Scoring
      the whole training window is 329k pairs (~3 h); a temporally adjacent prefix
      of 400 changes is ~62k pairs and is enough to fit the column contrast, with
      the baseline trained on exactly the same rows.
    """
    ds = datasets.marshmallow(order_seed=seed)
    split = splits.make_split(ds)
    candidate_sets = accessors.candidate_sets(ds, candidate_policy)
    rows = split.test_idx
    if cold_start_max_failures is not None:
        mask = subsets.cold_start_mask(ds, max_failures=cold_start_max_failures)
        rows = split.test_idx[mask[split.test_idx]]
    if train_prefix is not None:
        rows = split.train_idx[-train_prefix:]
    if exclude_scored is not None:
        # Score only the changes a previous condition has not already covered. Used to
        # build a superset condition incrementally: `failures <= 2` is a subset of
        # `failures <= 5`, so the 141-change condition only needs the 98 new changes.
        done = load_done_keys(exclude_scored)
        before = len(rows)
        rows = np.array(
            [
                int(r) for r in rows
                if not all((int(r), int(j)) in done for j in np.flatnonzero(candidate_sets[r]))
            ],
            dtype=np.int64,
        )
        print(f"excluded already-scored: {before - len(rows)} of {before} changes "
              f"(from {exclude_scored.name})")
    if feature_mode is None and include_features:
        feature_mode = "full"
    pair_set = build_pair_set(
        ds, rows, candidate_sets, shuffle=shuffle, seed=seed,
        feature_mode=feature_mode, placement=placement, instruction=instruction,
    )

    if out_path is None:
        if shuffle:
            out_path = config.ARTIFACTS / "semif_scores_shuffled.jsonl"
        elif feature_mode == "full":
            out_path = config.ARTIFACTS / "semif_scores_mirror.jsonl"
        else:
            out_path = config.SEMIF_SCORES_FILE

    print(f"orientation     : {orientation}")
    print(f"shuffle         : {shuffle}")
    print(f"feature mode    : {feature_mode}")
    print(f"placement       : {placement}")
    print(f"instruction     : {'default' if instruction is None else 'override'}")
    print(f"candidate_sets      : {candidate_policy}")
    print(f"cold_start <=      : {cold_start_max_failures}")
    print(f"train prefix    : {train_prefix}")
    print(f"changes         : {len(rows)}")
    print(f"pairs           : {len(pair_set.pairs):,}")
    print(f"output          : {out_path}")

    model, tokenizer, metadata = load_model()
    print(f"loaded          : {metadata['device']} {metadata['dtype']}", flush=True)
    stats = score_to_cache(
        model, tokenizer, ds, pair_set, out_path,
        batch_size=batch_size, max_tokens=max_tokens, orientation=orientation,
    )
    return stats


def run_controls(
    max_failures: int = 5,
    batch_size: int = 8,
    max_tokens: int | None = None,
    seed: int = config.SEED,
) -> dict:
    """Run the mirror-degradation diagnostics on the cold start subset.

    Depends on the existing text-only and full-mirror caches for the reference
    conditions; scores the remaining conditions sequentially in one process so the model is
    loaded once.
    """
    ds = datasets.marshmallow(order_seed=seed)
    split = splits.make_split(ds)
    candidate_sets = accessors.candidate_sets(ds, "coverage_restricted")
    mask = subsets.cold_start_mask(ds, max_failures=max_failures)
    rows = split.test_idx[mask[split.test_idx]]
    n_faults = int(sum(1 for i in rows if accessors.fault_mask(ds)[i]))

    print(f"subset          : cold_start failures<={max_failures}")
    print(f"changes         : {len(rows)}")
    print(f"faults          : {n_faults}")

    # (label, feature_mode, placement, output filename)
    conditions = [
        ("informative", "informative", "instruct", "semif_scores_ctl_informative.jsonl"),
        ("placebo", "placebo", "instruct", "semif_scores_ctl_placebo.jsonl"),
        ("shuffled", "shuffled", "instruct", "semif_scores_ctl_shuffled.jsonl"),
        ("after_document", "full", "after_document", "semif_scores_ctl_after.jsonl"),
    ]

    model, tokenizer, metadata = load_model()
    print(f"loaded          : {metadata['device']} {metadata['dtype']}", flush=True)

    stats: dict[str, dict] = {}
    for label, mode, placement, filename in conditions:
        out_path = config.ARTIFACTS / filename
        print(f"\n=== condition: {label} (mode={mode}, placement={placement}) ===", flush=True)
        pair_set = build_pair_set(
            ds, rows, candidate_sets,
            feature_mode=mode, placement=placement, seed=seed,
        )
        print(f"pairs: {len(pair_set.pairs):,}", flush=True)
        stats[label] = score_to_cache(
            model, tokenizer, ds, pair_set, out_path,
            batch_size=batch_size, max_tokens=max_tokens,
        )
        stats[label]["output"] = str(out_path)
        del pair_set
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pilot", type=int, default=0, help="score N held-out changes")
    parser.add_argument("--heldout", action="store_true", help="score all held-out changes")
    parser.add_argument("--controls", action="store_true",
                        help="run the mirror-degradation diagnostics on the cold_start subset")
    parser.add_argument("--shuffle", action="store_true", help="change-shuffle ablation")
    parser.add_argument("--mirror", action="store_true",
                        help="include the structured features in the prompt (fairness condition)")
    parser.add_argument("--max-failures", type=int, default=5,
                        help="cold_start subset threshold for --controls")
    parser.add_argument("--changes", type=int, default=10)
    parser.add_argument("--distractors", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--orientation", default="change_query",
                        choices=["change_query", "test_query"])
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--candidate-policy", default="coverage_restricted", choices=["coverage_restricted", "full"],
                        help="candidate set to score; 'full' is the whole 1187-test_suite")
    parser.add_argument("--cold_start", type=int, default=None,
                        help="restrict to the cold_start subset with this failure cap")
    parser.add_argument("--train-prefix", type=int, default=None,
                        help="score the last N training-window changes (for the P5 column)")
    parser.add_argument("--instruction", default="default",
                        choices=sorted(INSTRUCTION_VARIANTS),
                        help="question wording (P2 sweep)")
    parser.add_argument("--out", type=Path, default=None,
                        help="explicit output cache path")
    parser.add_argument("--exclude-scored", type=Path, default=None,
                        help="skip changes already fully scored in this cache")
    args = parser.parse_args()

    if args.controls:
        run_controls(
            max_failures=args.max_failures,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
        )
    elif args.heldout:
        instruction = (
            None if args.instruction == "default" else INSTRUCTION_VARIANTS[args.instruction]
        )
        score_heldout(
            batch_size=args.batch_size,
            orientation=args.orientation,
            max_tokens=args.max_tokens,
            shuffle=args.shuffle,
            include_features=args.mirror,
            out_path=args.out,
            candidate_policy=args.candidate_policy,
            cold_start_max_failures=args.cold_start,
            instruction=instruction,
            train_prefix=args.train_prefix,
            exclude_scored=args.exclude_scored,
        )
    elif args.pilot:
        pilot(
            n_changes=args.pilot,
            batch_size=args.batch_size,
            orientation=args.orientation,
            max_tokens=args.max_tokens,
        )
    else:
        smoke_test(
            n_changes=args.changes,
            n_distractors=args.distractors,
            batch_size=args.batch_size,
        )
