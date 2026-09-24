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

from . import config, dataset, features, source

INSTRUCTION = (
    "Given a code change, judge whether the test below exercises the changed "
    "behaviour, such that this change could make the test fail. Answer yes only "
    "if the test would plausibly need to run for this change."
)

# Structured features that never vary across a change's candidates under the
# `covered` mask, and so cannot help rank within that change. Verified empirically:
# path_distance is included because every test lives in the same directory tree, so
# it is constant per change rather than merely low-variance.
PER_CHANGE_CONSTANT = frozenset(
    {
        "covers_function",
        "n_covering_tests",
        "coverage_rank_prior",
        "path_distance",
        "change_size",
        "change_added_lines",
        "change_removed_lines",
    }
)

# Diagnostic arms for the mirror-degradation question. Each holds everything else
# fixed and varies one property of the injected block:
#   full          -- all 15 features (the original mirror treatment)
#   informative   -- only features that actually vary across candidates
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
) -> str:
    """Render one pair.

    ``orientation`` decides which side is the Query and which the Document.
    Rerankers are asymmetric, so this is a real design choice, not a detail:
    ``change_query`` treats the change as the information need and the test as the
    candidate document; ``test_query`` inverts it.

    ``features_block`` carries the structured features for the fairness arm.
    ``placement`` controls where it goes: inside ``<Instruct>`` (before the
    Query/Document, which also pushes the content further from the final position
    the reranker reads) or appended after ``<Document>``. The placement switch
    isolates a position/length effect from a content effect.
    """
    if orientation == "change_query":
        query, document = change_text, test_text
    elif orientation == "test_query":
        query, document = test_text, change_text
    else:
        raise ValueError(f"unknown orientation: {orientation!r}")
    if placement == "instruct":
        instruction = INSTRUCTION + (f"\n\n{features_block}" if features_block else "")
        body = (
            f"<Instruct>: {instruction}\n"
            f"<Query>: {query.strip()}\n"
            f"<Document>: {document.strip()}"
        )
    elif placement == "after_document":
        body = (
            f"<Instruct>: {INSTRUCTION}\n"
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
    ds: dataset.Dataset,
    X,
    names: list[str],
    row: int,
    col: int,
    mode: str = "full",
    value_col: int | None = None,
) -> str:
    """Serialize structured features for one pair into prompt text.

    ``mode`` produces the diagnostic variants:

    * ``full`` -- every feature, as XGBoost sees them (the mirror treatment).
    * ``informative`` -- only features that vary across a change's candidates, so
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
    lines = [
        f"- changed file: {change.file}",
        f"- changed function: {change.func_name}",
        f"- test file: {test_file}",
        f"- test name: {test_name}",
    ]
    if mode == "informative":
        selected = [i for i, n in enumerate(names) if n not in PER_CHANGE_CONSTANT]
    else:
        selected = list(range(len(names)))

    boolean = {"covers_function", "module_name_in_test_file"}
    integer = {
        "n_covering_tests", "n_tests_in_test_file", "test_n_lines", "test_n_tokens",
        "change_size", "change_added_lines", "change_removed_lines", "test_runs_cum",
        "path_distance",
    }
    for k in selected:
        name = names[k]
        if mode == "placebo":
            lines.append(f"- {name}: n/a")
            continue
        value = float(X[row, source, k])
        if name == "test_last_failure_age":
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
    return "Precomputed candidate metadata (from the test suite):\n" + "\n".join(lines)


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
    }
    return scores, stats


# --- dataset scoring -------------------------------------------------------


@dataclass
class PairSet:
    pairs: list[tuple[str, str]]
    index: list[tuple[int, int]]
    feature_blocks: list[str] | None = None
    placement: str = "instruct"


def build_pair_set(
    ds: dataset.Dataset,
    rows: np.ndarray,
    candidates: np.ndarray,
    shuffle: bool = False,
    seed: int = config.SEED,
    include_features: bool = False,
    feature_mode: str | None = None,
    placement: str = "instruct",
) -> PairSet:
    """Build pairs, optionally with a prompt metadata block.

    ``feature_mode`` selects the block variant, or ``None`` for text-only prompts.
    ``include_features=True`` is the older spelling of ``feature_mode="full"``.
    """
    if include_features and feature_mode is None:
        feature_mode = "full"

    infos = source.load_all(ds.test_ids)
    texts = [features.change_query_text(c) for c in ds.changes]
    if shuffle:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(texts))
        texts = [texts[i] for i in perm]

    X = names = None
    want_blocks = feature_mode is not None
    if want_blocks:
        X, names = features.structured_features(ds)

    pairs: list[tuple[str, str]] = []
    index: list[tuple[int, int]] = []
    blocks: list[str] | None = [] if want_blocks else None
    for r in rows:
        r = int(r)
        cols = [int(j) for j in np.flatnonzero(candidates[r])]
        if feature_mode == "shuffled":
            # Values come from a different candidate in the same change, so the
            # block keeps its format and value distribution but loses any
            # association with the candidate it is describing.
            rng_row = np.random.default_rng(seed + r)
            value_sources = [cols[p] for p in rng_row.permutation(len(cols))]
        else:
            value_sources = cols
        for pos, j in enumerate(cols):
            info = infos.get(ds.test_ids[j])
            pairs.append((texts[r], info.source if info else ""))
            index.append((r, j))
            if blocks is not None:
                blocks.append(
                    format_features(
                        ds, X, names, r, j,
                        mode=feature_mode, value_col=value_sources[pos],
                    )
                )
    return PairSet(pairs=pairs, index=index, feature_blocks=blocks, placement=placement)


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


def score_to_cache(
    model,
    tokenizer,
    ds: dataset.Dataset,
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
                        "change_id": ds.changes[row].change_id,
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
            placement=pair_set.placement,
        )
    finally:
        handle.close()

    print(f"[semif] wrote {len(scores)} new pairs to {out_path}")
    print(f"[semif] {stats}")
    return stats


# --- smoke test ------------------------------------------------------------


def pilot(
    n_changes: int = 50,
    batch_size: int = 16,
    orientation: str = "change_query",
    max_tokens: int | None = None,
    seed: int = config.SEED,
) -> dict:
    """Score N held-out fault changes on their covered candidate sets.

    Produces recall numbers directly comparable to the other selectors, plus real
    throughput so the full-run cost can be extrapolated instead of guessed.
    """
    from . import evaluate

    ds = dataset.build(seed=seed)
    candidates = dataset.candidate_mask(ds, "covered")
    fault_held = [int(i) for i in ds.test_fault_idx]
    rows = np.array(fault_held[:n_changes], dtype=np.int64)

    pair_set = build_pair_set(ds, rows, candidates)
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
        n_bootstrap=1000, seed=seed, candidates=candidates,
    )
    print(evaluate.format_table(f"semif_reranker ({orientation}, pilot n={len(rows)})", results))

    full_pairs = int(candidates[ds.test_idx].sum())
    print(f"\nfull held-out cost at this throughput: {full_pairs:,} pairs "
          f"-> {full_pairs / stats['pairs_per_second'] / 3600:.1f} h")
    return {"stats": stats, "results": evaluate.results_to_dicts(results)}


def smoke_test(n_changes: int = 10, n_distractors: int = 9, batch_size: int = 8, seed: int = config.SEED):
    """Rank a known killing test against distractors, for changes with a fault.

    This is the cheapest possible go/no-go: real data, known answer, no scoring of
    the full grid.
    """
    ds = dataset.build(seed=seed)
    rng = np.random.default_rng(seed)
    faults = [i for i in ds.test_fault_idx]
    picked = rng.choice(faults, size=min(n_changes, len(faults)), replace=False)

    print(f"loading {config.SEMIF_MODEL} @ {config.SEMIF_MODEL_REVISION[:12]} ...")
    model, tokenizer, metadata = load_model()
    print(f"loaded on {metadata['device']} dtype={metadata['dtype']} "
          f"torch={metadata['torch_version']} transformers={metadata['transformers_version']}")

    hits = 0
    for n, i in enumerate(picked, 1):
        killing = ds.changes[i].killing_tests[0]
        kill_col = ds.test_index[killing]
        pool = [c for c in ds.covered[i] if c != killing]
        distractors = rng.choice(pool, size=min(n_distractors, len(pool)), replace=False)
        cols = [kill_col] + [ds.test_index[ds.test_ids[c]] if isinstance(c, (int, np.integer)) else ds.test_index[c] for c in distractors]
        cols = [int(c) for c in cols]

        texts = features.change_query_text(ds.changes[i])
        infos = source.load_all([ds.test_ids[c] for c in cols])
        pairs = [(texts, infos[ds.test_ids[c]].source) for c in cols]
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
) -> dict:
    """Score every held-out change against its covered candidates.

    SemIf is zero-shot, so the training window is never needed: scoring only the
    held-out changes keeps all 464 faults and costs a fifth of the full grid.
    Writes the cache in the format ``rts.semif.load_scores`` consumes.

    ``include_features`` selects the fairness arm: with it, the prompt carries the
    same 15 structured features XGBoost receives. Without it the prompt is
    text-only, which is the control.
    """
    ds = dataset.build(seed=seed)
    candidates = dataset.candidate_mask(ds, "covered")
    rows = ds.test_idx
    pair_set = build_pair_set(
        ds, rows, candidates, shuffle=shuffle, seed=seed,
        include_features=include_features,
    )

    if out_path is None:
        if shuffle:
            out_path = config.ARTIFACTS / "semif_scores_shuffled.jsonl"
        elif include_features:
            out_path = config.ARTIFACTS / "semif_scores_mirror.jsonl"
        else:
            out_path = config.SEMIF_SCORES_FILE

    print(f"orientation     : {orientation}")
    print(f"shuffle         : {shuffle}")
    print(f"mirror features : {include_features}")
    print(f"held-out changes: {len(rows)}")
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
    """Run the mirror-degradation diagnostics on the starved subset.

    Depends on the existing text-only and full-mirror caches for the reference
    arms; scores the remaining arms sequentially in one process so the model is
    loaded once.
    """
    ds = dataset.build(seed=seed)
    candidates = dataset.candidate_mask(ds, "covered")
    mask = dataset.starved_mask(ds, max_failures=max_failures)
    rows = ds.test_idx[mask[ds.test_idx]]
    n_faults = int(sum(1 for i in rows if ds.changes[i].killing_tests))

    print(f"subset          : starved failures<={max_failures}")
    print(f"changes         : {len(rows)}")
    print(f"faults          : {n_faults}")

    # (label, feature_mode, placement, output filename)
    arms = [
        ("informative", "informative", "instruct", "semif_scores_ctl_informative.jsonl"),
        ("placebo", "placebo", "instruct", "semif_scores_ctl_placebo.jsonl"),
        ("shuffled", "shuffled", "instruct", "semif_scores_ctl_shuffled.jsonl"),
        ("after_document", "full", "after_document", "semif_scores_ctl_after.jsonl"),
    ]

    model, tokenizer, metadata = load_model()
    print(f"loaded          : {metadata['device']} {metadata['dtype']}", flush=True)

    stats: dict[str, dict] = {}
    for label, mode, placement, filename in arms:
        out_path = config.ARTIFACTS / filename
        print(f"\n=== arm: {label} (mode={mode}, placement={placement}) ===", flush=True)
        pair_set = build_pair_set(
            ds, rows, candidates,
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
                        help="run the mirror-degradation diagnostics on the starved subset")
    parser.add_argument("--shuffle", action="store_true", help="change-shuffle ablation")
    parser.add_argument("--mirror", action="store_true",
                        help="include the structured features in the prompt (fairness arm)")
    parser.add_argument("--max-failures", type=int, default=5,
                        help="starved subset threshold for --controls")
    parser.add_argument("--changes", type=int, default=10)
    parser.add_argument("--distractors", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--orientation", default="change_query",
                        choices=["change_query", "test_query"])
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    if args.controls:
        run_controls(
            max_failures=args.max_failures,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
        )
    elif args.heldout:
        score_heldout(
            batch_size=args.batch_size,
            orientation=args.orientation,
            max_tokens=args.max_tokens,
            shuffle=args.shuffle,
            include_features=args.mirror,
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
