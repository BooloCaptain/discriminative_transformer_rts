"""P1: SemIf direct mode, pairwise, for regression test selection.

The proposal
------------
SemIf's ``--mode direct`` is the configuration the model was designed for: state
the criterion, list candidate options, and read the native next-token logits on the
option letters. That *changes the task formulation* rather than the model -- the
prompt asks the model to apply a criterion and commit to an answer, instead of
scoring a pair for topical relevance the way the reranker path does. It is the one
proposal with a real chance of changing the verdict.

Why pairwise rather than one window of 16 options
------------------------------------------------
The repo's direct mode caps a decision at 16 options (``LETTERS``), so ranking a
~155-test candidate set needs either several windows or elimination rounds. A
16-option window is tempting because it is ~15x cheaper per change, but it is also
the wrong thing to measure here:

* It makes the prompt ~16 test sources large and *multi-topic*. The complexity
  ladder already showed this exact manipulation hurting SemIf (-0.115 recall at
  3.8 files) while leaving structural models untouched.
* The position controls showed ~200 tokens of even *uninformative* prefix text cost
  0.21 recall, so a several-thousand-token context is a confound stacked on top of
  the formulation change.
* Softmax over slots is only comparable *within* a window, so a global ranking needs
  cross-window normalisation or a tournament, both of which add their own
  assumptions.

Pairwise keeps the context the same size as the reranker arm's (change + one test),
so the only differences from the reranker arm are the model and the prompt
formulation. That is the clean, comparable experiment. ``--windowed`` implements the
tournament variant as a secondary arm for completeness.

Scoring
-------
Two options, always, so the readout is a fixed 2-slot softmax and every pair is on
one global scale:

    state    = {"change": <change text>, "test": <test source>}
    question = <the RTS criterion>
    options  = yes / no

Readout is ``logit(A) - logit(B)``, the same log-odds shape the reranker arm uses.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import config, dataset, features, source
from .semif_runner import PairSet, build_pair_set, load_done_keys

# The criterion for pairwise direct mode. Deliberately the same content as the
# reranker's INSTRUCTION default so the comparison isolates formulation + model
# rather than wording: this is the ``default`` variant of the P2 sweep expressed as
# a criterion rather than as a relevance question.
CRITERION = (
    "Given a code change, judge whether the test below exercises the changed "
    "behaviour, such that this change could make the test fail. Choose yes only "
    "if the test would plausibly need to run for this change."
)

YES_OPTION = {"id": "yes", "description": "The test would need to run for this change and could fail because of it."}
NO_OPTION = {"id": "no", "description": "The test is unrelated to this change and would behave the same."}


def load_direct_model(device: str = "auto"):
    """Load the direct-mode checkpoint pinned by the SemIf repo's manifest."""
    from semif_phase1.core import load_causal_model

    return load_causal_model(
        config.DIRECT_MODEL, config.DIRECT_MODEL_REVISION, device=device, dtype="bfloat16"
    )


def build_rows(
    ds: dataset.Dataset,
    pair_set: PairSet,
    criterion: str = CRITERION,
) -> list[dict]:
    """One direct-mode row per (change, test) pair."""
    rows: list[dict] = []
    for (change_row, test_col), (change_text, test_text) in zip(
        pair_set.index, pair_set.pairs
    ):
        change = ds.changes[change_row]
        rows.append(
            {
                "id": f"{change.change_id}|{ds.test_ids[test_col]}",
                "state": {"change": change_text, "test": test_text},
                "question": criterion,
                "options": [dict(YES_OPTION), dict(NO_OPTION)],
            }
        )
    return rows


def _prompts_and_slots(tokenizer, rows: list[dict]) -> tuple[list[str], list[int]]:
    """Render direct-mode chats and resolve the two answer-slot token ids."""
    from semif_phase1.core import direct_messages
    from semif_phase1.direct import _slot_ids

    prompts = [
        tokenizer.apply_chat_template(
            direct_messages(row), tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in rows
    ]
    return prompts, _slot_ids(tokenizer, 2)


def score_batch_direct(model, tokenizer, prompts: list[str], slots: list[int], max_tokens: int):
    """One padded forward pass; returns yes-vs-no log-odds and timing."""
    import torch

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
    kwargs = dict(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
        return_dict=True,
    )
    import inspect

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

    selected = logits[:, slots]
    log_odds = (selected[:, 0] - selected[:, 1]).tolist()
    timing = {
        "pairs": len(prompts),
        "forward_seconds": elapsed,
        "padded_tokens": int(input_ids.numel()),
        "max_prompt_tokens": width,
        "pairs_per_second": len(prompts) / elapsed if elapsed > 0 else float("nan"),
    }
    return log_odds, timing


def score_to_cache_direct(
    model,
    tokenizer,
    ds: dataset.Dataset,
    pair_set: PairSet,
    out_path: Path,
    batch_size: int = 8,
    max_tokens: int | None = None,
    criterion: str = CRITERION,
    resume: bool = True,
) -> dict:
    """Score pairs with direct mode, flushing per batch so the run is resumable.

    Writes the same record shape as ``rts.semif_runner.score_to_cache`` so
    ``rts.semif.load_scores`` consumes either arm without changes.
    """
    import torch

    max_tokens = max_tokens or config.SEMIF_MAX_TOKENS
    done = load_done_keys(out_path) if resume else set()
    keep = [i for i, (r, c) in enumerate(pair_set.index) if (r, c) not in done]
    if not keep:
        print(f"[direct] {out_path} already complete ({len(done)} pairs)")
        return {"pairs": 0, "already_done": len(done), "complete": True}

    rows = build_rows(ds, pair_set, criterion=criterion)
    prompts, slots = _prompts_and_slots(tokenizer, rows)
    print(f"[direct] mean prompt tokens ~{np.mean([len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts[:20]]):.0f} (sample of 20)", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = out_path.open("a")
    total_tokens = 0
    total_seconds = 0.0
    started = time.perf_counter()
    scores: list[float] = []

    for start in range(0, len(keep), batch_size):
        idx = keep[start : start + batch_size]
        chunk_prompts = [prompts[i] for i in idx]
        chunk_scores, timing = score_batch_direct(
            model, tokenizer, chunk_prompts, slots, max_tokens
        )
        total_tokens += timing["padded_tokens"]
        total_seconds += timing["forward_seconds"]
        for i, score in zip(idx, chunk_scores):
            row, col = pair_set.index[i]
            scores.append(score)
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
        if (start // batch_size) % 20 == 0:
            done_n = min(start + batch_size, len(keep))
            rate = done_n / max(time.perf_counter() - started, 1e-9)
            print(f"  [{done_n}/{len(keep)}] {rate:.1f} pairs/s", flush=True)
        if torch.cuda.is_available() and (start // batch_size) % 50 == 0:
            torch.cuda.empty_cache()
    handle.close()

    wall = time.perf_counter() - started
    stats = {
        "arm": "direct_pairwise",
        "pairs": len(keep),
        "batch_size": batch_size,
        "wall_seconds": wall,
        "forward_seconds": total_seconds,
        "mean_padded_tokens_per_pair": total_tokens / max(len(keep), 1),
        "pairs_per_second": len(keep) / max(wall, 1e-9),
        "output": str(out_path),
    }
    print(f"[direct] wrote {len(scores)} pairs to {out_path}")
    print(f"[direct] {stats}")
    return stats


def score_arm(
    batch_size: int = 8,
    out_path: Path | None = None,
    candidates_mode: str = "covered",
    starved_max_failures: int | None = None,
    max_tokens: int | None = None,
    criterion: str = CRITERION,
    seed: int = config.SEED,
    limit: int | None = None,
    device: str = "auto",
) -> dict:
    ds = dataset.build(seed=seed)
    candidates = dataset.candidate_mask(ds, candidates_mode)
    rows = ds.test_idx
    if starved_max_failures is not None:
        mask = dataset.starved_mask(ds, max_failures=starved_max_failures)
        rows = ds.test_idx[mask[ds.test_idx]]
    if limit is not None:
        rows = rows[:limit]
    pair_set = build_pair_set(ds, rows, candidates)
    if out_path is None:
        suffix = "starved" if starved_max_failures is not None else "heldout"
        out_path = config.ARTIFACTS / f"semif_direct_{suffix}_{candidates_mode}.jsonl"

    print(f"arm             : direct_pairwise (Qwen3.5-4B)")
    print(f"candidates      : {candidates_mode}")
    print(f"starved <=      : {starved_max_failures}")
    print(f"changes         : {len(rows)}")
    print(f"pairs           : {len(pair_set.pairs):,}")
    print(f"output          : {out_path}")
    model, tokenizer, metadata = load_direct_model(device=device)
    print(f"loaded          : {metadata['device']} {metadata['dtype']}", flush=True)
    return score_to_cache_direct(
        model, tokenizer, ds, pair_set, out_path,
        batch_size=batch_size, max_tokens=max_tokens, criterion=criterion,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="covered", choices=["covered", "full"])
    parser.add_argument("--starved", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N changes (smoke test)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    score_arm(
        batch_size=args.batch_size,
        out_path=args.out,
        candidates_mode=args.candidates,
        starved_max_failures=args.starved,
        max_tokens=args.max_tokens,
        limit=args.limit,
        device=args.device,
    )
