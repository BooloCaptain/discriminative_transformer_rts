"""P3: code-specialised embedding baseline.

Why this exists
---------------
BM25 is the lexical floor and the SemIf reranker is the "strong semantic" arm. The
embedding baseline sits strictly between them: it is a *semantic* similarity model
(so it can match paraphrases and renamed identifiers, which bag-of-words cannot)
but it is not a task-trained reranker (so it cannot exploit an instruction or a
criterion). It answers one question cheaply:

* If cosine similarity over code embeddings matches SemIf, a 4B reranker is
  unjustified for this task.
* If it beats BM25, there is a semantic signal worth pursuing with something
  stronger than either.

Method
------
Mean-pooled last hidden state, L2-normalised, cosine similarity between the change
text (the same text BM25 and SemIf receive) and each test's source. No training, no
instruction, no prompt: the same (change, test) pairs as every other selector, on
the same candidate mask, so the comparison is like-for-like.

The change and the test are both *code*, which is the regime where a code
embedding should be at its best -- so this is a favourable test of the semantic
hypothesis, unlike the natural-language reranker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config, dataset, features, source


def load_embedding_model(
    model_name: str | None = None,
    revision: str | None = None,
    device: str = "auto",
):
    """Load the pinned encoder.

    Defaults to CUDA when present, but ``device="cpu"`` is the recommended choice
    while a GPU scoring run is in flight: the encoder is ~110M parameters and the
    whole job is ~1700 short sequences, so CPU costs minutes and avoids contending
    with the reranker for VRAM and SM time.
    """
    import torch
    import transformers

    name = model_name or config.EMBED_MODEL
    rev = revision or config.EMBED_MODEL_REVISION
    tokenizer = transformers.AutoTokenizer.from_pretrained(name, revision=rev)
    model = transformers.AutoModel.from_pretrained(name, revision=rev)
    model.eval()
    target = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    torch_device = torch.device(target)
    model.to(torch_device)
    return model, tokenizer, {"source": name, "revision": rev, "device": str(torch_device)}


def embed_texts(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 16,
    max_length: int = 512,
    progress_every: int = 20,
) -> np.ndarray:
    """Mean-pooled, L2-normalised embeddings for a list of texts."""
    import torch

    device = next(model.parameters()).device
    out: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        chunk = [t if t.strip() else " " for t in texts[start : start + batch_size]]
        encoded = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state.float()
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        out.append(pooled.cpu().numpy())
        if progress_every and (start // batch_size) % progress_every == 0:
            print(f"  [embed {min(start + batch_size, len(texts))}/{len(texts)}]", flush=True)
    return np.vstack(out) if out else np.zeros((0, 1), dtype=np.float32)


def build_scores(
    ds: dataset.Dataset,
    batch_size: int = 16,
    max_length: int = 512,
    model=None,
    tokenizer=None,
    device: str = "auto",
) -> np.ndarray:
    """[n_changes, n_tests] cosine similarity between change text and test source."""
    if model is None:
        model, tokenizer, metadata = load_embedding_model(device=device)
        print(f"[embed] {metadata}", flush=True)
    infos = source.load_all(ds.test_ids)
    test_texts = [infos[t].source if t in infos else "" for t in ds.test_ids]
    change_texts = [features.change_query_text(c) for c in ds.changes]

    print(f"[embed] encoding {len(test_texts)} tests ...", flush=True)
    test_vecs = embed_texts(model, tokenizer, test_texts, batch_size, max_length)
    print(f"[embed] encoding {len(change_texts)} changes ...", flush=True)
    change_vecs = embed_texts(model, tokenizer, change_texts, batch_size, max_length)

    return (change_vecs @ test_vecs.T).astype(np.float32)


def run(batch_size: int = 16, max_length: int = 512, device: str = "auto") -> np.ndarray:
    from . import evaluate

    ds = dataset.build()
    scores = build_scores(ds, batch_size=batch_size, max_length=max_length, device=device)
    results = evaluate.evaluate(
        scores, ds, ds.test_idx, budgets=(0.01, 0.05, 0.1, 0.2),
        n_bootstrap=1000, candidates=dataset.candidate_mask(ds, "covered"),
    )
    print(evaluate.format_table("embed_codebert (covered candidates)", results))
    np.save(config.ARTIFACTS / "embed_scores.npy", scores)
    return scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"],
                        help="CPU is the default so the arm can run alongside GPU scoring")
    args = parser.parse_args()
    run(batch_size=args.batch_size, max_length=args.max_length, device=args.device)
