"""Complete the full-pool SemIf cache for the 1189-test pool.

The existing caches (`semif_scores_starved5_full.jsonl` and `_extra_full`) cover 141 changes x
1187 tests, because they were scored before node-id canonicalisation and before the three
mutmut-deselected tests were added to the pool. Under ``RTS_LABELS=full`` the pool is 1189, so
those three tests -- which happen to be the main out-of-coverage killers -- have no score.

This scores exactly the missing pairs (141 x 3 = 423) and writes one merged cache with every
pair for those 141 changes at the full 1189 pool, so the ladder needs a single file.

Cost: seconds at ~30 pairs/s. Usage:

    python scripts/complete_semif_full_cache.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from rts import artifacts, config, dataset, features, source  # noqa: E402

SOURCES = [
    config.ARTIFACTS / "semif_scores_starved5_full.jsonl",
    config.ARTIFACTS / "semif_scores_starved5_extra_full.jsonl",
]
OUT = config.ARTIFACTS / "semif_scores_ladder141_full.jsonl"


def main() -> None:
    config.set_labels("full")
    ds = dataset.build()

    existing: dict[tuple[str, str], float] = {}
    for path in SOURCES:
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                existing[(rec["change_id"], artifacts.canonical_nodeid(rec["test_nodeid"]))] = rec["score"]

    change_ids = sorted({cid for cid, _ in existing})
    print(f"[complete] {len(existing):,} cached pairs over {len(change_ids)} changes")

    pool = ds.test_ids
    missing: list[tuple[int, int]] = []
    for cid in change_ids:
        row = ds.change_index.get(cid)
        if row is None:
            continue
        for col, nodeid in enumerate(pool):
            if (cid, nodeid) not in existing:
                missing.append((row, col))

    print(f"[complete] {len(missing):,} missing pairs at the {len(pool)}-test pool")

    merged: list[dict] = []
    for cid in change_ids:
        row = ds.change_index.get(cid)
        if row is None:
            continue
        for col, nodeid in enumerate(pool):
            key = (cid, nodeid)
            score = existing.get(key)
            if score is None:
                continue
            merged.append(
                {
                    "change_row": row,
                    "test_col": col,
                    "change_id": cid,
                    "test_nodeid": nodeid,
                    "score": score,
                }
            )

    if missing:
        from rts import semif_runner

        model, tokenizer, _meta = semif_runner.load_model(device="auto")
        infos = source.load_all(pool)
        change_texts = [features.change_query_text(c) for c in ds.changes]
        pairs = [
            (change_texts[row], infos[pool[col]].source if pool[col] in infos else "")
            for row, col in missing
        ]
        scores, stats = semif_runner.score_pairs(
            model, tokenizer, pairs, batch_size=8, progress_every=1
        )
        print(f"[complete] scored {stats['pairs']} pairs at {stats['pairs_per_second']:.1f} pairs/s")
        for (row, col), score in zip(missing, scores):
            merged.append(
                {
                    "change_row": row,
                    "test_col": col,
                    "change_id": ds.changes[row].change_id,
                    "test_nodeid": pool[col],
                    "score": float(score),
                }
            )

    OUT.write_text("".join(json.dumps(r) + "\n" for r in merged))
    print(f"[complete] wrote {len(merged):,} pairs to {OUT.relative_to(WORKSPACE)}")


if __name__ == "__main__":
    main()
