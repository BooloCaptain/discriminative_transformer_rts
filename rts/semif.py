"""SemIf + Qwen reranker adapter: pair construction, cost estimation, score cache.

Pinned configuration (see implementation.md)
--------------------------------------------
* Mode ``reranker``, not ``direct``. Direct mode softmaxes over all options in a
  single prompt, so a test's score depends on which other tests were in that
  prompt and is capped at 2-16 options -- unusable for ranking a suite on one
  scale. Reranker mode scores each (change, test) pair independently.
* Model ``Qwen/Qwen3-Reranker-4B`` at a pinned revision.
* Raw log-odds, not the repo's normalized value.

Comparability trick
-------------------
SemIf requires 2-16 options per row and its readout is a softmax over those
options. To make scores comparable *across* rows we hold the option set constant:
every row offers exactly ``yes`` and ``no`` against the same question, and only
``state`` (change + test) varies. Because the softmax denominator is then the same
for every row, ``P(yes)`` is on one global scale.

Cost
----
The full grid is ``n_changes x n_tests`` pairs and is not affordable: at the
measured 1.86 decisions/s, 2651 x 1187 = 3.1M pairs is roughly 19 days. Scoring
can be restricted to the covered candidate set (mean ~155 tests per change)
without losing any fault, because every killing test covers the mutated function.
Use ``estimate_cost`` before committing.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np

from . import artifacts, config, dataset
# Measured on a 3090 with no prefix reuse (prefix reuse is a direct-mode feature).
DECISIONS_PER_SECOND = 1.86

QUESTION = (
    "Does this test exercise the code that changed, such that the change could "
    "make this test fail?"
)
OPTIONS = [
    {"id": "yes", "description": "Yes, this test is affected by the change."},
    {"id": "no", "description": "No, this test is not affected by the change."},
]


def build_state(change_text: str, test_text: str) -> str:
    """Render one (change, test) pair as the reranker's document."""
    return (
        "Changed code:\n"
        f"{change_text.strip()}\n\n"
        "Test code:\n"
        f"{test_text.strip()}\n"
    )


def build_pairs(
    ds: dataset.Dataset,
    change_rows: np.ndarray | None = None,
    candidates: np.ndarray | None = None,
    shuffle: bool = False,
    seed: int = config.SEED,
) -> tuple[list[dict], list[tuple[int, int]]]:
    """Build SemIf input rows.

    Returns ``(rows, index)`` where ``index[i] = (change_row, test_col)`` gives the
    matrix position each row's score belongs to.
    """
    from . import features, source

    if change_rows is None:
        change_rows = np.arange(ds.n_changes)
    if candidates is None:
        candidates = dataset.candidate_mask(ds, "covered")

    infos = source.load_all(ds.test_ids)
    texts = [features.change_query_text(c) for c in ds.changes]
    if shuffle:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(texts))
        texts = [texts[i] for i in perm]

    rows: list[dict] = []
    index: list[tuple[int, int]] = []
    for r in change_rows:
        row = int(r)
        for j in np.flatnonzero(candidates[row]):
            j = int(j)
            info = infos.get(ds.test_ids[j])
            test_text = info.source if info else ""
            rows.append(
                {
                    "id": f"{ds.changes[row].change_id}|{ds.test_ids[j]}",
                    "state": build_state(texts[row], test_text),
                    "question": QUESTION,
                    "options": OPTIONS,
                }
            )
            index.append((row, j))
    return rows, index


def estimate_cost(n_pairs: int) -> dict:
    seconds = n_pairs / DECISIONS_PER_SECOND
    return {
        "pairs": n_pairs,
        "seconds": seconds,
        "hours": seconds / 3600,
        "days": seconds / 86400,
    }


def write_pairs(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def score_command(input_path: Path, output_path: Path) -> list[str]:
    """The exact pinned ``semif-score`` invocation."""
    return [
        "semif-score",
        "--mode",
        "reranker",
        "--model",
        config.SEMIF_MODEL,
        "--revision",
        config.SEMIF_MODEL_REVISION,
        "--max-tokens",
        str(config.SEMIF_MAX_TOKENS),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]


def run_scoring(input_path: Path, output_path: Path) -> None:
    """Invoke SemIf. Requires the checkout on PATH and the checkpoint available."""
    cmd = score_command(input_path, output_path)
    print("[semif] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_semif_output(output_path: Path, index: list[tuple[int, int]]) -> list[dict]:
    """Convert SemIf's per-row probabilities into cached score records.

    Score = raw log-odds of ``yes`` over ``no``, so it stays on one scale across
    rows and can be ranked globally.
    """
    records: list[dict] = []
    with output_path.open() as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            probs = payload.get("probabilities") or []
            option_ids = payload.get("option_ids") or [o["id"] for o in OPTIONS]
            by_id = dict(zip(option_ids, probs))
            p_yes = float(by_id.get("yes", 0.0))
            p_no = float(by_id.get("no", 0.0))
            eps = 1e-9
            score = math.log(max(p_yes, eps)) - math.log(max(p_no, eps))
            change_row, test_col = index[i]
            records.append(
                {
                    "change_id": payload["id"].split("|", 1)[0],
                    "test_nodeid": payload["id"].split("|", 1)[1],
                    "change_row": change_row,
                    "test_col": test_col,
                    "p_yes": p_yes,
                    "score": score,
                }
            )
    return records


def write_scores(records: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return path


def load_scores(path: Path, ds: dataset.Dataset) -> np.ndarray:
    """Load cached scores into a ``[n_changes, n_tests]`` matrix.

    Unscored pairs get a very low score so they sort last. Restricting to the
    covered candidate set cannot lose a fault, since every killing test covers the
    mutated function.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"SemIf score cache not found: {path}\n"
            "Populate it with:\n"
            "  python -m rts.semif --build        # writes pairs + prints cost estimate\n"
            "  python -m rts.semif --score        # runs semif-score (needs checkout + checkpoint)\n"
            "See implementation.md for the pinned configuration."
        )

    out = np.full((ds.n_changes, ds.n_tests), -1e9, dtype=np.float32)
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            row = col = None
            # Prefer the stable identifiers over the row/col indices. The indices are only
            # valid for the exact dataset the cache was written against, so a cache silently
            # mis-maps if the candidate pool changes (e.g. under full-suite labels, where the
            # pool grows from 1187 to 1189 and every later column shifts).
            if "change_id" in record and "test_nodeid" in record:
                row = ds.change_index.get(record["change_id"])
                nodeid = record["test_nodeid"]
                if config.LABELS == "full":
                    nodeid = artifacts.canonical_nodeid(nodeid)
                col = ds.test_index.get(nodeid)
            if row is None or col is None:
                row = record.get("change_row")
                col = record.get("test_col")
            if row is None or col is None:
                continue
            out[int(row), int(col)] = record["score"]
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="write pair file and report cost")
    parser.add_argument("--score", action="store_true", help="invoke semif-score on the pair file")
    parser.add_argument("--candidates", default="covered", choices=["covered", "full"])
    parser.add_argument("--max-changes", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true", help="change-shuffle ablation")
    parser.add_argument("--suffix", default="", help="suffix for output filenames")
    args = parser.parse_args()

    ds = dataset.build()
    rows_n = np.arange(ds.n_changes)
    if args.max_changes:
        rows_n = rows_n[: args.max_changes]
    cand = dataset.candidate_mask(ds, args.candidates)

    rows, index = build_pairs(
        ds, change_rows=rows_n, candidates=cand, shuffle=args.shuffle
    )
    cost = estimate_cost(len(rows))
    print(f"candidates      : {args.candidates}")
    print(f"changes         : {len(rows_n)}")
    print(f"pairs           : {cost['pairs']:,}")
    print(f"estimated cost  : {cost['hours']:.1f} h ({cost['days']:.1f} d) at {DECISIONS_PER_SECOND}/s")
    print(f"mean tokens/pair: n/a until scored (report it after a pilot)")

    if args.build or args.score:
        tag = ("shuffled" if args.shuffle else "real") + args.suffix
        pairs_path = config.ARTIFACTS / f"semif_pairs_{tag}.jsonl"
        out_path = config.ARTIFACTS / f"semif_raw_{tag}.jsonl"
        write_pairs(rows, pairs_path)
        print(f"[semif] wrote {pairs_path}")
        (config.ARTIFACTS / f"semif_index_{tag}.json").write_text(json.dumps(index))
        if args.score:
            run_scoring(pairs_path, out_path)
            records = parse_semif_output(out_path, index)
            scores_path = config.ARTIFACTS / f"semif_scores_{tag}.jsonl"
            write_scores(records, scores_path)
            print(f"[semif] wrote {scores_path}")
