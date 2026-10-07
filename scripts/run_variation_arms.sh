#!/usr/bin/env bash
# SemIf variation arms that need the GPU, in priority order.
#
# Every arm writes a resumable JSONL score cache into artifacts/, so re-running the
# script after an interruption continues instead of restarting. Run one arm at a
# time: the GPU is a single 17 GB card holding a 4B model in BF16, so two arms
# cannot share it.
#
#   ./scripts/run_variation_arms.sh p2      # instruction sweep (4 wordings)
#   ./scripts/run_variation_arms.sh p2-orientation
#   ./scripts/run_variation_arms.sh p1      # pairwise direct mode
#   ./scripts/run_variation_arms.sh cold_start5-full
#   ./scripts/run_variation_arms.sh p5b     # reduced-window SemIf for the P5 column
set -euo pipefail

cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/home/noaha/hf_cache}"
PY="${PY:-/home/noaha/graphrnn_env/bin/python}"
BATCH="${BATCH:-8}"
mkdir -p artifacts/logs

run_semif() {  # name, args...
  local name="$1"; shift
  echo "=== arm: $name ==="
  "$PY" -u -m rts.model.semif_runner "$@" > "artifacts/logs/${name}.log" 2>&1
  echo "=== arm done: $name ""$(tail -1 "artifacts/logs/${name}.log")"
}

case "${1:-}" in
  p2)
    for variant in execution fault retrieval terse; do
      run_semif "instr_${variant}" --heldout --max-failures 5 \
        --instruction "$variant" --batch-size "$BATCH" \
        --out "artifacts/semif_scores_instr_${variant}_cold_start5.jsonl"
    done
    ;;
  p2-orientation)
    # The Query/Document choice was a docs/plan.md open question and was only ever
    # compared at n=10. This re-tests it at adequate n on the cold-start subset.
    run_semif "orientation_test_query" --heldout --max-failures 5 \
      --orientation test_query --batch-size "$BATCH" \
      --out "artifacts/semif_scores_testquery_cold_start5.jsonl"
    ;;
  p1)
    # Direct mode is ~1.3 pairs/s (Qwen3.5-4B falls back to reference kernels), so the
    # full held-out grid is a ~16 h job. The cold-start subset is the affordable condition.
    "$PY" -u -m rts.model.direct_runner --max-failures 2 --batch-size "$BATCH" \
      --out "artifacts/semif_direct_cold_start2_coverage_restricted.jsonl" \
      > artifacts/logs/direct_cold_start2.log 2>&1
    echo "=== arm done: direct_cold_start2"
    ;;
  p1-full)
    # ~72k pairs, ~16 h at the measured direct-mode throughput. Listed as outstanding.
    "$PY" -u -m rts.model.direct_runner --batch-size "$BATCH" \
      --out "artifacts/semif_direct_heldout_covered.jsonl" \
      > artifacts/logs/direct_heldout.log 2>&1
    echo "=== arm done: direct_heldout"
    ;;
  p3-arm)
    echo "p3 (embeddings) runs on CPU: python -m rts.model.embed --device cpu" >&2
    exit 2
    ;;
  cold_start5-full)
    # Strengthens the one positive result: the failures<=2 full-suite arm has only
    # 43 faults, so its intervals are wide. 141 faults at ~95 min is the decision
    # grade version.
    run_semif "cold_start5_full" --heldout --candidate-policy full --max-failures 5 \
      --out "artifacts/semif_scores_cold_start5_full.jsonl" --batch-size "$BATCH"
    ;;
  p5b)
    # A SemIf-scored prefix of the training window, so the P5 "add a column"
    # variant can train on rows that actually have the feature.
    run_semif "p5b_train_window" --train-prefix 400 --batch-size "$BATCH" \
      --out "artifacts/semif_scores_trainprefix400_covered.jsonl"
    ;;
  *)
    echo "usage: $0 {p2|p2-orientation|p1|p1-full|cold_start5-full|p5b}" >&2
    exit 2
    ;;
esac
