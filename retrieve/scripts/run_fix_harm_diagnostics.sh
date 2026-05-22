#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/reason/results/diagnostics"
mkdir -p "$LOG_DIR"

cd "$PROJECT_ROOT/reason"
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate reasoner

STAMP="$(date +%m%d_%H%M)"
LOG_FILE="$LOG_DIR/fix_harm_diagnostics_${STAMP}.log"
REFRESH_BASELINE_REASON="${REFRESH_BASELINE_REASON:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-463}"
BASELINE_TREE="${BASELINE_TREE:-$PROJECT_ROOT/retrieve/medmcqa_primekg_candidate_tree_fast_May19-15-38-51/tree_retrieval_result_val_top15.jsonl}"
if [[ ! -f "$BASELINE_TREE" ]]; then
  BASELINE_TREE="$PROJECT_ROOT/retrieve/medmcqa_primekg_candidate_tree_fast_May19-15-38-51/tree_retrieval_result_val.jsonl"
fi

TREE_PRED_ARGS=()
if [[ "$REFRESH_BASELINE_REASON" == "1" ]]; then
  REASON_CONFIG="${REASON_CONFIG:-configs/medmcqa_candidate_fast_api.yaml}"
  if [[ ! -f "$REASON_CONFIG" ]]; then
    echo "Reason config not found: $REASON_CONFIG" >&2
    echo "Set REASON_CONFIG to an existing local API config." >&2
    exit 7
  fi
  if [[ ! -f "$BASELINE_TREE" ]]; then
    echo "Baseline tree file not found: $BASELINE_TREE" >&2
    exit 8
  fi

  MARKER="$(mktemp)"
  python main.py \
    --config "$REASON_CONFIG" \
    --prompt_mode tree_10_mcq \
    -p "$BASELINE_TREE" \
    --thres 0.2 \
    --max_samples "$MAX_SAMPLES" \
    --skip_eval

  PRED_DIR="results/KGQA/medmcqa_primekg/SubgraphRAG/gpt-4.1-mini"
  TREE_PRED="$(
    find "$PRED_DIR" -type f -newer "$MARKER" \
      -name "*tree_10_mcq*val-first_${MAX_SAMPLES}-predictions.jsonl" \
      -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-
  )"
  rm -f "$MARKER"
  if [[ -z "$TREE_PRED" || ! -f "$TREE_PRED" ]]; then
    echo "Refreshed baseline prediction not found." >&2
    exit 9
  fi
  TREE_PRED_ARGS=(--tree-pred "$PROJECT_ROOT/reason/$TREE_PRED" --tree-data "$BASELINE_TREE")
fi

python metrics/diagnose_tree_fix_harm.py \
  --project-root "$PROJECT_ROOT" \
  --name "baseline_fast_tree10_thres0.2" \
  --out-prefix "baseline_fast_tree10_thres02_fix_harm" \
  "${TREE_PRED_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

echo "diagnostics_log=$LOG_FILE"
