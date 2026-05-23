#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-/home/ubuntu/project/SubgraphRAG-main}"
MAX_SAMPLES="${MAX_SAMPLES:-463}"
SPLIT="${SPLIT:-val}"
THRESHOLD="${THRESHOLD:-0.2}"
MODEL_NAME="${MODEL_NAME:-gpt-4.1-mini}"
REASON_CONFIG="${REASON_CONFIG:-configs/medmcqa_candidate_fast_api.yaml}"
FAST_TREE_FILE="${FAST_TREE_FILE:-$BASE_DIR/retrieve/medmcqa_primekg_candidate_tree_fast_May19-15-38-51/tree_retrieval_result_val.jsonl}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES

if [[ "$CUDA_VISIBLE_DEVICES" == "0" ]]; then
  echo "Refusing to run on GPU0. Set CUDA_VISIBLE_DEVICES=1." >&2
  exit 3
fi

if [[ ! -f "$FAST_TREE_FILE" ]]; then
  echo "FAST_TREE_FILE not found: $FAST_TREE_FILE" >&2
  exit 4
fi

cd "$BASE_DIR"
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh

STAMP="$(date +%m%d_%H%M)"
RUN_TAG="gate_${STAMP}"
LOG_DIR="$BASE_DIR/retrieve/logs"
RESULT_DIR="$BASE_DIR/reason/results"
GATE_DIR="$RESULT_DIR/gating/$RUN_TAG"
mkdir -p "$LOG_DIR" "$RESULT_DIR" "$GATE_DIR"

run_reason() {
  local label="$1"
  local prompt_mode="$2"
  local tree_file="${3:-}"
  local thres="${4:-0.0}"
  local log_file="$LOG_DIR/reason_${label}_${STAMP}.log"
  local marker
  marker="$(mktemp)"

  cd "$BASE_DIR/reason"
  conda activate reasoner

  echo "===== Reason: $label ====="
  echo "prompt_mode=$prompt_mode"
  echo "tree_file=$tree_file"
  echo "thres=$thres"

  if [[ -n "$tree_file" ]]; then
    python main.py \
      --config "$REASON_CONFIG" \
      --prompt_mode "$prompt_mode" \
      -p "$tree_file" \
      --thres "$thres" \
      --max_samples "$MAX_SAMPLES" \
      --model_name "$MODEL_NAME" \
      --output_tag "$RUN_TAG-$label" \
      --skip_eval \
      2>&1 | tee "$log_file"
  else
    python main.py \
      --config "$REASON_CONFIG" \
      --prompt_mode "$prompt_mode" \
      --thres "$thres" \
      --max_samples "$MAX_SAMPLES" \
      --model_name "$MODEL_NAME" \
      --output_tag "$RUN_TAG-$label" \
      --skip_eval \
      2>&1 | tee "$log_file"
  fi

  local pred_dir="results/KGQA/medmcqa_primekg/SubgraphRAG/$MODEL_NAME"
  local pred
  pred="$(
    find "$pred_dir" -type f -newer "$marker" \
      -name "*${prompt_mode}*${RUN_TAG}-${label}*predictions.jsonl" \
      -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-
  )"
  rm -f "$marker"

  if [[ -z "$pred" || ! -f "$pred" ]]; then
    echo "Prediction file not found for $label" >&2
    exit 5
  fi

  local summary="$RESULT_DIR/medmcqa_${label}_${RUN_TAG}_summary.txt"
  python metrics/evaluate_medmcqa_accuracy.py "$pred" \
    --split "$SPLIT" \
    --label-base zero \
    --show-wrong 0 \
    2>&1 | tee "$summary"

  local abs_pred="$BASE_DIR/reason/$pred"
  echo "$abs_pred" > "$GATE_DIR/${label}.pred_path"
  echo "$summary" > "$GATE_DIR/${label}.summary_path"
}

echo "============================================================"
echo "Gate/prompt experiments started at $(date)"
echo "BASE_DIR=$BASE_DIR"
echo "FAST_TREE_FILE=$FAST_TREE_FILE"
echo "RUN_TAG=$RUN_TAG"
echo "============================================================"

run_reason "noevi" "noevi_mcq" "" "0.0"
run_reason "tree10_standard" "tree_10_mcq" "$FAST_TREE_FILE" "$THRESHOLD"
run_reason "tree10_gate_prompt" "tree_10_gate_mcq" "$FAST_TREE_FILE" "$THRESHOLD"
run_reason "tree10_compare_prompt" "tree_10_compare_mcq" "$FAST_TREE_FILE" "$THRESHOLD"

cd "$BASE_DIR/reason"
conda activate reasoner

NOEVI_PRED="$(cat "$GATE_DIR/noevi.pred_path")"
STANDARD_PRED="$(cat "$GATE_DIR/tree10_standard.pred_path")"
GATE_PROMPT_PRED="$(cat "$GATE_DIR/tree10_gate_prompt.pred_path")"
COMPARE_PROMPT_PRED="$(cat "$GATE_DIR/tree10_compare_prompt.pred_path")"

python metrics/evidence_gate_ablation.py \
  --name "${RUN_TAG}_standard_vs_noevi" \
  --noevi-pred "$NOEVI_PRED" \
  --tree-pred "$STANDARD_PRED" \
  --tree-data "$FAST_TREE_FILE" \
  --out-dir "$GATE_DIR" \
  2>&1 | tee "$GATE_DIR/gate_ablation_standard.log"

python metrics/evidence_gate_ablation.py \
  --name "${RUN_TAG}_gate_prompt_vs_noevi" \
  --noevi-pred "$NOEVI_PRED" \
  --tree-pred "$GATE_PROMPT_PRED" \
  --tree-data "$FAST_TREE_FILE" \
  --out-dir "$GATE_DIR" \
  2>&1 | tee "$GATE_DIR/gate_ablation_gate_prompt.log"

python metrics/evidence_gate_ablation.py \
  --name "${RUN_TAG}_compare_prompt_vs_noevi" \
  --noevi-pred "$NOEVI_PRED" \
  --tree-pred "$COMPARE_PROMPT_PRED" \
  --tree-data "$FAST_TREE_FILE" \
  --out-dir "$GATE_DIR" \
  2>&1 | tee "$GATE_DIR/gate_ablation_compare_prompt.log"

cd "$BASE_DIR"
bash retrieve/scripts/show_latest_results.sh > "$GATE_DIR/experiment_report_snapshot.txt"

cat > "$GATE_DIR/MANIFEST.txt" <<EOF
run_tag=$RUN_TAG
fast_tree_file=$FAST_TREE_FILE
noevi_pred=$NOEVI_PRED
standard_tree_pred=$STANDARD_PRED
gate_prompt_pred=$GATE_PROMPT_PRED
compare_prompt_pred=$COMPARE_PROMPT_PRED
gate_dir=$GATE_DIR
latest_report=$BASE_DIR/reason/results/experiment_report_latest.md
EOF

echo "============================================================"
echo "Gate/prompt experiments completed at $(date)"
cat "$GATE_DIR/MANIFEST.txt"
echo "============================================================"

