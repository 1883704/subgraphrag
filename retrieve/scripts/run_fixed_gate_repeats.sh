#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-/home/ubuntu/project/SubgraphRAG-main}"
REPEATS="${REPEATS:-3}"
MAX_SAMPLES="${MAX_SAMPLES:-463}"
SPLIT="${SPLIT:-val}"
THRESHOLD="${THRESHOLD:-0.2}"
MODEL_NAME="${MODEL_NAME:-gpt-4.1-mini}"
REASON_CONFIG="${REASON_CONFIG:-configs/medmcqa_candidate_fast_api.yaml}"
FAST_TREE_FILE="${FAST_TREE_FILE:-$BASE_DIR/retrieve/medmcqa_primekg_candidate_tree_fast_May19-15-38-51/tree_retrieval_result_val.jsonl}"
PRIMARY_GATE_STRATEGY="${PRIMARY_GATE_STRATEGY:-tree_top_score}"
PRIMARY_GATE_THRESHOLD="${PRIMARY_GATE_THRESHOLD:-0.625}"
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
RUN_TAG="fixed_gate_repeats_${STAMP}"
LOG_DIR="$BASE_DIR/retrieve/logs"
RESULT_DIR="$BASE_DIR/reason/results"
RUN_DIR="$RESULT_DIR/gating/$RUN_TAG"
mkdir -p "$LOG_DIR" "$RESULT_DIR" "$RUN_DIR"

run_reason() {
  local repeat="$1"
  local label="$2"
  local prompt_mode="$3"
  local tree_file="${4:-}"
  local thres="${5:-0.0}"
  local repeat_dir="$RUN_DIR/repeat_${repeat}"
  local log_file="$LOG_DIR/reason_${RUN_TAG}_r${repeat}_${label}.log"
  local marker
  mkdir -p "$repeat_dir"
  marker="$(mktemp)"

  cd "$BASE_DIR/reason"
  conda activate reasoner

  echo "===== Repeat $repeat Reason: $label ====="
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
      --output_tag "$RUN_TAG-r${repeat}-${label}" \
      --skip_eval \
      2>&1 | tee "$log_file"
  else
    python main.py \
      --config "$REASON_CONFIG" \
      --prompt_mode "$prompt_mode" \
      --thres "$thres" \
      --max_samples "$MAX_SAMPLES" \
      --model_name "$MODEL_NAME" \
      --output_tag "$RUN_TAG-r${repeat}-${label}" \
      --skip_eval \
      2>&1 | tee "$log_file"
  fi

  local pred_dir="results/KGQA/medmcqa_primekg/SubgraphRAG/$MODEL_NAME"
  local pred
  pred="$(
    find "$pred_dir" -type f -newer "$marker" \
      -name "*${prompt_mode}*${RUN_TAG}-r${repeat}-${label}*predictions.jsonl" \
      -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-
  )"
  rm -f "$marker"

  if [[ -z "$pred" || ! -f "$pred" ]]; then
    echo "Prediction file not found for repeat=$repeat label=$label" >&2
    exit 5
  fi

  local summary="$RESULT_DIR/medmcqa_${RUN_TAG}_r${repeat}_${label}_summary.txt"
  python metrics/evaluate_medmcqa_accuracy.py "$pred" \
    --split "$SPLIT" \
    --label-base zero \
    --show-wrong 0 \
    2>&1 | tee "$summary"

  local abs_pred="$BASE_DIR/reason/$pred"
  echo "$abs_pred" > "$repeat_dir/${label}.pred_path"
  echo "$summary" > "$repeat_dir/${label}.summary_path"
}

echo "============================================================"
echo "Fixed-gate repeat experiment started at $(date)"
echo "BASE_DIR=$BASE_DIR"
echo "RUN_TAG=$RUN_TAG"
echo "REPEATS=$REPEATS"
echo "PRIMARY_GATE=$PRIMARY_GATE_STRATEGY:$PRIMARY_GATE_THRESHOLD"
echo "FAST_TREE_FILE=$FAST_TREE_FILE"
echo "============================================================"

for repeat in $(seq 1 "$REPEATS"); do
  repeat_dir="$RUN_DIR/repeat_${repeat}"
  mkdir -p "$repeat_dir"

  run_reason "$repeat" "noevi" "noevi_mcq" "" "0.0"
  run_reason "$repeat" "tree10_standard" "tree_10_mcq" "$FAST_TREE_FILE" "$THRESHOLD"

  cd "$BASE_DIR/reason"
  conda activate reasoner

  NOEVI_PRED="$(cat "$repeat_dir/noevi.pred_path")"
  STANDARD_PRED="$(cat "$repeat_dir/tree10_standard.pred_path")"

  python metrics/evidence_gate_ablation.py \
    --name "${RUN_TAG}_r${repeat}_standard_fixed" \
    --noevi-pred "$NOEVI_PRED" \
    --tree-pred "$STANDARD_PRED" \
    --tree-data "$FAST_TREE_FILE" \
    --out-dir "$repeat_dir" \
    --fixed-gate "${PRIMARY_GATE_STRATEGY}:${PRIMARY_GATE_THRESHOLD}" \
    --fixed-gate tree_top_score:0.60 \
    --fixed-gate tree_top_score:0.625 \
    --fixed-gate candidate_top_score:0.64 \
    --fixed-gate candidate_margin_no_generic_q:0.08 \
    2>&1 | tee "$repeat_dir/gate_ablation.log"

  PRIMARY_PRED="$(
  python3 - "$repeat_dir/${RUN_TAG}_r${repeat}_standard_fixed_gate_ablation.csv" "$PRIMARY_GATE_STRATEGY" "$PRIMARY_GATE_THRESHOLD" <<'PY'
import csv
import sys

csv_path, strategy, threshold = sys.argv[1], sys.argv[2], float(sys.argv[3])
with open(csv_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
for row in rows:
    if row.get("group") != "fixed_gate":
        continue
    if row.get("strategy") == strategy and abs(float(row.get("threshold") or 0) - threshold) < 1e-9:
        print(row["routed_pred_file"])
        break
else:
    raise SystemExit("primary fixed gate prediction not found")
PY
  )"

  python metrics/evaluate_medmcqa_accuracy.py "$PRIMARY_PRED" \
    --split "$SPLIT" \
    --label-base zero \
    --show-wrong 0 \
    2>&1 | tee "$RESULT_DIR/medmcqa_${RUN_TAG}_r${repeat}_primary_fixed_summary.txt"

  python metrics/diagnose_tree_fix_harm.py \
    --project-root "$BASE_DIR" \
    --name "${RUN_TAG}_r${repeat}_primary_fixed" \
    --tree-pred "$PRIMARY_PRED" \
    --tree-data "$FAST_TREE_FILE" \
    --out-prefix "${RUN_TAG}_r${repeat}_primary_fixed_fix_harm" \
    2>&1 | tee "$repeat_dir/primary_fixed_fix_harm.log"
done

cd "$BASE_DIR/reason"
python metrics/summarize_fixed_gate_repeats.py \
  --run-dir "$RUN_DIR" \
  --primary-strategy "$PRIMARY_GATE_STRATEGY" \
  --primary-threshold "$PRIMARY_GATE_THRESHOLD" \
  2>&1 | tee "$RUN_DIR/repeat_summary.log"

cd "$BASE_DIR"
bash retrieve/scripts/show_latest_results.sh > "$RUN_DIR/experiment_report_snapshot.txt"

cat > "$RUN_DIR/MANIFEST.txt" <<EOF
run_tag=$RUN_TAG
repeats=$REPEATS
primary_gate=$PRIMARY_GATE_STRATEGY:$PRIMARY_GATE_THRESHOLD
fast_tree_file=$FAST_TREE_FILE
run_dir=$RUN_DIR
repeat_summary=$RUN_DIR/repeat_summary.md
latest_report=$BASE_DIR/reason/results/experiment_report_latest.md
EOF

echo "============================================================"
echo "Fixed-gate repeat experiment completed at $(date)"
cat "$RUN_DIR/MANIFEST.txt"
echo "============================================================"

