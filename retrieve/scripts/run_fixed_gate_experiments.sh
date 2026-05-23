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
RUN_TAG="fixed_gate_${STAMP}"
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
echo "Fixed-gate experiment started at $(date)"
echo "BASE_DIR=$BASE_DIR"
echo "FAST_TREE_FILE=$FAST_TREE_FILE"
echo "RUN_TAG=$RUN_TAG"
echo "============================================================"

run_reason "noevi" "noevi_mcq" "" "0.0"
run_reason "tree10_standard" "tree_10_mcq" "$FAST_TREE_FILE" "$THRESHOLD"

cd "$BASE_DIR/reason"
conda activate reasoner

NOEVI_PRED="$(cat "$GATE_DIR/noevi.pred_path")"
STANDARD_PRED="$(cat "$GATE_DIR/tree10_standard.pred_path")"

python metrics/evidence_gate_ablation.py \
  --name "${RUN_TAG}_standard_fixed" \
  --noevi-pred "$NOEVI_PRED" \
  --tree-pred "$STANDARD_PRED" \
  --tree-data "$FAST_TREE_FILE" \
  --out-dir "$GATE_DIR" \
  --fixed-gate tree_top_score:0.55 \
  --fixed-gate tree_top_score:0.5805 \
  --fixed-gate tree_top_score:0.60 \
  --fixed-gate tree_top_score:0.625 \
  --fixed-gate candidate_top_score:0.64 \
  --fixed-gate candidate_top_score:0.7017 \
  --fixed-gate candidate_margin_no_generic_q:0.0588 \
  --fixed-gate candidate_margin_match_top:0.1625 \
  2>&1 | tee "$GATE_DIR/fixed_gate_ablation.log"

BEST_FIXED_PRED="$(
python3 - "$GATE_DIR/${RUN_TAG}_standard_fixed_gate_ablation.csv" <<'PY'
import csv
import sys

path = sys.argv[1]
rows = []
with open(path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("group") == "fixed_gate":
            rows.append(row)

if not rows:
    raise SystemExit("no fixed_gate rows")

rows.sort(
    key=lambda row: (
        float(row.get("accuracy") or 0),
        -float(row.get("tree_used_ratio") or 0),
    ),
    reverse=True,
)
print(rows[0]["routed_pred_file"])
PY
)"

echo "best_fixed_pred=$BEST_FIXED_PRED"

python metrics/evaluate_medmcqa_accuracy.py "$BEST_FIXED_PRED" \
  --split "$SPLIT" \
  --label-base zero \
  --show-wrong 0 \
  2>&1 | tee "$RESULT_DIR/medmcqa_${RUN_TAG}_best_fixed_summary.txt"

python metrics/diagnose_tree_fix_harm.py \
  --project-root "$BASE_DIR" \
  --name "${RUN_TAG}_best_fixed" \
  --tree-pred "$BEST_FIXED_PRED" \
  --tree-data "$FAST_TREE_FILE" \
  --out-prefix "${RUN_TAG}_best_fixed_fix_harm" \
  2>&1 | tee "$GATE_DIR/best_fixed_fix_harm.log"

cd "$BASE_DIR"
bash retrieve/scripts/show_latest_results.sh > "$GATE_DIR/experiment_report_snapshot.txt"

cat > "$GATE_DIR/MANIFEST.txt" <<EOF
run_tag=$RUN_TAG
fast_tree_file=$FAST_TREE_FILE
noevi_pred=$NOEVI_PRED
standard_tree_pred=$STANDARD_PRED
best_fixed_pred=$BEST_FIXED_PRED
gate_dir=$GATE_DIR
latest_report=$BASE_DIR/reason/results/experiment_report_latest.md
EOF

echo "============================================================"
echo "Fixed-gate experiment completed at $(date)"
cat "$GATE_DIR/MANIFEST.txt"
echo "============================================================"

