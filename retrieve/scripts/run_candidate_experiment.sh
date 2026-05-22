#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 EXP_LABEL RETRIEVE_CONFIG [PROMPT_MODE] [THRESHOLD]" >&2
  exit 2
fi

EXP_LABEL="$1"
RETRIEVE_CONFIG="$2"
PROMPT_MODE="${3:-tree_10_mcq}"
THRESHOLD="${4:-0.2}"
MAX_SAMPLES="${MAX_SAMPLES:-463}"
NUM_TREES="${NUM_TREES:-15}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RETRIEVE_DIR="$PROJECT_ROOT/retrieve"
REASON_DIR="$PROJECT_ROOT/reason"
STAMP="$(date +%m%d_%H%M)"
LOG_DIR="$RETRIEVE_DIR/logs"
mkdir -p "$LOG_DIR" "$REASON_DIR/results" "$REASON_DIR/results/diagnostics"

if [[ "$CUDA_VISIBLE_DEVICES" == "0" ]]; then
  echo "Refusing to run on GPU0. Set CUDA_VISIBLE_DEVICES=1." >&2
  exit 3
fi

cd "$RETRIEVE_DIR"
if [[ ! -f "$RETRIEVE_CONFIG" ]]; then
  echo "Retrieve config not found: $RETRIEVE_CONFIG" >&2
  exit 4
fi

source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate retriever

SAVE_PREFIX="$(
python - "$RETRIEVE_CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["train"]["save_prefix"])
PY
)"

TRAIN_LOG="$LOG_DIR/train_${EXP_LABEL}_${STAMP}.log"
INFER_LOG="$LOG_DIR/infer_${EXP_LABEL}_${STAMP}.log"
REASON_LOG="$LOG_DIR/reason_${EXP_LABEL}_${STAMP}.log"

echo "===== Train: $EXP_LABEL ====="
echo "config=$RETRIEVE_CONFIG"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
python train_candidate_tree.py --config "$RETRIEVE_CONFIG" 2>&1 | tee "$TRAIN_LOG"

CPT_DIR="$(
find . -maxdepth 1 -name "${SAVE_PREFIX}_*" -type d -printf '%T@ %p\n' \
  | sort -nr | head -1 | cut -d' ' -f2-
)"
if [[ -z "$CPT_DIR" || ! -f "$CPT_DIR/cpt.pth" ]]; then
  echo "Checkpoint not found for prefix: $SAVE_PREFIX" >&2
  exit 5
fi

echo "checkpoint_dir=$CPT_DIR"
echo "===== Inference: $EXP_LABEL ====="
python inference.py -p "$CPT_DIR/cpt.pth" --split val --num_trees "$NUM_TREES" 2>&1 | tee "$INFER_LOG"

TREE_FILE="$RETRIEVE_DIR/${CPT_DIR#./}/tree_retrieval_result_val.jsonl"
if [[ ! -f "$TREE_FILE" ]]; then
  echo "Tree result not found: $TREE_FILE" >&2
  exit 6
fi

cd "$REASON_DIR"
conda activate reasoner
REASON_CONFIG="${REASON_CONFIG:-configs/medmcqa_candidate_fast_api.yaml}"
if [[ ! -f "$REASON_CONFIG" ]]; then
  echo "Reason config not found: $REASON_CONFIG" >&2
  echo "Set REASON_CONFIG to an existing local API config." >&2
  exit 7
fi

MARKER="$(mktemp)"
echo "===== Reason QA: $EXP_LABEL ====="
python main.py \
  --config "$REASON_CONFIG" \
  --prompt_mode "$PROMPT_MODE" \
  -p "$TREE_FILE" \
  --thres "$THRESHOLD" \
  --max_samples "$MAX_SAMPLES" \
  --skip_eval \
  2>&1 | tee "$REASON_LOG"

PRED_DIR="results/KGQA/medmcqa_primekg/SubgraphRAG/gpt-4.1-mini"
PRED="$(
find "$PRED_DIR" -type f -newer "$MARKER" \
  -name "*${PROMPT_MODE}*val-first_${MAX_SAMPLES}-predictions.jsonl" \
  -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-
)"
rm -f "$MARKER"

if [[ -z "$PRED" || ! -f "$PRED" ]]; then
  echo "Prediction file not found after reason run." >&2
  exit 8
fi

THRESHOLD_TAG="${THRESHOLD//./}"
SUMMARY_FILE="results/medmcqa_${EXP_LABEL}_${PROMPT_MODE}_thres${THRESHOLD_TAG}_val${MAX_SAMPLES}_summary.txt"
python metrics/evaluate_medmcqa_accuracy.py "$PRED" \
  --split val \
  --label-base zero \
  --show-wrong 0 \
  2>&1 | tee "$SUMMARY_FILE"

python metrics/diagnose_tree_fix_harm.py \
  --project-root "$PROJECT_ROOT" \
  --name "$EXP_LABEL $PROMPT_MODE thres=$THRESHOLD" \
  --tree-pred "$REASON_DIR/$PRED" \
  --tree-data "$TREE_FILE" \
  --out-prefix "${EXP_LABEL}_${PROMPT_MODE}_thres${THRESHOLD_TAG}_fix_harm" \
  2>&1 | tee -a "$REASON_LOG"

echo "experiment=$EXP_LABEL"
echo "checkpoint=$CPT_DIR/cpt.pth"
echo "tree_file=$TREE_FILE"
echo "prediction=$REASON_DIR/$PRED"
echo "summary=$REASON_DIR/$SUMMARY_FILE"
echo "train_log=$TRAIN_LOG"
echo "infer_log=$INFER_LOG"
echo "reason_log=$REASON_LOG"
