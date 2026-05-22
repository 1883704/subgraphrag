#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STAMP="$(date +%m%d_%H%M)"
MASTER_LOG="$PROJECT_ROOT/retrieve/logs/overnight_${STAMP}.log"
REPORT="$PROJECT_ROOT/reason/results/overnight_experiment_report_${STAMP}.md"
MARKER="$(mktemp)"
mkdir -p "$PROJECT_ROOT/retrieve/logs" "$PROJECT_ROOT/reason/results"

exec > >(tee "$MASTER_LOG") 2>&1

echo "# Overnight experiment run"
echo "started_at=$(date)"
echo "project_root=$PROJECT_ROOT"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-1}"
echo

cd "$PROJECT_ROOT"
echo "===== Git status ====="
git status --short
echo

echo "===== Baseline fix/harm diagnostics ====="
bash "$SCRIPT_DIR/run_fix_harm_diagnostics.sh"
echo

echo "===== Experiment 1: clean filter ====="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" bash "$SCRIPT_DIR/run_clean_filter_experiment.sh"
echo

echo "===== Experiment 2: semantic support ====="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" bash "$SCRIPT_DIR/run_semantic_support_experiment.sh"
echo

{
  echo "# Overnight Experiment Report"
  echo
  echo "- Started marker: $(date -r "$MARKER" '+%Y-%m-%d %H:%M:%S')"
  echo "- Finished: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "- Master log: $MASTER_LOG"
  echo "- Current best before this run: fast scorer + tree_10_mcq + thres=0.2 = 361/463 = 0.7797"
  echo "- No-evidence baseline: 355/463 = 0.7667"
  echo
  echo "## New Summaries"
  echo
  find "$PROJECT_ROOT/reason/results" -maxdepth 1 -type f -newer "$MARKER" \
    -name "medmcqa_*_summary.txt" -print | sort | while read -r file; do
      echo "### $(basename "$file")"
      echo
      echo '```text'
      cat "$file"
      echo '```'
      echo
    done
  echo "## New Diagnostics"
  echo
  find "$PROJECT_ROOT/reason/results/diagnostics" -maxdepth 1 -type f -newer "$MARKER" \
    -name "*_summary.txt" -print | sort | while read -r file; do
      echo "### $(basename "$file")"
      echo
      echo '```text'
      cat "$file"
      echo '```'
      echo
    done
} > "$REPORT"

rm -f "$MARKER"
echo "report=$REPORT"
echo "finished_at=$(date)"
