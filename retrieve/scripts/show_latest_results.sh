#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-/home/ubuntu/project/SubgraphRAG-main}"
cd "$BASE_DIR"

python3 retrieve/scripts/summarize_experiment_results.py \
  --root "$BASE_DIR" \
  --output reason/results/experiment_report_latest.md \
  --csv reason/results/experiment_summary_latest.csv

