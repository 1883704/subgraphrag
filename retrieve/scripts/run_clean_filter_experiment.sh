#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/run_candidate_experiment.sh" \
  fast_clean_filter \
  configs/treescorer/medmcqa_primekg_candidate_fast_clean_filter.yaml \
  tree_10_mcq \
  0.2
