# Gate Experiment Operator Instructions

Role: run experiments only. Do not edit tracked source code. Do not commit. Do not push.

## Goal

Run the evidence-gating and prompt-ablation experiment prepared by Codex. The experiment uses GPU1 only and mostly calls the configured API LLM. It should:

1. Refresh no-evidence MCQ predictions.
2. Refresh standard `tree_10_mcq` predictions with the current best fast tree file.
3. Run two conservative tree prompt variants:
   - `tree_10_gate_mcq`
   - `tree_10_compare_mcq`
4. Run post-hoc evidence gate ablations against the no-evidence baseline.
5. Generate summary reports.

## Safety Rules

- Use GPU1 only. Never run on GPU0.
- Do not delete datasets, checkpoints, model directories, or result folders.
- Do not modify code or configs.
- Do not run `git reset`, `git checkout --`, or destructive cleanup commands.
- If a command fails, capture the last 120 lines of the log and continue only if the next step does not depend on the failed output.
- Do not ask the user for confirmation during the run.

## One Command

```bash
ssh ubuntu@202.38.78.248
cd /home/ubuntu/project/SubgraphRAG-main
git status --short
git pull --ff-only origin candidate-evidence-tree-scorer
chmod +x retrieve/scripts/run_gate_prompt_experiments.sh
CUDA_VISIBLE_DEVICES=1 bash retrieve/scripts/run_gate_prompt_experiments.sh 2>&1 | tee retrieve/logs/gate_prompt_operator_$(date +%m%d_%H%M).log
```

## Monitoring

Use these commands. They are read-only.

```bash
tmux ls || true
ps aux | grep -E 'python main.py|evidence_gate_ablation|run_gate_prompt' | grep -v grep || true
nvidia-smi
tail -n 120 retrieve/logs/gate_prompt_operator_*.log
```

## Expected Outputs

The script prints a manifest at the end. Important files:

```text
reason/results/gating/<run_tag>/MANIFEST.txt
reason/results/gating/<run_tag>/*_gate_ablation.md
reason/results/gating/<run_tag>/*_best_routed_predictions.jsonl
reason/results/experiment_report_latest.md
reason/results/experiment_summary_latest.csv
```

## What To Report Back

Report only these items:

- Whether the script completed.
- The `run_tag`.
- Accuracy for:
  - `noevi`
  - `tree10_standard`
  - `tree10_gate_prompt`
  - `tree10_compare_prompt`
- Best gate from each `*_gate_ablation.md`:
  - strategy
  - threshold
  - accuracy
  - cross-validated accuracy
- Whether any result exceeds the current reference `361/463 = 0.7797`.

