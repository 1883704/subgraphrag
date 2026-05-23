# Fixed Gate Experiment Operator Instructions

Role: run the prepared experiment only. Do not edit tracked source code. Do not commit. Do not push.

## Goal

Run a reproducible fixed-gate validation experiment. This is the follow-up to the `gate_0523_0405` run. It refreshes:

1. no-evidence MCQ predictions,
2. standard `tree_10_mcq` predictions,
3. fixed post-hoc evidence routing with thresholds chosen from the previous cross-validation run.

It does **not** train a new scorer and does **not** run the failed prompt variants.

## Safety Rules

- Use GPU1 only. Never run on GPU0.
- Do not delete datasets, checkpoints, model directories, or result folders.
- Do not modify code or configs.
- Do not run `git reset`, `git checkout --`, or destructive cleanup commands.
- Do not ask the user for confirmation during the run.
- If the script fails, save the last 160 log lines and report the failure.

## One Command

```bash
ssh ubuntu@202.38.78.248
cd /home/ubuntu/project/SubgraphRAG-main
git status --short
git pull --ff-only origin candidate-evidence-tree-scorer
chmod +x retrieve/scripts/run_fixed_gate_experiments.sh
CUDA_VISIBLE_DEVICES=1 bash retrieve/scripts/run_fixed_gate_experiments.sh \
  2>&1 | tee retrieve/logs/fixed_gate_operator_$(date +%m%d_%H%M).log
```

## Monitoring

Use read-only commands:

```bash
ps aux | grep -E 'python main.py|evidence_gate_ablation|run_fixed_gate' | grep -v grep || true
nvidia-smi
tail -n 160 retrieve/logs/fixed_gate_operator_*.log
```

## Expected Outputs

```text
reason/results/gating/fixed_gate_<timestamp>/MANIFEST.txt
reason/results/gating/fixed_gate_<timestamp>/*_gate_ablation.md
reason/results/gating/fixed_gate_<timestamp>/*_routed_predictions.jsonl
reason/results/medmcqa_fixed_gate_<timestamp>_best_fixed_summary.txt
reason/results/diagnostics/fixed_gate_<timestamp>_best_fixed_fix_harm_summary.txt
reason/results/experiment_report_latest.md
```

## What To Report Back

Report:

- `run_tag`
- no-evidence accuracy
- standard tree accuracy
- best fixed gate strategy and threshold
- best fixed gate accuracy
- fix/harm counts and McNemar p-value
- whether the best fixed gate exceeds `361/463 = 0.7797` and `366/463 = 0.7905`

