# Fixed Gate Repeats Operator Instructions

Role: run the prepared repeat experiment only. Do not edit tracked source code. Do not commit. Do not push.

## Goal

Run repeated fixed-gate validation to check whether the improvement is stable across API refreshes. This follows the `fixed_gate_0523_1433` result.

Primary rule:

```text
use tree answer when tree_top_score >= 0.625, otherwise use no-evidence answer
```

## Safety Rules

- Use GPU1 only. Never run on GPU0.
- Do not delete datasets, checkpoints, model directories, or result folders.
- Do not modify code or configs.
- Do not run `git reset`, `git checkout --`, or destructive cleanup commands.
- Do not ask the user for confirmation during the run.

## One Command

```bash
ssh ubuntu@202.38.78.248
cd /home/ubuntu/project/SubgraphRAG-main
git status --short
git pull --ff-only origin candidate-evidence-tree-scorer
chmod +x retrieve/scripts/run_fixed_gate_repeats.sh
REPEATS=3 CUDA_VISIBLE_DEVICES=1 bash retrieve/scripts/run_fixed_gate_repeats.sh \
  2>&1 | tee retrieve/logs/fixed_gate_repeats_operator_$(date +%m%d_%H%M).log
```

## Monitoring

Use read-only commands:

```bash
ps aux | grep -E 'python main.py|evidence_gate_ablation|run_fixed_gate_repeats' | grep -v grep || true
nvidia-smi
tail -n 160 retrieve/logs/fixed_gate_repeats_operator_*.log
```

## Expected Outputs

```text
reason/results/gating/fixed_gate_repeats_<timestamp>/MANIFEST.txt
reason/results/gating/fixed_gate_repeats_<timestamp>/repeat_summary.md
reason/results/gating/fixed_gate_repeats_<timestamp>/repeat_summary.csv
reason/results/gating/fixed_gate_repeats_<timestamp>/repeat_*/gate_ablation.log
reason/results/diagnostics/fixed_gate_repeats_<timestamp>_r*_primary_fixed_fix_harm_summary.txt
reason/results/experiment_report_latest.md
```

## What To Report Back

Report:

- `run_tag`
- per-repeat no-evidence, standard tree, and primary fixed gate accuracy
- mean/std/min/max of fixed gate accuracy
- how many repeats exceed:
  - old best `361/463 = 0.7797`
  - previous CV gate `366/463 = 0.7905`
- average fix/harm/net gain if available

