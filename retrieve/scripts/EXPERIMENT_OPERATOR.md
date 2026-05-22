# Experiment Operator Instructions

You are the experiment runner only. Do not modify code, YAML, or scripts on the server.

Allowed:
- `git fetch`, `git pull --ff-only`, `git status`
- run scripts under `retrieve/scripts/`
- monitor with `tmux`, `ps`, `nvidia-smi`, `free -h`, `df -h`, `tail`, `cat`, `grep`, `find`

Forbidden:
- editing Python/YAML/shell files
- `git reset --hard`, `git checkout -- .`, `git push`
- deleting raw data, processed data, checkpoints, or results
- using GPU0
- rebooting the server

Run sequence:

```bash
ssh ubuntu@202.38.78.248
cd /home/ubuntu/project/SubgraphRAG-main
git status --short
git fetch origin
git pull --ff-only
export CUDA_VISIBLE_DEVICES=1
tmux new-session -d -s overnight 'bash retrieve/scripts/run_overnight_experiments.sh'
```

Monitor:

```bash
tmux capture-pane -t overnight -p | tail -80
ps aux | grep -E "train_candidate|inference.py|python main.py" | grep -v grep
nvidia-smi
free -h
df -h
```

If a script is missing or `git pull --ff-only` fails, stop and report. Do not create or patch files yourself.
