# Stage 2: Reasoning

## Table of Contents

* [Installation](#installation)
* [Pre-processed Results for Reproducibility](#pre-processed-results-for-reproducibility)
* [Inference with LLMs](#inference-with-llms)

## Installation

```bash
conda create -n reasoner python=3.10.14 -y
conda activate reasoner
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.5.5 openai==1.50.2 wandb
```

## Reasoning (Inference)

### Using Pre-Processed Retrieval Results for Reproducibility

We provide pre-processed results for reproducibility of the paper experiments. To download them

```bash
huggingface-cli download siqim311/SubgraphRAG --revision main --local-dir ./
```

- `scored_triples` stores the pre-processed retrieval results.
- `results/KGQA` stores the reasoning results.

After downloading the pre-processed results, one can run `main.py` with proper paramerters. For example,

```
python main.py -d webqsp --prompt_mode scored_100
python main.py -d cwq --prompt_mode scored_100
```

### Using Alternative Retrieval Results

To use alternative retrieval results,

```
python main.py -d webqsp --prompt_mode scored_100 -p P
```
where `P` is the path to the retrieval results obtained from retrieval inference, e.g., `../retrieve/webqsp_Nov08-01:14:47/retrieval_result.pth`.

For TreeScorer reasoning-tree results, use a tree prompt mode and point `-p` to the generated tree result:

```
python main.py -d webqsp --prompt_mode tree_5 -p ../retrieve/webqsp_tree_Nov08-01-14-47/tree_retrieval_result.jsonl
```

For local datasets, `main.py` can now read local `raw/processed` subgraphs
directly, so you can run reasoning without a built-in `RoG-*` dataset as long
as you provide a retrieval result path:

```
python main.py -d medmcqa_primekg --prompt_mode scored_100 -p ../retrieve/medmcqa_primekg_Nov08-01-14-47/retrieval_result.pth --skip_eval
python main.py -d medmcqa_primekg --prompt_mode tree_5 -p ../retrieve/medmcqa_primekg_tree_Nov08-01-14-47/tree_retrieval_result.jsonl --skip_eval
```

If you already have a local `predictions.jsonl` input file, you can pass it
explicitly:

```
python main.py -d medmcqa_primekg --prompt_mode scored_100 -p ../retrieve/medmcqa_primekg_Nov08-01-14-47/retrieval_result.pth --pred_file_path ./results/KGQA/medmcqa_primekg/custom/predictions.jsonl --skip_eval
```

### Config

Our used config for each dataset can be found in `./config`.
