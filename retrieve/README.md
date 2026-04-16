# Stage 1: Retrieval

## Table of Contents

- [Supported Datasets](#supported-datasets)
- [1-1 Embedding Pre-Computation](#1-1-entity-and-relation-embedding-pre-computation)
    * [Installation](#installation)
    * [Inference (Embedding Computation)](#inference-embedding-computation)
- [1-2 Retriever Development](#1-2-retriever-development)
    * [Installation](#installation-1)
    * [Training](#training)
    * [Inference](#inference)
    * [Evaluation](#evaluation)

## Supported Datasets

We support two built-in multi-hop knowledge graph question answering (KGQA) datasets:

- `webqsp`
- `cwq`

## 1-1: Entity and Relation Embedding Pre-Computation

We first pre-compute and cache entity and relation embeddings for all samples to save time for later training and inference of retrievers.

### Installation

We use `gte-large-en-v1.5` for text encoder, hence the environment name.

```bash
conda create -n gte_large_en_v1-5 python=3.10 -y
conda activate gte_large_en_v1-5
pip install -r requirements/gte_large_en_v1-5.txt
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
```

### Inference (Embedding Computation)

```bash
python emb.py -d  webqsp
```
where `D` should be a dataset mentioned in ["Supported Datasets"](#supported-datasets).

## 1-2: Retriever Development

We now train a retriever, employ it for retrieval (inference), and evaluate the retrieval results.

### Installation

```bash
conda create -n retriever python=3.10 -y
conda activate retriever
pip install -r requirements/retriever.txt
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.5.3
pip install pyg_lib==0.3.1 torch_scatter==2.1.2 torch_sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

### Training

```bash
python train.py -d webqsp
```
where `D` should be a dataset mentioned in ["Supported Datasets"](#supported-datasets).

To train the alternative TreeScorer path retriever, use:

```bash
python train_tree.py -d webqsp
```

For logged learning curves, go to the corresponding Wandb interface. 

Once trained, there will be a folder in the current directory of the form `{dataset}_{time}` (e.g., `webqsp_Nov08-01:14:47/`) that stores the trained model checkpoint `cpt.pth`.

### Inference

```bash
python inference.py -p P
```
where `P` is the path to a saved model checkpoint. The predicted retrieval result will be stored in the same folder as the model checkpoint. For example, if `P` is `webqsp_Nov08-01:14:47/cpt.pth`, then the retrieval result will be saved as `webqsp_Nov08-01:14:47/retrieval_result.pth`.

`inference.py` also accepts checkpoints created by `train_tree.py`. TreeScorer scores root-to-leaf paths and aggregates the path scores back to ranked triples, producing the same `retrieval_result.pth` format as the default retriever.

To run inference with the newest TreeScorer checkpoint for a dataset:

```bash
python inference.py --latest -d webqsp
```

This selects the newest `webqsp_tree_*/cpt.pth` checkpoint in the current
directory. The equivalent short form is:

```bash
python inference.py -p latest -d webqsp
```

### Evaluation

```bash
python eval.py -d D -p P
```
where `D` should be a dataset mentioned in ["Supported Datasets"](#supported-datasets) and `P` is the path to [inference result](#inference), e.g., `webqsp_Nov08-01:14:47/retrieval_result.pth`.

For TreeScorer reasoning-tree outputs, evaluate tree-level answer coverage and
duplicate trees with:

```bash
python eval_tree.py -p P
```
where `P` is the generated `tree_retrieval_result.jsonl`, e.g.,
`webqsp_tree_Nov08-01:14:47/tree_retrieval_result.jsonl`.
