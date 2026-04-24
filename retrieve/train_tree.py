import os
import pickle
import random
import time
from collections import defaultdict

import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.config.treescorer import load_yaml
from src.dataset.treescorer import TreeScorerDataset
from src.model.TreeScorer import TreeScorer
from src.setup import set_seed


class QuestionBatchSampler:
    def __init__(self, dataset, questions_per_batch, shuffle):
        self.batches = [
            indices for _, indices in sorted(dataset.qid2indices.items())
            if len(indices) > 0
        ]
        self.questions_per_batch = max(1, questions_per_batch)
        self.shuffle = shuffle

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            random.shuffle(order)

        for start in range(0, len(order), self.questions_per_batch):
            batch_indices = []
            for batch_idx in order[start:start + self.questions_per_batch]:
                batch_indices.extend(self.batches[batch_idx])
            yield batch_indices

    def __len__(self):
        return (len(self.batches) + self.questions_per_batch - 1) // self.questions_per_batch


def listwise_question_loss(logits, labels, qids, temperature=1.0):
    logits = logits.view(-1)
    labels = labels.view(-1).float()
    qids = qids.view(-1).long()

    losses = []
    for qid in torch.unique(qids):
        mask = qids == qid
        q_logits = logits[mask] / temperature
        q_labels = labels[mask]
        pos_mask = q_labels > 0.5

        if pos_mask.sum() == 0:
            continue

        log_probs = q_logits - torch.logsumexp(q_logits, dim=0)
        losses.append(-log_probs[pos_mask].mean())

    if len(losses) == 0:
        return F.binary_cross_entropy_with_logits(logits, labels)

    return torch.stack(losses).mean()


def load_data_and_embs(dataset_name, text_encoder_name, split):
    base_dir = f"data_files/{dataset_name}"
    pkl_path = os.path.join(base_dir, "processed", f"{split}.pkl")
    emb_path = os.path.join(base_dir, "emb", text_encoder_name, f"{split}.pth")

    with open(pkl_path, "rb") as f:
        raw_samples = pickle.load(f)

    emb_dict = torch.load(emb_path, map_location="cpu")
    return raw_samples, emb_dict


def compute_question_hits(labels, scores, qids, k_list):
    q2items = defaultdict(list)
    for label, score, qid in zip(labels, scores, qids):
        q2items[qid].append((score, label))

    metrics = {f"hit@{k}": [] for k in k_list}
    metrics["mrr"] = []

    for items in q2items.values():
        items.sort(key=lambda item: item[0], reverse=True)
        sorted_labels = [label for _, label in items]

        first_pos_rank = None
        for rank, label in enumerate(sorted_labels, start=1):
            if label > 0.5:
                first_pos_rank = rank
                break

        metrics["mrr"].append(0.0 if first_pos_rank is None else 1.0 / first_pos_rank)
        for k in k_list:
            metrics[f"hit@{k}"].append(
                1.0 if any(label > 0.5 for label in sorted_labels[:k]) else 0.0
            )

    return {
        metric: sum(values) / max(len(values), 1)
        for metric, values in metrics.items()
    }


@torch.no_grad()
def eval_epoch(config, device, data_loader, model):
    model.eval()
    all_labels = []
    all_scores = []
    all_qids = []
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(data_loader, desc="Evaluating", leave=False):
        batch = batch.to(device)
        logits = model(batch).view(-1)
        labels = batch.y.view(-1).float()
        qids = batch.qid.view(-1).long()

        loss = listwise_question_loss(logits, labels, qids)
        total_loss += loss.item()
        num_batches += 1

        all_labels.extend(labels.detach().cpu().tolist())
        all_scores.extend(torch.sigmoid(logits).detach().cpu().tolist())
        all_qids.extend(qids.detach().cpu().tolist())

    metrics = compute_question_hits(
        all_labels, all_scores, all_qids, config["eval"]["k_list"]
    )
    metrics["loss"] = total_loss / max(num_batches, 1)
    return metrics


def train_epoch(device, train_loader, model, optimizer, grad_clip_norm):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        batch = batch.to(device)
        logits = model(batch).view(-1)
        labels = batch.y.view(-1).float()
        qids = batch.qid.view(-1).long()

        loss = listwise_question_loss(logits, labels, qids)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return {"loss": total_loss / max(num_batches, 1)}


def build_dataset(config, split, mode):
    dataset_name = config["dataset"]["name"]
    text_encoder_name = config["dataset"]["text_encoder_name"]
    raw_samples, emb_dict = load_data_and_embs(dataset_name, text_encoder_name, split)

    tree_config = config["treescorer"]
    cache_dir = os.path.join("data_files", dataset_name, "cache", "treescorer")
    cache_path = os.path.join(
        cache_dir,
        f"trees_{dataset_name}_{split}_h{tree_config['max_hops']}_{mode}.pt",
    )

    return TreeScorerDataset(
        raw_samples,
        emb_dict,
        max_hops=tree_config["max_hops"],
        max_paths_per_root=tree_config["max_paths_per_root"],
        max_paths_per_sample=tree_config["max_paths_per_sample"],
        add_reverse_edges=tree_config["add_reverse_edges"],
        mode=mode,
        max_neg_per_pos=tree_config["max_neg_per_pos"],
        max_neg_per_sample=tree_config["max_neg_per_sample"],
        cache_path=cache_path,
        use_cache=tree_config["use_cache"],
        cache_version=tree_config["cache_version"],
    )


def main(args):
    config_file = f"configs/treescorer/{args.dataset}.yaml"
    config = load_yaml(config_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(config["env"]["num_threads"])
    set_seed(config["env"]["seed"])

    ts = time.strftime("%b%d-%H-%M-%S", time.gmtime())
    exp_name = f"{config['train']['save_prefix']}_{ts}"
    os.makedirs(exp_name, exist_ok=True)

    config_df = pd.json_normalize(config, sep="/")
    wandb.init(
        project=f"{args.dataset}_TreeScorer",
        name=exp_name,
        config=config_df.to_dict(orient="records")[0],
    )

    train_set = build_dataset(config, split="train", mode="train")
    val_set = build_dataset(config, split="val", mode="eval")

    if len(train_set) == 0 or len(val_set) == 0:
        raise ValueError(
            f"TreeScorer dataset is empty: train={len(train_set)}, val={len(val_set)}"
        )

    train_loader = DataLoader(
        train_set,
        batch_sampler=QuestionBatchSampler(
            train_set,
            questions_per_batch=config["train"]["batch_size"],
            shuffle=True,
        ),
    )
    val_loader = DataLoader(
        val_set,
        batch_sampler=QuestionBatchSampler(
            val_set,
            questions_per_batch=config["train"]["batch_size"],
            shuffle=False,
        ),
    )

    emb_size = train_set[0].q_emb.shape[-1]
    tree_config = config["treescorer"]
    model = TreeScorer(
        emb_size=emb_size,
        hidden_size=tree_config["hidden_size"],
        num_layers=tree_config["num_layers"],
        heads=tree_config["heads"],
    ).to(device)
    optimizer = Adam(model.parameters(), **config["optimizer"])
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config["train"]["lr_scheduler_factor"],
        patience=config["train"]["lr_scheduler_patience"],
    )

    monitor_metric = config["train"]["monitor_metric"]
    if monitor_metric.startswith("val/"):
        monitor_metric = monitor_metric[len("val/"):]
    primary_metric = monitor_metric
    best_metric = -1.0
    num_patient_epochs = 0
    min_delta = config["train"]["min_delta"]
    grad_clip_norm = config["train"]["grad_clip_norm"]

    for epoch in range(config["train"]["num_epochs"]):
        train_log = train_epoch(
            device, train_loader, model, optimizer, grad_clip_norm)
        val_log = eval_epoch(config, device, val_loader, model)

        if primary_metric not in val_log:
            raise KeyError(
                f"Monitor metric '{primary_metric}' not found in validation "
                f"metrics: {sorted(val_log.keys())}"
            )
        target_metric = val_log[primary_metric]
        scheduler.step(target_metric)

        if target_metric > best_metric + min_delta:
            best_metric = target_metric
            num_patient_epochs = 0
            torch.save(
                {
                    "model_type": "treescorer",
                    "config": config,
                    "model_state_dict": model.state_dict(),
                    "best_metric": best_metric,
                    "best_metric_name": primary_metric,
                    "epoch": epoch,
                },
                os.path.join(exp_name, "cpt.pth"),
            )
        else:
            num_patient_epochs += 1

        log_dict = {
            "epoch": epoch,
            "num_patient_epochs": num_patient_epochs,
            "best_metric": best_metric,
            "lr": optimizer.param_groups[0]["lr"],
            "train/loss": train_log["loss"],
        }
        for key, val in val_log.items():
            log_dict[f"val/{key}"] = val
        wandb.log(log_dict)

        if num_patient_epochs >= config["train"]["patience"]:
            break


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Dataset name",
    )
    args = parser.parse_args()

    main(args)
