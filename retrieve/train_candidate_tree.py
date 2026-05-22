import os
import pickle
import random
import time
from collections import defaultdict

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from src.dataset.candidate_treescorer import CandidateTreeScorerDataset
from src.model.CandidateTreeScorer import CandidateTreeScorer
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


def load_yaml(config_file):
    with open(config_file, encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.loader.SafeLoader)
    task = config.pop("task")
    if task != "candidate_treescorer":
        raise ValueError(f"Expected task=candidate_treescorer, got {task}")
    if isinstance(config["eval"]["k_list"], str):
        config["eval"]["k_list"] = [
            int(k.strip()) for k in config["eval"]["k_list"].split(",") if k.strip()
        ]
    return config


def load_data_and_embs(dataset_name, text_encoder_name, split):
    base_dir = f"data_files/{dataset_name}"
    pkl_path = os.path.join(base_dir, "processed", f"{split}.pkl")
    emb_path = os.path.join(base_dir, "emb", text_encoder_name, f"{split}.pth")

    with open(pkl_path, "rb") as f:
        raw_samples = pickle.load(f)
    emb_dict = torch.load(emb_path, map_location="cpu")
    return raw_samples, emb_dict


def build_dataset(config, split, mode):
    dataset_name = config["dataset"]["name"]
    text_encoder_name = config["dataset"]["text_encoder_name"]
    raw_samples, emb_dict = load_data_and_embs(dataset_name, text_encoder_name, split)

    tree_config = config["candidate_treescorer"]
    cache_dir = os.path.join("data_files", dataset_name, "cache", "candidate_treescorer")
    cache_path = os.path.join(
        cache_dir,
        (
            f"candidate_trees_{dataset_name}_{split}"
            f"_h{tree_config['max_hops']}_{mode}.pt"
        ),
    )

    return CandidateTreeScorerDataset(
        raw_samples,
        emb_dict,
        max_hops=tree_config["max_hops"],
        max_paths_per_root=tree_config["max_paths_per_root"],
        max_paths_per_sample=tree_config["max_paths_per_sample"],
        max_paths_per_candidate=tree_config["max_paths_per_candidate"],
        max_pos_per_candidate=tree_config["max_pos_per_candidate"],
        max_hard_neg_per_candidate=tree_config["max_hard_neg_per_candidate"],
        add_reverse_edges=tree_config["add_reverse_edges"],
        label_base=config["dataset"].get("label_base", "zero"),
        num_options=config["dataset"].get("num_options", 4),
        mode=mode,
        cache_path=cache_path,
        use_cache=tree_config["use_cache"],
        cache_version=tree_config["cache_version"],
        filter_generic_entities=tree_config.get("filter_generic_entities", False),
        generic_entity_stoplist=tree_config.get("generic_entity_stoplist"),
        min_entity_text_len=tree_config.get("min_entity_text_len", 0),
        max_entity_degree=tree_config.get("max_entity_degree"),
        semantic_sort_records=tree_config.get("semantic_sort_records", False),
        semantic_positive_topk=tree_config.get("semantic_positive_topk", 0),
        semantic_positive_min_score=tree_config.get("semantic_positive_min_score"),
    )


def aggregate_candidate_scores(logits, qids, candidate_idxs, num_options, mode):
    q2scores = {}
    for qid in torch.unique(qids):
        q_mask = qids == qid
        scores = logits.new_full((num_options,), -1e4)
        for candidate_idx in range(num_options):
            c_mask = q_mask & (candidate_idxs == candidate_idx)
            if not c_mask.any():
                continue
            c_logits = logits[c_mask]
            if mode == "logsumexp":
                scores[candidate_idx] = torch.logsumexp(c_logits, dim=0)
            elif mode == "mean":
                scores[candidate_idx] = c_logits.mean()
            else:
                scores[candidate_idx] = c_logits.max()
        q2scores[int(qid.item())] = scores
    return q2scores


def candidate_option_loss(
    logits,
    qids,
    candidate_idxs,
    gold_idxs,
    num_options,
    aggregate_mode,
):
    q2scores = aggregate_candidate_scores(
        logits,
        qids,
        candidate_idxs,
        num_options,
        aggregate_mode,
    )

    losses = []
    for qid, scores in q2scores.items():
        q_mask = qids == qid
        gold_idx = int(gold_idxs[q_mask][0].item())
        if gold_idx < 0 or gold_idx >= num_options:
            continue
        target = torch.tensor([gold_idx], dtype=torch.long, device=logits.device)
        losses.append(F.cross_entropy(scores.view(1, -1), target))

    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def compute_metrics(logits, labels, qids, candidate_idxs, gold_idxs, config):
    num_options = config["dataset"].get("num_options", 4)
    aggregate_mode = config["loss"].get("aggregate_mode", "max")
    q2scores = aggregate_candidate_scores(
        logits,
        qids,
        candidate_idxs,
        num_options,
        aggregate_mode,
    )

    option_hits = []
    for qid, scores in q2scores.items():
        q_mask = qids == qid
        gold_idx = int(gold_idxs[q_mask][0].item())
        if gold_idx < 0 or gold_idx >= num_options:
            continue
        option_hits.append(float(int(scores.argmax().item()) == gold_idx))

    q2items = defaultdict(list)
    for label, score, qid in zip(labels.tolist(), logits.tolist(), qids.tolist()):
        q2items[int(qid)].append((float(score), float(label)))

    metrics = {
        "option_acc": sum(option_hits) / max(len(option_hits), 1),
        "num_eval_questions": len(option_hits),
    }
    for k in config["eval"]["k_list"]:
        hits = []
        for items in q2items.values():
            items.sort(key=lambda item: item[0], reverse=True)
            hits.append(float(any(label > 0.5 for _, label in items[:k])))
        metrics[f"support_hit@{k}"] = sum(hits) / max(len(hits), 1)

    return metrics


def compute_loss(config, logits, batch):
    labels = batch.y.view(-1).float()
    qids = batch.qid.view(-1).long()
    candidate_idxs = batch.candidate_idx.view(-1).long()
    gold_idxs = batch.option_target.view(-1).long()

    option_loss = candidate_option_loss(
        logits,
        qids,
        candidate_idxs,
        gold_idxs,
        config["dataset"].get("num_options", 4),
        config["loss"].get("aggregate_mode", "max"),
    )
    support_loss = F.binary_cross_entropy_with_logits(logits, labels)
    total_loss = (
        config["loss"].get("option_loss_weight", 1.0) * option_loss
        + config["loss"].get("support_loss_weight", 0.2) * support_loss
    )
    return total_loss, {
        "option_loss": option_loss.detach(),
        "support_loss": support_loss.detach(),
    }


def train_epoch(config, device, data_loader, model, optimizer):
    model.train()
    total_loss = 0.0
    total_option_loss = 0.0
    total_support_loss = 0.0
    num_batches = 0
    grad_clip_norm = config["train"].get("grad_clip_norm", 1.0)

    for batch in tqdm(data_loader, desc="Training", leave=False):
        batch = batch.to(device)
        logits = model(batch).view(-1)
        loss, loss_parts = compute_loss(config, logits, batch)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += float(loss.item())
        total_option_loss += float(loss_parts["option_loss"].item())
        total_support_loss += float(loss_parts["support_loss"].item())
        num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "option_loss": total_option_loss / max(num_batches, 1),
        "support_loss": total_support_loss / max(num_batches, 1),
    }


@torch.no_grad()
def eval_epoch(config, device, data_loader, model):
    model.eval()
    all_logits = []
    all_labels = []
    all_qids = []
    all_candidate_idxs = []
    all_gold_idxs = []
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(data_loader, desc="Evaluating", leave=False):
        batch = batch.to(device)
        logits = model(batch).view(-1)
        loss, _ = compute_loss(config, logits, batch)

        total_loss += float(loss.item())
        num_batches += 1

        all_logits.append(logits.detach().cpu())
        all_labels.append(batch.y.view(-1).float().detach().cpu())
        all_qids.append(batch.qid.view(-1).long().detach().cpu())
        all_candidate_idxs.append(batch.candidate_idx.view(-1).long().detach().cpu())
        all_gold_idxs.append(batch.option_target.view(-1).long().detach().cpu())

    if not all_logits:
        return {"loss": 0.0, "option_acc": 0.0}

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    qids = torch.cat(all_qids)
    candidate_idxs = torch.cat(all_candidate_idxs)
    gold_idxs = torch.cat(all_gold_idxs)

    metrics = compute_metrics(
        logits,
        labels,
        qids,
        candidate_idxs,
        gold_idxs,
        config,
    )
    metrics["loss"] = total_loss / max(num_batches, 1)
    return metrics


def main(args):
    config = load_yaml(args.config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(config["env"]["num_threads"])
    set_seed(config["env"]["seed"])

    train_set = build_dataset(config, split="train", mode="train")
    val_set = build_dataset(config, split="val", mode="eval")
    if len(train_set) == 0 or len(val_set) == 0:
        raise ValueError(
            f"CandidateTreeScorer dataset is empty: "
            f"train={len(train_set)}, val={len(val_set)}"
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
    tree_config = config["candidate_treescorer"]
    model = CandidateTreeScorer(
        emb_size=emb_size,
        hidden_size=tree_config["hidden_size"],
        num_layers=tree_config["num_layers"],
        heads=tree_config["heads"],
        dropout=tree_config["dropout"],
        node_extra_size=CandidateTreeScorerDataset.node_extra_size,
        edge_extra_size=CandidateTreeScorerDataset.edge_extra_size,
        path_extra_size=CandidateTreeScorerDataset.path_extra_size,
    ).to(device)

    optimizer = Adam(model.parameters(), **config["optimizer"])
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config["train"].get("lr_scheduler_factor", 0.5),
        patience=config["train"].get("lr_scheduler_patience", 2),
    )

    ts = time.strftime("%b%d-%H-%M-%S", time.gmtime())
    exp_name = f"{config['train']['save_prefix']}_{ts}"
    os.makedirs(exp_name, exist_ok=True)

    if wandb is not None:
        config_df = pd.json_normalize(config, sep="/")
        wandb.init(
            project=f"{config['dataset']['name']}_CandidateTreeScorer",
            name=exp_name,
            config=config_df.to_dict(orient="records")[0],
            mode=config["train"].get("wandb_mode", "disabled"),
        )

    monitor_metric = config["train"].get("monitor_metric", "option_acc")
    if monitor_metric.startswith("val/"):
        monitor_metric = monitor_metric[len("val/"):]
    best_metric = -1.0
    num_patient_epochs = 0
    min_delta = config["train"].get("min_delta", 0.0)

    for epoch in range(config["train"]["num_epochs"]):
        train_log = train_epoch(config, device, train_loader, model, optimizer)
        val_log = eval_epoch(config, device, val_loader, model)

        target_metric = val_log[monitor_metric]
        scheduler.step(target_metric)

        if target_metric > best_metric + min_delta:
            best_metric = target_metric
            num_patient_epochs = 0
            torch.save(
                {
                    "model_type": "candidate_treescorer",
                    "config": config,
                    "model_state_dict": model.state_dict(),
                    "best_metric": best_metric,
                    "best_metric_name": monitor_metric,
                    "epoch": epoch,
                    "dataset_stats": {
                        "train": train_set.stats,
                        "val": val_set.stats,
                    },
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
        }
        for key, val in train_log.items():
            log_dict[f"train/{key}"] = val
        for key, val in val_log.items():
            log_dict[f"val/{key}"] = val

        print(log_dict)
        if wandb is not None:
            wandb.log(log_dict)

        if num_patient_epochs >= config["train"]["patience"]:
            break


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser("Train candidate-aware reasoning tree scorer")
    parser.add_argument(
        "--config",
        default="configs/treescorer/medmcqa_primekg_candidate.yaml",
        help="Candidate TreeScorer YAML config.",
    )
    args = parser.parse_args()
    main(args)
