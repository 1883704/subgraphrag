import os
import time
import torch
import torch.nn.functional as F
import wandb
import pickle
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

from src.dataset.treescorer import TreeScorerDataset
from src.model.TreeScorer import TreeScorer


# -------------------------------------------------------
# -------------------------------------------------------
def listwise_question_loss(logits, labels, qids, temperature=1.0):
    """
    logits: [N]，所有路径的 logit
    labels: [N]，0/1
    qids:   [N]，question id（同一个问题的树具有相同 qid）

    对每个 qid 内做 softmax，将概率质量推向正样本：
        L_i = - mean_{j in Pos_i} log softmax(logit_{ij})
    最终返回所有 question 的平均 loss。
    """
    device = logits.device
    logits = logits.view(-1)
    labels = labels.view(-1).float()
    qids = qids.view(-1).long()

    unique_qids = torch.unique(qids)
    losses = []
    valid_q_cnt = 0

    for q in unique_qids:
        mask = qids == q
        lq = logits[mask] / temperature  # [K]
        yq = labels[mask]  # [K]

        pos_mask = yq > 0.5
        if pos_mask.sum() == 0:
            # 这个 question 没有正样本，跳过（也可以按需要处理）
            continue

        # log_softmax over this question
        log_probs = lq - torch.logsumexp(lq, dim=0)  # [K]
        pos_log_probs = log_probs[pos_mask]  # [#pos]

        loss_q = -pos_log_probs.mean()
        losses.append(loss_q)
        valid_q_cnt += 1

    if valid_q_cnt == 0:
        # 理论上不会发生，如果发生就退回普通 BCE 防止 NaN
        return F.binary_cross_entropy_with_logits(logits, labels)

    return torch.stack(losses).mean()


def load_data_and_embs(dataset_name, split):
    # 假设你的文件路径结构如下，请根据实际情况微调
    # data_files/webqsp/processed/train.pkl
    # data_files/webqsp/emb/gte-large-en-v1.5/train.pth

    base_dir = f"data_files/{dataset_name}"
    encoder_name = "gte-large-en-v1.5"  # 假设这是固定的

    # 加载 PKL
    pkl_path = os.path.join(base_dir, "processed", f"{split}.pkl")
    print(f"Loading structure from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        raw_samples = pickle.load(f)

    # 加载 PTH
    emb_path = os.path.join(base_dir, "emb", encoder_name, f"{split}.pth")
    print(f"Loading embeddings from {emb_path}...")
    if not os.path.exists(emb_path):
        # 兼容处理：有些文件名可能叫 validation.pth 而不是 val.pth
        alt_path = emb_path.replace("val", "validation")
        if os.path.exists(alt_path):
            emb_path = alt_path
        else:
            raise FileNotFoundError(f"Embedding file not found: {emb_path}")

    t0 = time.time()
    emb_dict = torch.load(emb_path, map_location="cpu", mmap=True)
    print(f"Loaded embeddings in {time.time()-t0:.1f}s. keys={len(emb_dict)}")

    return raw_samples, emb_dict


def compute_question_hits(labels, scores, qids, ks=(1, 3, 5)):
    """
    labels, scores, qids: 均为 Python list，长度相同
    返回 (hit@1, hit@3, hit@5)，单位是 [0,1] 之间的比例
    """
    from collections import defaultdict

    q2items = defaultdict(list)
    for y, s, q in zip(labels, scores, qids):
        q2items[q].append((s, y))

    hits = {k: 0 for k in ks}
    total_q = 0

    for q, items in q2items.items():
        # items: List[(score, label)]
        # 按 score 降序
        items.sort(key=lambda x: x[0], reverse=True)
        labels_sorted = [y for _, y in items]
        total_q += 1

        for k in ks:
            topk = labels_sorted[:k]
            if any(y > 0.5 for y in topk):
                hits[k] += 1

    if total_q == 0:
        return (0.0,) * len(ks)

    return tuple(hits[k] / total_q for k in ks)


# -------------------------------------------------------
# 2. 训练与评估函数
# -------------------------------------------------------
@torch.no_grad()
def eval_epoch(device, data_loader, model):
    model.eval()
    all_labels = []
    all_scores = []
    all_qids = []

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(data_loader, desc="Evaluating", leave=False)
    for batch in pbar:
        batch = batch.to(device)

        logits = model(batch).view(-1)
        labels = batch.y.view(-1).float()
        qids = batch.qid.view(-1).long()

        loss = listwise_question_loss(logits, labels, qids, temperature=1.0)
        total_loss += loss.item()
        num_batches += 1

        probs = torch.sigmoid(logits)
        all_labels.extend(labels.detach().cpu().tolist())
        all_scores.extend(probs.detach().cpu().tolist())
        all_qids.extend(qids.detach().cpu().tolist())

    avg_loss = total_loss / max(num_batches, 1)

    try:
        auc = roc_auc_score(all_labels, all_scores)
    except ValueError:
        auc = 0.5

    # 可选：再加一个 question 级 top-k 指标（后面 2.4 给代码）
    hit1, hit3, hit5 = compute_question_hits(all_labels, all_scores, all_qids)

    return {"loss": avg_loss, "auc": auc, "hit@1": hit1, "hit@3": hit3, "hit@5": hit5}


def train_epoch(device, train_loader, model, optimizer):
    model.train()
    epoch_loss = 0.0
    num_q = 0  # 统计参与 loss 的 question 数

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch).view(-1)  # [num_graphs]
        labels = batch.y.view(-1).float()  # [num_graphs]
        qids = batch.qid.view(-1).long()  # [num_graphs]  <- 依赖上面 Dataset 中的 qid

        loss = listwise_question_loss(logits, labels, qids, temperature=1.0)
        loss.backward()
        optimizer.step()

        # 这里 epoch_loss 可以简单做加和平均（以 question 为单位也可以）
        epoch_loss += loss.item()
        num_q += 1

        pbar.set_postfix({"loss": loss.item()})

    return {"loss": epoch_loss / max(num_q, 1)}


# -------------------------------------------------------
# 3. 主程序
# -------------------------------------------------------
def main(args):
    # 配置参数 (你可以改为 argparse 或 yaml 读取)
    config = {
        "max_hops": 3,
        "batch_size": 8,
        "hidden_size": 256,
        "lr": 1e-4,
        "epochs": 5,
        "patience": 3,
        "dataset": args.dataset,
        "cache_dir": "data_files/{dataset}/cache/treescorer",
    }
    wandb.init(
        project="TreeScorer",  # 可以换成你自己的项目名
        config=config,
        name=f"{args.dataset}_h{config['max_hops']}_hs{config['hidden_size']}",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载数据
    train_raw, train_embs = load_data_and_embs(args.dataset, "train")
    val_raw, val_embs = load_data_and_embs(args.dataset, "val")  # 注意此处 split 名称
    # 控制训练样本的数量
    # DEBUG_SIZE = 2500
    # print(f"⚠️ DEBUG MODE ON: Slicing dataset to top {DEBUG_SIZE} samples.")

    # 2. 构建 Dataset
    # 自动探测 Embedding 维度
    sample_key = list(train_embs.keys())[0]
    emb_dim = train_embs[sample_key]["q_emb"].shape[-1]
    print(f"Detected Embedding Dimension: {emb_dim}")
    print(f"Train Set Size: {len(train_raw)}")
    cache_dir = config["cache_dir"].format(dataset=args.dataset)
    train_cache_path = os.path.join(
        cache_dir,
        f"trees_{args.dataset}_train_h{config['max_hops']}_"
        f"negpos10_negsample60.pt",
    )
    val_cache_path = os.path.join(
        cache_dir,
        f"trees_{args.dataset}_val_h{config['max_hops']}_eval.pt",
    )

    train_set = TreeScorerDataset(
        train_raw,
        train_embs,
        max_hops=config["max_hops"],
        mode="train",
        max_neg_per_pos=10,  # 每个正样本配 10 个负样本
        max_neg_per_sample=60,  # 每个问题最多 60 个负样本
        cache_path=train_cache_path,
        use_cache=True,
    )

    val_set = TreeScorerDataset(
        val_raw,
        val_embs,
        max_hops=config["max_hops"],
        mode="eval",  # 验证集不做负采样，保持真实分布
        cache_path=val_cache_path,
        use_cache=True,
    )
    print("len(train_raw) =", len(train_raw))
    print("len(train_set) =", len(train_set))
    print("len(val_raw)   =", len(val_raw))
    print("len(val_set)   =", len(val_set))

    # 看第一个样本长什么样
    if len(train_set) > 0:
        d0 = train_set[0]
        print("First train sample Data:", d0)
        print("  x.shape:", d0.x.shape)
        print("  edge_index.shape:", d0.edge_index.shape)
        print("  y:", d0.y, "qid:", d0.qid, "num_nodes:", d0.num_nodes)
        train_loader = DataLoader(
            train_set, batch_size=config["batch_size"], shuffle=True
        )
        val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)

    # 3. 初始化模型
    model = TreeScorer(
        emb_size=emb_dim, hidden_size=config["hidden_size"], num_layers=2
    ).to(device)

    optimizer = Adam(model.parameters(), lr=config["lr"])
    wandb.watch(model, log="gradients", log_freq=100)
    # 4. 训练循环
    # wandb.init(project="TreeScorer", config=config) # 如果需要 WandB 请取消注释

    best_auc = 0
    patience_cnt = 0

    print("\n🚀 Start Training...")
    for epoch in range(config["epochs"]):
        start_time = time.time()

        # Train
        train_metrics = train_epoch(device, train_loader, model, optimizer)

        # Eval
        val_metrics = eval_epoch(device, val_loader, model)

        duration = time.time() - start_time
        print(
            f"Epoch {epoch+1:02d} | Time: {duration:.1f}s | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f} | "
            f"Hit@1: {val_metrics['hit@1']:.4f} | "
            f"Hit@3: {val_metrics['hit@3']:.4f} | "
            f"Hit@5: {val_metrics['hit@5']:.4f}"
        )
        # Log to WandB
        # 【新】把本 epoch 的指标打到 wandb
        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": train_metrics["loss"],
                "val/loss": val_metrics["loss"],
                "val/auc": val_metrics["auc"],
                "val/hit@1": val_metrics["hit@1"],
                "val/hit@3": val_metrics["hit@3"],
                "val/hit@5": val_metrics["hit@5"],
                "lr": optimizer.param_groups[0]["lr"],
                "time/epoch_sec": duration,
            }
        )

        # Checkpoint
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            patience_cnt = 0
            save_path = f"best_model_{args.dataset}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  🌟 New Best Model Saved! (AUC: {best_auc:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= config["patience"]:
                print("  🛑 Early Stopping Triggered.")
                break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", type=str, default="webqsp", help="Dataset name"
    )
    args = parser.parse_args()

    main(args)
