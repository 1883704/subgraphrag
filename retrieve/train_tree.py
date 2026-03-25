import os
import time
import torch
import torch.nn.functional as F
import pandas as pd
import wandb
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

# 关键变化：使用 PyG 的 DataLoader
from torch_geometric.loader import DataLoader 

# 假设你已经定义好了这些模块
# from src.config.tree import load_yaml  # 假设你有新的 tree 配置文件
from src.dataset.treescorer import TreeScorerDataset # 上一步实现的 Dataset
from src.model.TreeScorer import TreeScorer # 上上步实现的 Model
from src.setup import set_seed

@torch.no_grad()
def eval_epoch(device, data_loader, model):
    model.eval()
    
    all_labels = []
    all_scores = []
    total_loss = 0
    
    for batch in tqdm(data_loader, desc="Evaluating"):
        # 1. 搬运数据到 GPU
        batch = batch.to(device)
        
        # 2. 前向传播
        # 注意：q_emb 需要在 Dataset 构建时放入 Data 对象中
        # 此时 batch.q_emb 的形状是 [Batch_Size, Emb_Dim]
        scores = model(batch, batch.q_emb).squeeze(-1) # [Batch_Size]
        
        # 3. 计算 Loss
        labels = batch.y # [Batch_Size]
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        total_loss += loss.item() * batch.num_graphs
        
        # 4. 收集结果用于计算指标
        probs = torch.sigmoid(scores)
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(probs.cpu().numpy())
    
    # 计算整个 Epoch 的平均 Loss
    avg_loss = total_loss / len(data_loader.dataset)
    
    # 计算分类指标
    # AUC: 衡量模型区分正负样本的能力
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except ValueError:
        auc = 0.5 # 防止只有一个类别报错
        
    # Accuracy: 简单的准确率 (阈值 0.5)
    preds = [1 if p > 0.5 else 0 for p in all_scores]
    acc = accuracy_score(all_labels, preds)

    metric_dict = {
        'loss': avg_loss,
        'auc': auc,
        'acc': acc
    }
    
    return metric_dict

def train_epoch(device, train_loader, model, optimizer):
    model.train()
    epoch_loss = 0
    num_samples = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)
        
        # 1. 前向传播
        # TreeScorer 的 forward 接收 batch 和 q_emb
        pred_scores = model(batch, batch.q_emb).squeeze(-1) # [Batch_Size]
        
        # 2. 计算 Loss
        # batch.y 是我们在 Dataset 里打的标签 (0 或 1)
        labels = batch.y
        loss = F.binary_cross_entropy_with_logits(pred_scores, labels)
        
        # 3. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录加权 Loss
        epoch_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs
    
    avg_loss = epoch_loss / num_samples
    return {'loss': avg_loss}

def main(args):
    # 1. 配置加载
    config_file = f'configs/treescorer/{args.dataset}.yaml' # 注意修改配置文件路径
    config = load_yaml(config_file)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(config['env']['num_threads'])
    set_seed(config['env']['seed'])

    # 2. WandB 初始化
    ts = time.strftime('%b%d-%H-%M-%S', time.gmtime())
    exp_prefix = config['train']['save_prefix']
    exp_name = f'Tree_{exp_prefix}_{ts}'
    
    wandb.init(
        project=f'{args.dataset}_Tree',
        name=exp_name,
        config=config
    )
    os.makedirs(exp_name, exist_ok=True)

    # 3. 数据集准备 (假设 raw_samples 和 emb 已经准备好)
    # 在实际代码中，你需要先加载 raw_samples 和 embedding 矩阵
    # 这里省略加载 raw_samples, ent_embs, rel_embs 的代码
    print("Loading datasets...")
    # 示例占位符：你需要实现一个函数来加载预处理好的 pickle/pt 文件
    # raw_samples_train, ent_embs, rel_embs = load_processed_data(args.dataset, 'train')
    
    train_set = TreeScorerDataset(
        raw_samples=raw_samples_train, # 需外部传入
        entity_embs=ent_embs, 
        relation_embs=rel_embs,
        max_hops=config['data']['max_hops']
    )
    
    val_set = TreeScorerDataset(
        raw_samples=raw_samples_val, # 需外部传入
        entity_embs=ent_embs, 
        relation_embs=rel_embs,
        max_hops=config['data']['max_hops']
    )

    # 🌟 关键：使用 PyG 的 DataLoader
    # 它会自动把多棵树 collate 成一个 Batch 对象
    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config['train']['batch_size'], shuffle=False)
    
    # 4. 模型初始化
    emb_size = ent_embs.shape[-1]
    model = TreeScorer(
        emb_size=emb_size, 
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers']
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=float(config['optimizer']['lr']))

    # 5. 训练循环
    print("Start Training...")
    best_val_metric = 0 # 这里我们用 AUC 作为最佳指标
    num_patient_epochs = 0
    
    for epoch in range(config['train']['num_epochs']):
        # --- Eval ---
        val_metrics = eval_epoch(device, val_loader, model)
        target_val_metric = val_metrics['auc'] # 关注 AUC
        
        print(f"Epoch {epoch} | Val AUC: {val_metrics['auc']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        
        # --- Checkpoint 保存 ---
        if target_val_metric > best_val_metric:
            num_patient_epochs = 0
            best_val_metric = target_val_metric
            best_state_dict = {
                'config': config,
                'model_state_dict': model.state_dict(),
                'best_auc': best_val_metric
            }
            torch.save(best_state_dict, os.path.join(exp_name, 'best_tree_model.pth'))
        else:
            num_patient_epochs += 1

        # --- Logging ---
        log_dict = {'epoch': epoch}
        for k, v in val_metrics.items():
            log_dict[f'val/{k}'] = v
        wandb.log(log_dict)

        # --- Train ---
        train_metrics = train_epoch(device, train_loader, model, optimizer)
        wandb.log({f'train/{k}': v for k, v in train_metrics.items()})
        
        # Early Stopping
        if num_patient_epochs >= config['train']['patience']:
            print(f"Early stopping at epoch {epoch}")
            break

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Dataset name')
    args = parser.parse_args()
    main(args)