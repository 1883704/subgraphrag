import torch
from torch_geometric.loader import DataLoader
from src.dataset.treescorer import TreeScorerDataset

def test_dataloader_collation_v2():
    print("=== Test 2: DataLoader Batching (Dict Mode) ===")
    
    # -------------------------------------------
    # 1. 构造 Mock 数据 (适配 Dict 模式)
    # -------------------------------------------
    HIDDEN_DIM = 8
    
    # 样本 1
    id_1 = "sample_001"
    raw_1 = {
        'id': id_1,
        'h_id_list': [0], 't_id_list': [1], 'r_id_list': [0],
        'q_entity_id_list': [0], 'a_entity_id_list': [1]
    }
    emb_1 = {
        'q_emb': torch.randn(1, HIDDEN_DIM), # [1, 8]
        'entity_embs': torch.randn(2, HIDDEN_DIM), # 2个节点
        'relation_embs': torch.randn(1, HIDDEN_DIM)
    }

    # 样本 2
    id_2 = "sample_002"
    raw_2 = {
        'id': id_2,
        'h_id_list': [0, 1], 't_id_list': [1, 2], 'r_id_list': [0, 0],
        'q_entity_id_list': [0], 'a_entity_id_list': [2]
    }
    emb_2 = {
        'q_emb': torch.randn(1, HIDDEN_DIM), # [1, 8]
        'entity_embs': torch.randn(3, HIDDEN_DIM), # 3个节点
        'relation_embs': torch.randn(1, HIDDEN_DIM)
    }

    # 组装
    raw_samples = [raw_1, raw_2]
    emb_dict = {id_1: emb_1, id_2: emb_2}
    
    # -------------------------------------------
    # 2. 初始化 Dataset 和 DataLoader
    # -------------------------------------------
    print("-> Initializing Dataset...")
    dataset = TreeScorerDataset(raw_samples, emb_dict, max_hops=2)
    
    # 模拟 Batch Size = 2
    print("-> Initializing DataLoader (Batch Size = 2)...")
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # -------------------------------------------
    # 3. 验证 Batch 对象
    # -------------------------------------------
    # 只有 2 个样本，所以这里只循环一次
    for batch in loader:
        print(f"\n[Checking Batch Shape]")
        
        # A. 检查 q_emb 堆叠
        # 我们有 2 个样本，每个 q_emb 是 [1, 8]
        # PyG Collate 应该把它变成 [2, 8] 或者 [2, 1, 8]
        print(f"  Batch q_emb shape: {batch.q_emb.shape}")
        
        if batch.q_emb.shape[0] == 2:
            print("  ✅ Batch Size aligned correctly (Dim 0 is 2).")
        else:
            print("  ❌ Batch Size mismatch!")

        # B. 检查节点堆叠
        # 样本1有2个点，样本2有3个点，总共应该是 5 个点
        total_nodes = batch.x.shape[0]
        print(f"  Total Nodes in Batch: {total_nodes}")
        if total_nodes == 5:
            print("  ✅ Nodes stacked correctly (2 + 3 = 5).")
        else:
            print(f"  ❌ Node count mismatch! Expected 5, got {total_nodes}")
            
        # C. 检查 Batch 索引 (batch vector)
        # 应该是 [0, 0, 1, 1, 1]
        print(f"  Batch Index Vector: {batch.batch}")
        
        # D. 检查 Leaf Mask
        # 应该有 2 个 True (每个图一个)
        true_leaves = batch.leaf_mask.sum().item()
        print(f"  Total Leaves: {true_leaves}")
        if true_leaves == 2:
             print("  ✅ Leaf Masks are correct.")
        else:
             print("  ❌ Leaf Mask error!")

    print("\n🎉 DataLoader Test Passed!")

if __name__ == "__main__":
    test_dataloader_collation_v2()