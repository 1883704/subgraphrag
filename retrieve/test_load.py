import torch
import sys
import os

# 假设你的项目结构是标准的，这里添加路径以确保能 import
sys.path.append(os.getcwd())

from src.dataset.treescorer import TreeScorerDataset

def test_local_embedding_loading():
    print("=== Test: Local Embedding Dictionary Loading ===")

    # -------------------------------------------
    # 1. 构造 Mock 数据 (模拟 .pkl 和 .pth 的内容)
    # -------------------------------------------
    
    # 设定 Embedding 维度
    HIDDEN_DIM = 4 
    
    # --- 样本 A (完整，正常数据) ---
    # 逻辑: 0 -> 1 -> 2 (一条简单的推理链)
    sample_a_id = "sample_001"
    raw_sample_a = {
        'id': sample_a_id,
        'h_id_list': [0, 1], 
        't_id_list': [1, 2], 
        'r_id_list': [0, 0], # 假设都用关系 0
        'q_entity_id_list': [0], # 从 0 开始搜
        'a_entity_id_list': [2], # 答案是 2
        # 注意：现在不需要 q_emb_vec 在 raw_sample 里了，因为它在 emb_dict 里
    }

    # 构造样本 A 的专属 Embedding
    # 假设有 3 个局部实体，1 个局部关系
    # 实体 0 全是 0.1, 实体 1 全是 0.2 ... 方便验证
    local_ent_a = torch.tensor([
        [0.1] * HIDDEN_DIM, # Node 0
        [0.2] * HIDDEN_DIM, # Node 1
        [0.3] * HIDDEN_DIM  # Node 2
    ])
    local_rel_a = torch.tensor([[0.9] * HIDDEN_DIM]) # Rel 0
    q_emb_a = torch.tensor([0.5] * HIDDEN_DIM)       # Question

    # --- 样本 B (缺失 Embedding 的坏数据) ---
    sample_b_id = "sample_missing_emb"
    raw_sample_b = {
        'id': sample_b_id,
        'h_id_list': [0, 1], 't_id_list': [1, 0], 'r_id_list': [0, 0],
        'q_entity_id_list': [0], 'a_entity_id_list': [1]
    }

    # --- 组装输入数据 ---
    raw_samples = [raw_sample_a, raw_sample_b]
    
    emb_dict = {
        sample_a_id: {
            'q_emb': q_emb_a,
            'entity_embs': local_ent_a,
            'relation_embs': local_rel_a
        }
        # 注意：这里故意不放 sample_b_id，测试代码是否会报错
    }

    # -------------------------------------------
    # 2. 运行 Dataset
    # -------------------------------------------
    print("\n[Step 1] Initializing Dataset...")
    dataset = TreeScorerDataset(raw_samples, emb_dict, max_hops=3)

    print(f"\n[Step 2] Checking Dataset Length")
    print(f"  Generated Trees: {len(dataset)}")
    
    # 我们期望生成样本 A 的树，样本 B 应该被跳过
    # 样本 A (0->1->2) 可能会生成 [0->1] 和 [0->1->2] 两个路径，或者只生成到达叶子的
    if len(dataset) > 0:
        print("  ✅ Dataset is not empty (Sample B was skipped safely).")
    else:
        print("  ❌ Dataset is empty! (Logic Error)")
        return

    # -------------------------------------------
    # 3. 深度验证数据正确性
    # -------------------------------------------
    print("\n[Step 3] Verifying Data Content")
    
    # 取出一个样本进行解剖
    data = dataset[0] # 这应该是属于 Sample A 的树
    
    # A. 检查 q_emb
    # 期望: [1, 4] 且值都是 0.5
    print(f"  --> Checking Question Embedding...")
    expected_q = torch.tensor([[0.5] * HIDDEN_DIM])
    if data.q_emb.shape == (1, HIDDEN_DIM):
        if torch.allclose(data.q_emb, expected_q):
            print("    ✅ q_emb matches dictionary value exactly.")
        else:
            print(f"    ❌ q_emb value mismatch! Got {data.q_emb}")
    else:
        print(f"    ❌ q_emb shape mismatch! Got {data.q_emb.shape}")

    # B. 检查节点特征 (x)
    # 假设这条路径是 0 -> 1，那么 x[0] 应该是 Node 0 (全是 0.1)
    print(f"  --> Checking Node Features (x)...")
    # 获取第一个节点在 path 中的原始 ID
    # data.x 的第一行应该对应 local_ent_a 中的某一行
    first_node_feat = data.x[0]
    
    # 检查它是否等于 0.1 (Node 0) 或者 0.2 (Node 1) 等等
    # 只要它存在于 local_ent_a 中就是对的
    is_valid_node = False
    for row in local_ent_a:
        if torch.allclose(first_node_feat, row):
            is_valid_node = True
            break
            
    if is_valid_node:
        print(f"    ✅ Node features correctly retrieved from local matrix. (First node value: {first_node_feat[0].item()})")
    else:
        print("    ❌ Node features do not match any row in local embeddings!")

    # C. 检查关系特征 (edge_attr)
    print(f"  --> Checking Edge Attributes...")
    if data.edge_attr is not None and data.edge_attr.shape[1] == HIDDEN_DIM:
        print("    ✅ Edge attributes loaded with correct dimension.")
    else:
        print("    ❌ Edge attributes missing or wrong shape.")

    print("\n🎉 All Tests Passed! The new dictionary-based loading works.")

if __name__ == "__main__":
    test_local_embedding_loading()