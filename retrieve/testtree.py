# ... import ...

def load_data_and_embs(dataset_name, split):
    """
    读取 processed pkl 和 emb pth
    """
    base_dir = f'data_files/{dataset_name}'
    
    # 1. 加载原始样本 (Raw Samples)
    pkl_path = os.path.join(base_dir, 'processed', f'{split}.pkl')
    print(f"Loading structure from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        # 假设这里是个 List[Dict]
        raw_samples = pickle.load(f)
        
    # 2. 加载 Embeddings (Emb Dict)
    # 根据你的代码，路径是 data_files/{dataset}/emb/{encoder_name}/{split}.pth
    encoder_name = 'gte-large-en-v1.5' # 或者从 config 读取
    emb_path = os.path.join(base_dir, 'emb', encoder_name, f'{split}.pth')
    
    print(f"Loading embeddings from {emb_path}...")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embedding file not found: {emb_path}")
        
    # 这里的 emb_dict 就是你 get_emb 生成的字典
    emb_dict = torch.load(emb_path, map_location='cpu')
    
    return raw_samples, emb_dict

def main(args):
    # ... 配置加载代码 ...
    
    # --- 数据加载 ---
    train_raw, train_embs = load_data_and_embs(args.dataset, 'train')
    val_raw, val_embs = load_data_and_embs(args.dataset, 'val') # 或者是 validation
    
    # --- Dataset 实例化 ---
    # 注意：现在不需要传 global_ent_embs 了，传对应的字典
    train_set = TreeScorerDataset(
        raw_samples=train_raw, 
        emb_dict=train_embs, 
        max_hops=config['data']['max_hops']
    )
    
    val_set = TreeScorerDataset(
        raw_samples=val_raw, 
        emb_dict=val_embs, 
        max_hops=config['data']['max_hops']
    )
    
    # ... DataLoader 和 训练代码不变 ...