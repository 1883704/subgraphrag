import os
import torch
import pandas as pd
import glob

# 设置 Huggingface 镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_URL'] = 'https://hf-mirror.com'

from datasets import load_dataset
from tqdm import tqdm

from src.config.emb import load_yaml
from src.dataset.emb import EmbInferDataset


def print_dataset_info(dataset_name, train_set, val_set, test_set, entity_identifiers):
    """打印数据集的详细信息"""
    print("=" * 60)
    print(f"WebQSP 数据集信息统计")
    print("=" * 60)

    # 数据集大小
    print(f"训练集大小: {len(train_set)}")
    print(f"验证集大小: {len(val_set)}")
    print(f"测试集大小: {len(test_set)}")
    print(f"总样本数: {len(train_set) + len(val_set) + len(test_set)}")
    print()

    # 统计实体和关系
    all_entities = set()
    all_relations = set()
    all_questions = []

    # 从训练集中统计
    for i in range(min(100, len(train_set))):  # 只统计前100个样本
        sample = train_set[i]
        question = sample['question']
        triples = sample['graph']
        all_questions.append(question)

        for (h, r, t) in triples:
            all_entities.add(h)
            all_entities.add(t)
            all_relations.add(r)

    print(f"实体总数: {len(all_entities)}")
    print(f"关系总数: {len(all_relations)}")
    print(f"非文本实体标识符数量: {len(entity_identifiers)}")
    print()

    # 打印一些示例问题
    print("示例问题:")
    for i, question in enumerate(all_questions[:5]):
        print(f"  {i + 1}. {question}")
    print()

    # 打印一些示例实体
    print("示例实体 (前10个):")
    entity_list = sorted(list(all_entities))[:10]
    for i, entity in enumerate(entity_list):
        print(f"  {i + 1}. {entity}")
    print()

    # 打印一些示例关系
    print("示例关系 (前10个):")
    relation_list = sorted(list(all_relations))[:10]
    for i, relation in enumerate(relation_list):
        print(f"  {i + 1}. {relation}")
    print()

    # 统计三元组数量
    total_triples = 0
    for i in range(min(100, len(train_set))):
        sample = train_set[i]
        triples = sample['graph']
        total_triples += len(triples)

    print(f"前100个样本的三元组总数: {total_triples}")
    print(f"平均每个样本的三元组数: {total_triples / min(100, len(train_set)):.2f}")
    print()

    # 统计答案实体
    answer_entities = set()
    for i in range(min(100, len(train_set))):
        sample = train_set[i]
        answer_entities.update(sample['a_entity'])

    print(f"前100个样本的答案实体总数: {len(answer_entities)}")
    print()

    print("=" * 60)


def load_local_parquet_sharded(base_path, split_name):
    """加载分片的 parquet 文件并转换为标准格式"""
    # 构建文件模式，匹配所有分片
    pattern = os.path.join(base_path, f'{split_name}-*.parquet')
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No parquet files found for pattern: {pattern}")

    print(f"Loading {split_name} data from {len(files)} files: {[os.path.basename(f) for f in files]}")

    # 读取所有分片并合并
    all_data = []
    for file in files:
        df = pd.read_parquet(file)
        all_data.append(df)

    # 合并所有 DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)

    # 转换为标准格式，确保与 huggingface 数据集格式一致
    data = []
    for _, row in combined_df.iterrows():
        # 处理每个样本，确保数据格式正确
        sample = {}

        # 复制基本字段
        for key in row.index:
            value = row[key]

            # 处理列表类型的字段
            if key in ['q_entity', 'a_entity', 'answer']:
                # 检查是否为空值
                if hasattr(value, '__iter__') and not isinstance(value, str):
                    # 如果是数组类型，检查是否全为空
                    if hasattr(value, 'any'):
                        is_empty = value.any() == False if hasattr(value, 'dtype') and value.dtype == bool else pd.isna(
                            value).all()
                    else:
                        is_empty = all(pd.isna(v) for v in value)
                else:
                    is_empty = pd.isna(value)

                if is_empty:
                    sample[key] = []
                elif hasattr(value, 'tolist'):
                    sample[key] = value.tolist()
                elif isinstance(value, str):
                    # 如果是字符串，尝试解析为列表
                    try:
                        import ast
                        sample[key] = ast.literal_eval(value)
                    except:
                        sample[key] = [value] if value else []
                elif not isinstance(value, list):
                    sample[key] = [value] if value is not None else []
                else:
                    sample[key] = value
            else:
                # 其他字段直接复制
                if hasattr(value, '__iter__') and not isinstance(value, str):
                    # 如果是数组类型，检查是否全为空
                    if hasattr(value, 'any'):
                        is_empty = value.any() == False if hasattr(value, 'dtype') and value.dtype == bool else pd.isna(
                            value).all()
                    else:
                        is_empty = all(pd.isna(v) for v in value)
                else:
                    is_empty = pd.isna(value)

                if is_empty:
                    sample[key] = None
                else:
                    sample[key] = value

        # 确保 a_entity 和 answer 字段存在且格式一致
        if 'a_entity' not in sample:
            sample['a_entity'] = sample.get('answer', [])
        if 'answer' not in sample:
            sample['answer'] = sample.get('a_entity', [])

        # 确保 q_entity 字段存在
        if 'q_entity' not in sample:
            sample['q_entity'] = []

        # 确保 graph 字段是列表格式
        if 'graph' in sample:
            graph = sample['graph']
            if hasattr(graph, 'tolist'):
                sample['graph'] = graph.tolist()
            elif not isinstance(graph, list):
                sample['graph'] = [graph] if graph is not None else []

        data.append(sample)

    print(f"Loaded {len(data)} samples for {split_name}")
    return data


def load_pickle_split(base_dirs, split_names):
    for base_dir in base_dirs:
        for split_name in split_names:
            path = os.path.join(base_dir, f"{split_name}.pkl")
            if os.path.exists(path):
                import pickle as _pkl
                with open(path, "rb") as f:
                    return _pkl.load(f), path

    tried = [
        os.path.join(base_dir, f"{split_name}.pkl")
        for base_dir in base_dirs
        for split_name in split_names
    ]
    raise FileNotFoundError(
        "Could not find split pickle. Tried:\n  " + "\n  ".join(tried)
    )


def get_emb(subset, text_encoder, save_file):
    emb_dict = dict()
    for i in tqdm(range(len(subset))):
        item = subset[i]
        if len(item) == 4:
            id, q_text, text_entity_list, relation_list = item
            question_stem = q_text
            option_texts = []
        else:
            id, q_text, question_stem, text_entity_list, relation_list, option_texts = item

        q_emb, entity_embs, relation_embs = text_encoder(
            q_text, text_entity_list, relation_list)
        question_stem_emb = text_encoder.embed([question_stem])
        option_embs = text_encoder.embed(option_texts)
        emb_dict_i = {
            'q_emb': q_emb,
            'question_stem_emb': question_stem_emb,
            'option_embs': option_embs,
            'entity_embs': entity_embs,
            'relation_embs': relation_embs
        } 
        emb_dict[id] = emb_dict_i

    torch.save(emb_dict, save_file)


def main(args):
    # Modify the config file for advanced settings and extensions.
    config_file = f'configs/emb/gte-large-en-v1.5/{args.dataset}.yaml'
    config = load_yaml(config_file)

    torch.set_num_threads(config['env']['num_threads'])

    if args.dataset == 'cwq':
        input_file = os.path.join('rmanluo', 'RoG-cwq')
        # 对于 cwq，保持原来的 load_dataset 方式
        train_set = load_dataset(input_file, split='train')
        val_set = load_dataset(input_file, split='validation')
        test_set = load_dataset(input_file, split='test')
    elif args.dataset == 'webqsp':
        # 对于 webqsp，使用本地 parquet 文件
        input_dir = os.path.join('data', 'webqsp')
        print(f"Loading WebQSP data from local directory: {input_dir}")

        train_set = load_local_parquet_sharded(input_dir, 'train')
        val_set = load_local_parquet_sharded(input_dir, 'validation')
        test_set = load_local_parquet_sharded(input_dir, 'test')
    elif args.dataset == 'chatdoctor5k':
        raw_dirs = [
            os.path.join('data_files', 'chatdoctor5k', 'raw'),
            os.path.join('data', 'chatdoctor5k'),
        ]
        train_set, train_path = load_pickle_split(raw_dirs, ['train'])
        val_set, val_path = load_pickle_split(raw_dirs, ['val', 'validation'])
        test_set, test_path = load_pickle_split(raw_dirs, ['test'])
        print("Loading chatdoctor5k raw data from:")
        print(f"  train: {train_path}")
        print(f"  val:   {val_path}")
        print(f"  test:  {test_path}")
    else:
        raw_dirs = [
            os.path.join('data_files', args.dataset, 'raw'),
            os.path.join('data', args.dataset),
        ]
        train_set, train_path = load_pickle_split(raw_dirs, ['train'])
        val_set, val_path = load_pickle_split(raw_dirs, ['val', 'validation'])
        test_set, test_path = load_pickle_split(raw_dirs, ['test'])
        print(f"Loading {args.dataset} raw data from:")
        print(f"  train: {train_path}")
        print(f"  val:   {val_path}")
        print(f"  test:  {test_path}")

    entity_identifiers = []
    with open(config['entity_identifier_file'], 'r') as f:
        for line in f:
            entity_identifiers.append(line.strip())
    entity_identifiers = set(entity_identifiers)

    # 打印数据集信息
    if args.dataset == 'webqsp':
        print_dataset_info(args.dataset, train_set, val_set, test_set, entity_identifiers)

    save_dir = f'data_files/{args.dataset}/processed'
    os.makedirs(save_dir, exist_ok=True)
    N_train = 100000
    N_val = 200000
    N_test = 500000
    # pickle 列表不支持 .select，统一用切片
    small_train_set = train_set[:min(N_train, len(train_set))]
    small_val_set = val_set[:min(N_val, len(val_set))]
    small_test_set = test_set[:min(N_test, len(test_set))]
    train_set = EmbInferDataset(
        small_train_set,
        entity_identifiers,
        os.path.join(save_dir, 'train.pkl'))

    val_set = EmbInferDataset(
        small_val_set,
        entity_identifiers,
        os.path.join(save_dir, 'val.pkl'))

    test_set = EmbInferDataset(
        small_test_set,
        entity_identifiers,
        os.path.join(save_dir, 'test.pkl'),
        skip_no_topic=False,
        skip_no_ans=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    text_encoder_name = config['text_encoder']['name']
    if text_encoder_name == 'gte-large-en-v1.5':
        from src.model.text_encoders import GTELargeEN
        text_encoder = GTELargeEN(device)
    else:
        raise NotImplementedError(text_encoder_name)

    emb_save_dir = f'data_files/{args.dataset}/emb/{text_encoder_name}'
    os.makedirs(emb_save_dir, exist_ok=True)

    # Check if embedding files already exist and skip if they do
    train_emb_path = os.path.join(emb_save_dir, 'train.pth')
    val_emb_path = os.path.join(emb_save_dir, 'val.pth')
    test_emb_path = os.path.join(emb_save_dir, 'test.pth')

    if os.path.exists(train_emb_path):
        print(f"Train embeddings already exist at {train_emb_path}, skipping...")
    else:
        get_emb(train_set, text_encoder, train_emb_path)

    if os.path.exists(val_emb_path):
        print(f"Validation embeddings already exist at {val_emb_path}, skipping...")
    else:
        get_emb(val_set, text_encoder, val_emb_path)

    if os.path.exists(test_emb_path):
        print(f"Test embeddings already exist at {test_emb_path}, skipping...")
    else:
        get_emb(test_set, text_encoder, test_emb_path)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('Text Embedding Pre-Computation for Retrieval')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Dataset name')
    args = parser.parse_args()

    main(args)
