import os
import re
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from .prepare_prompts import unique_preserve_order


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_ground_truth(row):
    if row.get("ground_truth") is not None:
        return row

    for key in ("a_entity_in_graph", "a_entity", "answer"):
        value = row.get(key)
        if value:
            row["ground_truth"] = value if isinstance(value, list) else [value]
            return row

    row["ground_truth"] = []
    return row


def raw_item_from_subgraph(sample):
    row = normalize_local_subgraph_sample(sample)
    return normalize_ground_truth({
        "id": row["id"],
        "question": row.get("question", ""),
        "ground_truth": row.get("a_entity", []),
        "a_entity": row.get("a_entity", []),
        "graph": row.get("graph", []),
    })


def graph_from_processed_sample(sample):
    entity_list = sample.get("text_entity_list", []) + sample.get("non_text_entity_list", [])
    relation_list = sample.get("relation_list", [])
    graph = []
    for h_id, r_id, t_id in zip(
        sample.get("h_id_list", []),
        sample.get("r_id_list", []),
        sample.get("t_id_list", []),
    ):
        try:
            graph.append((
                entity_list[h_id],
                relation_list[r_id],
                entity_list[t_id],
            ))
        except IndexError:
            continue
    return graph


def normalize_local_subgraph_sample(sample):
    row = dict(sample)
    if "graph" not in row:
        row["graph"] = graph_from_processed_sample(row)
    row["graph"] = [tuple(each) for each in row.get("graph", [])]
    if "a_entity" not in row:
        row["a_entity"] = row.get("answer", [])
    return row


def local_subgraph_candidates(dataset_name, split):
    split_aliases = [split]
    if split == "validation":
        split_aliases.append("val")
    elif split == "val":
        split_aliases.append("validation")

    roots = [
        os.path.join("..", "retrieve", "data_files", dataset_name),
        os.path.join("data_files", dataset_name),
        os.path.join("..", "retrieve", "data", dataset_name),
        os.path.join("data", dataset_name),
    ]
    stages = ["processed", "raw", ""]

    candidates = []
    for root in roots:
        for stage in stages:
            base = os.path.join(root, stage) if stage else root
            for split_name in split_aliases:
                candidates.append(os.path.join(base, f"{split_name}.pkl"))
                candidates.append(os.path.join(base, f"{split_name}.jsonl"))
                candidates.append(os.path.join(base, f"{split_name}.json"))
    return candidates


def load_local_subgraphs(dataset_name, split):
    for path in local_subgraph_candidates(dataset_name, split):
        if not os.path.exists(path):
            continue

        if path.endswith(".pkl"):
            with open(path, "rb") as f:
                rows = pickle.load(f)
        elif path.endswith(".jsonl"):
            rows = load_jsonl(path)
        else:
            with open(path, "r", encoding="utf-8") as f:
                rows = json.load(f)

        print(f"Loaded local subgraphs from: {path}")
        return [normalize_local_subgraph_sample(row) for row in rows]

    return None


def get_subgraphs(dataset_name, split):
    local_rows = load_local_subgraphs(dataset_name, split)
    if local_rows is not None:
        return local_rows

    input_file = os.path.join("rmanluo", f"RoG-{dataset_name}")
    return load_dataset(input_file, split=split)


def extract_reasoning_paths(text):
    pattern = r"Reasoning Paths:(.*?)\n\nQuestion:"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        reasoning_paths = match.group(1).strip()
        return reasoning_paths
    else:
        return None


def add_good_triplets_from_rog(data):
    print("Adding good triplets from ROG...")
    total_good_triplets = 0
    total_good_triplets_in_graph = 0
    total_good_triplets_not_in_graph = 0
    for idx, each_qa in enumerate(tqdm(data)):
        all_paths = extract_reasoning_paths(each_qa["input"]).split("\n")
        data[idx]["good_paths_rog"] = all_paths
        all_good_triplets = []
        for each_path in all_paths:
            each_path = each_path.split(" -> ")
            good_triplets = []
            i = 0
            while i < len(each_path):
                if i + 2 < len(each_path):
                    triplet = (each_path[i], each_path[i + 1], each_path[i + 2])
                    temp_triplet = (each_path[i + 2], each_path[i + 1], each_path[i])
                    total_good_triplets += 1
                    # if triplet in each_qa["graph"] or temp_triplet in each_qa["graph"]:
                    #     total_good_triplets_in_graph += 1
                    # else:
                    #     total_good_triplets_not_in_graph += 1
                    good_triplets.append(triplet)
                i += 2
            all_good_triplets.extend(good_triplets)
        data[idx]["good_triplets_rog"] = unique_preserve_order(all_good_triplets)
    return data


def add_gt_if_not_present(triple_score_dict):
    st = [','.join(list(each)[:3]) for each in triple_score_dict['scored_triples']]
    tt = [','.join(list(each)[:3]) for each in triple_score_dict['target_relevant_triples']]
    for each in tt:
        if each in st:
            continue
        else:
            # put at the beginning
            triple_score_dict["scored_triples"].insert(0, tuple(each.split(',')))
    return triple_score_dict["scored_triples"]


def add_scored_triplets(data, score_dict_path, prompt_mode):
    print("Adding scored triplets...")
    new_data = []
    cnt = 0
    triple_score_dict = torch.load(score_dict_path, weights_only=False)

    running_baselines = False
    if 'triples' in triple_score_dict[next(iter(triple_score_dict))]:
        running_baselines = True
        for k, v in tqdm(triple_score_dict.items()):
            triple_score_dict[k]['scored_triples'] = v['triples']

    for each_qa in tqdm(data):
        if each_qa["id"] in triple_score_dict:
            if 'gt' in prompt_mode:
                scored_triples = add_gt_if_not_present(triple_score_dict[each_qa["id"]])
            else:
                scored_triples = triple_score_dict[each_qa["id"]]["scored_triples"]
            each_qa['scored_triplets'] = scored_triples
            new_data.append(each_qa)
        else:
            print(f"Triplets not found for {each_qa['id']}")
            if running_baselines:
                each_qa['scored_triplets'] = [('', '', '')]
                new_data.append(each_qa)
            elif 'gt' not in prompt_mode:
                raise ValueError
            else:
                cnt += 1
    print(f"Triplets not found for {cnt} questions")
    return new_data


def load_tree_results(tree_result_path):
    if tree_result_path.endswith(".jsonl"):
        tree_dict = {}
        with open(tree_result_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = row.pop("id")
                tree_dict[sample_id] = row
        return tree_dict

    return torch.load(tree_result_path, weights_only=False)


def add_scored_trees(data, tree_result_path):
    if tree_result_path is None:
        raise ValueError("Tree prompt modes require -p/--score_dict_path to point to a tree retrieval result.")

    print("Adding scored reasoning trees...")
    tree_dict = load_tree_results(tree_result_path)
    new_data = []
    missing = 0

    for each_qa in tqdm(data):
        tree_sample = tree_dict.get(each_qa["id"])
        if tree_sample is None:
            missing += 1
            continue

        each_qa["scored_trees"] = tree_sample.get("scored_trees", [])
        each_qa["q_entity_in_graph"] = tree_sample.get("q_entity_in_graph", [])
        each_qa["a_entity_in_graph"] = tree_sample.get("a_entity_in_graph", [])
        each_qa["max_path_length"] = tree_sample.get("max_path_length")
        new_data.append(each_qa)

    print(f"Tree results not found for {missing} questions")
    return new_data


def sample_random_triplets(data, num_triplets, seed=0):
    print(f"Sampling {num_triplets} random triplets...")
    np.random.seed(seed)
    for idx, each_qa in enumerate(tqdm(data)):
        all_triplets = np.array(each_qa["graph"])
        sampled_triplets = np.random.permutation(all_triplets)[:num_triplets]
        data[idx][f"sampled_triplets_{num_triplets}"] = sampled_triplets.tolist()
    return data


def get_data(dataset_name, pred_file_path, score_dict_path, split, prompt_mode, seed=0, triplets_to_sample=[50, 100, 200, 300]):
    subgraphs = None

    if pred_file_path and os.path.exists(pred_file_path):
        raw_data = load_jsonl(pred_file_path)
    elif "tree" in prompt_mode and score_dict_path:
        print(
            "Prediction file not found; using tree retrieval result as local "
            "reasoning input."
        )
        tree_dict = load_tree_results(score_dict_path)
        raw_data = []
        for sample_id, row in tree_dict.items():
            item = {"id": sample_id}
            item.update(row)
            raw_data.append(normalize_ground_truth(item))
    elif "rog" not in prompt_mode:
        print(
            "Prediction file not found; using local subgraphs as reasoning "
            "input."
        )
        subgraphs = get_subgraphs(dataset_name, split)
        raw_data = [raw_item_from_subgraph(row) for row in subgraphs]
    else:
        raise FileNotFoundError(f"Prediction file not found: {pred_file_path}")

    print("Loading subgraphs...")
    if subgraphs is None:
        subgraphs = get_subgraphs(dataset_name, split)
    subgraph_by_id = {row["id"]: row for row in subgraphs if "id" in row}

    print("Adding subgraphs to data...")
    data = []
    for i, each_qa in enumerate(tqdm(raw_data)):
        subgraph = subgraph_by_id.get(each_qa["id"])
        if subgraph is None and i < len(subgraphs):
            subgraph = subgraphs[i]
        if subgraph is None:
            each_qa.setdefault("graph", [])
            each_qa.setdefault("a_entity", each_qa.get("ground_truth", []))
        else:
            each_qa["graph"] = [tuple(each) for each in subgraph.get("graph", [])]
            each_qa['a_entity'] = subgraph.get('a_entity', each_qa.get("ground_truth", []))
        each_qa = normalize_ground_truth(each_qa)
        data.append(each_qa)

    if 'tree' in prompt_mode:
        data = add_scored_trees(data, score_dict_path)
        return data

    if 'rog' in prompt_mode:
        data = add_good_triplets_from_rog(data)
    data = add_scored_triplets(data, score_dict_path, prompt_mode)
    # for num_triplets in triplets_to_sample:
    #     data = sample_random_triplets(data, num_triplets, seed)

    return data
