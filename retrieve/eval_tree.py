import json
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pandas as pd


def parse_k_list(k_list):
    return [int(k) for k in k_list.split(",") if k.strip()]


def normalize_entity(entity, lower):
    if not isinstance(entity, str):
        entity = str(entity)
    entity = entity.strip()
    return entity.lower() if lower else entity


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num}: {exc}") from exc
    return rows


def get_answers(sample, answer_field, lower):
    answers = sample.get(answer_field) or []
    if answer_field == "a_entity_in_graph" and not answers:
        answers = sample.get("a_entity") or []
    return {
        normalize_entity(answer, lower)
        for answer in answers
        if str(answer).strip()
    }


def tree_entities(tree, lower):
    entities = set()

    for key in ("root", "leaf"):
        if tree.get(key):
            entities.add(normalize_entity(tree[key], lower))

    for path in tree.get("paths", []):
        for key in ("root", "leaf"):
            if path.get(key):
                entities.add(normalize_entity(path[key], lower))
        for node in path.get("nodes", []):
            entities.add(normalize_entity(node, lower))
        for triple in path.get("triples", []):
            if len(triple) >= 3:
                entities.add(normalize_entity(triple[0], lower))
                entities.add(normalize_entity(triple[2], lower))

    for triple in tree.get("triples", []):
        if len(triple) >= 3:
            entities.add(normalize_entity(triple[0], lower))
            entities.add(normalize_entity(triple[2], lower))

    return entities


def tree_leaf_entities(tree, lower):
    leaves = set()
    if tree.get("leaf"):
        leaves.add(normalize_entity(tree["leaf"], lower))
    for path in tree.get("paths", []):
        if path.get("leaf"):
            leaves.add(normalize_entity(path["leaf"], lower))
    return leaves


def canonical_tree_key(tree, lower):
    triples = []
    for triple in tree.get("triples", []):
        if len(triple) >= 3:
            triples.append(tuple(normalize_entity(part, lower) for part in triple[:3]))

    if triples:
        return ("triples", tuple(sorted(triples)))

    paths = []
    for path in tree.get("paths", []):
        nodes = tuple(normalize_entity(node, lower) for node in path.get("nodes", []))
        triples_in_path = tuple(
            tuple(normalize_entity(part, lower) for part in triple[:3])
            for triple in path.get("triples", [])
            if len(triple) >= 3
        )
        paths.append((nodes, triples_in_path))
    return ("paths", tuple(sorted(paths)))


def duplicate_rate(trees, lower):
    if not trees:
        return 0.0
    unique_keys = {canonical_tree_key(tree, lower) for tree in trees}
    return (len(trees) - len(unique_keys)) / len(trees)


def evaluate(rows, k_list, answer_field, lower):
    metrics = defaultdict(list)
    stats = {
        "samples": len(rows),
        "samples_with_answers": 0,
        "samples_with_trees": 0,
        "evaluated_samples": 0,
    }

    for sample in rows:
        answers = get_answers(sample, answer_field, lower)
        trees = sample.get("scored_trees") or []

        if answers:
            stats["samples_with_answers"] += 1
        if trees:
            stats["samples_with_trees"] += 1
        if not answers or not trees:
            continue

        stats["evaluated_samples"] += 1

        for k in k_list:
            top_trees = trees[:k]
            if not top_trees:
                continue

            entities_k = set()
            leaves_k = set()
            relevant_tree_count = 0
            path_count = 0
            triple_count = 0

            for tree in top_trees:
                entities = tree_entities(tree, lower)
                leaves = tree_leaf_entities(tree, lower)
                entities_k.update(entities)
                leaves_k.update(leaves)
                relevant_tree_count += int(bool(answers & entities))
                path_count += len(tree.get("paths", []))
                triple_count += len(tree.get("triples", []))

            answer_recall = len(answers & entities_k) / len(answers)
            leaf_answer_recall = len(answers & leaves_k) / len(answers)

            metrics[f"tree_hit@{k}"].append(float(answer_recall > 0))
            metrics[f"tree_answer_recall@{k}"].append(answer_recall)
            metrics[f"tree_all_answer_hit@{k}"].append(float(answer_recall == 1.0))
            metrics[f"leaf_hit@{k}"].append(float(leaf_answer_recall > 0))
            metrics[f"leaf_answer_recall@{k}"].append(leaf_answer_recall)
            metrics[f"relevant_tree_rate@{k}"].append(
                relevant_tree_count / len(top_trees)
            )
            metrics[f"duplicate_tree_rate@{k}"].append(
                duplicate_rate(top_trees, lower)
            )
            metrics[f"avg_paths_per_tree@{k}"].append(
                path_count / len(top_trees)
            )
            metrics[f"avg_triples_per_tree@{k}"].append(
                triple_count / len(top_trees)
            )

    return stats, {metric: float(np.mean(values)) for metric, values in metrics.items()}


def main(args):
    rows = load_jsonl(args.path)
    k_list = parse_k_list(args.k_list)
    stats, metrics = evaluate(
        rows,
        k_list=k_list,
        answer_field=args.answer_field,
        lower=not args.case_sensitive,
    )

    print("Tree-level evaluation")
    print(f"path: {args.path}")
    print(
        "samples: {samples}, with_answers: {samples_with_answers}, "
        "with_trees: {samples_with_trees}, evaluated: {evaluated_samples}".format(
            **stats
        )
    )

    table = {
        "K": k_list,
        "tree_hit": [round(metrics.get(f"tree_hit@{k}", 0.0), 3) for k in k_list],
        "tree_answer_recall": [
            round(metrics.get(f"tree_answer_recall@{k}", 0.0), 3)
            for k in k_list
        ],
        "leaf_hit": [round(metrics.get(f"leaf_hit@{k}", 0.0), 3) for k in k_list],
        "leaf_answer_recall": [
            round(metrics.get(f"leaf_answer_recall@{k}", 0.0), 3)
            for k in k_list
        ],
        "duplicate_tree_rate": [
            round(metrics.get(f"duplicate_tree_rate@{k}", 0.0), 3)
            for k in k_list
        ],
        "relevant_tree_rate": [
            round(metrics.get(f"relevant_tree_rate@{k}", 0.0), 3)
            for k in k_list
        ],
        "avg_paths_per_tree": [
            round(metrics.get(f"avg_paths_per_tree@{k}", 0.0), 3)
            for k in k_list
        ],
        "avg_triples_per_tree": [
            round(metrics.get(f"avg_triples_per_tree@{k}", 0.0), 3)
            for k in k_list
        ],
    }
    print(pd.DataFrame(table).to_string(index=False))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=True,
        help="Path to tree_retrieval_result.jsonl",
    )
    parser.add_argument(
        "--k_list",
        type=str,
        default="1,3,5",
        help="Comma-separated list of K values for top-K tree evaluation",
    )
    parser.add_argument(
        "--answer_field",
        type=str,
        default="a_entity_in_graph",
        choices=["a_entity_in_graph", "a_entity"],
        help="Answer field used for matching tree entities",
    )
    parser.add_argument(
        "--case_sensitive",
        action="store_true",
        help="Use case-sensitive entity-name matching",
    )
    main(parser.parse_args())
