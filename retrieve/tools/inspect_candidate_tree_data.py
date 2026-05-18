import argparse
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset.candidate_treescorer import CandidateTreeScorerDataset


def load_split(dataset_name, text_encoder_name, split):
    base_dir = os.path.join("data_files", dataset_name)
    processed_path = os.path.join(base_dir, "processed", f"{split}.pkl")
    emb_path = os.path.join(base_dir, "emb", text_encoder_name, f"{split}.pth")

    with open(processed_path, "rb") as f:
        rows = pickle.load(f)
    emb_dict = torch.load(emb_path, map_location="cpu")
    return rows, emb_dict, processed_path, emb_path


def main():
    parser = argparse.ArgumentParser(
        description="Inspect candidate-aware TreeScorer data alignment."
    )
    parser.add_argument("-d", "--dataset", default="medmcqa_primekg")
    parser.add_argument("--text_encoder", default="gte-large-en-v1.5")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--label_base", default="zero", choices=["zero", "one"])
    args = parser.parse_args()

    rows, emb_dict, processed_path, emb_path = load_split(
        args.dataset,
        args.text_encoder,
        args.split,
    )
    rows = rows[: args.max_samples]

    dataset = CandidateTreeScorerDataset(
        rows,
        emb_dict,
        max_hops=args.max_hops,
        max_paths_per_root=80,
        max_paths_per_sample=240,
        max_paths_per_candidate=16,
        max_pos_per_candidate=8,
        max_hard_neg_per_candidate=8,
        label_base=args.label_base,
        mode="eval",
        use_cache=False,
    )

    print(f"processed: {processed_path}")
    print(f"emb:       {emb_path}")
    print(f"raw rows inspected: {len(rows)}")
    print(f"candidate path examples: {len(dataset)}")
    print(f"dataset stats: {dataset.stats}")

    if len(dataset) == 0:
        return

    first = dataset[0]
    print("\nfirst example shapes:")
    print(f"  x:             {tuple(first.x.shape)}")
    print(f"  edge_attr:     {tuple(first.edge_attr.shape)}")
    print(f"  edge_extra:    {tuple(first.edge_extra.shape)}")
    print(f"  node_extra:    {tuple(first.node_extra.shape)}")
    print(f"  path_extra:    {tuple(first.path_extra.shape)}")
    print(f"  q_emb:         {tuple(first.q_emb.shape)}")
    print(f"  candidate_emb: {tuple(first.candidate_emb.shape)}")

    emb_dim = first.q_emb.shape[-1]
    checks = {
        "x_dim_matches_q": first.x.shape[-1] == emb_dim,
        "edge_attr_dim_matches_q": first.edge_attr.shape[-1] == emb_dim,
        "candidate_dim_matches_q": first.candidate_emb.shape[-1] == emb_dim,
        "node_extra_width": first.node_extra.shape[-1]
        == CandidateTreeScorerDataset.node_extra_size,
        "edge_extra_width": first.edge_extra.shape[-1]
        == CandidateTreeScorerDataset.edge_extra_size,
        "path_extra_width": first.path_extra.shape[-1]
        == CandidateTreeScorerDataset.path_extra_size,
    }
    print("\nalignment checks:")
    for key, value in checks.items():
        print(f"  {key}: {value}")

    if not all(checks.values()):
        raise SystemExit("Alignment check failed.")

    qid_to_candidates = defaultdict(set)
    labels = Counter()
    for item in dataset:
        qid_to_candidates[int(item.qid.item())].add(int(item.candidate_idx.item()))
        labels[int(float(item.y.item()) > 0.5)] += 1

    missing_candidate_questions = sum(
        int(len(candidates) < 4) for candidates in qid_to_candidates.values()
    )
    print("\nquestion/candidate coverage:")
    print(f"  questions: {len(qid_to_candidates)}")
    print(f"  questions_missing_any_candidate: {missing_candidate_questions}")
    print(f"  support labels: {dict(labels)}")


if __name__ == "__main__":
    main()
