import os
import json
import random
import pickle
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple, Union


def ensure_list_from_str(s: Union[str, List[str], None]) -> List[str]:
    if s is None:
        return []
    if isinstance(s, list):
        return [x.strip() for x in s if isinstance(x, str) and x.strip()]
    if not isinstance(s, str):
        return []
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def load_json_or_jsonl(input_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Expected a JSON array at top-level.")
            return data
        else:
            rows = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
            return rows


def to_triple_list(graph_obj: Any) -> List[Tuple[str, str, str]]:
    triples = []
    if not isinstance(graph_obj, Iterable):
        return triples
    for item in graph_obj:
        if isinstance(item, dict):
            h = str(item.get("head", "")).strip()
            r = str(item.get("relation", "")).strip()
            t = str(item.get("tail", "")).strip()
            if h and r and t:
                triples.append((h, r, t))
        else:
            try:
                h, r, t = item
                triples.append((str(h), str(r), str(t)))
            except Exception:
                continue
    return triples


def convert_sample(raw: Dict[str, Any], idx: int) -> Dict[str, Any]:
    question = str(raw.get("question", "")).strip()
    q_entity_list = ensure_list_from_str(raw.get("q_entity"))
    a_entity_list = ensure_list_from_str(raw.get("a_entity"))

    # 强制 answer 与 a_entity 完全一致（列表、顺序一致）
    answer_list = list(a_entity_list)

    triples = to_triple_list(raw.get("graph", []))

    sample_id = raw.get("id")
    if sample_id is None or str(sample_id).strip() == "":
        sample_id = f"chatdoctor5k_{idx}"

    return {
        "id": str(sample_id),
        "question": question,
        "q_entity": q_entity_list,
        "a_entity": a_entity_list,
        "answer": answer_list,
        "graph": triples,
    }


def split_data(data: List[Dict[str, Any]], seed: int = 42, ratio=(0.8, 0.1, 0.1)):
    assert abs(sum(ratio) - 1.0) < 1e-6
    n = len(data)
    rnd = random.Random(seed)
    rnd.shuffle(data)
    n_train = int(n * ratio[0])
    n_val = int(n * ratio[1])
    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]
    return train, val, test


def save_pickle(obj: Any, path: str):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert chatdoctor5k.json to SubgraphRAG raw format (answer == a_entity).")
    parser.add_argument("--input", type=str, required=True, help="Path to chatdoctor5k.json (JSON array or JSONL).")
    parser.add_argument("--dataset_name", type=str, default="chatdoctor5k", help="Target dataset name under out_root/")
    parser.add_argument("--out_root", type=str, default="data", help="Output root directory relative to script parent (default: data)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("Train/val/test ratios must sum to 1.0.")

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    rows = load_json_or_jsonl(args.input)
    converted = [convert_sample(r, i) for i, r in enumerate(rows)]

    # 保留有问题文本且图非空的样本
    filtered = [s for s in converted if s["question"] and len(s["graph"]) > 0]

    train, val, test = split_data(
        filtered, seed=args.seed,
        ratio=(args.train_ratio, args.val_ratio, args.test_ratio)
    )

    # 保存到脚本上一级目录下的 out_root/<dataset_name>/
    repo_dir = Path(__file__).resolve().parents[1]
    base_dir = repo_dir / args.out_root / args.dataset_name
    save_pickle(train, str(base_dir / "train.pkl"))
    save_pickle(val, str(base_dir / "validation.pkl"))
    save_pickle(test, str(base_dir / "test.pkl"))

    print("Done. Saved raw data to:")
    print(f"  {base_dir / 'train.pkl'}   ({len(train)} samples)")
    print(f"  {base_dir / 'validation.pkl'} ({len(val)} samples)")
    print(f"  {base_dir / 'test.pkl'}    ({len(test)} samples)")


if __name__ == "__main__":
    main()