import argparse
import glob
import json
import os
import pickle
import re
import string
from pathlib import Path


LABEL_MAP = {
    "1": "A",
    "2": "B",
    "3": "C",
    "4": "D",
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
}


def normalize_text(value):
    text = str(value or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def normalize_label(value):
    return LABEL_MAP.get(str(value or "").strip().upper(), "")


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_gold(raw_path):
    with open(raw_path, "rb") as f:
        rows = pickle.load(f)
    return {row["id"]: row for row in rows}


def get_gold_label(row):
    metadata = row.get("metadata", {})
    return normalize_label(metadata.get("answer_label"))


def get_options(row):
    options = row.get("metadata", {}).get("options", [])
    return [(normalize_label(label), str(text or "")) for label, text in options]


def extract_label_from_prediction(prediction, options):
    text = str(prediction or "")
    if not text.strip():
        return "", "empty"

    lowered = text.lower()
    if "ans: not available" in lowered or "no sufficient information" in lowered:
        return "", "not_available"

    patterns = [
        r"\bans\s*:\s*[\(\[]?\s*([ABCD1-4])\b",
        r"\banswer\s*:\s*[\(\[]?\s*([ABCD1-4])\b",
        r"\boption\s*[\(\[]?\s*([ABCD1-4])\b",
        r"\bthe answer is\s*[\(\[]?\s*([ABCD1-4])\b",
        r"\bcorrect answer is\s*[\(\[]?\s*([ABCD1-4])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            label = normalize_label(match.group(1))
            if label:
                return label, "label"

    normalized_prediction = normalize_text(text)
    for label, option_text in options:
        normalized_option = normalize_text(option_text)
        if label and normalized_option and normalized_option in normalized_prediction:
            return label, "option_text"

    return "", "parse_failed"


def evaluate_file(pred_path, gold):
    rows = load_jsonl(pred_path)
    total = 0
    correct = 0
    missing_gold = 0
    parse_failed = 0
    method_counts = {}
    wrong = []

    for row in rows:
        sample_id = row.get("id")
        gold_row = gold.get(sample_id)
        if gold_row is None:
            missing_gold += 1
            continue

        gold_label = get_gold_label(gold_row)
        options = get_options(gold_row)
        pred_label, method = extract_label_from_prediction(
            row.get("prediction", ""), options
        )

        total += 1
        method_counts[method] = method_counts.get(method, 0) + 1
        if not pred_label:
            parse_failed += 1

        is_correct = pred_label == gold_label
        correct += int(is_correct)

        if not is_correct:
            wrong.append({
                "id": sample_id,
                "gold": gold_label,
                "pred": pred_label,
                "parse_method": method,
                "question": gold_row.get("question", ""),
                "options": options,
                "prediction": str(row.get("prediction", "")),
            })

    accuracy = correct / total if total else 0.0
    return {
        "pred_file": pred_path,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "parse_failed": parse_failed,
        "missing_gold": missing_gold,
        "parse_methods": method_counts,
        "wrong": wrong,
    }


def find_latest_prediction(dataset_name, results_root):
    pattern = os.path.join(
        results_root, "KGQA", dataset_name, "**", "*predictions.jsonl"
    )
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No predictions found with pattern: {pattern}")
    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return candidates[0]


def default_raw_path(dataset_name, split):
    return os.path.join(
        "..", "retrieve", "data_files", dataset_name, "raw", f"{split}.pkl"
    )


def print_result(result, show_wrong):
    print("=" * 80)
    print(f"pred_file:    {result['pred_file']}")
    print(f"total:        {result['total']}")
    print(f"correct:      {result['correct']}")
    print(f"accuracy:     {result['accuracy']:.4f}")
    print(f"parse_failed: {result['parse_failed']}")
    print(f"missing_gold: {result['missing_gold']}")
    print(f"parse_methods:{json.dumps(result['parse_methods'], ensure_ascii=False)}")

    if show_wrong:
        print("\nwrong examples:")
        for item in result["wrong"][:show_wrong]:
            preview = item["prediction"].replace("\n", " ")[:300]
            print(
                json.dumps(
                    {
                        "id": item["id"],
                        "gold": item["gold"],
                        "pred": item["pred"],
                        "parse_method": item["parse_method"],
                        "prediction": preview,
                    },
                    ensure_ascii=False,
                )
            )


def write_wrong_examples(path, results):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for result in results:
            for item in result["wrong"]:
                item = dict(item)
                item["pred_file"] = result["pred_file"]
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MedMCQA-style multiple-choice predictions."
    )
    parser.add_argument(
        "pred_files",
        nargs="*",
        help="Prediction JSONL files. If omitted, use --latest.",
    )
    parser.add_argument("-d", "--dataset-name", default="medmcqa_primekg")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--raw-path",
        default=None,
        help="Gold raw split pickle. Defaults to ../retrieve/data_files/<dataset>/raw/<split>.pkl",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root containing KGQA prediction folders.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Evaluate the newest predictions.jsonl under results/KGQA/<dataset>.",
    )
    parser.add_argument("--show-wrong", type=int, default=5)
    parser.add_argument(
        "--wrong-out",
        default=None,
        help="Optional JSONL path for all wrong examples.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print compact JSON summary instead of human-readable output.",
    )
    args = parser.parse_args()

    pred_files = list(args.pred_files)
    if args.latest:
        pred_files.append(find_latest_prediction(args.dataset_name, args.results_root))
    if not pred_files:
        raise SystemExit("Provide prediction files or pass --latest.")

    raw_path = args.raw_path or default_raw_path(args.dataset_name, args.split)
    gold = load_gold(raw_path)
    results = [evaluate_file(pred_path, gold) for pred_path in pred_files]

    if args.json:
        summaries = [
            {key: value for key, value in result.items() if key != "wrong"}
            for result in results
        ]
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
    else:
        print(f"gold_file:    {raw_path}")
        for result in results:
            print_result(result, args.show_wrong)

    if args.wrong_out:
        write_wrong_examples(args.wrong_out, results)
        print(f"\nwrong examples saved to: {args.wrong_out}")


if __name__ == "__main__":
    main()
