#!/usr/bin/env python
"""Diagnose where KG reasoning trees help or hurt MedMCQA MCQ predictions."""

import argparse
import glob
import json
import math
import os
import pickle
import re
import string
from collections import Counter
from pathlib import Path


LABEL_MAP = {"A": "A", "B": "B", "C": "C", "D": "D"}
ONE_BASED = {1: "A", 2: "B", 3: "C", 4: "D"}
ZERO_BASED = {0: "A", 1: "B", 2: "C", 3: "D"}

GENERIC_ENTITIES = {
    "all",
    "patient",
    "patients",
    "disease",
    "diseases",
    "medicine",
    "medical",
    "treatment",
    "therapy",
    "drug",
    "drugs",
    "human",
    "normal",
    "abnormal",
    "positive",
    "negative",
    "clinical",
    "surgery",
    "procedure",
    "syndrome",
    "disorder",
    "condition",
    "finding",
    "symptom",
    "sign",
    "test",
    "diagnosis",
    "management",
    "cause",
    "risk",
    "factor",
    "complication",
    "effect",
    "unknown",
    "various",
    "multiple",
    "common",
    "rare",
}


def normalize_text(value):
    text = str(value or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def normalize_entity(value):
    text = str(value or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def normalize_option_label(value):
    text = str(value or "").strip().upper()
    if text in LABEL_MAP:
        return LABEL_MAP[text]
    try:
        number = float(text)
        if number.is_integer():
            return ONE_BASED.get(int(number), "")
    except ValueError:
        pass
    return ""


def normalize_gold_label(value, numeric_base):
    text = str(value or "").strip().upper()
    if text in LABEL_MAP:
        return LABEL_MAP[text]
    try:
        number = float(text)
        if number.is_integer():
            num = int(number)
            return ZERO_BASED.get(num, "") if numeric_base == "zero" else ONE_BASED.get(num, "")
    except ValueError:
        pass
    return ""


def get_options(row):
    options = row.get("metadata", {}).get("options", []) or row.get("options", [])
    normalized = []
    for item in options:
        if isinstance(item, dict):
            label = normalize_option_label(item.get("label"))
            text = str(item.get("text") or "")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            label = normalize_option_label(item[0])
            text = str(item[1] or "")
        else:
            continue
        if label:
            normalized.append((label, text))
    return normalized


def get_gold_label(row, numeric_base, options):
    metadata = row.get("metadata", {}) or {}
    for key in ("answer_label_letter", "answer_label"):
        label = normalize_gold_label(row.get(key) or metadata.get(key), numeric_base)
        if label:
            return label
    answer_text = metadata.get("answer_text") or row.get("answer_text") or ""
    if answer_text:
        for label, text in options:
            if normalize_text(text) == normalize_text(answer_text):
                return label
    return ""


def extract_label_from_prediction(prediction, options):
    text = str(prediction or "")
    if not text.strip():
        return "", "empty"
    lowered = text.lower()
    if "ans: not available" in lowered:
        return "", "not_available"
    patterns = [
        r"\bans\s*:\s*[\(\[]?\s*([ABCD1-4])\b",
        r"\banswer\s*:\s*[\(\[]?\s*([ABCD1-4])\b",
        r"\boption\s*[\(\[]?\s*([ABCD1-4])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            label = normalize_option_label(match.group(1))
            if label:
                return label, "label"
    normalized_pred = normalize_text(text)
    for label, option_text in options:
        normalized_option = normalize_text(option_text)
        if normalized_option and normalized_option in normalized_pred:
            return label, "option_text"
    return "", "parse_failed"


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_gold(path):
    with open(path, "rb") as handle:
        rows = pickle.load(handle)
    return {row["id"]: row for row in rows}


def latest_file(patterns, project_root):
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(str(project_root / pattern), recursive=True))
    matches = [path for path in matches if os.path.isfile(path)]
    if not matches:
        return None
    matches.sort(key=lambda path: (os.path.getmtime(path), path), reverse=True)
    return matches[0]


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        if "|" in value:
            return [item.strip() for item in value.split("|") if item.strip()]
        return [value] if value.strip() else []
    return [str(value)]


def is_generic(entity_text):
    text = normalize_entity(entity_text)
    return not text or len(text) <= 1 or text in GENERIC_ENTITIES


def iter_tree_paths(trees):
    for tree in trees or []:
        for path in tree.get("paths", []) or []:
            yield path
        if not tree.get("paths"):
            yield {
                "nodes": [],
                "triples": tree.get("triples", []) or [],
                "score": tree.get("score", 0.0),
            }


def relation_counts(trees):
    counts = Counter()
    for path in iter_tree_paths(trees):
        for triple in path.get("triples", []) or []:
            if len(triple) >= 2:
                counts[str(triple[1])] += 1
    return counts


def entity_counts(trees):
    counts = Counter()
    for tree in trees or []:
        if tree.get("root"):
            counts[str(tree["root"])] += 1
    for path in iter_tree_paths(trees):
        for node in path.get("nodes", []) or []:
            counts[str(node)] += 1
        for triple in path.get("triples", []) or []:
            if len(triple) >= 3:
                counts[str(triple[0])] += 1
                counts[str(triple[2])] += 1
    return counts


def candidate_rank(candidate_scores, gold_label):
    if not candidate_scores:
        return None
    sorted_scores = sorted(
        candidate_scores,
        key=lambda item: item.get("score", 0.0),
        reverse=True,
    )
    for rank, item in enumerate(sorted_scores):
        if item.get("candidate_label") == gold_label:
            return rank
    return None


def exact_mcnemar_p(tree_fix, tree_harm):
    n = tree_fix + tree_harm
    if n == 0:
        return 1.0
    k = min(tree_fix, tree_harm)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * prob)


def build_diagnostics(args):
    project_root = Path(args.project_root).resolve()
    gold_path = Path(args.gold) if args.gold else project_root / "retrieve/data_files/medmcqa_primekg/raw/val.pkl"
    noevi_path = Path(args.noevi_pred) if args.noevi_pred else latest_file(
        [
            "reason/results/KGQA/medmcqa_primekg/SubgraphRAG/gpt-4.1-mini/*noevi*val-first_463-predictions.jsonl",
            "reason/results/KGQA/medmcqa_primekg/SubgraphRAG/gpt-4.1-mini/*noevi*predictions.jsonl",
        ],
        project_root,
    )
    tree_pred_path = Path(args.tree_pred) if args.tree_pred else latest_file(
        [
            "reason/results/KGQA/medmcqa_primekg/SubgraphRAG/gpt-4.1-mini/*tree_10_mcq*thres_0.2*val-first_463-predictions.jsonl",
            "reason/results/KGQA/medmcqa_primekg/SubgraphRAG/gpt-4.1-mini/*tree_10_mcq*thr*_0.2*val-first_463-predictions.jsonl",
            "reason/results/KGQA/medmcqa_primekg/SubgraphRAG/gpt-4.1-mini/*tree_10_mcq*val-first_463-predictions.jsonl",
        ],
        project_root,
    )
    tree_data_path = Path(args.tree_data) if args.tree_data else latest_file(
        [
            "retrieve/medmcqa_primekg_candidate_tree_fast_May19-15-38-51/tree_retrieval_result_val_top15.jsonl",
            "retrieve/medmcqa_primekg_candidate_tree_fast_May19-15-38-51/tree_retrieval_result_val.jsonl",
            "retrieve/medmcqa_primekg_candidate_tree_fast_*/tree_retrieval_result_val*.jsonl",
        ],
        project_root,
    )

    missing = [
        ("gold", gold_path),
        ("noevi_pred", noevi_path),
        ("tree_pred", tree_pred_path),
        ("tree_data", tree_data_path),
    ]
    missing = [(name, path) for name, path in missing if not path or not Path(path).exists()]
    if missing:
        details = "\n".join(f"  {name}: {path}" for name, path in missing)
        raise FileNotFoundError("Required input file(s) not found:\n" + details)

    out_dir = project_root / "reason/results/diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = args.out_prefix or "tree10_fast_thres02_fix_harm"
    jsonl_out = out_dir / f"{out_prefix}.jsonl"
    summary_out = out_dir / f"{out_prefix}_summary.txt"

    gold_rows = read_gold(gold_path)
    noevi_rows = {row["id"]: row for row in read_jsonl(noevi_path)}
    tree_rows = {row["id"]: row for row in read_jsonl(tree_pred_path)}
    tree_data_rows = {row["id"]: row for row in read_jsonl(tree_data_path)}

    samples = []
    for sample_id in sorted(set(noevi_rows) & set(tree_rows)):
        gold_row = gold_rows.get(sample_id)
        if gold_row is None:
            continue

        options = get_options(gold_row)
        gold_label = get_gold_label(gold_row, args.label_base, options)
        noevi_text = noevi_rows[sample_id].get("prediction", "")
        tree_text = tree_rows[sample_id].get("prediction", "")
        noevi_pred, noevi_method = extract_label_from_prediction(noevi_text, options)
        tree_pred, tree_method = extract_label_from_prediction(tree_text, options)

        noevi_correct = bool(noevi_pred) and noevi_pred == gold_label
        tree_correct = bool(tree_pred) and tree_pred == gold_label
        if noevi_correct and tree_correct:
            case_type = "both_correct"
        elif not noevi_correct and not tree_correct:
            case_type = "both_wrong"
        elif tree_correct:
            case_type = "tree_fix"
        else:
            case_type = "tree_harm"

        tree_data = tree_data_rows.get(sample_id, {})
        candidate_scores = tree_data.get("candidate_scores", []) or []
        top_trees = tree_data.get("scored_trees", []) or []
        row = {
            "id": sample_id,
            "case_type": case_type,
            "gold": gold_label,
            "noevi_pred": noevi_pred,
            "tree_pred": tree_pred,
            "noevi_parse_method": noevi_method,
            "tree_parse_method": tree_method,
            "noevi_prediction": noevi_text,
            "tree_prediction": tree_text,
            "question": gold_row.get("question", tree_data.get("question", "")),
            "options": options,
            "q_entity": tree_data.get("q_entity", []),
            "a_entity": tree_data.get("a_entity", []),
            "candidate_scores": candidate_scores,
            "gold_candidate_rank": candidate_rank(candidate_scores, gold_label),
            "num_trees": len(top_trees),
            "top_trees": top_trees[: args.max_trees_dump],
        }
        samples.append(row)

    counts = Counter(row["case_type"] for row in samples)
    fixes = [row for row in samples if row["case_type"] == "tree_fix"]
    harms = [row for row in samples if row["case_type"] == "tree_harm"]

    with jsonl_out.open("w", encoding="utf-8") as handle:
        for row in samples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def collect(rows, key):
        counts_local = Counter()
        for row in rows:
            for item in as_list(row.get(key)):
                if item:
                    counts_local[str(item)] += 1
        return counts_local

    fix_q = collect(fixes, "q_entity")
    harm_q = collect(harms, "q_entity")
    fix_a = collect(fixes, "a_entity")
    harm_a = collect(harms, "a_entity")
    fix_tree_entities = Counter()
    harm_tree_entities = Counter()
    fix_relations = Counter()
    harm_relations = Counter()
    for row in fixes:
        fix_tree_entities.update(entity_counts(row["top_trees"]))
        fix_relations.update(relation_counts(row["top_trees"]))
    for row in harms:
        harm_tree_entities.update(entity_counts(row["top_trees"]))
        harm_relations.update(relation_counts(row["top_trees"]))

    rank_fix = Counter(row["gold_candidate_rank"] for row in fixes if row["gold_candidate_rank"] is not None)
    rank_harm = Counter(row["gold_candidate_rank"] for row in harms if row["gold_candidate_rank"] is not None)
    harm_generic_q = sum(any(is_generic(ent) for ent in as_list(row.get("q_entity"))) for row in harms)
    harm_generic_a = sum(any(is_generic(ent) for ent in as_list(row.get("a_entity"))) for row in harms)
    harm_generic_tree = [(ent, count) for ent, count in harm_tree_entities.most_common(50) if is_generic(ent)]

    paired_total = len(samples)
    noevi_correct = counts["both_correct"] + counts["tree_harm"]
    tree_correct = counts["both_correct"] + counts["tree_fix"]
    net_gain = counts["tree_fix"] - counts["tree_harm"]
    p_value = exact_mcnemar_p(counts["tree_fix"], counts["tree_harm"])

    lines = [
        "=" * 80,
        f"Fix/Harm Diagnosis: {args.name}",
        "=" * 80,
        "",
        f"gold_file:    {gold_path}",
        f"noevi_pred:   {noevi_path}",
        f"tree_pred:    {tree_pred_path}",
        f"tree_data:    {tree_data_path}",
        f"jsonl_out:    {jsonl_out}",
        "",
        f"paired_total: {paired_total}",
        f"both_correct: {counts['both_correct']}",
        f"both_wrong:   {counts['both_wrong']}",
        f"tree_fix:     {counts['tree_fix']}",
        f"tree_harm:    {counts['tree_harm']}",
        f"net_gain:     {net_gain}",
        f"mcnemar_exact_p: {p_value:.6f}",
        f"accuracy_noevi: {noevi_correct}/{paired_total} = {noevi_correct / max(paired_total, 1):.4f}",
        f"accuracy_tree:  {tree_correct}/{paired_total} = {tree_correct / max(paired_total, 1):.4f}",
        "",
        "--- Fix Cases ---",
        f"gold_candidate_rank: {dict(sorted(rank_fix.items(), key=lambda item: item[0]))}",
        f"top_q_entities: {fix_q.most_common(15)}",
        f"top_a_entities: {fix_a.most_common(15)}",
        f"top_tree_entities: {fix_tree_entities.most_common(20)}",
        f"top_relations: {fix_relations.most_common(20)}",
        "",
        "--- Harm Cases ---",
        f"gold_candidate_rank: {dict(sorted(rank_harm.items(), key=lambda item: item[0]))}",
        f"cases_with_generic_q_entity: {harm_generic_q}",
        f"cases_with_generic_a_entity: {harm_generic_a}",
        f"top_q_entities: {harm_q.most_common(15)}",
        f"top_a_entities: {harm_a.most_common(15)}",
        f"top_tree_entities: {harm_tree_entities.most_common(20)}",
        f"top_relations: {harm_relations.most_common(20)}",
        f"generic_entities_in_harm_top50: {harm_generic_tree[:20]}",
        "",
        "--- Interpretation Guardrail ---",
        "Treat this as a diagnostic file. A small net gain or high p-value is not enough for a strong paper claim.",
    ]

    summary = "\n".join(lines)
    summary_out.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"\nWrote diagnostic JSONL: {jsonl_out}")
    print(f"Wrote summary: {summary_out}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--name", default="tree_10_mcq_thres0.2_fast")
    parser.add_argument("--gold", default=None)
    parser.add_argument("--noevi-pred", default=None)
    parser.add_argument("--tree-pred", default=None)
    parser.add_argument("--tree-data", default=None)
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--label-base", choices=["zero", "one"], default="zero")
    parser.add_argument("--max-trees-dump", type=int, default=10)
    return parser.parse_args()


def main():
    build_diagnostics(parse_args())


if __name__ == "__main__":
    main()
