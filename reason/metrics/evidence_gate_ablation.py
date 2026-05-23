#!/usr/bin/env python3
"""Ablate post-hoc evidence gates for MedMCQA predictions.

This script compares a no-evidence MCQ prediction file with a tree-evidence
prediction file, extracts confidence/noise features from tree retrieval results,
and evaluates simple routing rules:

    use tree answer if feature >= threshold, otherwise use no-evidence answer.

It is meant for controlled diagnosis. It does not call an LLM and does not train
the tree scorer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any, Callable

from evaluate_medmcqa_accuracy import (
    default_raw_path,
    evaluate_file,
    extract_label_from_prediction,
    get_gold_label,
    get_options,
    infer_gold_numeric_base,
    load_gold,
    load_jsonl,
    normalize_option_label,
)


GENERIC_TERMS = {
    "all",
    "disease",
    "diseases",
    "disorder",
    "disorders",
    "symptom",
    "symptoms",
    "patient",
    "patients",
    "person",
    "persons",
    "human",
    "humans",
    "cell",
    "cells",
    "protein",
    "proteins",
    "gene",
    "genes",
    "drug",
    "drugs",
    "treatment",
    "therapy",
}


def stable_fold(sample_id: str, folds: int) -> int:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def is_generic(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in GENERIC_TERMS


def load_tree_jsonl(path: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("id")
            if sample_id:
                rows[sample_id] = row
    return rows


def pred_by_id(path: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in load_jsonl(path)}


def candidate_features(row: dict[str, Any]) -> dict[str, Any]:
    scores = []
    for item in row.get("candidate_scores", []) or []:
        label = normalize_option_label(item.get("candidate_label", ""))
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        if label:
            scores.append((label, score))

    scores.sort(key=lambda pair: pair[1], reverse=True)
    top_label = scores[0][0] if scores else ""
    top_score = scores[0][1] if scores else 0.0
    second_score = scores[1][1] if len(scores) > 1 else 0.0
    return {
        "candidate_top_label": top_label,
        "candidate_top_score": top_score,
        "candidate_margin": top_score - second_score,
    }


def iter_tree_entities(tree: dict[str, Any]):
    root = tree.get("root")
    if root:
        yield root
    for path in tree.get("paths", []) or []:
        for triple in path.get("triples", []) or []:
            if len(triple) >= 3:
                yield triple[0]
                yield triple[2]
    for triple in tree.get("triples", []) or []:
        if len(triple) >= 3:
            yield triple[0]
            yield triple[2]


def tree_features(row: dict[str, Any], thres: float = 0.2) -> dict[str, Any]:
    trees = row.get("scored_trees", []) or []
    kept = []
    scores = []
    generic_mentions = 0
    generic_top_root = 0
    for idx, tree in enumerate(trees):
        try:
            score = float(tree.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        scores.append(score)
        if score >= thres:
            kept.append(tree)
        if idx == 0 and is_generic(tree.get("root", "")):
            generic_top_root = 1
        generic_mentions += sum(1 for entity in iter_tree_entities(tree) if is_generic(entity))

    scores.sort(reverse=True)
    top_score = scores[0] if scores else 0.0
    second_score = scores[1] if len(scores) > 1 else 0.0
    q_entities = row.get("q_entity_in_graph") or row.get("q_entity") or []
    a_entities = row.get("a_entity_in_graph") or row.get("a_entity") or []
    generic_q = int(any(is_generic(entity) for entity in q_entities))
    generic_a = int(any(is_generic(entity) for entity in a_entities))
    return {
        "tree_top_score": top_score,
        "tree_margin": top_score - second_score,
        "tree_count": len(trees),
        "tree_count_ge_thres": len(kept),
        "generic_q": generic_q,
        "generic_a": generic_a,
        "generic_top_root": generic_top_root,
        "generic_mentions": generic_mentions,
    }


def build_records(args) -> list[dict[str, Any]]:
    gold = load_gold(args.raw_path)
    inferred_base, _ = infer_gold_numeric_base(gold)
    numeric_base = inferred_base if args.label_base == "auto" else args.label_base
    noevi_rows = pred_by_id(args.noevi_pred)
    tree_rows = pred_by_id(args.tree_pred)
    tree_data = load_tree_jsonl(args.tree_data)

    records = []
    for sample_id, gold_row in gold.items():
        if sample_id not in noevi_rows or sample_id not in tree_rows:
            continue
        options = get_options(gold_row)
        gold_label = get_gold_label(gold_row, numeric_base, options)
        if not gold_label:
            continue

        noevi_label, noevi_method = extract_label_from_prediction(
            noevi_rows[sample_id].get("prediction", ""), options
        )
        tree_label, tree_method = extract_label_from_prediction(
            tree_rows[sample_id].get("prediction", ""), options
        )
        trow = tree_data.get(sample_id, {})
        features = {}
        features.update(candidate_features(trow))
        features.update(tree_features(trow, args.tree_threshold))
        features["tree_matches_candidate_top"] = int(
            bool(tree_label) and tree_label == features.get("candidate_top_label")
        )
        features["noevi_matches_candidate_top"] = int(
            bool(noevi_label) and noevi_label == features.get("candidate_top_label")
        )
        features["predictions_disagree"] = int(
            bool(noevi_label) and bool(tree_label) and noevi_label != tree_label
        )
        records.append(
            {
                "id": sample_id,
                "gold": gold_label,
                "noevi_label": noevi_label,
                "tree_label": tree_label,
                "noevi_correct": noevi_label == gold_label,
                "tree_correct": tree_label == gold_label,
                "noevi_row": noevi_rows[sample_id],
                "tree_row": tree_rows[sample_id],
                "noevi_method": noevi_method,
                "tree_method": tree_method,
                **features,
            }
        )
    return records


def accuracy(rows: list[dict[str, Any]], chooser: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    correct = 0
    tree_used = 0
    routed_rows = []
    for row in rows:
        source = chooser(row)
        if source == "tree" and row["tree_label"]:
            pred = row["tree_label"]
            tree_used += 1
        else:
            source = "noevi"
            pred = row["noevi_label"]
        is_correct = pred == row["gold"]
        correct += int(is_correct)
        routed_rows.append((row, source, pred, is_correct))

    total = len(rows)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "tree_used": tree_used,
        "tree_used_ratio": tree_used / total if total else 0.0,
        "routed_rows": routed_rows,
    }


def thresholds_for(rows: list[dict[str, Any]], feature: str) -> list[float]:
    values = sorted({float(row.get(feature, 0.0) or 0.0) for row in rows})
    if not values:
        return [0.0]
    candidates = set(values)
    for left, right in zip(values, values[1:]):
        candidates.add((left + right) / 2)
    candidates.add(min(values) - 1e-6)
    candidates.add(max(values) + 1e-6)
    if len(candidates) > 120:
        ordered = sorted(candidates)
        step = max(1, len(ordered) // 120)
        candidates = set(ordered[::step] + [ordered[-1]])
    return sorted(candidates)


def gate_specs() -> list[dict[str, Any]]:
    return [
        {"name": "candidate_margin", "feature": "candidate_margin", "kind": "min"},
        {"name": "candidate_top_score", "feature": "candidate_top_score", "kind": "min"},
        {"name": "tree_top_score", "feature": "tree_top_score", "kind": "min"},
        {"name": "tree_margin", "feature": "tree_margin", "kind": "min"},
        {
            "name": "candidate_margin_match_top",
            "feature": "candidate_margin",
            "kind": "min",
            "requires": lambda row: row.get("tree_matches_candidate_top") == 1,
        },
        {
            "name": "candidate_margin_no_generic_q",
            "feature": "candidate_margin",
            "kind": "min",
            "requires": lambda row: row.get("generic_q") == 0,
        },
        {
            "name": "candidate_margin_no_generic_root",
            "feature": "candidate_margin",
            "kind": "min",
            "requires": lambda row: row.get("generic_top_root") == 0,
        },
        {
            "name": "candidate_margin_low_generic_mentions",
            "feature": "candidate_margin",
            "kind": "min",
            "requires": lambda row: row.get("generic_mentions", 0) <= 5,
        },
        {
            "name": "tree_score_no_generic_q",
            "feature": "tree_top_score",
            "kind": "min",
            "requires": lambda row: row.get("generic_q") == 0,
        },
    ]


def gate_spec_by_name(name: str) -> dict[str, Any]:
    for spec in gate_specs():
        if spec["name"] == name:
            return spec
    valid = ", ".join(spec["name"] for spec in gate_specs())
    raise ValueError(f"Unknown gate strategy: {name}. Valid strategies: {valid}")


def parse_fixed_gate(value: str) -> tuple[str, float]:
    for sep in ("=", ":"):
        if sep in value:
            name, threshold = value.split(sep, 1)
            return name.strip(), float(threshold.strip())
    raise ValueError(
        f"Invalid fixed gate '{value}'. Use STRATEGY:THRESHOLD, "
        "for example tree_top_score:0.58"
    )


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def chooser_for(spec: dict[str, Any], threshold: float) -> Callable[[dict[str, Any]], str]:
    feature = spec["feature"]
    requires = spec.get("requires")

    def choose(row: dict[str, Any]) -> str:
        if row.get("tree_label") == row.get("noevi_label"):
            return "tree"
        if requires is not None and not requires(row):
            return "noevi"
        value = float(row.get(feature, 0.0) or 0.0)
        return "tree" if value >= threshold else "noevi"

    return choose


def best_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = None
    for spec in gate_specs():
        for threshold in thresholds_for(rows, spec["feature"]):
            result = accuracy(rows, chooser_for(spec, threshold))
            item = {
                "strategy": spec["name"],
                "threshold": threshold,
                "correct": result["correct"],
                "accuracy": result["accuracy"],
                "tree_used": result["tree_used"],
                "tree_used_ratio": result["tree_used_ratio"],
            }
            if best is None or (
                item["accuracy"],
                -item["tree_used_ratio"],
            ) > (
                best["accuracy"],
                -best["tree_used_ratio"],
            ):
                best = item
    assert best is not None
    return best


def cross_validate(rows: list[dict[str, Any]], folds: int) -> list[dict[str, Any]]:
    results = []
    for spec in gate_specs():
        routed = []
        thresholds = []
        for fold in range(folds):
            train = [row for row in rows if stable_fold(row["id"], folds) != fold]
            valid = [row for row in rows if stable_fold(row["id"], folds) == fold]
            best = None
            for threshold in thresholds_for(train, spec["feature"]):
                result = accuracy(train, chooser_for(spec, threshold))
                item = (result["accuracy"], -result["tree_used_ratio"], threshold)
                if best is None or item > best:
                    best = item
            assert best is not None
            threshold = best[2]
            thresholds.append(threshold)
            routed.extend(accuracy(valid, chooser_for(spec, threshold))["routed_rows"])
        correct = sum(int(item[3]) for item in routed)
        tree_used = sum(1 for _, source, _, _ in routed if source == "tree")
        total = len(routed)
        results.append(
            {
                "strategy": spec["name"],
                "cv_correct": correct,
                "cv_accuracy": correct / total if total else 0.0,
                "cv_tree_used": tree_used,
                "cv_tree_used_ratio": tree_used / total if total else 0.0,
                "threshold_mean": statistics.mean(thresholds) if thresholds else 0.0,
                "thresholds": ",".join(f"{value:.6g}" for value in thresholds),
            }
        )
    results.sort(key=lambda row: (row["cv_accuracy"], -row["cv_tree_used_ratio"]), reverse=True)
    return results


def evaluate_fixed_gates(
    rows: list[dict[str, Any]],
    fixed_gates: list[str],
    out_dir: Path,
    name: str,
    gold: dict[str, Any],
    numeric_base: str,
) -> list[dict[str, Any]]:
    results = []
    for raw_gate in fixed_gates:
        strategy, threshold = parse_fixed_gate(raw_gate)
        spec = gate_spec_by_name(strategy)
        result = accuracy(rows, chooser_for(spec, threshold))
        pred_path = out_dir / f"{name}_fixed_{safe_name(strategy)}_{threshold:g}_routed_predictions.jsonl"
        write_routed_predictions(pred_path, result["routed_rows"])
        eval_result = evaluate_file(str(pred_path), gold, numeric_base)
        results.append(
            {
                "strategy": strategy,
                "threshold": threshold,
                "correct": result["correct"],
                "accuracy": result["accuracy"],
                "tree_used": result["tree_used"],
                "tree_used_ratio": result["tree_used_ratio"],
                "routed_eval_correct": eval_result["correct"],
                "routed_eval_accuracy": eval_result["accuracy"],
                "routed_pred_file": str(pred_path),
            }
        )
    results.sort(key=lambda row: (row["accuracy"], -row["tree_used_ratio"]), reverse=True)
    return results


def baseline_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    no = accuracy(rows, lambda _: "noevi")
    tree = accuracy(rows, lambda _: "tree")
    oracle_correct = sum(1 for row in rows if row["noevi_correct"] or row["tree_correct"])
    both_correct = sum(1 for row in rows if row["noevi_correct"] and row["tree_correct"])
    both_wrong = sum(1 for row in rows if not row["noevi_correct"] and not row["tree_correct"])
    tree_fix = sum(1 for row in rows if (not row["noevi_correct"]) and row["tree_correct"])
    tree_harm = sum(1 for row in rows if row["noevi_correct"] and (not row["tree_correct"]))
    total = len(rows)
    return [
        {
            "name": "noevi",
            "correct": no["correct"],
            "accuracy": no["accuracy"],
            "tree_used": no["tree_used"],
        },
        {
            "name": "tree",
            "correct": tree["correct"],
            "accuracy": tree["accuracy"],
            "tree_used": tree["tree_used"],
        },
        {
            "name": "oracle_choose_noevi_or_tree",
            "correct": oracle_correct,
            "accuracy": oracle_correct / total if total else 0.0,
            "tree_used": "",
        },
        {
            "name": "fix_harm",
            "correct": "",
            "accuracy": "",
            "tree_used": "",
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "tree_fix": tree_fix,
            "tree_harm": tree_harm,
            "net_gain": tree_fix - tree_harm,
        },
    ]


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_routed_predictions(path: Path, routed_rows: list[tuple[dict[str, Any], str, str, bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row, source, pred, is_correct in routed_rows:
            source_row = row["tree_row"] if source == "tree" else row["noevi_row"]
            out = dict(source_row)
            out["prediction"] = f"ans: {pred}" if pred else ""
            out["routed_source"] = source
            out["routed_gold"] = row["gold"]
            out["routed_correct"] = is_correct
            out["gate_features"] = {
                key: row.get(key)
                for key in [
                    "candidate_top_label",
                    "candidate_top_score",
                    "candidate_margin",
                    "tree_top_score",
                    "tree_margin",
                    "tree_count",
                    "tree_count_ge_thres",
                    "generic_q",
                    "generic_top_root",
                    "generic_mentions",
                    "tree_matches_candidate_top",
                    "noevi_matches_candidate_top",
                    "predictions_disagree",
                ]
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noevi-pred", required=True)
    parser.add_argument("--tree-pred", required=True)
    parser.add_argument("--tree-data", required=True)
    parser.add_argument("-d", "--dataset-name", default="medmcqa_primekg")
    parser.add_argument("--split", default="val")
    parser.add_argument("--raw-path", default=None)
    parser.add_argument("--label-base", choices=["auto", "zero", "one"], default="zero")
    parser.add_argument("--tree-threshold", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--name", default="evidence_gate")
    parser.add_argument("--out-dir", default="results/gating")
    parser.add_argument(
        "--fixed-gate",
        action="append",
        default=[],
        help=(
            "Evaluate a fixed gate in STRATEGY:THRESHOLD format. "
            "Can be repeated. Example: --fixed-gate tree_top_score:0.58"
        ),
    )
    args = parser.parse_args()

    args.raw_path = args.raw_path or default_raw_path(args.dataset_name, args.split)
    records = build_records(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baselines = baseline_rows(records)
    best = best_gate(records)
    cv = cross_validate(records, args.folds)
    best_spec = next(spec for spec in gate_specs() if spec["name"] == best["strategy"])
    routed = accuracy(records, chooser_for(best_spec, best["threshold"]))
    routed_path = out_dir / f"{args.name}_best_routed_predictions.jsonl"
    write_routed_predictions(routed_path, routed["routed_rows"])

    gold = load_gold(args.raw_path)
    inferred_base, _ = infer_gold_numeric_base(gold)
    numeric_base = inferred_base if args.label_base == "auto" else args.label_base
    eval_result = evaluate_file(str(routed_path), gold, numeric_base)
    best["routed_pred_file"] = str(routed_path)
    best["routed_eval_correct"] = eval_result["correct"]
    best["routed_eval_accuracy"] = eval_result["accuracy"]
    fixed = evaluate_fixed_gates(
        records,
        args.fixed_gate,
        out_dir,
        args.name,
        gold,
        numeric_base,
    )

    all_rows = []
    for row in baselines:
        copied = {"group": "baseline"}
        copied.update(row)
        all_rows.append(copied)
    all_rows.append({"group": "best_all", **best})
    for row in fixed:
        all_rows.append({"group": "fixed_gate", **row})
    for row in cv:
        all_rows.append({"group": "cross_validation", **row})

    csv_path = out_dir / f"{args.name}_gate_ablation.csv"
    write_csv(csv_path, all_rows)

    report = [
        f"# Evidence Gate Ablation: {args.name}",
        "",
        f"- records: `{len(records)}`",
        f"- no-evidence predictions: `{args.noevi_pred}`",
        f"- tree predictions: `{args.tree_pred}`",
        f"- tree data: `{args.tree_data}`",
        "",
        "## Baselines",
        "",
        md_table(baselines, ["name", "correct", "accuracy", "tree_used", "tree_fix", "tree_harm", "net_gain"]),
        "",
        "## Best Gate Fitted On All Rows",
        "",
        md_table([best], ["strategy", "threshold", "correct", "accuracy", "tree_used", "tree_used_ratio", "routed_eval_correct", "routed_eval_accuracy"]),
        "",
        "## Fixed Gates",
        "",
        md_table(fixed, ["strategy", "threshold", "correct", "accuracy", "tree_used", "tree_used_ratio", "routed_eval_correct", "routed_eval_accuracy", "routed_pred_file"]),
        "",
        "## Cross-Validated Gates",
        "",
        md_table(cv, ["strategy", "cv_correct", "cv_accuracy", "cv_tree_used", "cv_tree_used_ratio", "threshold_mean", "thresholds"]),
        "",
        "## Outputs",
        "",
        f"- routed predictions: `{routed_path}`",
        f"- csv: `{csv_path}`",
        "",
    ]
    report_path = out_dir / f"{args.name}_gate_ablation.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"report={report_path}")
    print(f"csv={csv_path}")
    print(f"routed_predictions={routed_path}")


if __name__ == "__main__":
    main()
