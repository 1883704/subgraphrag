#!/usr/bin/env python3
"""Summarize repeated fixed-gate experiments."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_repeat_name(path: Path) -> str:
    match = re.search(r"_r(\d+)_", path.name)
    if match:
        return f"r{match.group(1)}"
    for part in path.parts:
        match = re.match(r"repeat_(\d+)$", part)
        if match:
            return f"r{match.group(1)}"
    return path.parent.name


def parse_summary_file(path: Path) -> dict[str, str]:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def parse_accuracy_fraction(value: str) -> str:
    match = re.search(r"=\s*([0-9.]+)", value)
    return match.group(1) if match else value


def find_diagnostic(run_dir: Path, repeat: str) -> dict[str, str]:
    if run_dir.parent.name == "gating":
        results_dir = run_dir.parent.parent
    else:
        results_dir = run_dir
    diag_dir = results_dir / "diagnostics"
    candidates = list(diag_dir.glob(f"*_{repeat}_*_fix_harm_summary.txt"))
    if not candidates:
        candidates = list(diag_dir.glob(f"*{repeat}*fix_harm_summary.txt"))
    if not candidates:
        return {}
    path = max(candidates, key=lambda p: p.stat().st_mtime)
    values = parse_summary_file(path)
    values["diagnostic_file"] = str(path)
    if values.get("accuracy_noevi"):
        values["accuracy_noevi"] = parse_accuracy_fraction(values["accuracy_noevi"])
    if values.get("accuracy_tree"):
        values["accuracy_tree"] = parse_accuracy_fraction(values["accuracy_tree"])
    return values


def collect_rows(run_dir: Path, primary_strategy: str, primary_threshold: float) -> list[dict[str, Any]]:
    rows = []
    for csv_path in sorted(run_dir.glob("repeat_*/*gate_ablation.csv")):
        repeat = parse_repeat_name(csv_path)
        csv_rows = read_csv(csv_path)
        record: dict[str, Any] = {"repeat": repeat, "csv": str(csv_path)}
        for row in csv_rows:
            group = row.get("group")
            if group == "baseline" and row.get("name") == "noevi":
                record["noevi_correct"] = int(row.get("correct") or 0)
                record["noevi_accuracy"] = float(row.get("accuracy") or 0)
            elif group == "baseline" and row.get("name") == "tree":
                record["tree_correct"] = int(row.get("correct") or 0)
                record["tree_accuracy"] = float(row.get("accuracy") or 0)
            elif group == "fixed_gate":
                strategy = row.get("strategy")
                threshold = float(row.get("threshold") or 0)
                if strategy == primary_strategy and abs(threshold - primary_threshold) < 1e-9:
                    record["fixed_strategy"] = strategy
                    record["fixed_threshold"] = threshold
                    record["fixed_correct"] = int(row.get("correct") or 0)
                    record["fixed_accuracy"] = float(row.get("accuracy") or 0)
                    record["fixed_tree_used"] = int(row.get("tree_used") or 0)
                    record["fixed_tree_used_ratio"] = float(row.get("tree_used_ratio") or 0)
                    record["fixed_pred_file"] = row.get("routed_pred_file", "")
            elif group == "best_all":
                record["best_all_strategy"] = row.get("strategy", "")
                record["best_all_threshold"] = float(row.get("threshold") or 0)
                record["best_all_correct"] = int(row.get("correct") or 0)
                record["best_all_accuracy"] = float(row.get("accuracy") or 0)

        diag = find_diagnostic(run_dir, repeat)
        for key in ["tree_fix", "tree_harm", "net_gain", "mcnemar_exact_p", "diagnostic_file"]:
            if key in diag:
                if key in {"tree_fix", "tree_harm", "net_gain"}:
                    record[key] = int(diag[key])
                elif key == "mcnemar_exact_p":
                    record[key] = float(diag[key])
                else:
                    record[key] = diag[key]
        rows.append(record)
    return rows


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": "", "std": "", "min": "", "max": ""}
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


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
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--primary-strategy", default="tree_top_score")
    parser.add_argument("--primary-threshold", type=float, default=0.625)
    parser.add_argument("--reference-correct", type=int, default=361)
    parser.add_argument("--cv-reference-correct", type=int, default=366)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows = collect_rows(run_dir, args.primary_strategy, args.primary_threshold)
    if not rows:
        raise SystemExit(f"No repeat gate CSV files found under: {run_dir}")

    summary_rows = []
    for metric in ["noevi_accuracy", "tree_accuracy", "fixed_accuracy", "best_all_accuracy"]:
        values = [float(row[metric]) for row in rows if metric in row]
        item = {"metric": metric}
        item.update(stats(values))
        summary_rows.append(item)

    count_rows = []
    for metric in ["noevi_correct", "tree_correct", "fixed_correct", "tree_fix", "tree_harm", "net_gain"]:
        values = [float(row[metric]) for row in rows if metric in row and row[metric] != ""]
        item = {"metric": metric}
        item.update(stats(values))
        count_rows.append(item)

    fixed_corrects = [int(row["fixed_correct"]) for row in rows if "fixed_correct" in row]
    wins_old = sum(1 for value in fixed_corrects if value > args.reference_correct)
    wins_cv = sum(1 for value in fixed_corrects if value > args.cv_reference_correct)
    ties_cv = sum(1 for value in fixed_corrects if value == args.cv_reference_correct)

    report = [
        f"# Fixed Gate Repeat Summary",
        "",
        f"- run_dir: `{run_dir}`",
        f"- primary_gate: `{args.primary_strategy}:{args.primary_threshold:g}`",
        f"- repeats: `{len(rows)}`",
        f"- fixed_gate > {args.reference_correct}/463: `{wins_old}/{len(rows)}`",
        f"- fixed_gate > {args.cv_reference_correct}/463: `{wins_cv}/{len(rows)}`",
        f"- fixed_gate == {args.cv_reference_correct}/463: `{ties_cv}/{len(rows)}`",
        "",
        "## Per Repeat",
        "",
        md_table(
            rows,
            [
                "repeat",
                "noevi_correct",
                "tree_correct",
                "fixed_correct",
                "fixed_accuracy",
                "fixed_tree_used",
                "tree_fix",
                "tree_harm",
                "net_gain",
                "mcnemar_exact_p",
                "best_all_strategy",
                "best_all_correct",
            ],
        ),
        "",
        "## Aggregate Accuracy",
        "",
        md_table(summary_rows, ["metric", "mean", "std", "min", "max"]),
        "",
        "## Aggregate Counts",
        "",
        md_table(count_rows, ["metric", "mean", "std", "min", "max"]),
        "",
    ]

    report_path = run_dir / "repeat_summary.md"
    csv_path = run_dir / "repeat_summary.csv"
    report_path.write_text("\n".join(report), encoding="utf-8")
    write_csv(csv_path, rows)
    print("\n".join(report))
    print(f"report={report_path}")
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main()
