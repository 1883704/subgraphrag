#!/usr/bin/env python3
"""Collect training/evaluation artifacts into one compact report.

The script is intentionally dependency-free so it can run on the server
without activating a specific conda environment.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import os
import re
from pathlib import Path
from typing import Any


SUMMARY_KEYS = (
    "pred_file",
    "total_pred",
    "evaluated",
    "correct",
    "accuracy",
    "parse_failed",
)

DIAG_KEYS = (
    "paired_total",
    "both_correct",
    "both_wrong",
    "tree_fix",
    "tree_harm",
    "net_gain",
    "mcnemar_exact_p",
    "accuracy_noevi",
    "accuracy_tree",
)

TRAIN_KEYS = (
    "epoch",
    "best_metric",
    "val/option_acc",
    "val/support_hit@1",
    "val/support_hit@3",
    "val/support_hit@5",
    "val/loss",
    "train/loss",
)


def strip_suffix(text: str, suffix: str) -> str:
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return text


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def parse_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key:
            values[key] = value
    return values


def parse_accuracy_fraction(value: str) -> str:
    match = re.search(r"=\s*([0-9.]+)", value)
    return match.group(1) if match else value


def collect_summaries(root: Path) -> list[dict[str, Any]]:
    results_dir = root / "reason" / "results"
    rows: list[dict[str, Any]] = []
    if not results_dir.exists():
        return rows

    for path in sorted(results_dir.glob("*summary.txt"), key=lambda p: p.stat().st_mtime):
        if path.parent.name == "diagnostics":
            continue
        values = parse_key_value_file(path)
        if not any(k in values for k in SUMMARY_KEYS):
            continue
        row = {
            "file": rel(path, root),
            "experiment": strip_suffix(path.name, "_summary.txt"),
            "mtime": dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        }
        row.update({k: values.get(k, "") for k in SUMMARY_KEYS})
        rows.append(row)
    return rows


def collect_diagnostics(root: Path) -> list[dict[str, Any]]:
    diag_dir = root / "reason" / "results" / "diagnostics"
    rows: list[dict[str, Any]] = []
    if not diag_dir.exists():
        return rows

    for path in sorted(diag_dir.glob("*fix_harm_summary.txt"), key=lambda p: p.stat().st_mtime):
        values = parse_key_value_file(path)
        row = {
            "file": rel(path, root),
            "experiment": strip_suffix(path.name, "_fix_harm_summary.txt"),
            "mtime": dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        }
        row.update({k: values.get(k, "") for k in DIAG_KEYS})
        if row.get("accuracy_noevi"):
            row["accuracy_noevi"] = parse_accuracy_fraction(str(row["accuracy_noevi"]))
        if row.get("accuracy_tree"):
            row["accuracy_tree"] = parse_accuracy_fraction(str(row["accuracy_tree"]))
        rows.append(row)
    return rows


def parse_train_log(path: Path) -> dict[str, Any] | None:
    epoch_rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not (line.startswith("{") and "'epoch'" in line):
            continue
        try:
            parsed = ast.literal_eval(line)
        except (SyntaxError, ValueError):
            continue
        if isinstance(parsed, dict):
            epoch_rows.append(parsed)

    if not epoch_rows:
        return None

    best = max(epoch_rows, key=lambda row: row.get("val/option_acc", float("-inf")))
    last = epoch_rows[-1]
    row: dict[str, Any] = {
        "file": path.as_posix(),
        "experiment": strip_suffix(path.name, ".log"),
        "mtime": dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        "num_epochs_logged": len(epoch_rows),
        "best_epoch": best.get("epoch", ""),
    }
    for key in TRAIN_KEYS:
        if key in best:
            row[f"best/{key}"] = best[key]
        if key in last:
            row[f"last/{key}"] = last[key]
    return row


def collect_training_logs(root: Path) -> list[dict[str, Any]]:
    logs_dir = root / "retrieve" / "logs"
    if not logs_dir.exists():
        return []
    rows = []
    for path in sorted(logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime):
        parsed = parse_train_log(path)
        if parsed is not None:
            parsed["file"] = rel(path, root)
            rows.append(parsed)
    return rows


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No records found._\n"

    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        text = str(value)
        return text.replace("\n", " ").replace("|", "\\|")

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(root: Path) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    summaries = collect_summaries(root)
    diagnostics = collect_diagnostics(root)
    train_logs = collect_training_logs(root)

    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Experiment Results",
        "",
        f"- Generated: `{generated}`",
        f"- Root: `{root}`",
        "",
        "## QA Accuracy Summaries",
        "",
        md_table(
            summaries,
            ["mtime", "experiment", "correct", "accuracy", "evaluated", "parse_failed", "file"],
        ),
        "## Fix/Harm Diagnostics",
        "",
        md_table(
            diagnostics,
            [
                "mtime",
                "experiment",
                "accuracy_noevi",
                "accuracy_tree",
                "tree_fix",
                "tree_harm",
                "net_gain",
                "mcnemar_exact_p",
                "file",
            ],
        ),
        "## Training Logs",
        "",
        md_table(
            train_logs,
            [
                "mtime",
                "experiment",
                "best_epoch",
                "best/val/option_acc",
                "best/val/support_hit@1",
                "best/val/support_hit@3",
                "best/val/support_hit@5",
                "num_epochs_logged",
                "file",
            ],
        ),
        "## Quick Commands",
        "",
        "```bash",
        "python3 retrieve/scripts/summarize_experiment_results.py",
        "less -S reason/results/experiment_report_latest.md",
        "column -s, -t < reason/results/experiment_summary_latest.csv | less -S",
        "```",
        "",
    ]
    return "\n".join(lines), {
        "summaries": summaries,
        "diagnostics": diagnostics,
        "train_logs": train_logs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None, help="Repository root. Defaults to auto-detected root.")
    parser.add_argument("--output", default=None, help="Markdown report path.")
    parser.add_argument("--csv", default=None, help="CSV summary path.")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    root = Path(args.root).resolve() if args.root else script_path.parents[2]
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) if args.output else root / "reason" / "results" / "experiment_report_latest.md"
    csv_output = Path(args.csv) if args.csv else root / "reason" / "results" / "experiment_summary_latest.csv"

    report, groups = build_report(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")

    timestamped_output = output.with_name(f"experiment_report_{timestamp}.md")
    timestamped_output.write_text(report, encoding="utf-8")

    csv_rows: list[dict[str, Any]] = []
    for group_name, rows in groups.items():
        for row in rows:
            copied = {"group": group_name}
            copied.update(row)
            csv_rows.append(copied)
    write_csv(csv_output, csv_rows)

    print(f"Wrote report: {output}")
    print(f"Wrote snapshot: {timestamped_output}")
    print(f"Wrote csv: {csv_output}")
    print("")
    print(report)


if __name__ == "__main__":
    main()
