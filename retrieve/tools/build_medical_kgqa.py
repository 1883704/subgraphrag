import argparse
import csv
import json
import os
import pickle
import random
import re
from collections import Counter, defaultdict, deque
from pathlib import Path


def normalize_text(text):
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def load_json_or_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def load_kg_edges(path, head_col, rel_col, tail_col, delimiter):
    ext = Path(path).suffix.lower()
    rows = []

    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                head = str(row[head_col]).strip()
                rel = str(row[rel_col]).strip()
                tail = str(row[tail_col]).strip()
                if head and rel and tail:
                    rows.append((head, rel, tail))
        return rows

    with open(path, "r", encoding="utf-8", newline="") as f:
        if delimiter is None:
            delimiter = "\t" if ext in {".tsv", ".txt"} else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            head = str(row[head_col]).strip()
            rel = str(row[rel_col]).strip()
            tail = str(row[tail_col]).strip()
            if head and rel and tail:
                rows.append((head, rel, tail))
    return rows


def load_alias_map(path, entity_col, alias_col, delimiter):
    alias_to_entity = {}
    entity_to_aliases = defaultdict(set)

    if not path:
        return alias_to_entity, entity_to_aliases

    with open(path, "r", encoding="utf-8", newline="") as f:
        if delimiter is None:
            delimiter = "\t" if Path(path).suffix.lower() in {".tsv", ".txt"} else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            entity = str(row[entity_col]).strip()
            alias = str(row[alias_col]).strip()
            if not entity or not alias:
                continue
            alias_norm = normalize_text(alias)
            alias_to_entity[alias_norm] = entity
            entity_to_aliases[entity].add(alias)

    return alias_to_entity, entity_to_aliases


def build_entity_catalog(edges, entity_to_aliases):
    entities = set()
    for head, _, tail in edges:
        entities.add(head)
        entities.add(tail)

    phrase_to_entities = defaultdict(set)
    max_phrase_tokens = 1
    for entity in entities:
        entity_norm = normalize_text(entity)
        if entity_norm:
            phrase_to_entities[entity_norm].add(entity)
            max_phrase_tokens = max(max_phrase_tokens, len(entity_norm.split()))
        entity_to_aliases[entity].add(entity)
        for alias in entity_to_aliases[entity]:
            alias_norm = normalize_text(alias)
            if not alias_norm:
                continue
            phrase_to_entities[alias_norm].add(entity)
            max_phrase_tokens = max(max_phrase_tokens, len(alias_norm.split()))
    return entities, phrase_to_entities, max_phrase_tokens


def build_adjacency(edges, add_reverse_edges):
    adjacency = defaultdict(list)
    for head, rel, tail in edges:
        adjacency[head].append((rel, tail, head, rel, tail))
        if add_reverse_edges:
            adjacency[tail].append((f"{rel}__reverse", head, tail, rel, head))
    return adjacency


def find_entities_in_text(text, phrase_to_entities, max_phrase_tokens, max_entities):
    text_norm = normalize_text(text)
    if not text_norm:
        return []

    tokens = text_norm.split()
    matched = []
    seen_phrases = set()
    for start in range(len(tokens)):
        for width in range(1, min(max_phrase_tokens, len(tokens) - start) + 1):
            phrase = " ".join(tokens[start:start + width])
            if phrase in seen_phrases:
                continue
            entities = phrase_to_entities.get(phrase)
            if not entities:
                continue
            seen_phrases.add(phrase)
            for entity in sorted(entities):
                matched.append((width, entity))

    matched.sort(key=lambda item: (-item[0], item[1]))
    ordered_entities = []
    used = set()
    for _, entity in matched:
        if entity in used:
            continue
        used.add(entity)
        ordered_entities.append(entity)
        if len(ordered_entities) >= max_entities:
            break
    return ordered_entities


def bfs_collect_triples(adjacency, start_entities, max_hops, max_triples):
    if not start_entities:
        return []

    visited_depth = {}
    queue = deque()
    collected = []
    seen_triples = set()

    for entity in start_entities:
        queue.append((entity, 0))
        visited_depth.setdefault(entity, 0)

    while queue and len(collected) < max_triples:
        entity, depth = queue.popleft()
        if depth >= max_hops:
            continue

        for _, neighbor, triple_head, triple_rel, triple_tail in adjacency.get(entity, []):
            triple = (triple_head, triple_rel, triple_tail)
            if triple not in seen_triples:
                seen_triples.add(triple)
                collected.append(triple)
                if len(collected) >= max_triples:
                    break

            next_depth = depth + 1
            prev_depth = visited_depth.get(neighbor)
            if prev_depth is None or next_depth < prev_depth:
                visited_depth[neighbor] = next_depth
                queue.append((neighbor, next_depth))

    return collected


def build_question_text(question, options):
    if not options:
        return question.strip()

    lines = [question.strip(), "", "Options:"]
    for label, option_text in options:
        lines.append(f"{label}. {option_text}")
    return "\n".join(lines).strip()


def option_label_to_index(label):
    if label is None:
        return None

    label = str(label).strip()
    if not label:
        return None

    upper = label.upper()
    if upper in {"A", "B", "C", "D"}:
        return ord(upper) - ord("A")

    try:
        value = int(label)
    except ValueError:
        return None

    # MedMCQA stores cop as a zero-based option index: 0=A, 1=B, 2=C, 3=D.
    if 0 <= value <= 3:
        return value

    return None


def parse_medmcqa_row(row):
    option_pairs = []
    for label, key in (("A", "opa"), ("B", "opb"), ("C", "opc"), ("D", "opd")):
        value = str(row.get(key, "")).strip()
        if value:
            option_pairs.append((label, value))

    correct_value = row.get("cop")
    answer_text = ""
    correct_index = option_label_to_index(correct_value)
    answer_label_letter = ""
    if correct_index is not None and 0 <= correct_index < len(option_pairs):
        answer_label_letter = option_pairs[correct_index][0]
        answer_text = option_pairs[correct_index][1]

    return {
        "id": str(row.get("id", "")),
        "question_stem": str(row.get("question", "")).strip(),
        "question": build_question_text(str(row.get("question", "")).strip(), option_pairs),
        "options": option_pairs,
        "answer_text": answer_text,
        "answer_label": str(correct_value).strip(),
        "answer_label_letter": answer_label_letter,
        "explanation": str(row.get("exp", "")).strip(),
    }


def parse_medqa_row(row):
    options_raw = row.get("options", {})
    option_pairs = []
    if isinstance(options_raw, dict):
        for label in sorted(options_raw):
            value = str(options_raw[label]).strip()
            if value:
                option_pairs.append((label, value))

    answer_key = str(row.get("answer_idx", row.get("answer", ""))).strip()
    answer_text = ""
    for label, option_text in option_pairs:
        if label == answer_key:
            answer_text = option_text
            break

    return {
        "id": str(row.get("id", "")),
        "question_stem": str(row.get("question", "")).strip(),
        "question": build_question_text(str(row.get("question", "")).strip(), option_pairs),
        "options": option_pairs,
        "answer_text": answer_text,
        "answer_label": answer_key,
        "answer_label_letter": answer_key,
        "explanation": str(row.get("explanation", "")).strip(),
    }


def parse_generic_row(row, question_field, answer_field):
    answer_text = row.get(answer_field, "")
    if isinstance(answer_text, list):
        answer_items = [str(item).strip() for item in answer_text if str(item).strip()]
        answer_text = answer_items[0] if answer_items else ""
    return {
        "id": str(row.get("id", "")),
        "question_stem": str(row.get(question_field, "")).strip(),
        "question": str(row.get(question_field, "")).strip(),
        "options": [],
        "answer_text": str(answer_text).strip(),
        "answer_label": "",
        "answer_label_letter": "",
        "explanation": "",
    }


def parse_sample(row, qa_format, question_field, answer_field):
    if qa_format == "medmcqa":
        return parse_medmcqa_row(row)
    if qa_format == "medqa":
        return parse_medqa_row(row)
    return parse_generic_row(row, question_field, answer_field)


def assign_sample_id(sample, split_name, index):
    sample_id = sample.get("id", "").strip()
    if sample_id:
        return sample_id
    return f"{split_name}_{index}"


def convert_split(
    rows,
    split_name,
    qa_format,
    adjacency,
    phrase_to_entities,
    max_phrase_tokens,
    max_question_entities,
    max_answer_entities,
    max_hops,
    max_triples,
    min_graph_size,
    question_field,
    answer_field,
):
    converted = []
    stats = Counter()

    for index, row in enumerate(rows):
        parsed = parse_sample(row, qa_format, question_field, answer_field)
        sample_id = assign_sample_id(parsed, split_name, index)
        question_text = parsed["question"]
        question_stem = parsed["question_stem"]
        answer_text = parsed["answer_text"]

        q_entity = find_entities_in_text(
            question_stem,
            phrase_to_entities,
            max_phrase_tokens,
            max_question_entities,
        )
        candidate_entities = []
        for option_label, option_text in parsed["options"]:
            option_entities = find_entities_in_text(
                option_text,
                phrase_to_entities,
                max_phrase_tokens,
                max_answer_entities,
            )
            candidate_entities.append({
                "label": option_label,
                "text": option_text,
                "entities": option_entities,
            })

        answer_label_letter = parsed.get("answer_label_letter", "")
        a_entity = []
        for candidate in candidate_entities:
            if candidate["label"] == answer_label_letter:
                a_entity = list(candidate["entities"])
                break

        if not a_entity:
            a_entity = find_entities_in_text(
                answer_text,
                phrase_to_entities,
                max_phrase_tokens,
                max_answer_entities,
            )

        option_seed_entities = []
        for candidate in candidate_entities:
            option_seed_entities.extend(candidate["entities"])

        graph_seeds = list(dict.fromkeys(q_entity + option_seed_entities))
        if not graph_seeds:
            graph_seeds = list(dict.fromkeys(q_entity + a_entity))
        graph = bfs_collect_triples(
            adjacency,
            graph_seeds,
            max_hops=max_hops,
            max_triples=max_triples,
        )

        stats["total"] += 1
        stats["with_q_entity"] += int(bool(q_entity))
        stats["with_a_entity"] += int(bool(a_entity))
        stats["with_graph"] += int(len(graph) >= min_graph_size)

        if len(graph) < min_graph_size:
            continue

        converted.append({
            "id": sample_id,
            "question": question_text,
            "q_entity": q_entity,
            "a_entity": a_entity,
            "answer": list(a_entity),
            "graph": graph,
            "metadata": {
                "split": split_name,
                "qa_format": qa_format,
                "question_stem": question_stem,
                "answer_text": answer_text,
                "answer_label": parsed["answer_label"],
                "answer_label_letter": parsed.get("answer_label_letter", ""),
                "options": parsed["options"],
                "candidate_entities": candidate_entities,
                "explanation": parsed["explanation"],
            },
        })

    return converted, stats


def save_pickle(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_entity_identifiers(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("")


def split_rows(rows, seed, train_ratio, val_ratio):
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    train_end = int(len(rows) * train_ratio)
    val_end = train_end + int(len(rows) * val_ratio)
    return {
        "train": rows[:train_end],
        "val": rows[train_end:val_end],
        "test": rows[val_end:],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build a local medical KGQA dataset in SubgraphRAG raw format."
    )
    parser.add_argument("--dataset_name", required=True, help="Output dataset name")
    parser.add_argument("--kg_path", required=True, help="Path to KG edge file")
    parser.add_argument("--qa_format", choices=["medmcqa", "medqa", "generic"], default="medmcqa")
    parser.add_argument("--train_path", help="Path to train split file")
    parser.add_argument("--val_path", help="Path to val split file")
    parser.add_argument("--test_path", help="Path to test split file")
    parser.add_argument("--input_path", help="Single input file to be randomly split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--output_root", default="data_files", help="Output root under retrieve/")
    parser.add_argument("--head_col", default="x_name", help="KG head column")
    parser.add_argument("--rel_col", default="relation", help="KG relation column")
    parser.add_argument("--tail_col", default="y_name", help="KG tail column")
    parser.add_argument("--kg_delimiter", default=None, help="KG delimiter, defaults by file extension")
    parser.add_argument("--alias_path", help="Optional alias file")
    parser.add_argument("--alias_entity_col", default="entity", help="Alias file entity column")
    parser.add_argument("--alias_alias_col", default="alias", help="Alias file alias column")
    parser.add_argument("--alias_delimiter", default=None, help="Alias file delimiter")
    parser.add_argument("--max_hops", type=int, default=2, help="BFS hop limit for local graph construction")
    parser.add_argument("--max_triples", type=int, default=200, help="Maximum triples kept per sample")
    parser.add_argument("--min_graph_size", type=int, default=1, help="Minimum triples required to keep a sample")
    parser.add_argument("--max_question_entities", type=int, default=8, help="Maximum linked question entities")
    parser.add_argument("--max_answer_entities", type=int, default=4, help="Maximum linked answer entities")
    parser.add_argument("--question_field", default="question", help="Question field for generic format")
    parser.add_argument("--answer_field", default="answer", help="Answer field for generic format")
    parser.add_argument("--add_reverse_edges", action="store_true", help="Add reverse edges during BFS")
    args = parser.parse_args()

    if not any([args.train_path, args.val_path, args.test_path, args.input_path]):
        raise ValueError("Provide either split files or --input_path.")

    repo_dir = Path(__file__).resolve().parents[1]
    output_dir = repo_dir / args.output_root / args.dataset_name
    raw_dir = output_dir / "raw"

    print("Loading KG edges...")
    edges = load_kg_edges(
        args.kg_path,
        head_col=args.head_col,
        rel_col=args.rel_col,
        tail_col=args.tail_col,
        delimiter=args.kg_delimiter,
    )
    print(f"Loaded {len(edges)} KG edges")

    _, entity_to_aliases = load_alias_map(
        args.alias_path,
        entity_col=args.alias_entity_col,
        alias_col=args.alias_alias_col,
        delimiter=args.alias_delimiter,
    )
    _, phrase_to_entities, max_phrase_tokens = build_entity_catalog(
        edges,
        entity_to_aliases,
    )
    adjacency = build_adjacency(edges, add_reverse_edges=args.add_reverse_edges)

    if args.input_path:
        all_rows = load_json_or_jsonl(args.input_path)
        split_rows_dict = split_rows(
            all_rows,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
    else:
        split_rows_dict = {}
        if args.train_path:
            split_rows_dict["train"] = load_json_or_jsonl(args.train_path)
        if args.val_path:
            split_rows_dict["val"] = load_json_or_jsonl(args.val_path)
        if args.test_path:
            split_rows_dict["test"] = load_json_or_jsonl(args.test_path)

    all_stats = {
        "dataset_name": args.dataset_name,
        "qa_format": args.qa_format,
        "kg_edges": len(edges),
        "entity_catalog_size": len(phrase_to_entities),
        "splits": {},
    }

    for split_name in ("train", "val", "test"):
        rows = split_rows_dict.get(split_name, [])
        converted, stats = convert_split(
            rows,
            split_name=split_name,
            qa_format=args.qa_format,
            adjacency=adjacency,
            phrase_to_entities=phrase_to_entities,
            max_phrase_tokens=max_phrase_tokens,
            max_question_entities=args.max_question_entities,
            max_answer_entities=args.max_answer_entities,
            max_hops=args.max_hops,
            max_triples=args.max_triples,
            min_graph_size=args.min_graph_size,
            question_field=args.question_field,
            answer_field=args.answer_field,
        )
        save_pickle(converted, raw_dir / f"{split_name}.pkl")
        all_stats["splits"][split_name] = {
            "input_rows": len(rows),
            "kept_rows": len(converted),
            "with_q_entity": stats["with_q_entity"],
            "with_a_entity": stats["with_a_entity"],
            "with_graph": stats["with_graph"],
        }
        print(
            f"{split_name}: kept {len(converted)} / {len(rows)} | "
            f"q_entity={stats['with_q_entity']} | "
            f"a_entity={stats['with_a_entity']} | "
            f"graph={stats['with_graph']}"
        )

    write_entity_identifiers(output_dir / "entity_identifiers.txt")
    save_json(all_stats, output_dir / "build_stats.json")
    print(f"Saved dataset to: {raw_dir}")


if __name__ == "__main__":
    main()
