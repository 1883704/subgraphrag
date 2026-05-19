import os
import random
from collections import deque

import networkx as nx
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm


LABEL_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


def _normalize_options(sample):
    options = sample.get("options")
    if not options:
        metadata = sample.get("metadata", {}) or {}
        options = metadata.get("options", [])

    normalized = []
    for item in options or []:
        if isinstance(item, dict):
            label = str(item.get("label", "")).strip().upper()
            text = str(item.get("text", "")).strip()
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            label = str(item[0]).strip().upper()
            text = str(item[1]).strip()
        else:
            continue
        if label and text:
            normalized.append((label, text))
    return normalized


def _gold_index(sample, label_base="zero"):
    metadata = sample.get("metadata", {}) or {}
    letter = str(
        sample.get("answer_label_letter") or metadata.get("answer_label_letter") or ""
    ).strip().upper()
    if letter in LABEL_TO_INDEX:
        return LABEL_TO_INDEX[letter]

    raw_label = sample.get("answer_label")
    if raw_label is None or raw_label == "":
        raw_label = metadata.get("answer_label", "")

    raw_label = str(raw_label).strip()
    if raw_label.upper() in LABEL_TO_INDEX:
        return LABEL_TO_INDEX[raw_label.upper()]
    if not raw_label:
        return -1

    try:
        value = int(raw_label)
    except ValueError:
        return -1

    if label_base == "one":
        value -= 1

    if 0 <= value <= 3:
        return value
    return -1


class CandidateTreeScorerDataset(Dataset):
    """Build candidate-conditioned question/path examples.

    Each item is one (question, answer candidate, reasoning path) example.
    The training loss aggregates path scores back to four candidate option
    scores at question level.
    """

    node_extra_size = 4
    edge_extra_size = 3
    path_extra_size = 4

    def __init__(
        self,
        raw_samples,
        emb_dict,
        max_hops=3,
        max_paths_per_root=200,
        max_paths_per_sample=1000,
        max_paths_per_candidate=64,
        max_pos_per_candidate=16,
        max_hard_neg_per_candidate=24,
        add_reverse_edges=True,
        label_base="zero",
        num_options=4,
        mode="train",
        cache_path=None,
        use_cache=True,
        cache_version="candidate_v2",
    ):
        self.emb_dict = emb_dict
        self.max_hops = max_hops
        self.max_paths_per_root = max_paths_per_root
        self.max_paths_per_sample = max_paths_per_sample
        self.max_paths_per_candidate = max_paths_per_candidate
        self.max_pos_per_candidate = max_pos_per_candidate
        self.max_hard_neg_per_candidate = max_hard_neg_per_candidate
        self.add_reverse_edges = add_reverse_edges
        self.label_base = label_base
        self.num_options = num_options
        self.mode = mode
        self.cache_path = cache_path
        self.use_cache = use_cache
        self.cache_version = cache_version

        self.data_list = []
        self.qid2indices = {}
        self.stats = {}

        if self._maybe_load_cache():
            return

        print(
            "Processing "
            f"{len(raw_samples)} samples for candidate-aware TreeScorer..."
        )
        self._process(raw_samples)
        self._maybe_save_cache()

    def _cache_config(self):
        return {
            "max_hops": self.max_hops,
            "max_paths_per_root": self.max_paths_per_root,
            "max_paths_per_sample": self.max_paths_per_sample,
            "max_paths_per_candidate": self.max_paths_per_candidate,
            "max_pos_per_candidate": self.max_pos_per_candidate,
            "max_hard_neg_per_candidate": self.max_hard_neg_per_candidate,
            "add_reverse_edges": self.add_reverse_edges,
            "label_base": self.label_base,
            "num_options": self.num_options,
            "mode": self.mode,
        }

    def _maybe_load_cache(self):
        if not self.use_cache or not self.cache_path:
            return False
        if not os.path.exists(self.cache_path):
            return False
        payload = torch.load(self.cache_path, map_location="cpu")
        if (
            payload.get("cache_version") != self.cache_version
            or payload.get("config") != self._cache_config()
        ):
            print(f"Cache config mismatch, rebuilding: {self.cache_path}")
            return False
        self.data_list = payload.get("data_list", [])
        self.qid2indices = payload.get("qid2indices", {})
        self.stats = payload.get("stats", {})
        print(f"Loaded cached CandidateTreeScorerDataset: {self.cache_path}")
        return True

    def _maybe_save_cache(self):
        if not self.use_cache or not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        payload = {
            "cache_version": self.cache_version,
            "config": self._cache_config(),
            "data_list": self.data_list,
            "qid2indices": self.qid2indices,
            "stats": self.stats,
        }
        torch.save(payload, self.cache_path)
        print(f"Saved CandidateTreeScorerDataset cache to: {self.cache_path}")

    def _process(self, raw_samples):
        missing_embs = 0
        missing_options = 0
        missing_option_embs = 0
        missing_gold = 0
        raw_paths = 0
        support_pos = 0
        support_total = 0

        for sample_idx, sample in enumerate(tqdm(raw_samples)):
            sample_id = sample["id"]
            if sample_id not in self.emb_dict:
                missing_embs += 1
                continue

            options = _normalize_options(sample)
            if len(options) < self.num_options:
                missing_options += 1
                continue
            options = options[: self.num_options]

            gold_idx = _gold_index(sample, self.label_base)
            if gold_idx < 0 and self.mode != "test":
                missing_gold += 1
                continue

            sample_embs = self.emb_dict[sample_id]
            option_embs = sample_embs.get("option_embs")
            if option_embs is None or option_embs.shape[0] < len(options):
                missing_option_embs += 1
                continue

            q_emb = sample_embs.get("question_stem_emb", sample_embs["q_emb"])
            q_emb = q_emb.view(1, -1).cpu()
            option_embs = option_embs[: len(options)].cpu()
            local_ent_embs = sample_embs["entity_embs"].cpu()
            local_rel_embs = sample_embs["relation_embs"].cpu()

            num_entities = len(sample.get("text_entity_list", [])) + len(
                sample.get("non_text_entity_list", [])
            )
            local_ent_embs = self._pad_non_text_entity_embs(
                local_ent_embs, num_entities, q_emb.shape[-1]
            )

            nx_g = self._build_nx_graph(sample)
            root_ids = sample.get("q_entity_id_list", [])
            if not root_ids:
                continue

            option_entity_id_lists = self._option_entity_id_lists(sample, options)
            question_path_records = self._enumerate_paths_from_roots(nx_g, root_ids)
            raw_paths += len(question_path_records)

            start_idx = len(self.data_list)
            for candidate_idx, (candidate_label, _) in enumerate(options):
                candidate_entity_ids = set(option_entity_id_lists[candidate_idx])
                candidate_path_records = self._enumerate_paths_from_roots(
                    nx_g,
                    sorted(candidate_entity_ids),
                )
                path_records = self._dedupe_records(
                    question_path_records + candidate_path_records
                )
                raw_paths += len(candidate_path_records)
                selected_records = self._select_records_for_candidate(
                    path_records,
                    candidate_entity_ids,
                    candidate_idx,
                    gold_idx,
                )
                if not selected_records:
                    selected_records = [
                        self._null_record(root_ids[0], sample.get("a_entity_id_list", []))
                    ]

                candidate_emb = option_embs[candidate_idx].view(1, -1)
                for record in selected_records:
                    data = self._path_to_data(
                        record,
                        candidate_idx=candidate_idx,
                        candidate_entity_ids=candidate_entity_ids,
                        gold_idx=gold_idx,
                        q_emb=q_emb,
                        candidate_emb=candidate_emb,
                        local_ent_embs=local_ent_embs,
                        local_rel_embs=local_rel_embs,
                        sample_idx=sample_idx,
                    )
                    if data is None:
                        continue
                    data.qid = torch.tensor([sample_idx], dtype=torch.long)
                    self.data_list.append(data)
                    support_total += 1
                    support_pos += int(float(data.y.item()) > 0.5)

            end_idx = len(self.data_list)
            if end_idx > start_idx:
                self.qid2indices[sample_idx] = list(range(start_idx, end_idx))

        self.stats = {
            "missing_embs": missing_embs,
            "missing_options": missing_options,
            "missing_option_embs": missing_option_embs,
            "missing_gold": missing_gold,
            "raw_paths": raw_paths,
            "support_pos": support_pos,
            "support_total": support_total,
            "support_pos_ratio": support_pos / max(support_total, 1),
        }
        print(f"[CandidateTreeGen] stats={self.stats}")

    def _option_entity_id_lists(self, sample, options):
        option_entity_id_lists = sample.get("option_entity_id_lists", [])
        if len(option_entity_id_lists) >= len(options):
            return option_entity_id_lists[: len(options)]

        entity_list = list(sample.get("text_entity_list", [])) + list(
            sample.get("non_text_entity_list", [])
        )
        entity2id = {entity: idx for idx, entity in enumerate(entity_list)}

        metadata = sample.get("metadata", {}) or {}
        candidate_entities = metadata.get("candidate_entities", [])
        candidate_by_label = {}
        if isinstance(candidate_entities, list):
            for candidate in candidate_entities:
                if not isinstance(candidate, dict):
                    continue
                label = str(candidate.get("label", "")).strip().upper()
                entities = candidate.get("entities", []) or []
                candidate_by_label[label] = [
                    entity2id[entity]
                    for entity in entities
                    if entity in entity2id
                ]

        return [
            candidate_by_label.get(label, [])
            for label, _ in options
        ]

    def _select_records_for_candidate(
        self,
        path_records,
        candidate_entity_ids,
        candidate_idx,
        gold_idx,
    ):
        if not path_records:
            return []

        positives = []
        anchored = []
        other_negatives = []
        for record in path_records:
            path_nodes = set(record["path_nodes"])
            leaf_is_candidate = record["leaf_id"] in candidate_entity_ids
            path_has_candidate = bool(path_nodes & candidate_entity_ids)
            is_positive = (
                candidate_idx == gold_idx
                and path_has_candidate
                and bool(candidate_entity_ids)
            )
            if is_positive:
                positives.append(record)
            elif path_has_candidate or leaf_is_candidate:
                anchored.append(record)
            else:
                other_negatives.append(record)

        if self.mode == "train":
            if len(positives) > self.max_pos_per_candidate:
                positives = random.sample(positives, self.max_pos_per_candidate)
            selected = list(positives)

            hard_budget = min(
                self.max_hard_neg_per_candidate,
                max(self.max_paths_per_candidate - len(selected), 0),
            )
            if len(anchored) > hard_budget:
                selected.extend(random.sample(anchored, hard_budget))
            else:
                selected.extend(anchored)

            budget = max(self.max_paths_per_candidate - len(selected), 0)
            if len(other_negatives) > budget:
                selected.extend(random.sample(other_negatives, budget))
            else:
                selected.extend(other_negatives[:budget])
            return selected

        selected = []
        selected.extend(positives[: self.max_pos_per_candidate])
        selected.extend(anchored[: self.max_hard_neg_per_candidate])
        budget = max(self.max_paths_per_candidate - len(selected), 0)
        selected.extend(other_negatives[:budget])
        return selected[: self.max_paths_per_candidate]

    def _build_nx_graph(self, sample):
        g = nx.MultiDiGraph()
        h_list = sample["h_id_list"]
        t_list = sample["t_id_list"]
        r_list = sample["r_id_list"]
        for i in range(len(h_list)):
            g.add_edge(
                h_list[i],
                t_list[i],
                key=f"f_{i}",
                r_id=r_list[i],
                triple_id=i,
                direction=1,
            )
            if self.add_reverse_edges:
                g.add_edge(
                    t_list[i],
                    h_list[i],
                    key=f"r_{i}",
                    r_id=r_list[i],
                    triple_id=i,
                    direction=-1,
                )
        return g

    def _enumerate_paths_from_roots(self, nx_g, root_ids):
        records = []
        seen = set()
        for root_id in root_ids:
            if root_id not in nx_g:
                continue
            remaining_budget = self.max_paths_per_sample - len(records)
            if remaining_budget <= 0:
                break
            for record in self._enumerate_paths_from_root(
                nx_g,
                root_id,
                max_paths=min(self.max_paths_per_root, remaining_budget),
            ):
                key = tuple(record["path_triple_ids"]) + tuple(record["path_dirs"])
                if key in seen:
                    continue
                seen.add(key)
                records.append(record)
                if len(records) >= self.max_paths_per_sample:
                    break
        return records

    def _dedupe_records(self, records):
        deduped = []
        seen = set()
        for record in records:
            key = (
                tuple(record["path_nodes"]),
                tuple(record["path_triple_ids"]),
                tuple(record["path_dirs"]),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        return deduped

    def _enumerate_paths_from_root(self, nx_g, root_id, max_paths):
        path_records = []
        queue = deque([(root_id, [root_id], [], [], [])])

        while queue and len(path_records) < max_paths:
            cur_node, node_path, rel_path, triple_path, dir_path = queue.popleft()
            if len(rel_path) >= self.max_hops:
                continue

            out_edges = list(nx_g.out_edges(cur_node, keys=True, data=True))
            out_edges.sort(
                key=lambda edge: (
                    edge[3]["triple_id"],
                    -edge[3]["direction"],
                    edge[1],
                )
            )

            for _, next_node, _, edge_data in out_edges:
                if next_node in node_path:
                    continue

                next_node_path = node_path + [next_node]
                next_rel_path = rel_path + [edge_data["r_id"]]
                next_triple_path = triple_path + [edge_data["triple_id"]]
                next_dir_path = dir_path + [edge_data["direction"]]
                path_records.append({
                    "root_id": root_id,
                    "leaf_id": next_node,
                    "path_nodes": next_node_path,
                    "path_rel_ids": next_rel_path,
                    "path_triple_ids": next_triple_path,
                    "path_dirs": next_dir_path,
                })

                if len(path_records) >= max_paths:
                    break
                if len(next_rel_path) < self.max_hops:
                    queue.append(
                        (
                            next_node,
                            next_node_path,
                            next_rel_path,
                            next_triple_path,
                            next_dir_path,
                        )
                    )

        return path_records

    def _null_record(self, root_id, answer_ids):
        return {
            "root_id": root_id,
            "leaf_id": root_id,
            "path_nodes": [root_id],
            "path_rel_ids": [],
            "path_triple_ids": [],
            "path_dirs": [],
        }

    def _pad_non_text_entity_embs(self, local_ent_embs, num_entities, emb_dim):
        if local_ent_embs.shape[0] >= num_entities:
            return local_ent_embs
        num_missing = num_entities - local_ent_embs.shape[0]
        pad = torch.zeros(num_missing, emb_dim, dtype=local_ent_embs.dtype)
        return torch.cat([local_ent_embs, pad], dim=0)

    def _path_to_data(
        self,
        record,
        candidate_idx,
        candidate_entity_ids,
        gold_idx,
        q_emb,
        candidate_emb,
        local_ent_embs,
        local_rel_embs,
        sample_idx,
    ):
        path_nodes = record["path_nodes"]
        edge_r_ids = record["path_rel_ids"]
        path_dirs = record["path_dirs"]
        root_id = record["root_id"]
        leaf_id = record["leaf_id"]

        node_ids_tensor = torch.tensor(path_nodes, dtype=torch.long)
        try:
            x = local_ent_embs[node_ids_tensor]
        except IndexError:
            return None

        num_nodes = len(path_nodes)
        if num_nodes > 1:
            edge_index = torch.tensor(
                [list(range(num_nodes - 1)), list(range(1, num_nodes))],
                dtype=torch.long,
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        edge_r_tensor = torch.tensor(edge_r_ids, dtype=torch.long)
        try:
            edge_attr = local_rel_embs[edge_r_tensor]
        except IndexError:
            return None

        if len(edge_r_ids) == 0:
            edge_attr = local_rel_embs.new_zeros((0, local_rel_embs.shape[-1]))

        leaf_mask = torch.zeros(num_nodes, dtype=torch.bool)
        leaf_mask[-1] = True
        root_mask = torch.zeros(num_nodes, dtype=torch.bool)
        root_mask[0] = True

        node_extra = []
        candidate_node_set = set(candidate_entity_ids)
        for hop_idx, node_id in enumerate(path_nodes):
            node_extra.append([
                1.0 if node_id == root_id else 0.0,
                1.0 if node_id == leaf_id else 0.0,
                1.0 if node_id in candidate_node_set else 0.0,
                hop_idx / max(self.max_hops, 1),
            ])
        node_extra = torch.tensor(node_extra, dtype=torch.float)

        edge_extra = []
        for hop_idx, direction in enumerate(path_dirs):
            edge_extra.append([
                float(direction),
                (hop_idx + 1) / max(self.max_hops, 1),
                1.0 if direction < 0 else 0.0,
            ])
        if edge_extra:
            edge_extra = torch.tensor(edge_extra, dtype=torch.float)
        else:
            edge_extra = torch.zeros((0, self.edge_extra_size), dtype=torch.float)

        path_node_set = set(path_nodes)
        leaf_is_candidate = leaf_id in candidate_node_set
        path_has_candidate = bool(path_node_set & candidate_node_set)
        candidate_has_entity = bool(candidate_node_set)
        path_extra = torch.tensor([[
            len(edge_r_ids) / max(self.max_hops, 1),
            1.0 if leaf_is_candidate else 0.0,
            1.0 if path_has_candidate else 0.0,
            1.0 if candidate_has_entity else 0.0,
        ]], dtype=torch.float)

        label = float(
            candidate_idx == gold_idx
            and path_has_candidate
            and candidate_has_entity
        )

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_extra=edge_extra,
            node_extra=node_extra,
            path_extra=path_extra,
            leaf_mask=leaf_mask,
            root_mask=root_mask,
            q_emb=q_emb,
            candidate_emb=candidate_emb,
            y=torch.tensor([label], dtype=torch.float),
            option_target=torch.tensor([gold_idx], dtype=torch.long),
            candidate_idx=torch.tensor([candidate_idx], dtype=torch.long),
            sample_idx=torch.tensor([sample_idx], dtype=torch.long),
            root_id=torch.tensor([root_id], dtype=torch.long),
            leaf_id=torch.tensor([leaf_id], dtype=torch.long),
            path_node_ids=torch.tensor(path_nodes, dtype=torch.long),
            path_rel_ids=torch.tensor(edge_r_ids, dtype=torch.long),
            path_dir_ids=torch.tensor(path_dirs, dtype=torch.long),
            path_triple_ids=torch.tensor(record["path_triple_ids"], dtype=torch.long),
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
