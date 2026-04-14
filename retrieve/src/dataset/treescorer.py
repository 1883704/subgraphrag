import os
import torch
import networkx as nx
from collections import deque
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm
import random


class TreeScorerDataset(Dataset):
    def __init__(
        self,
        raw_samples,
        emb_dict,
        max_hops=3,
        max_paths_per_root=200,
        max_paths_per_sample=1000,
        add_reverse_edges=True,
        mode="train",
        max_neg_per_pos=20,
        max_neg_per_sample=80,
        cache_path=None,
        use_cache=True,
        cache_version="v3",
    ):
        """
        mode: "train" 时对每个问题做负样本下采样；"eval" / "test" 时不过采样
        max_neg_per_pos: 每个正样本最多配多少个负样本
        max_neg_per_sample: 每个问题最多保留多少个负样本（兜底上限）
        """
        self.emb_dict = emb_dict
        self.max_hops = max_hops
        self.max_paths_per_root = max_paths_per_root
        self.max_paths_per_sample = max_paths_per_sample
        self.add_reverse_edges = add_reverse_edges
        self.mode = mode
        self.max_neg_per_pos = max_neg_per_pos
        self.max_neg_per_sample = max_neg_per_sample
        self.cache_path = cache_path
        self.use_cache = use_cache
        self.cache_version = cache_version

        self.tree_data_list = []
        # 可选：question 级索引
        self.qid2indices = {}

        if self._maybe_load_cache():
            return

        print(f"Processing {len(raw_samples)} samples with pre-computed embeddings...")
        self._process(raw_samples)
        self._maybe_save_cache()

    def _cache_config(self):
        return {
            "max_hops": self.max_hops,
            "max_paths_per_root": self.max_paths_per_root,
            "max_paths_per_sample": self.max_paths_per_sample,
            "add_reverse_edges": self.add_reverse_edges,
            "mode": self.mode,
            "max_neg_per_pos": self.max_neg_per_pos,
            "max_neg_per_sample": self.max_neg_per_sample,
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

        self.tree_data_list = payload.get("tree_data_list", [])
        self.qid2indices = payload.get("qid2indices", {})
        self.num_pos = payload.get("num_pos", 0)
        self.num_total = payload.get("num_total", 0)
        self.pos_ratio = payload.get("pos_ratio", 0.0)
        print(f"Loaded cached TreeScorerDataset: {self.cache_path}")
        return True

    def _maybe_save_cache(self):
        if not self.use_cache or not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        payload = {
            "cache_version": self.cache_version,
            "config": self._cache_config(),
            "tree_data_list": self.tree_data_list,
            "qid2indices": self.qid2indices,
            "num_pos": getattr(self, "num_pos", 0),
            "num_total": getattr(self, "num_total", 0),
            "pos_ratio": getattr(self, "pos_ratio", 0.0),
        }
        torch.save(payload, self.cache_path)
        print(f"Saved TreeScorerDataset cache to: {self.cache_path}")

    def _process(self, raw_samples):
        missing_embs = 0

        raw_total_paths = 0  # 未采样前的路径总数
        pos_global = 0  # 采样后全局正样本数
        neg_global = 0  # 采样后全局负样本数

        for sample_idx, sample in enumerate(tqdm(raw_samples)):
            sample_id = sample["id"]

            # 1) 检查 embedding 是否存在
            if sample_id not in self.emb_dict:
                missing_embs += 1
                continue

            sample_embs = self.emb_dict[sample_id]
            q_emb = sample_embs["q_emb"].view(1, -1).cpu()
            local_ent_embs = sample_embs["entity_embs"].cpu()
            local_rel_embs = sample_embs["relation_embs"].cpu()

            num_entities = len(sample.get("text_entity_list", [])) + len(
                sample.get("non_text_entity_list", [])
            )
            local_ent_embs = self._pad_non_text_entity_embs(
                local_ent_embs, num_entities, q_emb.shape[-1]
            )

            # 2) 构建 NetworkX 图
            nx_g = self._build_nx_graph(sample)

            # 3) 枚举 root→leaf 路径，先放入 per-question 列表
            root_ids = sample["q_entity_id_list"]
            ans_ids = set(sample["a_entity_id_list"])

            pos_list = []  # 该 question 下的所有正路径
            neg_list = []  # 该 question 下的所有负路径

            for root_id in root_ids:
                if root_id not in nx_g:
                    continue

                remaining_budget = self.max_paths_per_sample - len(pos_list) - len(neg_list)
                if remaining_budget <= 0:
                    break

                path_records = self._enumerate_paths_from_root(
                    nx_g,
                    root_id,
                    max_paths=min(self.max_paths_per_root, remaining_budget),
                )

                for leaf_id, path_nodes, edge_r_ids, triple_ids in path_records:

                    tree_data = self._path_to_pyg_data(
                        path_nodes,
                        edge_r_ids,
                        triple_ids,
                        root_id,
                        leaf_id,
                        ans_ids,
                        q_emb,
                        local_ent_embs,
                        local_rel_embs,
                        sample_idx,
                    )
                    if tree_data is None:
                        continue

                    # 打上 qid：用于后续 per-question 排序 loss
                    tree_data.qid = torch.tensor([sample_idx], dtype=torch.long)

                    raw_total_paths += 1

                    label = float(tree_data.y.item())
                    if label == 1.0:
                        pos_list.append(tree_data)
                    else:
                        neg_list.append(tree_data)

            # 4) 该 question 内做负样本下采样（只在 train 模式）
            if self.mode == "train":
                kept_pos = pos_list  # 正样本全部保留

                if len(pos_list) > 0:
                    max_neg_by_ratio = self.max_neg_per_pos * len(pos_list)
                else:
                    # 没有正样本的问题：可以给一个固定较小上限，避免撑爆
                    max_neg_by_ratio = self.max_neg_per_sample

                max_neg_for_sample = min(max_neg_by_ratio, self.max_neg_per_sample)

                if len(neg_list) > max_neg_for_sample:
                    kept_neg = random.sample(neg_list, max_neg_for_sample)
                else:
                    kept_neg = neg_list
            else:
                # eval/test 模式：不做采样，全部保留
                kept_pos = pos_list
                kept_neg = neg_list

            # 把该 question 的样本写入大列表，并记录 qid→indices
            start_idx = len(self.tree_data_list)
            self.tree_data_list.extend(kept_pos)
            self.tree_data_list.extend(kept_neg)

            end_idx = len(self.tree_data_list)
            if end_idx > start_idx:
                self.qid2indices[sample_idx] = list(range(start_idx, end_idx))

            pos_global += len(kept_pos)
            neg_global += len(kept_neg)

        print(
            f"[TreeGen] raw_paths={raw_total_paths} from raw_samples={len(raw_samples)}"
        )
        if missing_embs > 0:
            print(
                f"⚠️ Warning: {missing_embs} samples skipped due to missing embeddings."
            )

        total_kept = pos_global + neg_global
        if total_kept > 0:
            pos_ratio = pos_global / total_kept
            print(
                f"[LabelStats] kept_paths={total_kept}, "
                f"pos={pos_global}, neg={neg_global}, "
                f"pos_ratio={pos_ratio:.6f}"
            )
            self.num_pos = pos_global
            self.num_total = total_kept
            self.pos_ratio = pos_ratio
        else:
            self.num_pos = 0
            self.num_total = 0
            self.pos_ratio = 0.0

    def _build_nx_graph(self, sample):
        g = nx.MultiDiGraph()
        h_list = sample["h_id_list"]
        t_list = sample["t_id_list"]
        r_list = sample["r_id_list"]
        # 这里假设 h_list 里的 id 已经是对应 local_ent_embs 的索引
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

    def _enumerate_paths_from_root(self, nx_g, root_id, max_paths):
        path_records = []
        queue = deque([(root_id, [root_id], [], [])])

        while queue and len(path_records) < max_paths:
            cur_node, node_path, rel_path, triple_path = queue.popleft()
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
                path_records.append((
                    next_node,
                    next_node_path,
                    next_rel_path,
                    next_triple_path,
                ))

                if len(path_records) >= max_paths:
                    break
                if len(next_rel_path) < self.max_hops:
                    queue.append((
                        next_node,
                        next_node_path,
                        next_rel_path,
                        next_triple_path,
                    ))

        return path_records

    def _pad_non_text_entity_embs(self, local_ent_embs, num_entities, emb_dim):
        """
        The text encoder only embeds text-bearing entities. Processed entity IDs
        put non-text entities after text entities, so pad them with zero vectors
        instead of dropping paths that touch them.
        """
        if local_ent_embs.shape[0] >= num_entities:
            return local_ent_embs

        num_missing = num_entities - local_ent_embs.shape[0]
        pad = torch.zeros(num_missing, emb_dim, dtype=local_ent_embs.dtype)
        return torch.cat([local_ent_embs, pad], dim=0)

    def _path_to_pyg_data(
        self,
        path_nodes,
        edge_r_ids,
        triple_ids,
        root_id,
        leaf_id,
        ans_ids,
        q_emb,
        local_ent_embs,
        local_rel_embs,
        sample_idx,
    ):
        """
        参数变化：接收 local_ent_embs 和 local_rel_embs
        """
        # --- A. ID Mapping ---
        node_to_local_idx = {global_id: i for i, global_id in enumerate(path_nodes)}
        local_leaf_idx = node_to_local_idx[leaf_id]
        num_nodes = len(path_nodes)

        # --- B. Edge Extraction ---
        src_list = list(range(num_nodes - 1))
        dst_list = list(range(1, num_nodes))

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        # --- C. Features (关键修改) ---
        # 这里的 path_nodes 本身就是局部 ID (对应 local_ent_embs 的行号)
        # 所以直接用 path_nodes 取值即可
        node_ids_tensor = torch.tensor(path_nodes, dtype=torch.long)

        # 从局部矩阵中取出节点特征
        try:
            x = local_ent_embs[node_ids_tensor]
        except IndexError:
            # 保护机制：如果图构建时的ID超出了Embedding矩阵范围
            return None

        # 从局部矩阵中取出关系特征
        edge_r_tensor = torch.tensor(edge_r_ids, dtype=torch.long)
        try:
            edge_attr = local_rel_embs[edge_r_tensor]
        except IndexError:
            return None

        # --- D. Mask & Label ---
        leaf_mask = torch.zeros(num_nodes, dtype=torch.bool)
        leaf_mask[local_leaf_idx] = True

        label = 1.0 if leaf_id in ans_ids else 0.0
        y = torch.tensor(label, dtype=torch.float)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            leaf_mask=leaf_mask,
            q_emb=q_emb,
            y=y,
            sample_idx=torch.tensor([sample_idx], dtype=torch.long),
            root_id=torch.tensor([root_id], dtype=torch.long),
            leaf_id=torch.tensor([leaf_id], dtype=torch.long),
            path_node_ids=torch.tensor(path_nodes, dtype=torch.long),
            path_rel_ids=torch.tensor(edge_r_ids, dtype=torch.long),
            path_triple_ids=torch.tensor(triple_ids, dtype=torch.long),
            num_nodes=num_nodes,
        )
        return data

    def __len__(self):
        return len(self.tree_data_list)

    def __getitem__(self, idx):
        return self.tree_data_list[idx]
