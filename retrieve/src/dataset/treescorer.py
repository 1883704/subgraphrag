import os
import torch
import networkx as nx
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
        mode="train",
        max_neg_per_pos=20,
        max_neg_per_sample=80,
        cache_path=None,
        use_cache=True,
        cache_version="v1",
    ):
        """
        mode: "train" 时对每个问题做负样本下采样；"eval" / "test" 时不过采样
        max_neg_per_pos: 每个正样本最多配多少个负样本
        max_neg_per_sample: 每个问题最多保留多少个负样本（兜底上限）
        """
        self.emb_dict = emb_dict
        self.max_hops = max_hops
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

                try:
                    paths = nx.single_source_shortest_path(
                        nx_g, root_id, cutoff=self.max_hops
                    )
                except Exception:
                    continue

                for leaf_id, path_nodes in paths.items():
                    # 跳过 trivial path（只有 root 自己）
                    if leaf_id == root_id and len(path_nodes) == 1:
                        continue

                    tree_data = self._path_to_pyg_data(
                        nx_g,
                        path_nodes,
                        root_id,
                        leaf_id,
                        ans_ids,
                        q_emb,
                        local_ent_embs,
                        local_rel_embs,
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
            g.add_edge(h_list[i], t_list[i], r_id=r_list[i])
        return g

    def _path_to_pyg_data(
        self,
        nx_g,
        path_nodes,
        root_id,
        leaf_id,
        ans_ids,
        q_emb,
        local_ent_embs,
        local_rel_embs,
    ):
        """
        参数变化：接收 local_ent_embs 和 local_rel_embs
        """
        # --- A. ID Mapping ---
        node_to_local_idx = {global_id: i for i, global_id in enumerate(path_nodes)}
        local_leaf_idx = node_to_local_idx[leaf_id]
        num_nodes = len(path_nodes)

        # --- B. Edge Extraction ---
        src_list = []
        dst_list = []
        edge_r_ids = []

        for i in range(len(path_nodes) - 1):
            u = path_nodes[i]
            v = path_nodes[i + 1]

            edges_dict = nx_g.get_edge_data(u, v)
            first_key = list(edges_dict.keys())[0]
            r_id = edges_dict[first_key]["r_id"]  # 这是局部关系 ID

            src_list.append(node_to_local_idx[u])
            dst_list.append(node_to_local_idx[v])
            edge_r_ids.append(r_id)

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
            num_nodes=num_nodes,
        )
        return data

    def __len__(self):
        return len(self.tree_data_list)

    def __getitem__(self, idx):
        return self.tree_data_list[idx]
