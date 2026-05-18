import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool, global_max_pool, global_mean_pool


def _graph_matrix(value):
    if value.dim() == 3 and value.size(1) == 1:
        return value.squeeze(1)
    return value


class CandidateTreeScorer(nn.Module):
    """Candidate-conditioned reasoning path scorer.

    The model scores a (question, candidate option, evidence path) tuple.
    Question-level training aggregates path scores into four option scores.
    """

    def __init__(
        self,
        emb_size,
        hidden_size,
        num_layers=2,
        heads=2,
        dropout=0.2,
        node_extra_size=4,
        edge_extra_size=3,
        path_extra_size=4,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.path_extra_size = path_extra_size

        node_in_size = 5 * emb_size + node_extra_size
        edge_in_size = 3 * emb_size + edge_extra_size

        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.convs = nn.ModuleList([
            GATConv(
                hidden_size,
                hidden_size,
                heads=heads,
                concat=False,
                edge_dim=hidden_size,
                add_self_loops=False,
            )
            for _ in range(num_layers)
        ])
        self.conv_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

        self.pool_gate = nn.Sequential(
            nn.Linear(hidden_size + 2 * emb_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        graph_in_size = (
            5 * hidden_size
            + 4 * emb_size
            + path_extra_size
        )
        self.scorer = nn.Sequential(
            nn.Linear(graph_in_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, data):
        x = data.x
        device = x.device
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        q_emb = _graph_matrix(data.q_emb)
        candidate_emb = _graph_matrix(data.candidate_emb)
        if q_emb.size(0) == 1 and num_graphs > 1:
            q_emb = q_emb.expand(num_graphs, -1)
        if candidate_emb.size(0) == 1 and num_graphs > 1:
            candidate_emb = candidate_emb.expand(num_graphs, -1)

        node_extra = getattr(data, "node_extra", None)
        if node_extra is None:
            node_extra = x.new_zeros((x.size(0), 4))

        q_node = q_emb[batch]
        c_node = candidate_emb[batch]
        node_input = torch.cat(
            [
                x,
                x * q_node,
                x * c_node,
                torch.abs(x - q_node),
                torch.abs(x - c_node),
                node_extra,
            ],
            dim=-1,
        )
        h = self.node_encoder(node_input)

        edge_attr = data.edge_attr
        edge_extra = getattr(data, "edge_extra", None)
        if edge_extra is None:
            edge_extra = edge_attr.new_zeros((edge_attr.size(0), 3))

        if edge_attr.numel() > 0:
            edge_batch = batch[data.edge_index[0]]
            q_edge = q_emb[edge_batch]
            c_edge = candidate_emb[edge_batch]
            edge_input = torch.cat(
                [
                    edge_attr,
                    edge_attr * q_edge,
                    edge_attr * c_edge,
                    edge_extra,
                ],
                dim=-1,
            )
            edge_h = self.edge_encoder(edge_input)
        else:
            edge_h = h.new_zeros((0, self.hidden_size))

        for conv, norm in zip(self.convs, self.conv_norms):
            residual = h
            msg = conv(h, data.edge_index, edge_attr=edge_h)
            h = norm(residual + F.dropout(F.relu(msg), p=self.dropout, training=self.training))

        gate_input = torch.cat([h, q_emb[batch], candidate_emb[batch]], dim=-1)
        gate = torch.sigmoid(self.pool_gate(gate_input))
        attn_pool = global_add_pool(h * gate, batch, size=num_graphs)
        gate_sum = global_add_pool(gate, batch, size=num_graphs).clamp_min(1e-6)
        attn_pool = attn_pool / gate_sum

        mean_pool = global_mean_pool(h, batch, size=num_graphs)
        max_pool = global_max_pool(h, batch, size=num_graphs)

        leaf_batch = batch[data.leaf_mask]
        leaf_pool = global_mean_pool(h[data.leaf_mask], leaf_batch, size=num_graphs)
        root_batch = batch[data.root_mask]
        root_pool = global_mean_pool(h[data.root_mask], root_batch, size=num_graphs)

        path_extra = getattr(data, "path_extra", None)
        if path_extra is None:
            path_extra = h.new_zeros((num_graphs, self.path_extra_size))
        path_extra = _graph_matrix(path_extra)

        graph_input = torch.cat(
            [
                attn_pool,
                mean_pool,
                max_pool,
                leaf_pool,
                root_pool,
                q_emb,
                candidate_emb,
                q_emb * candidate_emb,
                torch.abs(q_emb - candidate_emb),
                path_extra,
            ],
            dim=-1,
        )
        return self.scorer(graph_input)
