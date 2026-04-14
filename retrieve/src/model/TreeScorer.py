import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class TreeScorer(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers=2, heads=1):
        super(TreeScorer, self).__init__()

        self.convs = nn.ModuleList()

        # 第 1 层：Input(Emb) -> Output(Hidden)
        # 注意：这里开启了 edge_dim 来利用关系特征
        self.convs.append(
            GATConv(emb_size, hidden_size, heads=heads, concat=False, edge_dim=emb_size)
        )

        # 后续层：Input(Hidden) -> Output(Hidden)
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(
                    hidden_size,
                    hidden_size,
                    heads=heads,
                    concat=False,
                    edge_dim=emb_size,
                )
            )

        # 评分头 (MLP)
        self.scorer = nn.Sequential(
            nn.Linear(
                hidden_size + emb_size, hidden_size
            ),  # hidden_size(树) + emb_size(问题)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, data):
        """
        [修改点] 只接收 data 一个参数
        """
        x, edge_index = data.x, data.edge_index

        # [修改点] 从 data 中自动提取 edge_attr 和 q_emb
        # 确保 Dataset 里正确传递了这些属性
        edge_attr = data.edge_attr
        q_emb = data.q_emb

        # 1. GNN Message Passing (Root -> Leaf)
        for conv in self.convs:
            # 传入 edge_attr
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        # 2. Leaf Extraction
        # 利用 leaf_mask 选出每棵树的叶子节点
        leaf_features = x[data.leaf_mask]

        # 3. Fusion with Question
        # 此时 q_emb 可能是 [Batch, 1, Hidden] 或者 [Batch, Hidden]
        # 我们统一 reshape 成 [Batch, Hidden]
        if len(q_emb.shape) == 3:
            q_emb = q_emb.squeeze(1)

        # 拼接: [Batch, Hidden] + [Batch, Emb] -> [Batch, Hidden+Emb]
        # 注意：这里需要确保 leaf_features 和 q_emb 的行数(Batch Size)一致
        final_input = torch.cat([leaf_features, q_emb], dim=1)

        # 4. Scoring
        logits = self.scorer(final_input)  # [Batch, 1]

        return logits
