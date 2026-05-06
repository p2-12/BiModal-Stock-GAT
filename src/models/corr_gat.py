from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class CorrWeightedGAT(nn.Module):
    """
    Correlation-weighted GAT over stocks.

    Important:
    - `edge_attr` must be provided and `edge_dim` must match.
    - We use GATv2 rather than standard GAT to reduce the 'static attention' issue.

    Inputs (per snapshot):
        price:        [S, L, F]
        text:         [S, D_text]
        text_mask:    [S]
        edge_index:   [2, E]
        edge_attr:    [E, edge_attr_dim]
    """

    def __init__(
        self,
        price_encoder: nn.Module,
        text_proj: nn.Module,
        fusion: nn.Module,
        fusion_dim: int,
        gat_hidden: int,
        gat_heads: int,
        gat_layers: int,
        edge_attr_dim: int,
        dropout: float,
        out_dim: int,
    ):
        super().__init__()
        self.price_encoder = price_encoder
        self.text_proj = text_proj
        self.fusion = fusion

        self.pre = nn.Sequential(
            nn.Linear(fusion_dim, gat_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        convs = []
        in_dim = gat_hidden
        for _ in range(gat_layers):
            convs.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=gat_hidden // gat_heads,
                    heads=gat_heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_attr_dim,
                    add_self_loops=True,
                    fill_value=0.0,
                )
            )
            in_dim = gat_hidden
        self.convs = nn.ModuleList(convs)

        self.head = nn.Sequential(
            nn.Linear(gat_hidden + fusion_dim, gat_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gat_hidden, out_dim),
        )

    def forward(self, price, text, text_mask, edge_index, edge_attr):
        price_h = self.price_encoder(price)  # [S, Dp]
        text_h = self.text_proj(text)  # [S, D]
        fused = self.fusion(price_h, text_h, text_mask)  # [S, D]

        g = self.pre(fused)
        for conv in self.convs:
            g = conv(g, edge_index, edge_attr=edge_attr)
            g = F.gelu(g)

        logits = self.head(torch.cat([fused, g], dim=-1))
        return logits
