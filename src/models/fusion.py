from __future__ import annotations

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """Fuse price and text embeddings with a learned gate and a missing-news mask.

    price_h: [S, D]
    text_h:  [S, D]
    mask:    [S] (1 if text is present, else 0)

    The gate is constrained to downweight text when mask=0.
    """

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * dim + 1, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, price_h: torch.Tensor, text_h: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if mask.dim() == 1:
            mask = mask.unsqueeze(-1)  # [S,1]

        g = self.gate(torch.cat([price_h, text_h, mask], dim=-1))  # [S,1]
        # If no news, force gate toward price by zeroing the effective text contribution
        g_eff = g * mask  # when mask=0 -> 0
        out = price_h + g_eff * (text_h - price_h)
        return out
