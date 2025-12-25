from __future__ import annotations

import torch
import torch.nn as nn


class PriceLSTMEncoder(nn.Module):
    """Encode a [S, L, F] sequence into [S, H]."""
    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=1, batch_first=True, bidirectional=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, price_seq: torch.Tensor) -> torch.Tensor:
        # price_seq: [S, L, F] or [B*S, L, F] depending on packing
        _, (h, _) = self.lstm(price_seq)
        out = h[-1]  # [*, H]
        return self.drop(out)


class TextProjector(nn.Module):
    """Project fixed embeddings (e.g., FinBERT CLS) into fusion space."""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
