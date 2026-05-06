from __future__ import annotations

from pathlib import Path
import subprocess

import torch

from src.models.corr_gat import CorrWeightedGAT
from src.models.encoders import PriceLSTMEncoder, TextProjector
from src.models.fusion import GatedFusion


def current_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build_model(cfg, out_dim: int):
    price = PriceLSTMEncoder(cfg.model.price_feat_dim, cfg.model.price_hidden, cfg.model.dropout)
    text = TextProjector(cfg.model.text_dim, cfg.model.fusion_dim, cfg.model.dropout)
    fusion = GatedFusion(cfg.model.fusion_dim, cfg.model.dropout)
    return CorrWeightedGAT(
        price_encoder=price,
        text_proj=text,
        fusion=fusion,
        fusion_dim=cfg.model.fusion_dim,
        gat_hidden=cfg.model.gat_hidden,
        gat_heads=cfg.model.gat_heads,
        gat_layers=cfg.model.gat_layers,
        edge_attr_dim=cfg.model.edge_attr_dim,
        dropout=cfg.model.dropout,
        out_dim=out_dim,
    )


def load_dataset(path: str | Path):
    return torch.load(Path(path), map_location="cpu")
