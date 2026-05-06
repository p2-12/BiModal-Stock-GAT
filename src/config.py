from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    lookback: int = 60
    horizon: int = 15
    stride: int = 5
    corr_window: int = 60
    topk: int = 6
    use_abs_corr: bool = True
    min_history: int = 220
    dataset_version: str = "v1"
    feature_schema_version: str = "v1"
    label_mode: str = "std"
    threshold_k: float = 0.5


@dataclass(frozen=True)
class ModelConfig:
    price_feat_dim: int = 5
    price_hidden: int = 64
    text_dim: int = 768
    fusion_dim: int = 128
    gat_hidden: int = 128
    gat_heads: int = 4
    gat_layers: int = 2
    dropout: float = 0.2
    edge_attr_dim: int = 2


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    epochs: int = 30
    patience: int = 8
    task: str = "classify"
    experiment_name: str = "bimodal-stock-gat"
    mlflow_tracking_uri: str | None = None
    run_name: str | None = None


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        out = yaml.safe_load(f) or {}
    if not isinstance(out, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(out).__name__}")
    return out


def load_config(config_dir: str | Path = "configs") -> AppConfig:
    config_dir = Path(config_dir)
    data = DataConfig(**_read_yaml(config_dir / "data.yaml"))
    model = ModelConfig(**_read_yaml(config_dir / "model.yaml"))
    train = TrainConfig(**_read_yaml(config_dir / "train.yaml"))
    cfg = AppConfig(data=data, model=model, train=train)
    validate_config(cfg)
    return cfg


def validate_config(cfg: AppConfig) -> None:
    required = {
        "dataset_version": cfg.data.dataset_version,
        "feature_schema_version": cfg.data.feature_schema_version,
        "label_mode": cfg.data.label_mode,
    }
    missing = [k for k, v in required.items() if v in (None, "")]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    if cfg.data.label_mode not in {"std", "quantile"}:
        raise ValueError("data.label_mode must be one of {'std', 'quantile'}")
    if cfg.train.task not in {"classify", "regress"}:
        raise ValueError("train.task must be one of {'classify', 'regress'}")

    if cfg.model.fusion_dim != cfg.model.price_hidden:
        raise ValueError("Dimension mismatch: model.fusion_dim must equal model.price_hidden")

    if cfg.model.gat_hidden % cfg.model.gat_heads != 0:
        raise ValueError("model.gat_hidden must be divisible by model.gat_heads")

    positive_ints = {
        "data.lookback": cfg.data.lookback,
        "data.horizon": cfg.data.horizon,
        "data.stride": cfg.data.stride,
        "model.gat_layers": cfg.model.gat_layers,
        "train.batch_size": cfg.train.batch_size,
        "train.epochs": cfg.train.epochs,
    }
    bad = [k for k, v in positive_ints.items() if int(v) <= 0]
    if bad:
        raise ValueError(f"Expected strictly positive integers for: {bad}")
