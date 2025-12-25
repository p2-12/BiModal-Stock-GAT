from dataclasses import dataclass

@dataclass(frozen=True)
class DataConfig:
    lookback: int = 60
    horizon: int = 15
    stride: int = 5  # step between graph snapshots (reduces overlap)
    corr_window: int = 60
    topk: int = 6
    use_abs_corr: bool = True
    min_history: int = 220  # drop tickers with less data than this (after feature eng)

    # label mapping happens after split to avoid leakage:
    # classify via quantiles or std-based thresholds from TRAIN only
    label_mode: str = "std"  # "std" or "quantile"
    threshold_k: float = 0.5 # for std mode: +/- k * std(train_future_ret)

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
    edge_attr_dim: int = 2  # (|corr|, sign)

@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    epochs: int = 30
    patience: int = 8  # early stopping
