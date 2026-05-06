from __future__ import annotations

import numpy as np


def thresholds_from_train(train_future_ret: np.ndarray, mode: str = "std", k: float = 0.5):
    """Compute leakage-safe thresholds using TRAIN ONLY.

    train_future_ret: [N_train_snapshots, S] or flattened.
    """
    r = np.asarray(train_future_ret).reshape(-1)
    r = r[np.isfinite(r)]
    if mode == "std":
        s = float(np.std(r))
        return (-k * s, +k * s)
    if mode == "quantile":
        lo = float(np.quantile(r, 0.33))
        hi = float(np.quantile(r, 0.67))
        return (lo, hi)
    raise ValueError(f"Unknown mode: {mode}")


def to_3class_labels(future_ret: np.ndarray, lo: float, hi: float) -> np.ndarray:
    y = np.zeros_like(future_ret, dtype=np.int64)
    y[future_ret > hi] = 2
    y[(future_ret >= lo) & (future_ret <= hi)] = 1
    y[future_ret < lo] = 0
    return y
