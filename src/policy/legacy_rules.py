from __future__ import annotations

import numpy as np
import pandas as pd


def apply_legacy_rules(features: pd.DataFrame, max_position: float = 0.1) -> pd.DataFrame:
    """Legacy action and sizing heuristics migrated from scoreram.py."""
    signals = features.copy()
    momentum = signals["ret_5d"].fillna(0.0)
    signals["action"] = np.where(momentum > 0.01, "BUY", np.where(momentum < -0.01, "SELL", "HOLD"))
    raw_size = momentum.abs() * 2
    signals["target_weight"] = raw_size.clip(0.0, max_position)
    signals.loc[signals["action"] == "HOLD", "target_weight"] = 0.0
    return signals
