from __future__ import annotations

import pandas as pd


def _close_series(price_frame: pd.DataFrame, symbol: str) -> pd.Series:
    if isinstance(price_frame.columns, pd.MultiIndex):
        return price_frame[symbol]["Close"].astype(float)
    return price_frame["Close"].astype(float)


def compute_technical_features(price_frame: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Compute simple technical indicators from OHLCV history."""
    rows: list[dict[str, float | str]] = []
    for symbol in symbols:
        close = _close_series(price_frame, symbol)
        rows.append(
            {
                "symbol": symbol,
                "ret_1d": float(close.pct_change().iloc[-1]),
                "ret_5d": float(close.pct_change(5).iloc[-1]),
                "sma_20": float(close.rolling(20).mean().iloc[-1]),
                "sma_50": float(close.rolling(50).mean().iloc[-1]),
                "vol_20d": float(close.pct_change().rolling(20).std().iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def compute_fundamental_features(symbols: list[str]) -> pd.DataFrame:
    """Placeholder for fundamental features.

    This keeps the legacy scoreram module split by ownership domain.
    """
    return pd.DataFrame({"symbol": symbols})
