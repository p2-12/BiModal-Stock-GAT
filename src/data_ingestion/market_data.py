from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def load_watchlist(path: str | Path) -> list[str]:
    """Load ticker symbols from a newline-delimited watchlist file."""
    symbols = [line.strip().upper() for line in Path(path).read_text().splitlines()]
    return [symbol for symbol in symbols if symbol and not symbol.startswith("#")]


def fetch_watchlist_prices(
    symbols: list[str], period: str = "1y", interval: str = "1d"
) -> pd.DataFrame:
    """Download OHLCV history for all symbols using yfinance."""
    if not symbols:
        return pd.DataFrame()
    df = yf.download(
        tickers=symbols,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
    )
    return df
