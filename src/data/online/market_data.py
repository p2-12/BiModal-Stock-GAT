from __future__ import annotations

import datetime as dt
import time
from typing import Iterable, Protocol

import pandas as pd
import yfinance as yf

from ..contracts.schemas import OhlcvBar
from ..contracts.validation import validate_ohlcv_bar


class MarketDataProvider(Protocol):
    name: str

    def fetch_bars(
        self, ticker: str, start_utc: dt.datetime, end_utc: dt.datetime
    ) -> list[OhlcvBar]: ...


class YFinanceProvider:
    name = "yfinance"

    def fetch_bars(
        self, ticker: str, start_utc: dt.datetime, end_utc: dt.datetime
    ) -> list[OhlcvBar]:
        df = yf.download(
            ticker, start=start_utc.date(), end=end_utc.date(), progress=False, auto_adjust=False
        )
        events = []
        for idx, row in df.iterrows():
            bar = OhlcvBar(
                ticker=ticker,
                timestamp_utc=pd.Timestamp(idx).to_pydatetime(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
                source=self.name,
                ingestion_time_utc=dt.datetime.now(dt.timezone.utc),
                confidence=0.9,
                quality_flag="verified",
            )
            events.append(validate_ohlcv_bar(bar))
        return events


class MarketDataClient:
    def __init__(
        self,
        providers: list[MarketDataProvider],
        max_retries: int = 3,
        rate_limit_seconds: float = 0.2,
    ):
        self.providers = providers
        self.max_retries = max_retries
        self.rate_limit_seconds = rate_limit_seconds

    def fetch_with_failover(
        self, ticker: str, start_utc: dt.datetime, end_utc: dt.datetime
    ) -> list[OhlcvBar]:
        last_error = None
        for provider in self.providers:
            for _ in range(self.max_retries):
                try:
                    out = provider.fetch_bars(ticker, start_utc, end_utc)
                    if out:
                        return out
                except Exception as exc:  # noqa: PERF203
                    last_error = exc
                time.sleep(self.rate_limit_seconds)
        if last_error:
            raise RuntimeError(f"all providers failed for {ticker}") from last_error
        return []

    def backfill(
        self, tickers: Iterable[str], start_utc: dt.datetime, end_utc: dt.datetime
    ) -> dict[str, list[OhlcvBar]]:
        return {ticker: self.fetch_with_failover(ticker, start_utc, end_utc) for ticker in tickers}
