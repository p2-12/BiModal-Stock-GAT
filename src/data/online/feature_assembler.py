from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..news import NewsAggregator
from ..price_features import FEATURE_COLS, engineer_features
from .market_data import MarketDataClient


class TrainingServingSkewError(RuntimeError):
    pass


class OnlineFeatureAssembler:
    def __init__(self, market_client: MarketDataClient, news_aggregator: NewsAggregator):
        self.market_client = market_client
        self.news_aggregator = news_aggregator

    def assemble(
        self, ticker: str, now_utc: dt.datetime, lookback_days: int = 200
    ) -> dict[str, float]:
        start_utc = now_utc - dt.timedelta(days=lookback_days)
        bars = self.market_client.fetch_with_failover(ticker, start_utc, now_utc)
        if not bars:
            raise RuntimeError(f"no market bars returned for {ticker}")

        df = (
            pd.DataFrame(
                [
                    {
                        "Date": b.timestamp_utc,
                        "Open": b.open,
                        "High": b.high,
                        "Low": b.low,
                        "Close": b.close,
                        "Volume": b.volume,
                    }
                    for b in bars
                ]
            )
            .set_index("Date")
            .sort_index()
        )

        feat = engineer_features(df)
        latest = feat.iloc[-1]
        self._enforce_parity(latest)

        news = self.news_aggregator.fetch_filtered(
            [ticker], now_utc - dt.timedelta(days=3), now_utc
        )
        return {
            **{k: float(latest[k]) for k in FEATURE_COLS if k in latest.index},
            "news_count_3d": float(len(news)),
        }

    def _enforce_parity(self, latest: pd.Series) -> None:
        missing = [col for col in FEATURE_COLS if col not in latest.index]
        if missing:
            raise TrainingServingSkewError(f"missing features at serving time: {missing}")
        vals = latest[FEATURE_COLS].astype(float).to_numpy()
        if np.isnan(vals).any() or np.isinf(vals).any():
            raise TrainingServingSkewError("serving features contain NaN/Inf")
