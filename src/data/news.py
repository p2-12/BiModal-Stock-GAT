from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from hashlib import sha1
from typing import Iterable, Protocol, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .contracts.schemas import NewsEvent, QualityFlag
from .contracts.validation import validate_news_event


class NewsProvider(Protocol):
    name: str

    def fetch(
        self, tickers: Iterable[str], start_utc: dt.datetime, end_utc: dt.datetime
    ) -> list[NewsEvent]: ...


@dataclass
class NewsConfig:
    dataset_name: str = "oliverwang15/us_stock_news_with_price"
    text_cols: Tuple[str, str] = ("title", "content")
    ticker_col: str = "stock"
    date_col: str = "exact_trading_date"
    lookback_days: int = 3


class OpenSourceNewsProvider:
    name = "opensource_dataset"

    def __init__(self, cfg: NewsConfig):
        self.cfg = cfg

    def fetch(
        self, tickers: Iterable[str], start_utc: dt.datetime, end_utc: dt.datetime
    ) -> list[NewsEvent]:
        ds = load_dataset(self.cfg.dataset_name)
        df = ds["train"].to_pandas()
        df = df[[self.cfg.ticker_col, self.cfg.date_col, *self.cfg.text_cols]].dropna().copy()
        df.rename(columns={self.cfg.ticker_col: "ticker", self.cfg.date_col: "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df[df["ticker"].isin(list(tickers))]
        df = df[(df["date"] >= pd.Timestamp(start_utc)) & (df["date"] <= pd.Timestamp(end_utc))]
        events = []
        for row in df.itertuples(index=False):
            ev = NewsEvent(
                ticker=row.ticker,
                timestamp_utc=row.date.to_pydatetime(),
                headline=getattr(row, self.cfg.text_cols[0]),
                content=getattr(row, self.cfg.text_cols[1]),
                language="en",
                source=self.name,
                ingestion_time_utc=dt.datetime.now(dt.timezone.utc),
                confidence=0.7,
                quality_flag=QualityFlag.ESTIMATED,
                article_id=None,
            )
            events.append(validate_news_event(ev))
        return events


class ScraperNewsProvider:
    name = "scraper"

    def fetch(
        self, tickers: Iterable[str], start_utc: dt.datetime, end_utc: dt.datetime
    ) -> list[NewsEvent]:
        return []


class PaidApiNewsProvider:
    name = "paid_api"

    def fetch(
        self, tickers: Iterable[str], start_utc: dt.datetime, end_utc: dt.datetime
    ) -> list[NewsEvent]:
        return []


class NewsAggregator:
    def __init__(self, providers: list[NewsProvider], source_rank: dict[str, int] | None = None):
        self.providers = providers
        self.source_rank = source_rank or {"paid_api": 3, "scraper": 2, "opensource_dataset": 1}

    def fetch_filtered(
        self,
        tickers: Iterable[str],
        start_utc: dt.datetime,
        end_utc: dt.datetime,
        language: str = "en",
        min_chars: int = 40,
    ) -> list[NewsEvent]:
        events = []
        for provider in self.providers:
            events.extend(provider.fetch(tickers, start_utc, end_utc))

        filtered = [
            e
            for e in events
            if e.language.lower() == language.lower() and len(e.content.strip()) >= min_chars
        ]
        dedup: dict[str, NewsEvent] = {}
        for event in filtered:
            key = (
                event.article_id
                or sha1(
                    f"{event.ticker}|{event.headline.strip().lower()}|{event.timestamp_utc.isoformat()}".encode()
                ).hexdigest()
            )
            existing = dedup.get(key)
            if existing is None or self.source_rank.get(event.source, 0) > self.source_rank.get(
                existing.source, 0
            ):
                dedup[key] = event
        return list(dedup.values())


class NewsIndex:
    def __init__(self, events: Iterable[NewsEvent], lookback_days: int = 3):
        self.lookback_days = lookback_days
        rows = []
        for e in events:
            rows.append(
                {
                    "ticker": e.ticker,
                    "date": pd.Timestamp(e.timestamp_utc).normalize(),
                    "text": f"{e.headline}. {e.content}",
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            self.daily = pd.DataFrame(columns=["ticker", "date", "text"])
            return
        daily = (
            df.groupby(["ticker", "date"], sort=True)["text"]
            .apply(lambda x: " ".join(x))
            .reset_index()
        )
        self.daily = daily.set_index(["ticker", "date"]).sort_index()

    def get_window_text(self, ticker: str, date: pd.Timestamp) -> str:
        if self.daily.empty:
            return "[NO_NEWS]"
        date = pd.to_datetime(date).normalize()
        start = date - pd.Timedelta(days=self.lookback_days)
        parts = [
            self.daily.loc[(ticker, d), "text"]
            for d in pd.date_range(start, date, freq="D")
            if (ticker, d) in self.daily.index
        ]
        return " ".join(parts) if parts else "[NO_NEWS]"


class FinBERTEmbedder:
    def __init__(self, model_name: str = "ProsusAI/finbert", device: str | None = None):
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def encode_texts(
        self, texts: list[str], batch_size: int = 32, max_length: int = 256
    ) -> np.ndarray:
        outs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT encode"):
            enc = self.tokenizer(
                texts[i : i + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc).last_hidden_state
            outs.append(out[:, 0, :].detach().cpu().numpy())
        return np.concatenate(outs, axis=0)
