from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm


@dataclass
class NewsConfig:
    dataset_name: str = "oliverwang15/us_stock_news_with_price"
    text_cols: Tuple[str, str] = ("title", "content")
    ticker_col: str = "stock"
    date_col: str = "exact_trading_date"
    lookback_days: int = 3

class NewsIndex:
    """Loads news rows and exposes fast (ticker, date) -> aggregated text.

    Design note:
    - We aggregate first to daily text, then build per-day rolling windows in `get_window_text`.
    """
    def __init__(self, tickers: Iterable[str], cfg: NewsConfig):
        self.cfg = cfg
        ds = load_dataset(cfg.dataset_name)
        df = ds["train"].to_pandas()

        df = df[[cfg.ticker_col, cfg.date_col, *cfg.text_cols]].dropna().copy()
        df.rename(columns={cfg.ticker_col: "ticker", cfg.date_col: "date"}, inplace=True)
        df["ticker"] = df["ticker"].astype(str)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

        df = df[df["ticker"].isin(list(tickers))].copy()
        if df.empty:
            self.daily = pd.DataFrame(columns=["ticker", "date", "text"])
            return

        df["text"] = df[cfg.text_cols[0]].fillna("") + ". " + df[cfg.text_cols[1]].fillna("")
        daily = df.groupby(["ticker", "date"], sort=True)["text"].apply(lambda x: " ".join(x)).reset_index()

        # store as a MultiIndex series for fast lookup
        self.daily = daily.set_index(["ticker", "date"]).sort_index()

    def get_window_text(self, ticker: str, date: pd.Timestamp) -> str:
        """Aggregate text in [date - lookback_days, date] inclusive."""
        if self.daily.empty:
            return "[NO_NEWS]"

        date = pd.to_datetime(date).normalize()
        start = date - pd.Timedelta(days=self.cfg.lookback_days)
        # Collect by iterating days; this is fast enough for small lookback_days.
        parts = []
        for d in pd.date_range(start, date, freq="D"):
            key = (ticker, d.normalize())
            if key in self.daily.index:
                parts.append(self.daily.loc[key, "text"])
        if not parts:
            return "[NO_NEWS]"
        # newest first is optional; keep chronological by default for readability
        return " ".join(parts)


class FinBERTEmbedder:
    """Minimal embedder to turn aggregated text into a single vector.

    This uses the [CLS] token from the final hidden state, with mean pooling as a fallback.
    Intended for *precomputation* (not in-training) to keep experiments tractable.
    """
    def __init__(self, model_name: str = "ProsusAI/finbert", device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 32, max_length: int = 256) -> np.ndarray:
        outs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT encode"):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc).last_hidden_state  # [B, T, H]
            cls = out[:, 0, :]  # [B, H]
            outs.append(cls.detach().cpu().numpy())
        return np.concatenate(outs, axis=0)
