"""Data ingestion utilities for daily pipeline."""

from .market_data import fetch_watchlist_prices, load_watchlist

__all__ = ["load_watchlist", "fetch_watchlist_prices"]
