from __future__ import annotations

from pathlib import Path

from src.data_ingestion import fetch_watchlist_prices, load_watchlist
from src.features import compute_fundamental_features, compute_technical_features
from src.policy import apply_legacy_rules
from src.reporting import build_email_payload, render_html_report


def run_daily_pipeline(watchlist_path: str | Path, recipients: list[str]) -> dict[str, object]:
    """Thin orchestration entrypoint for the daily legacy workflow."""
    symbols = load_watchlist(watchlist_path)
    prices = fetch_watchlist_prices(symbols)
    technical = compute_technical_features(prices, symbols)
    fundamentals = compute_fundamental_features(symbols)
    features = technical.merge(fundamentals, on="symbol", how="left")
    signals = apply_legacy_rules(features)
    html = render_html_report(signals)
    return build_email_payload("Daily Signal Report", html, recipients)
