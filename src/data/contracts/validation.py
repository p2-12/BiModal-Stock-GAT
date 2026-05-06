from __future__ import annotations

from datetime import datetime

from .schemas import (
    CorporateAction,
    EmbeddingPayload,
    NewsEvent,
    OhlcvBar,
    ensure_utc,
    validate_confidence,
)


def _validate_required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")
    return value.strip()


def validate_ohlcv_bar(bar: OhlcvBar) -> OhlcvBar:
    _validate_required_text(bar.ticker, "ticker")
    _validate_required_text(bar.source, "source")
    validate_confidence(bar.confidence)
    if bar.high < max(bar.open, bar.close) or bar.low > min(bar.open, bar.close):
        raise ValueError("invalid OHLC bar: high/low bounds inconsistent")
    return OhlcvBar(**{**bar.__dict__, "timestamp_utc": ensure_utc(bar.timestamp_utc), "ingestion_time_utc": ensure_utc(bar.ingestion_time_utc)})


def validate_corporate_action(event: CorporateAction) -> CorporateAction:
    _validate_required_text(event.ticker, "ticker")
    _validate_required_text(event.action_type, "action_type")
    _validate_required_text(event.source, "source")
    validate_confidence(event.confidence)
    return CorporateAction(**{**event.__dict__, "timestamp_utc": ensure_utc(event.timestamp_utc), "ingestion_time_utc": ensure_utc(event.ingestion_time_utc)})


def validate_news_event(event: NewsEvent) -> NewsEvent:
    _validate_required_text(event.ticker, "ticker")
    _validate_required_text(event.headline, "headline")
    _validate_required_text(event.content, "content")
    _validate_required_text(event.language, "language")
    _validate_required_text(event.source, "source")
    validate_confidence(event.confidence)
    return NewsEvent(**{**event.__dict__, "timestamp_utc": ensure_utc(event.timestamp_utc), "ingestion_time_utc": ensure_utc(event.ingestion_time_utc)})


def validate_embedding_payload(payload: EmbeddingPayload) -> EmbeddingPayload:
    _validate_required_text(payload.ticker, "ticker")
    _validate_required_text(payload.source, "source")
    validate_confidence(payload.confidence)
    if not payload.vector:
        raise ValueError("vector is required")
    return EmbeddingPayload(**{**payload.__dict__, "timestamp_utc": ensure_utc(payload.timestamp_utc), "ingestion_time_utc": ensure_utc(payload.ingestion_time_utc)})


def is_stale(timestamp_utc: datetime, max_age_seconds: int, now_utc: datetime) -> bool:
    return (now_utc - ensure_utc(timestamp_utc)).total_seconds() > max_age_seconds
