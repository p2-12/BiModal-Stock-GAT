from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping, Optional


class QualityFlag(str, Enum):
    VERIFIED = "verified"
    ESTIMATED = "estimated"
    LOW_CONFIDENCE = "low_confidence"
    STALE = "stale"


@dataclass(frozen=True)
class OhlcvBar:
    ticker: str
    timestamp_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str
    ingestion_time_utc: datetime
    confidence: float
    quality_flag: QualityFlag


@dataclass(frozen=True)
class CorporateAction:
    ticker: str
    timestamp_utc: datetime
    action_type: str
    value: float
    source: str
    ingestion_time_utc: datetime
    confidence: float
    quality_flag: QualityFlag


@dataclass(frozen=True)
class NewsEvent:
    ticker: str
    timestamp_utc: datetime
    headline: str
    content: str
    language: str
    source: str
    ingestion_time_utc: datetime
    confidence: float
    quality_flag: QualityFlag
    article_id: Optional[str] = None


@dataclass(frozen=True)
class EmbeddingPayload:
    ticker: str
    timestamp_utc: datetime
    source: str
    ingestion_time_utc: datetime
    confidence: float
    quality_flag: QualityFlag
    vector: tuple[float, ...] = field(default_factory=tuple)
    metadata: Mapping[str, str] = field(default_factory=dict)


def ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def validate_confidence(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"confidence must be in [0,1], got {value}")
    return value
