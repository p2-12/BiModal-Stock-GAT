from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QualityFlag(str, Enum):
    VERIFIED = "verified"
    ESTIMATED = "estimated"
    LOW_CONFIDENCE = "low_confidence"
    STALE = "stale"


@dataclass(frozen=True)
class SchemaMetadata:
    schema_version: str
    model_version: str
    data_version: str
    policy_profile: str


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


@dataclass(frozen=True)
class FeatureRow:
    metadata: SchemaMetadata
    ticker: str
    as_of_date: date
    features: Mapping[str, float] = field(default_factory=dict)
    data_quality_flags: tuple[QualityFlag, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AlphaOutput:
    metadata: SchemaMetadata
    ticker: str
    as_of_date: date
    expected_return: float
    uncertainty: float
    horizon: str


@dataclass(frozen=True)
class RiskOutput:
    metadata: SchemaMetadata
    as_of_date: date
    covariance: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    factor_exposures: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    specific_risk: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizerInput:
    metadata: SchemaMetadata
    as_of_date: date
    constraints: Mapping[str, float] = field(default_factory=dict)
    target_weights: Mapping[str, float] = field(default_factory=dict)
    projected_turnover: float = 0.0


@dataclass(frozen=True)
class OptimizerOutput:
    metadata: SchemaMetadata
    as_of_date: date
    constraints: Mapping[str, float] = field(default_factory=dict)
    target_weights: Mapping[str, float] = field(default_factory=dict)
    projected_turnover: float = 0.0


class RecommendationLabel(str, Enum):
    NEW_BUY = "NEW_BUY"
    ADD = "ADD"
    REDUCE = "REDUCE"
    EXIT = "EXIT"
    HOLD = "HOLD"


@dataclass(frozen=True)
class RecommendationOutput:
    metadata: SchemaMetadata
    ticker: str
    as_of_date: date
    recommendation: RecommendationLabel
    rationale: str


class SchemaMetadataModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    model_version: str
    data_version: str
    policy_profile: str


class FeatureRowModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: SchemaMetadataModel
    ticker: str
    as_of_date: date
    features: dict[str, float] = Field(default_factory=dict)
    data_quality_flags: list[QualityFlag] = Field(default_factory=list)


class AlphaOutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: SchemaMetadataModel
    ticker: str
    as_of_date: date
    expected_return: float
    uncertainty: float = Field(ge=0.0)
    horizon: str


class RiskOutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: SchemaMetadataModel
    as_of_date: date
    covariance: dict[str, dict[str, float]] = Field(default_factory=dict)
    factor_exposures: dict[str, dict[str, float]] = Field(default_factory=dict)
    specific_risk: dict[str, float] = Field(default_factory=dict)


class OptimizerInputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: SchemaMetadataModel
    as_of_date: date
    constraints: dict[str, float] = Field(default_factory=dict)
    target_weights: dict[str, float] = Field(default_factory=dict)
    projected_turnover: float = Field(ge=0.0, default=0.0)


class OptimizerOutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: SchemaMetadataModel
    as_of_date: date
    constraints: dict[str, float] = Field(default_factory=dict)
    target_weights: dict[str, float] = Field(default_factory=dict)
    projected_turnover: float = Field(ge=0.0, default=0.0)


class RecommendationOutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: SchemaMetadataModel
    ticker: str
    as_of_date: date
    recommendation: RecommendationLabel
    rationale: str = Field(min_length=1)

    @field_validator("rationale")
    @classmethod
    def _trimmed_rationale(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("rationale cannot be empty")
        return trimmed


def ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def validate_confidence(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"confidence must be in [0,1], got {value}")
    return value
