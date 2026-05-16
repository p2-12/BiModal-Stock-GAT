from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureQualityConfig:
    """Configuration for fundamental feature quality checks and transforms."""

    null_rate_threshold: float = 0.2
    stale_window_days: int = 5
    outlier_zscore_threshold: float = 5.0


DEFAULT_RELIABILITY_BY_SOURCE: dict[str, float] = {
    "sec_filing": 1.0,
    "company_report": 0.95,
    "vendor": 0.9,
    "scraped": 0.6,
    "unknown": 0.5,
}


class FeatureQualityProcessor:
    """Applies quality checks and deterministic feature transforms.

    Expected input schema:
      - `as_of_date`: date-like column for partitioning.
      - one or more fundamental feature columns (listed in `fundamental_features`).
      - optional `<feature>__source` columns with source labels used for reliability scores.
    """

    def __init__(
        self,
        fundamental_features: list[str],
        config: FeatureQualityConfig | None = None,
        source_reliability: dict[str, float] | None = None,
    ) -> None:
        self.fundamental_features = fundamental_features
        self.config = config or FeatureQualityConfig()
        self.source_reliability = source_reliability or DEFAULT_RELIABILITY_BY_SOURCE

    def run(
        self,
        df: pd.DataFrame,
        snapshot_root: str | Path = "data/features",
    ) -> tuple[pd.DataFrame, dict[str, dict[str, float | bool]]]:
        transformed = self._add_transform_columns(df)
        checks = self._quality_checks(transformed)
        self.persist_daily_snapshots(transformed, snapshot_root=snapshot_root)
        return transformed, checks

    def _add_transform_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for feature in self.fundamental_features:
            if feature not in out.columns:
                out[feature] = np.nan

            missing_mask = out[feature].isna()
            out[f"{feature}__missing"] = missing_mask.astype(np.int8)

            imputed = out[feature].astype(float).copy()
            median = float(imputed.median()) if not imputed.dropna().empty else 0.0
            imputed = imputed.fillna(median)
            out[f"{feature}__imputed"] = imputed

            source_col = f"{feature}__source"
            if source_col in out.columns:
                reliability = (
                    out[source_col]
                    .map(self.source_reliability)
                    .fillna(self.source_reliability["unknown"])
                )
            else:
                reliability = pd.Series(self.source_reliability["unknown"], index=out.index)
            out[f"{feature}__source_reliability"] = reliability.astype(float)

        return out

    def _quality_checks(self, df: pd.DataFrame) -> dict[str, dict[str, float | bool]]:
        checks: dict[str, dict[str, float | bool]] = {}

        as_of = (
            pd.to_datetime(df["as_of_date"], errors="coerce")
            if "as_of_date" in df.columns
            else None
        )

        for feature in self.fundamental_features:
            missing_col = f"{feature}__missing"
            imputed_col = f"{feature}__imputed"

            null_rate = float(df[missing_col].mean()) if missing_col in df.columns else 1.0
            completeness_ok = null_rate <= self.config.null_rate_threshold

            stale_rate = 0.0
            staleness_ok = True
            if as_of is not None and not as_of.isna().all():
                valid_dates = as_of[~df[missing_col].astype(bool)]
                if not valid_dates.empty:
                    days_old = (as_of.max() - valid_dates).dt.days
                    stale_mask = days_old > self.config.stale_window_days
                    stale_rate = float(stale_mask.mean()) if not stale_mask.empty else 0.0
                    staleness_ok = stale_rate == 0.0

            outlier_rate = 0.0
            outliers_ok = True
            vals = df[imputed_col].astype(float)
            std = float(vals.std(ddof=0))
            if std > 0.0:
                z = ((vals - float(vals.mean())) / std).abs()
                outlier_mask = z > self.config.outlier_zscore_threshold
                outlier_rate = float(outlier_mask.mean())
                outliers_ok = outlier_rate == 0.0

            checks[feature] = {
                "null_rate": null_rate,
                "completeness_ok": completeness_ok,
                "stale_rate": stale_rate,
                "staleness_ok": staleness_ok,
                "outlier_rate": outlier_rate,
                "outliers_ok": outliers_ok,
            }

        return checks

    def persist_daily_snapshots(
        self,
        df: pd.DataFrame,
        snapshot_root: str | Path = "data/features",
    ) -> None:
        if "as_of_date" not in df.columns:
            raise ValueError("`as_of_date` column is required to persist partitioned snapshots")

        root = Path(snapshot_root)
        snapshot = df.copy()
        snapshot["as_of_date"] = pd.to_datetime(snapshot["as_of_date"], errors="coerce").dt.date

        for as_of_date, part in snapshot.groupby("as_of_date", dropna=True):
            part_dir = root / f"as_of_date={as_of_date.isoformat()}"
            part_dir.mkdir(parents=True, exist_ok=True)
            part.to_parquet(part_dir / "features.parquet", index=False)


def compute_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible fundamental feature hook.

    Currently returns a copy of the provided dataframe so pipeline imports that
    reference this symbol continue to type-check while the dedicated
    fundamental feature implementation lives elsewhere.
    """

    return df.copy()


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible technical feature hook.

    Currently returns a copy of the provided dataframe so pipeline imports that
    reference this symbol continue to type-check while the dedicated
    technical feature implementation lives elsewhere.
    """

    return df.copy()
