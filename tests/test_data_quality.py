import pandas as pd
import numpy as np


def null_rate(df: pd.DataFrame) -> pd.Series:
    return df.isna().mean()


def stale_timestamps(df: pd.DataFrame, ts_col: str, max_age_days: int, now: pd.Timestamp) -> int:
    return int((now - pd.to_datetime(df[ts_col], utc=True)).dt.days.gt(max_age_days).sum())


def duplicate_headlines(df: pd.DataFrame) -> int:
    return int(df["headline"].str.lower().duplicated().sum())


def outlier_returns(close: pd.Series, z_thresh: float = 6.0) -> int:
    r = np.log(close / close.shift(1)).dropna()
    z = (r - r.mean()) / (r.std() + 1e-12)
    return int((z.abs() > z_thresh).sum())


def feature_drift_vs_baseline(curr: pd.DataFrame, baseline: pd.DataFrame, thresh: float = 0.2) -> list[str]:
    drift = []
    for col in sorted(set(curr.columns).intersection(baseline.columns)):
        mu_curr = float(curr[col].mean())
        mu_base = float(baseline[col].mean())
        denom = abs(mu_base) + 1e-12
        if abs(mu_curr - mu_base) / denom > thresh:
            drift.append(col)
    return drift


def test_quality_functions_smoke():
    df = pd.DataFrame({"headline": ["A", "a", "B"], "ts": ["2026-01-01", "2026-01-02", "2026-05-01"]})
    assert duplicate_headlines(df) == 1
    assert stale_timestamps(df, "ts", max_age_days=20, now=pd.Timestamp("2026-05-06", tz="UTC")) >= 1
