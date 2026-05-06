import numpy as np
import pandas as pd

FEATURE_COLS = [
    "Log_Ret_1",
    "Log_Ret_5",
    "Log_Ret_10",
    "Mom_20",
    "Mom_60",
    "RV_10",
    "RV_20",
    "RV_60",
    "Downside_RV_20",
    "Z_Close_20",
    "Z_Close_60",
    "EMA_Ratio_12_26",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "TrueRange",
    "ATR_14",
    "HL_Range",
    "OC_Range",
    "Log_Vol_Change",
    "Vol_Z_20",
    "RSI_14",
    "StochK_14",
    "StochD_14",
    "BB_Pos_20",
    "MKT_RV_20",
    "MKT_Trend_60",
    "MKT_Drawdown_126",
    "MKT_RelVol_20_126",
]


def compute_regime_features(market_df):
    m = market_df.copy().sort_index()
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in m.columns]
    m = m[cols].replace([np.inf, -np.inf], np.nan).ffill().bfill()

    close = m["Close"]
    r = np.log(close / close.shift(1))

    mkt_rv_20 = r.rolling(20).std()

    ma_60 = close.rolling(60).mean()
    sd_60 = close.rolling(60).std()
    mkt_trend_60 = (close - ma_60) / (sd_60 + 1e-12)

    roll_max_126 = close.rolling(126).max()
    mkt_drawdown_126 = (close / (roll_max_126 + 1e-12)) - 1.0

    mkt_rv_126 = r.rolling(126).std()
    mkt_relvol_20_126 = mkt_rv_20 / (mkt_rv_126 + 1e-12)

    out = pd.DataFrame(
        {
            "MKT_RV_20": mkt_rv_20,
            "MKT_Trend_60": mkt_trend_60,
            "MKT_Drawdown_126": mkt_drawdown_126,
            "MKT_RelVol_20_126": mkt_relvol_20_126,
        },
        index=m.index,
    )

    return out.replace([np.inf, -np.inf], np.nan)


def engineer_features(df, regime_df=None):
    """
    Create per-ticker OHLCV features and optionally join market regime features.

    Expects df with at least: Open, High, Low, Close, Volume.
    Returns a frame indexed by date with FEATURE_COLS + Close.
    """
    df = df.copy().sort_index()

    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].replace([np.inf, -np.inf], np.nan).ffill().bfill()

    close = df["Close"]
    high = df["High"] if "High" in df.columns else close
    low = df["Low"] if "Low" in df.columns else close
    open_ = df["Open"] if "Open" in df.columns else close
    vol = df["Volume"] if "Volume" in df.columns else pd.Series(1.0, index=df.index)

    # returns / momentum
    df["Log_Ret_1"] = np.log(close / close.shift(1))
    df["Log_Ret_5"] = np.log(close / close.shift(5))
    df["Log_Ret_10"] = np.log(close / close.shift(10))
    df["Mom_20"] = close / close.shift(20) - 1.0
    df["Mom_60"] = close / close.shift(60) - 1.0

    # volatility
    r = df["Log_Ret_1"]
    df["RV_10"] = r.rolling(10).std()
    df["RV_20"] = r.rolling(20).std()
    df["RV_60"] = r.rolling(60).std()
    r_neg = r.where(r < 0.0, 0.0)
    df["Downside_RV_20"] = r_neg.rolling(20).std()

    # trend / mean reversion
    ma_20 = close.rolling(20).mean()
    sd_20 = close.rolling(20).std()
    ma_60 = close.rolling(60).mean()
    sd_60 = close.rolling(60).std()
    df["Z_Close_20"] = (close - ma_20) / (sd_20 + 1e-12)
    df["Z_Close_60"] = (close - ma_60) / (sd_60 + 1e-12)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["EMA_Ratio_12_26"] = ema_12 / (ema_26 + 1e-12) - 1.0

    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # range / ATR
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    df["TrueRange"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR_14"] = df["TrueRange"].rolling(14).mean()
    df["HL_Range"] = (high - low) / (close + 1e-12)
    df["OC_Range"] = (close - open_) / (open_ + 1e-12)

    # volume proxies
    vol_safe = vol.replace(0.0, 1.0)
    df["Log_Vol_Change"] = np.log(vol_safe / vol_safe.shift(1))
    df["Vol_Z_20"] = (vol_safe - vol_safe.rolling(20).mean()) / (vol_safe.rolling(20).std() + 1e-12)

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0.0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0.0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    df["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))

    # stochastic
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    df["StochK_14"] = 100.0 * (close - low_14) / (high_14 - low_14 + 1e-12)
    df["StochD_14"] = df["StochK_14"].rolling(3).mean()

    # Bollinger position
    df["BB_Pos_20"] = (close - (ma_20 - 2.0 * sd_20)) / (4.0 * sd_20 + 1e-12)

    # Join regime features (market-wide)
    if regime_df is not None:
        df = df.join(regime_df, how="left")
        df = df.ffill()

    keep = ["Close"] + [c for c in FEATURE_COLS if c in df.columns]
    df = df[keep].replace([np.inf, -np.inf], np.nan).dropna()
    return df


def future_log_return(close, horizon):
    return np.log(close.shift(-horizon) / close)


def future_log_return_curve(close, horizon):
    out = {}
    for k in range(1, horizon + 1):
        out[k] = np.log(close.shift(-k) / close)
    return pd.DataFrame(out, index=close.index)  # columns: 1..H
