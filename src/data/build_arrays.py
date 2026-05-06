import os
import time
import random
import numpy as np
import pandas as pd

from defeatbeta_api.data.ticker import Ticker
from .price_features import (
    engineer_features,
    FEATURE_COLS,
    future_log_return_curve,
    compute_regime_features,
)
from .news import NewsConfig, NewsIndex, FinBERTEmbedder
from .graph_dataset import GraphArrays


def _cache_piece_path(cache_path, tag):
    if cache_path is None:
        return None
    root, ext = os.path.splitext(str(cache_path))
    if ext == "":
        ext = ".pkl"
    safe_tag = str(tag).replace(os.sep, "_")
    return f"{root}.{safe_tag}{ext}"


def _defeatbeta_price_one(ticker, start, end, max_retries, base_sleep, jitter):
    last_err = None
    for attempt in range(max_retries):
        try:
            df = Ticker(ticker).price()
            if df is None or len(df) == 0:
                raise RuntimeError("Empty price() result")

            # expected columns per README example: symbol, report_date, open, close, high, low, volume :contentReference[oaicite:1]{index=1}
            if "report_date" not in df.columns:
                raise RuntimeError(f"Missing report_date for {ticker}")

            df = df.copy()
            df["report_date"] = pd.to_datetime(df["report_date"]).dt.tz_localize(None)

            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            df = df[(df["report_date"] >= start_dt) & (df["report_date"] < end_dt)]

            # normalize to your expected OHLCV columns
            ren = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
            for k in list(ren.keys()):
                if k not in df.columns:
                    raise RuntimeError(f"Missing {k} for {ticker}")

            out = df[["report_date", "open", "high", "low", "close", "volume"]].rename(columns=ren)
            out = out.drop_duplicates(subset=["report_date"]).sort_values("report_date")
            out = out.set_index("report_date")
            out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill()

            return out

        except Exception as e:
            last_err = e
            sleep_s = base_sleep * (2**attempt) + random.random() * jitter
            time.sleep(sleep_s)

    raise RuntimeError(f"defeatbeta price download failed for {ticker}: {last_err}")


def download_prices(
    tickers,
    start,
    end,
    cache_path=None,
    auto_adjust=True,
    batch_size=20,
    max_retries=4,
    base_sleep=0.5,
    jitter=0.25,
    inter_batch_sleep=0.05,
):
    """
    defeatbeta-api downloader (yfinance replacement) that preserves your downstream expectations:

      - returns a DataFrame with MultiIndex columns (ticker, field)
      - each per-ticker field frame has columns: Open, High, Low, Close, Volume
      - supports disk caching:
          * combined cache_path (all tickers)
          * per-ticker caches to avoid refetching if only a few names change

    Notes:
      - auto_adjust is kept only for signature compatibility (defeatbeta already serves adjusted-like series in many cases).
      - batching params are used only for pacing + cache grouping (not for a single bulk API call).
    """
    tickers = list(dict.fromkeys(tickers))
    if len(tickers) == 0:
        return pd.DataFrame()

    # combined cache fast-path
    if cache_path is not None:
        try:
            cached = pd.read_pickle(cache_path)
            if isinstance(cached.columns, pd.MultiIndex) and not cached.empty:
                have = set(cached.columns.levels[0])
                if set(tickers).issubset(have):
                    return cached
        except FileNotFoundError:
            pass

    # pull per ticker (cached)
    per = {}
    batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]
    for bi, batch in enumerate(batches):
        for t in batch:
            tcache = (
                _cache_piece_path(cache_path, f"ticker.{t}") if cache_path is not None else None
            )

            if tcache is not None:
                try:
                    df = pd.read_pickle(tcache)
                    if df is not None and not df.empty:
                        per[t] = df
                        continue
                except FileNotFoundError:
                    pass

            df = _defeatbeta_price_one(
                t, start, end, max_retries=max_retries, base_sleep=base_sleep, jitter=jitter
            )
            per[t] = df

            if tcache is not None:
                df.to_pickle(tcache)

            time.sleep(inter_batch_sleep + random.random() * jitter)

    # combine into MultiIndex dataframe like yf.download(..., group_by="ticker")
    parts = {}
    for t, df in per.items():
        if df is None or df.empty:
            continue
        parts[t] = df[["Open", "High", "Low", "Close", "Volume"]]

    if len(parts) == 0:
        return pd.DataFrame()

    out = pd.concat(parts, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]

    if cache_path is not None:
        out.to_pickle(cache_path)

    return out


def to_per_ticker_frames(raw, tickers):
    out = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if t not in raw.columns.levels[0]:
                continue
            df = raw[t].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[~df.index.duplicated(keep="first")].sort_index()
            out[t] = df
    else:
        # single ticker
        t = tickers[0]
        df = raw.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        out[t] = df
    return out


def _market_proxy_from_universe(per_ticker_frames):
    # Build an equal-weighted market proxy from available closes/volumes
    closes = []
    vols = []

    for t, df in per_ticker_frames.items():
        if df is None or df.empty:
            continue
        if "Close" not in df.columns:
            continue
        c = df["Close"].rename(t)
        closes.append(c)

        if "Volume" in df.columns:
            v = df["Volume"].rename(t)
            vols.append(v)

    if len(closes) == 0:
        raise RuntimeError(
            "Cannot build market proxy: no usable Close series in universe download."
        )

    close_df = pd.concat(closes, axis=1).sort_index()
    # mean across tickers (ignore missing)
    proxy_close = close_df.mean(axis=1)

    if len(vols):
        vol_df = pd.concat(vols, axis=1).reindex(close_df.index)
        proxy_vol = vol_df.sum(axis=1, min_count=1).fillna(0.0)
    else:
        proxy_vol = pd.Series(0.0, index=proxy_close.index)

    # fabricate OHLC from Close (compute_regime_features mainly uses Close anyway)
    proxy = pd.DataFrame(
        {
            "Open": proxy_close,
            "High": proxy_close,
            "Low": proxy_close,
            "Close": proxy_close,
            "Volume": proxy_vol,
        },
        index=proxy_close.index,
    )

    proxy = proxy.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return proxy


def build_aligned_feature_panel(tickers, start, end, cache_prices=None):
    raw = download_prices(tickers, start, end, cache_path=cache_prices, auto_adjust=True)
    per = to_per_ticker_frames(raw, tickers)

    # regime from universe proxy (no SPY call)
    proxy_df = _market_proxy_from_universe(per)
    regime_df = compute_regime_features(proxy_df)

    feats = {}
    bad = []
    for t in tickers:
        if t not in per:
            bad.append((t, "missing_raw"))
            continue
        f = engineer_features(per[t], regime_df=regime_df)
        if f is None or len(f) == 0:
            bad.append((t, "empty_features"))
            continue
        feats[t] = f

    if len(feats) == 0:
        raise RuntimeError("No tickers produced usable engineered features.")

    # 3) Align by intersection of dates to avoid NaN graphs
    common = None
    for t, f in feats.items():
        idx = f.index
        common = idx if common is None else common.intersection(idx)

    if common is None or len(common) == 0:
        msg = "No common dates across tickers after feature engineering."
        if bad:
            msg += f" Bad tickers: {bad[:10]}{'...' if len(bad)>10 else ''}"
        raise RuntimeError(msg)

    common = common.sort_values()

    # keep only common dates
    for t in list(feats.keys()):
        feats[t] = feats[t].loc[common].copy()

    # If some tickers failed, we still return the aligned panel for those that work
    if bad:
        kept = list(feats.keys())
        print(f"[build_aligned_feature_panel] Dropping {len(bad)} tickers with missing/empty data.")
        print(f"[build_aligned_feature_panel] Kept {len(kept)} tickers out of {len(tickers)}.")

    return common, feats


def build_graph_arrays(
    tickers,
    start,
    end,
    lookback,
    horizon,
    stride,
    use_text,
    news_lookback_days=3,
    finbert_model="ProsusAI/finbert",
    cache_prices=None,
    max_text_length=256,
):
    """
    Create arrays for dynamic-graph snapshots aligned by date.

    Outputs unchanged:
      - price:      [T, S, L, F]
      - text:       [T, S, D]
      - text_mask:  [T, S]
      - future_ret: [T, S]
      - dates:      list[str]
      - tickers:    list[str] (may be reduced only if some tickers truly have no usable data)
    """
    dates, feats = build_aligned_feature_panel(tickers, start, end, cache_prices=cache_prices)

    # Use only tickers that actually produced features
    tickers_used = [t for t in tickers if t in feats]
    if len(tickers_used) == 0:
        raise RuntimeError("No tickers available after feature engineering.")

    S = len(tickers_used)
    F = len(FEATURE_COLS)

    # Future return CURVE per ticker (aligned to common dates)
    future_ret_by_t = {}
    for t in tickers_used:
        f = feats[t]
        future_ret_by_t[t] = future_log_return_curve(f["Close"], horizon=horizon)  # [T_total, H]

    # snapshot indices (ensure enough history + enough future)
    start_i = lookback
    end_i = len(dates) - horizon
    idxs = list(range(start_i, end_i, stride))
    if len(idxs) == 0:
        raise RuntimeError(
            "No snapshots available: increase date range or reduce lookback/horizon/stride."
        )

    price_arr = np.zeros((len(idxs), S, lookback, F), dtype=np.float32)
    ret_arr = np.zeros((len(idxs), S, horizon), dtype=np.float32)

    for ti, tkr in enumerate(tickers_used):
        f = feats[tkr]

        missing = [c for c in FEATURE_COLS if c not in f.columns]
        if missing:
            raise RuntimeError(f"Ticker {tkr} missing engineered columns: {missing}")

        X = f[FEATURE_COLS].to_numpy(np.float32)  # [T_total, F]
        rcurve = future_ret_by_t[tkr].to_numpy(np.float32)  # [T_total, H]

        for si, i in enumerate(idxs):
            price_arr[si, ti] = X[i - lookback : i, :]
            ret_arr[si, ti, :] = rcurve[i, :]

    # Optional news
    D_text = 768
    text_arr = np.zeros((len(idxs), S, D_text), dtype=np.float32)
    mask_arr = np.zeros((len(idxs), S), dtype=bool)

    if use_text:
        ncfg = NewsConfig(lookback_days=news_lookback_days)
        nidx = NewsIndex(tickers=tickers_used, cfg=ncfg)

        texts = []
        keys = []
        for si, i in enumerate(idxs):
            d = dates[i]
            for ti, tkr in enumerate(tickers_used):
                txt = nidx.get_window_text(tkr, d)
                keys.append((si, ti))
                texts.append(txt)

        for (si, ti), txt in zip(keys, texts):
            mask_arr[si, ti] = txt != "[NO_NEWS]"

        embedder = FinBERTEmbedder(model_name=finbert_model)
        embs = embedder.encode_texts(texts, batch_size=24, max_length=max_text_length)

        k = 0
        for si in range(len(idxs)):
            for ti in range(S):
                text_arr[si, ti] = embs[k]
                k += 1

        text_arr[~mask_arr] = 0.0

    return GraphArrays(
        price=price_arr,
        text=text_arr,
        text_mask=mask_arr,
        future_ret=ret_arr,
        dates=[dates[i].strftime("%Y-%m-%d") for i in idxs],
        tickers=tickers_used,
    )
