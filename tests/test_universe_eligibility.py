import types

import numpy as np
import pandas as pd


def _import_build_arrays_with_stub():
    import sys

    sys.modules.setdefault("defeatbeta_api", types.ModuleType("defeatbeta_api"))
    sys.modules.setdefault("defeatbeta_api.data", types.ModuleType("defeatbeta_api.data"))
    ticker_mod = types.ModuleType("defeatbeta_api.data.ticker")
    ticker_mod.Ticker = object
    sys.modules.setdefault("defeatbeta_api.data.ticker", ticker_mod)

    from src.data import build_arrays

    return build_arrays


def test_graph_arrays_persist_eligibility_masks(tmp_path):
    from src.data.graph_dataset import GraphArrays, load_graph_arrays, save_graph_arrays

    arr = GraphArrays(
        price=np.zeros((2, 1, 3, 2), dtype=np.float32),
        text=np.zeros((2, 1, 4), dtype=np.float32),
        text_mask=np.zeros((2, 1), dtype=bool),
        future_ret=np.zeros((2, 1, 1), dtype=np.float32),
        dates=["2026-01-01", "2026-01-02"],
        tickers=["AAA"],
        eligibility_mask=np.array([[True], [False]]),
        unavailable_mask=np.array([[False], [False]]),
        eligibility_liquidity=1_000_000,
        eligibility_price_floor=5,
        eligibility_market_cap_floor=100_000_000,
    )
    out = tmp_path / "arrays.npz"
    save_graph_arrays(str(out), arr)
    loaded = load_graph_arrays(str(out))
    assert loaded.eligibility_mask is not None
    assert loaded.unavailable_mask is not None
    assert loaded.eligibility_mask.tolist() == [[True], [False]]
    assert loaded.unavailable_mask.tolist() == [[False], [False]]


def test_build_graph_arrays_applies_historical_eligibility(monkeypatch):
    build_arrays = _import_build_arrays_with_stub()
    from src.data.price_features import FEATURE_COLS

    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"])
    feat = pd.DataFrame({c: [0.1, 0.2, 0.3, 0.4] for c in FEATURE_COLS}, index=dates)
    feat["Close"] = [10.0, 2.0, 10.0, 10.0]
    per = {
        "AAA": pd.DataFrame(
            {
                "Close": [10.0, 2.0, 10.0, 10.0],
                "Volume": [200_000, 200_000, np.nan, 200_000],
                "MarketCap": [200_000_000, 200_000_000, 200_000_000, 200_000_000],
            },
            index=dates,
        )
    }

    monkeypatch.setattr(
        build_arrays, "build_aligned_feature_panel", lambda *a, **k: (dates, {"AAA": feat})
    )
    monkeypatch.setattr(build_arrays, "to_per_ticker_frames", lambda *a, **k: per)
    monkeypatch.setattr(build_arrays, "download_prices", lambda *a, **k: None)

    arrays = build_arrays.build_graph_arrays(
        tickers=["AAA"],
        start="2026-01-01",
        end="2026-01-05",
        lookback=1,
        horizon=1,
        stride=1,
        use_text=False,
        eligibility_price_floor=5.0,
        eligibility_liquidity=1_000_000.0,
        eligibility_market_cap_floor=100_000_000.0,
    )
    assert arrays.eligibility_mask.tolist() == [[False], [False]]
    assert arrays.unavailable_mask.tolist() == [[False], [True]]
    assert float(arrays.price[0, 0].sum()) == 0.0
    assert float(arrays.price[1, 0].sum()) == 0.0
