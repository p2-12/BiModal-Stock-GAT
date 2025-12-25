# Bimodal Correlation-Weighted GAT for Stock Movement Prediction

This project scaffolds a **bimodal, per-day dynamic graph** model:

- **Nodes**: stocks (tickers)
- **Node features**:
  - **Price**: a lookback window of technical features (e.g., log-returns, RSI, MACD)
  - **Text**: a FinBERT-style embedding of aggregated news for that stock/date (optional)
- **Edges**: learned relationships approximated by **rolling return correlation** computed *only from the past lookback window*
- **Edge attributes**: (1) |corr|, (2) sign(corr)
- **Model**: (Price encoder) + (Text projection/gating) -> **GATv2** with `edge_attr` -> per-node classifier/regressor

## Scope / Next Steps

- **Still in progress**
- Add ablations: no-text, no-graph, static-graph vs dynamic-graph.
- Sweep `CORR_W` and `STRIDE`; correlation graphs are often unstable at short windows.
- Hyperparameter tuning and alternative loss function
- Persist metrics + plots (train/val curves, calibration, attention diagnostics).
- Finally, a more robust analysis/evaluation of the model/results

