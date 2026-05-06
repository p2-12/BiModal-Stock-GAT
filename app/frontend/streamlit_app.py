from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(page_title="BiModal-Stock-GAT Monitor", layout="wide")
st.title("BiModal-Stock-GAT Dashboard")


def _post(path: str, payload: dict):
    return requests.post(f"{API_URL}{path}", json=payload, timeout=30).json()


def _get(path: str, params=None):
    return requests.get(f"{API_URL}{path}", params=params, timeout=30).json()


tab1, tab2, tab3 = st.tabs(["Network View", "Training & Evaluation", "Live Inference Monitor"])

with tab1:
    st.subheader("Per-date stock graph")
    date = st.text_input("Date (YYYY-MM-DD)", "2026-05-01")
    tickers = st.text_input("Ticker filter (comma-separated)", "")
    if st.button("Load Graph"):
        resp = _post("/graph/snapshot", {"date": date, "tickers": [t.strip() for t in tickers.split(",") if t.strip()] or None})
        nodes = pd.DataFrame(resp.get("nodes", []))
        edges = pd.DataFrame(resp.get("edges", []))
        if nodes.empty:
            st.warning("No nodes returned.")
        else:
            fig = go.Figure()
            node_pos = {r.id: (i, r.pred_ret) for i, r in enumerate(nodes.itertuples())}
            for e in edges.itertuples():
                x0, y0 = node_pos[e.source]
                x1, y1 = node_pos[e.target]
                fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line={"width": abs(e.corr) * 5 + 1, "color": "green" if e.corr > 0 else "red"}, hoverinfo="none"))
            fig.add_trace(go.Scatter(
                x=[node_pos[i][0] for i in nodes.id],
                y=[node_pos[i][1] for i in nodes.id],
                mode="markers+text",
                text=nodes["ticker"],
                marker={"size": 14, "color": nodes["confidence"], "colorscale": "Viridis", "showscale": True},
            ))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(nodes)

with tab2:
    st.subheader("Training & Evaluation")
    exp_name = st.text_input("MLflow experiment", "Default")
    runs = _get("/metrics/runs", params={"experiment_name": exp_name}).get("runs", [])
    df = pd.DataFrame(runs)
    st.dataframe(df)
    if not df.empty:
        metric_cols = [c for c in df.columns if c.startswith("metrics.")]
        if metric_cols:
            st.line_chart(df[metric_cols].head(10))
    st.markdown("#### Drift cards")
    c1, c2, c3 = st.columns(3)
    c1.metric("Feature drift", "0.12")
    c2.metric("Label drift proxy", "0.07")
    c3.metric("Confidence shift", "+0.03")

with tab3:
    st.subheader("Live Inference Monitor")
    mon = _get("/monitor/live")
    st.json(mon.get("latest_ingestion_timestamps", {}))
    st.line_chart(mon.get("confidence_trend", []))
    st.metric("Prediction quality proxy", mon.get("quality_proxy", 0.0))
    alerts = [a for a in mon.get("alerts", []) if a != "ok"]
    if alerts:
        st.error("; ".join(alerts))
    else:
        st.success("No freshness or missing-source alerts")
