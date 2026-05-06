from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from app.backend.schemas import GraphSnapshotRequest, PredictRequest
from app.backend.service import AppService, AppSettings

settings = AppSettings(
    dataset_path=os.getenv("GRAPH_ARRAYS_PATH", "artifacts/graph_arrays.npz"),
    mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
)
svc = AppService(settings)
app = FastAPI(title="BiModal Stock GAT App API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return svc.predict(req.ticker, req.date)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/graph/snapshot")
def graph_snapshot(req: GraphSnapshotRequest):
    try:
        return svc.snapshot(req.date, req.tickers)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/metrics/runs")
def metrics_runs(experiment_name: str = "Default", max_results: int = 20):
    return {"runs": svc.mlflow_runs(experiment_name=experiment_name, max_results=max_results)}


@app.get("/monitor/live")
def monitor_live():
    return svc.live_monitor()
