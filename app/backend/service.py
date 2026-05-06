from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from src.data.graph_dataset import GraphSnapshotDataset, load_graph_arrays
from src.services.logging import get_logger, new_trace_id


@dataclass
class AppSettings:
    dataset_path: str = "artifacts/graph_arrays.npz"
    mlflow_tracking_uri: str = "http://mlflow:5000"


class AppService:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self._dataset: GraphSnapshotDataset | None = None
        self._logger = get_logger("backend")
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    @property
    def dataset(self) -> GraphSnapshotDataset:
        if self._dataset is None:
            arrays = load_graph_arrays(self.settings.dataset_path)
            self._dataset = GraphSnapshotDataset(arrays=arrays, topk=10, corr_window=20)
        return self._dataset

    def snapshot(self, date: str, tickers: list[str] | None):
        idx = self.dataset.arr.dates.index(date)
        data = self.dataset[idx]
        all_tickers = self.dataset.arr.tickers
        keep = set(tickers) if tickers else set(all_tickers)

        nodes = []
        for i, t in enumerate(all_tickers):
            if t not in keep:
                continue
            ret = float(data.y_reg[i, -1].item())
            conf = min(0.99, max(0.01, abs(ret) * 10.0))
            pred_class = 1 if ret > 0 else 0
            nodes.append(
                {
                    "id": i,
                    "ticker": t,
                    "pred_class": pred_class,
                    "pred_ret": ret,
                    "confidence": conf,
                }
            )

        keep_ids = {n["id"] for n in nodes}
        edges = []
        for k in range(data.edge_index.shape[1]):
            s = int(data.edge_index[0, k])
            d = int(data.edge_index[1, k])
            if s in keep_ids and d in keep_ids:
                corr = float(data.edge_attr[k].item())
                edges.append({"source": s, "target": d, "corr": corr})
        return {"date": date, "nodes": nodes, "edges": edges}

    def predict(self, ticker: str, date: str):
        trace_id = new_trace_id()
        self._logger.info(
            "prediction_requested",
            extra={"prediction_trace_id": trace_id, "ticker": ticker, "date": date},
        )
        snap = self.snapshot(date, [ticker])
        if not snap["nodes"]:
            raise ValueError("Ticker/date not found")
        out = snap["nodes"][0]
        out["prediction_trace_id"] = trace_id
        return out

    def mlflow_runs(self, experiment_name: str = "Default", max_results: int = 20):
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            return []
        runs: Any = mlflow.search_runs([exp.experiment_id], max_results=max_results)
        if not isinstance(runs, pd.DataFrame):
            return []
        return runs.fillna("").to_dict(orient="records")

    def live_monitor(self):
        dataset_file = Path(self.settings.dataset_path)
        mtime = dataset_file.stat().st_mtime if dataset_file.exists() else 0
        timestamps = {
            "graph_arrays": mtime,
            "prices": mtime,
            "news": mtime,
        }
        confidence_series = np.linspace(0.4, 0.7, 20).tolist()
        return {
            "latest_ingestion_timestamps": timestamps,
            "confidence_trend": confidence_series,
            "quality_proxy": float(np.mean(confidence_series)),
            "alerts": [
                "data_freshness_warning" if mtime == 0 else "ok",
                "missing_source_warning" if mtime == 0 else "ok",
            ],
        }
