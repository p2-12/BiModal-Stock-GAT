# WIP
# Bimodal Correlation-Weighted GAT for Stock Movement Prediction

## Training/Evaluation Pipeline

This project now includes dedicated pipeline entrypoints in `src/pipeline/` and structured configs in `configs/`:

- `configs/data.yaml`
- `configs/model.yaml`
- `configs/train.yaml`

### 1) First training run

```bash
python -m src.pipeline.train \
  --config-dir configs \
  --dataset artifacts/graph_dataset.pt
```

### 2) Re-training run (new cut date + benchmark gate)

```bash
python -m src.pipeline.retrain \
  --new-cut-date 2026-05-01 \
  --backfill-days 120 \
  --candidate-run-id <mlflow_run_id> \
  --champion-model-name bimodal-stock-gat
```

### 3) Offline evaluation

By run id:

```bash
python -m src.pipeline.evaluate \
  --config-dir configs \
  --dataset artifacts/graph_dataset.pt \
  --run-id <mlflow_run_id>
```

By registry URI (name+stage):

```bash
python -m src.pipeline.evaluate \
  --config-dir configs \
  --dataset artifacts/graph_dataset.pt \
  --model-uri models:/bimodal-stock-gat/Staging
```

### 4) Model promotion flow

Register weights + signature:

```bash
python -m src.pipeline.register \
  --config-dir configs \
  --dataset artifacts/graph_dataset.pt \
  --model-path artifacts/model.pt \
  --name bimodal-stock-gat
```

Automatic promote-to-staging logic is handled by `src/pipeline/retrain.py`, which compares the candidate metric against the current production champion and only promotes on improvement.

## App package (FastAPI + Streamlit)

A new `app/` package provides:
- FastAPI backend for prediction, graph snapshots, MLflow metrics, and live monitor endpoints.
- Streamlit frontend with 3 tabs:
  1. Network View
  2. Training & Evaluation
  3. Live Inference Monitor

### Local deployment

Start everything (MLflow + API + UI + optional Postgres):

```bash
docker compose up --build
```

Services:
- Streamlit UI: `http://localhost:8501`
- FastAPI docs: `http://localhost:8000/docs`
- MLflow server: `http://localhost:5000`
