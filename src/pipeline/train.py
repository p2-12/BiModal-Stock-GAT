from __future__ import annotations

import argparse
import json

import mlflow

from src.config import load_config
from src.pipeline.common import build_model, current_git_commit, load_dataset
from src.train.train_node import set_seed, train_node_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--dataset", required=True)
    args = p.parse_args()

    cfg = load_config(args.config_dir)
    if cfg.train.mlflow_tracking_uri:
        mlflow.set_tracking_uri(cfg.train.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.train.experiment_name)

    dataset = load_dataset(args.dataset)
    out_dim = 3 if cfg.train.task == "classify" else 1
    model = build_model(cfg, out_dim=out_dim)

    set_seed(cfg.train.seed)
    with mlflow.start_run(run_name=cfg.train.run_name):
        mlflow.log_dict(cfg.to_dict(), "config/full_config.json")
        mlflow.log_text(json.dumps(cfg.to_dict(), indent=2), "config/full_config_pretty.json")
        mlflow.log_param("git_commit", current_git_commit())
        mlflow.log_param("seed", cfg.train.seed)
        mlflow.log_param("dataset_version", cfg.data.dataset_version)
        mlflow.log_param("feature_schema_version", cfg.data.feature_schema_version)

        result, splits, thresholds, _ = train_node_model(
            model=model,
            dataset=dataset,
            task=cfg.train.task,
            batch_size=cfg.train.batch_size,
            epochs=cfg.train.epochs,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            patience=cfg.train.patience,
            label_mode=cfg.data.label_mode,
            threshold_k=cfg.data.threshold_k,
        )
        mlflow.log_metrics({"best_val_loss": min(result.history["val_loss"])})
        mlflow.log_dict(result.history, "metrics/history.json")
        mlflow.log_dict({"splits": [s.tolist() for s in splits], "thresholds": thresholds}, "artifacts/splits_thresholds.json")


if __name__ == "__main__":
    main()
