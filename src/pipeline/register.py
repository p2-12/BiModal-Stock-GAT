from __future__ import annotations

import argparse

import mlflow
import torch
from mlflow.models import infer_signature

from src.config import load_config
from src.pipeline.common import build_model, load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--name", required=True)
    args = p.parse_args()

    cfg = load_config(args.config_dir)
    dataset = load_dataset(args.dataset)
    model = build_model(cfg, out_dim=3 if cfg.train.task == "classify" else 1)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()

    sample = dataset[0]
    with torch.no_grad():
        out = model(sample.price, sample.text, sample.text_mask, sample.edge_index, sample.edge_attr)
    sig = infer_signature(sample.price.numpy(), out.numpy())

    with mlflow.start_run(run_name="register-model") as run:
        mlflow.log_artifact(args.model_path, "weights")
        info = mlflow.pytorch.log_model(model, artifact_path="model", signature=sig, registered_model_name=args.name)
        mv = mlflow.register_model(info.model_uri, args.name)
        client = mlflow.tracking.MlflowClient()
        client.set_model_version_tag(args.name, mv.version, "data_version", cfg.data.dataset_version)
        client.set_model_version_tag(args.name, mv.version, "feature_schema_version", cfg.data.feature_schema_version)
        client.set_model_version_tag(args.name, mv.version, "label_mode", cfg.data.label_mode)
        print(run.info.run_id)


if __name__ == "__main__":
    main()
