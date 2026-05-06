from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error

from src.config import load_config
from src.pipeline.common import build_model, load_dataset
from src.services.logging import get_logger, new_trace_id
from src.train.train_node import _collect_batch_outputs, make_time_splits, pick_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--dataset", required=True)
    p.add_argument("--run-id")
    p.add_argument("--model-uri")
    args = p.parse_args()

    cfg = load_config(args.config_dir)
    logger = get_logger("evaluate")
    run_trace_id = new_trace_id()
    dataset = load_dataset(args.dataset)
    out_dim = 3 if cfg.train.task == "classify" else 1
    model = build_model(cfg, out_dim=out_dim)

    if args.run_id:
        artifact = mlflow.artifacts.download_artifacts(run_id=args.run_id, artifact_path="model.pt")
        model.load_state_dict(torch.load(artifact, map_location="cpu"))
    elif args.model_uri:
        pyfunc = mlflow.pyfunc.load_model(args.model_uri)
        native = pyfunc._model_impl.python_model.model
        model.load_state_dict(native.state_dict())
    else:
        raise ValueError("Specify --run-id or --model-uri")

    _, _, te_idx = make_time_splits(len(dataset))
    te_loader = torch.utils.data.DataLoader(
        [dataset[i] for i in te_idx], batch_size=cfg.train.batch_size
    )
    device = pick_device()
    y, y_reg, pred, probs = _collect_batch_outputs(
        model.to(device), te_loader, device, cfg.train.task
    )
    logger.info(
        "evaluation_outputs_collected",
        extra={"run_trace_id": run_trace_id, "dataset_version": cfg.data.dataset_version},
    )

    with mlflow.start_run(run_name="offline-eval"):
        if cfg.train.task == "classify":
            report = classification_report(y, pred, output_dict=True)
            flat = {
                f"cls_{k}_{m}": v
                for k, d in report.items()
                if isinstance(d, dict)
                for m, v in d.items()
            }
            mlflow.log_metrics(flat)

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(np.zeros((3, 3)))
            ax.set_title("Confusion Matrix Placeholder")
            fp = Path("confusion_matrix.png")
            fig.savefig(fp)
            plt.close(fig)
            mlflow.log_artifact(str(fp), "plots")

            frac_pos, mean_pred = calibration_curve((y == 1).astype(int), probs[:, 1], n_bins=10)
            cal_fig, cal_ax = plt.subplots()
            cal_ax.plot(mean_pred, frac_pos)
            cal_ax.set_title("Calibration")
            cfp = Path("calibration.png")
            cal_fig.savefig(cfp)
            plt.close(cal_fig)
            mlflow.log_artifact(str(cfp), "plots")
        else:
            mlflow.log_metric("rmse", float(np.sqrt(mean_squared_error(y_reg, pred))))
            mlflow.log_metric("mae", float(mean_absolute_error(y_reg, pred)))

        drift = float(abs(np.mean(pred) - np.mean(y_reg if y_reg is not None else y)))
        mlflow.log_metric("prediction_target_mean_drift", drift)


if __name__ == "__main__":
    main()
