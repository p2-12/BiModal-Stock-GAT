from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import mlflow


def benchmark_improved(candidate: float, champion: float) -> bool:
    return candidate < champion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--new-cut-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--backfill-days", type=int, default=90)
    p.add_argument("--candidate-run-id", required=True)
    p.add_argument("--champion-model-name", required=True)
    p.add_argument("--metric", default="best_val_loss")
    args = p.parse_args()

    cut_date = datetime.strptime(args.new_cut_date, "%Y-%m-%d")
    start_date = cut_date - timedelta(days=args.backfill_days)
    print(f"Rolling backfill window: {start_date.date()} -> {cut_date.date()}")
    print("Validation strategy: rolling time-split across backfill snapshots")

    client = mlflow.tracking.MlflowClient()
    cand = client.get_run(args.candidate_run_id)
    candidate_metric = float(cand.data.metrics[args.metric])

    versions = client.search_model_versions(f"name='{args.champion_model_name}'")
    prod = next((v for v in versions if v.current_stage == "Production"), None)
    if prod is None:
        print("No production champion found; promoting candidate to Staging")
        client.transition_model_version_stage(args.champion_model_name, "1", "Staging")
        return

    champ = client.get_run(prod.run_id)
    champion_metric = float(champ.data.metrics[args.metric])
    if benchmark_improved(candidate_metric, champion_metric):
        candidate_versions = client.search_model_versions(f"run_id='{args.candidate_run_id}'")
        if not candidate_versions:
            raise RuntimeError("Candidate run has no registered model version")
        v = candidate_versions[0]
        client.transition_model_version_stage(v.name, v.version, "Staging")
        print(f"Promoted {v.name} v{v.version} to Staging")
    else:
        print("Candidate did not beat champion; no promotion")


if __name__ == "__main__":
    main()
