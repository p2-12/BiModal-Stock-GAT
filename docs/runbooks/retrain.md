# Retrain Runbook
1. Verify provider freshness and dataset build success.
2. Launch `python -m src.pipeline.train --dataset <path>`.
3. Confirm MLflow metrics and model artifacts.
4. Execute offline eval before promotion.
