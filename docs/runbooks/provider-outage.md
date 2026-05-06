# Provider Outage Runbook
1. Detect outage via freshness/ingestion alerts.
2. Freeze retraining and keep prior model active.
3. Backfill missed windows once provider recovers.
4. Re-run data quality tests and retrain.
