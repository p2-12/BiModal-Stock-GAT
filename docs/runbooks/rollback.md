# Rollback Runbook
1. Identify last known good model run in MLflow.
2. Repoint serving model URI to previous artifact.
3. Validate prediction and monitoring endpoints.
4. Document rollback reason and incident timeline.
