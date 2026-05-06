# Architecture

## Layered Components
- `domain/`: core entities, schemas, business invariants.
- `services/`: training, evaluation, feature assembly, inference orchestration.
- `adapters/`: external integrations (data providers, storage, model registry, UI/API).
- `interfaces/`: user/operator entrypoints (CLI and HTTP handlers).

## Data Flow
1. Adapters ingest raw market/news data.
2. Services assemble features and snapshots.
3. Domain schemas validate arrays/graph contracts.
4. Services run training/evaluation and register artifacts.
5. Interfaces expose predictions and monitoring.

## Logging
Structured events include:
- `run_trace_id` for training/evaluation runs.
- `dataset_version` for reproducibility.
- `prediction_trace_id` for per-request inference tracing.
