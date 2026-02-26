# aumos-federated-learning

Enterprise federated learning without data sharing — part of the AumOS platform.

## Overview

`aumos-federated-learning` enables multiple organizations to collaboratively train machine learning models without ever sharing raw data. It provides:

- **Framework-agnostic FL** via [Flower](https://flower.ai/): FedAvg, FedProx, SCAFFOLD strategies
- **Differential privacy**: Formal (ε, δ)-DP guarantees on aggregated model updates via Gaussian mechanism
- **Cryptographic secure aggregation**: MPC-based protocol so the coordinator never sees individual participant updates
- **Cross-organization coordination**: Multi-tenant participant scheduling and lifecycle management
- **Automatic synthetic fallback**: Falls back to privacy-engine synthetic data when participation is insufficient
- **Per-round metrics**: Convergence tracking, contribution weights, round history

## Architecture

```
src/aumos_federated_learning/
├── api/
│   ├── router.py        # FastAPI routes (job lifecycle, participants, rounds, model)
│   └── schemas.py       # Pydantic v2 request/response models
├── core/
│   ├── models.py        # SQLAlchemy domain models (FederatedJob, Participant, AggregationRound)
│   ├── interfaces.py    # Protocol definitions (strategy, aggregator, coordinator)
│   └── services.py      # Business logic (JobService, TrainingService, AggregationService, ...)
├── adapters/
│   ├── strategies/
│   │   ├── fedavg.py    # Federated Averaging via Flower
│   │   ├── fedprox.py   # FedProx with proximal term
│   │   └── scaffold.py  # SCAFFOLD for heterogeneous data
│   ├── dp_aggregator.py          # Differentially private aggregation
│   ├── secure_aggregation.py     # Cryptographic (MPC) secure aggregation
│   ├── coordinator.py            # Cross-org coordination
│   ├── synthetic_fallback.py     # Synthetic data fallback via privacy-engine
│   ├── repositories.py           # SQLAlchemy async repositories
│   ├── kafka.py                  # Kafka event publisher/consumer
│   ├── storage.py                # Model artifact storage
│   └── privacy_client.py        # HTTP client for privacy-engine
├── main.py       # FastAPI application factory
└── settings.py   # Pydantic BaseSettings
```

## Quick Start

```bash
cp .env.example .env
docker compose -f docker-compose.dev.yml up -d
```

API available at `http://localhost:8005`. OpenAPI docs at `http://localhost:8005/docs`.

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| POST | `/fl/jobs` | Create a federated learning job |
| GET | `/fl/jobs/{id}` | Job status and progress |
| POST | `/fl/jobs/{id}/start` | Start training |
| POST | `/fl/participants` | Join as a participant |
| GET | `/fl/jobs/{id}/rounds` | Round history |
| POST | `/fl/jobs/{id}/rounds/{num}/submit` | Submit model update |
| GET | `/fl/jobs/{id}/model` | Download aggregated model |
| GET | `/live` | Liveness probe |
| GET | `/ready` | Readiness probe |

## Configuration

See `.env.example` for all configuration variables. Key settings:

| Variable | Description |
|----------|-------------|
| `FL_DEFAULT_STRATEGY` | `fedavg`, `fedprox`, or `scaffold` |
| `FL_DEFAULT_DP_EPSILON` | Privacy budget ε (lower = more private) |
| `FL_DEFAULT_DP_DELTA` | Privacy budget δ |
| `FL_SECURE_AGG_ENABLED` | Enable MPC-based secure aggregation |
| `PRIVACY_ENGINE_URL` | URL of aumos-privacy-engine service |
| `MODEL_REGISTRY_URL` | URL of aumos-model-registry service |

## Dependencies

- **aumos-common**: Auth, DB sessions, events, errors, health
- **aumos-proto**: Protobuf event schemas
- **aumos-privacy-engine**: DP noise calibration and budget accounting
- **aumos-model-registry**: Model artifact storage and versioning

## Table Prefix

All database tables use the `fed_` prefix.

## License

Apache 2.0 — see [LICENSE](LICENSE).
