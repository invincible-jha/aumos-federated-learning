# aumos-federated-learning — Agent Context

## Service Purpose

Enterprise federated learning enabling multi-organization collaborative model training without data sharing. Provides differentially private aggregation, cryptographic secure aggregation (MPC), and cross-organization training coordination.

## Tier & Dependencies

- **Tier**: B (Data Science / AI)
- **Depends on**: aumos-common, aumos-proto, aumos-privacy-engine, aumos-model-registry
- **Depended on by**: aumos-agent-framework (trained models), aumos-platform-core (orchestration)

## Database

- **Table prefix**: `fed_`
- Tables: `fed_jobs`, `fed_participants`, `fed_aggregation_rounds`
- RLS tenant isolation: SET `app.current_tenant` on every session (via aumos-common TenantMixin)

## Architecture

Hexagonal layout:
```
api/        → FastAPI routers + Pydantic schemas (system boundary)
core/       → Domain models (SQLAlchemy), Protocol interfaces, Services
adapters/   → DB repos, Kafka, Flower strategies, DP aggregator, secure agg, coordinator, fallback
```

## FL Strategy Notes

- **FedAvg**: Standard weighted averaging. Fast, simple, assumes IID data.
- **FedProx**: Adds proximal term μ‖w - w_t‖² to client objective. More stable on non-IID data.
- **SCAFFOLD**: Uses control variates to correct client drift on heterogeneous data. Most communication-efficient for non-IID.

## Differential Privacy

- Gaussian mechanism applied to aggregated gradients/weights.
- ε and δ are per-job, per-round configurable.
- Budget accounting delegated to `aumos-privacy-engine` via `privacy_client.py`.
- Default: ε=1.0, δ=1e-5, noise_multiplier=1.1, max_grad_norm=1.0.

## Secure Aggregation

- MPC-based: participants mask their updates with secret-shared random vectors that cancel out in sum.
- Coordinator only ever sees the aggregate — never individual participant updates.
- Threshold: configurable fraction of participants must complete for aggregation to proceed.

## Synthetic Fallback

- When actual_participants < min_participants, FallbackService calls privacy-engine to generate synthetic training data.
- Job status changes to `training` with a synthetic_fallback flag in metadata.

## Key Interfaces (core/interfaces.py)

- `FLStrategyProtocol` — configure/aggregate strategy
- `AggregatorProtocol` — plain aggregation
- `DPAggregatorProtocol` — DP-aware aggregation
- `CoordinatorProtocol` — cross-org scheduling
- `SyntheticFallbackProtocol` — fallback trigger

## Environment Variables

See `.env.example`. Critical:
- `PRIVACY_ENGINE_URL` — required for DP budget accounting
- `MODEL_REGISTRY_URL` — required for model artifact storage
- `FLOWER_SSL_ENABLED` — must be true in production
- `FL_SECURE_AGG_ENABLED` — must be true in production

## Coding Conventions

- All function signatures have type hints
- Pydantic v2 for all API schemas and settings
- SQLAlchemy 2.x async sessions
- Conventional commits; PR titles explain WHY
- Test coverage gate: 80%
