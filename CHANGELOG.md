# Changelog

All notable changes to aumos-federated-learning will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-26

### Added
- Initial scaffolding for aumos-federated-learning service
- Hexagonal architecture: api/, core/, adapters/ layers
- Core domain models: FederatedJob, Participant, AggregationRound (SQLAlchemy with TenantMixin)
- Protocol interfaces: FLStrategyProtocol, AggregatorProtocol, DPAggregatorProtocol, CoordinatorProtocol, SyntheticFallbackProtocol
- Core services: JobService, TrainingService, AggregationService, CoordinationService, FallbackService, MetricsService
- FL strategies via Flower framework: FedAvg, FedProx, SCAFFOLD
- Differentially private secure aggregation adapter
- Cryptographic (MPC-based) secure aggregation adapter
- Cross-organization coordination adapter
- Synthetic data fallback adapter
- REST API: job lifecycle, participant management, round submission, model download
- Pydantic v2 request/response schemas
- Repository, Kafka, storage, and privacy client adapters
- FastAPI application with health endpoints
- Settings via Pydantic BaseSettings
- Docker multi-stage build
- GitHub Actions CI workflow
- docker-compose.dev.yml for local development
- Full standard deliverables: CLAUDE.md, README, pyproject.toml, Makefile, etc.
