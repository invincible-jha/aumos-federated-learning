# Contributing to aumos-federated-learning

Thank you for your interest in contributing to the AumOS Federated Learning service.

## Development Setup

```bash
git clone https://github.com/muveraai/aumos-federated-learning
cd aumos-federated-learning
pip install -e ".[dev]"
```

## Running Tests

```bash
make test
```

## Code Style

We use `ruff` for linting and formatting, and `mypy` for type checking:

```bash
make lint
make format
make typecheck
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`: `feature/`, `fix/`, `docs/`
2. Write tests alongside your implementation
3. Ensure all checks pass: `make all`
4. Submit a PR with a clear description of the change and motivation

## Commit Messages

Follow Conventional Commits:
- `feat:` — new feature
- `fix:` — bug fix
- `refactor:` — code restructuring
- `docs:` — documentation only
- `test:` — test changes
- `chore:` — build/tooling changes

Commit messages should explain WHY, not just WHAT.

## Privacy and Security

This service handles sensitive federated learning workloads. Any changes to:
- Differential privacy mechanisms
- Secure aggregation protocols
- Cryptographic operations
- Participant data handling

...must be reviewed by a security-aware maintainer. See SECURITY.md for reporting vulnerabilities.

## Architecture

Follow the hexagonal architecture pattern:
- `api/` — FastAPI routers and Pydantic schemas (system boundary)
- `core/` — Domain models, services, and protocol interfaces
- `adapters/` — External integrations (DB, Kafka, Flower, Privacy Engine)
