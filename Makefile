.PHONY: install test test-quick lint format typecheck clean all

all: lint typecheck test

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=aumos_federated_learning --cov-report=term-missing

test-quick:
	pytest tests/ -x -q --no-header

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/aumos_federated_learning/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	rm -rf dist/ build/ *.egg-info

docker-build:
	docker build -t aumos/federated-learning:dev .

docker-run:
	docker compose -f docker-compose.dev.yml up -d
