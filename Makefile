.PHONY: help install fmt lint test up down mlflow clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create the virtualenv and install all deps
	uv sync --all-groups

fmt:  ## Format code
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Lint + type-check
	uv run ruff check .
	uv run ruff format --check .

test:  ## Run the test suite
	uv run pytest -q

up:  ## Start local infra (MLflow)
	docker compose up -d

down:  ## Stop local infra
	docker compose down

mlflow:  ## Open the MLflow UI
	open http://localhost:5000

clean:  ## Remove caches and local run artifacts
	rm -rf .ruff_cache .mypy_cache .pytest_cache eval/reports/*.json
