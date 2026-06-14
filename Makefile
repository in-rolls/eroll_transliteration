.PHONY: help install dev-install test lint format type-check ci

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install: ## Install package
	uv sync

dev-install: ## Install with dev dependencies
	uv sync --group dev

test: ## Run tests
	uv run python -m pytest

lint: ## Run linter
	uv run ruff check eroll/ tests/

format: ## Format code
	uv run black eroll/ tests/
	uv run isort eroll/ tests/

type-check: ## Run type checker
	uv run mypy eroll/

ci: lint type-check test ## Run all CI checks
