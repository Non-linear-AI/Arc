.PHONY: help install install-dev run test test-watch lint format clean docs docs-serve

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install:  ## Install production dependencies
	uv sync

install-dev:  ## Install development dependencies
	uv sync --dev

run:  ## Run Arc CLI in interactive mode
	uv run arc chat

test:  ## Run tests with coverage
	uv run pytest

test-watch:  ## Run tests in watch mode
	uv run pytest-watcher .

lint:  ## Run linting checks
	uv run ruff check .

lint-all:  ## Run linting and format checks (CI mode)
	uv run ruff check .
	uv run ruff format --check .

format:  ## Format code and fix linting issues
	uv run ruff format .
	uv run ruff check . --fix

format-check:  ## Check formatting without modifying files
	uv run ruff format --check .

clean:  ## Clean build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf dist/ build/ *.egg-info/ .coverage htmlcov/ .pytest_cache/ .ruff_cache/

all: format lint test  ## Run all checks

ci: lint-all test  ## Run CI checks locally (non-modifying)

docs:
	uv run --group docs mkdocs build

docs-serve:
	uv run --group docs mkdocs serve
