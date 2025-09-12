.PHONY: help install install-dev test test-watch lint format type-check clean pre-commit

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install:  ## Install production dependencies
	uv sync

install-dev:  ## Install development dependencies
	uv sync --dev
	uv run pre-commit install

test:  ## Run tests with coverage
	uv run pytest

test-watch:  ## Run tests in watch mode
	uv run pytest-watch

lint:  ## Run linting checks
	uv run ruff check .

format:  ## Format code
	uv run ruff format .

clean:  ## Clean build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf dist/ build/ *.egg-info/ .coverage htmlcov/ .pytest_cache/ .ruff_cache/

pre-commit:  ## Run pre-commit hooks on all files
	uv run pre-commit run --all-files

all: format lint test  ## Run all checks
