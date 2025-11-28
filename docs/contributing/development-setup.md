# Development Setup

Quick guide to set up Arc for development.

## Prerequisites

- **Python 3.12 or higher**
- **Git** for version control
- **uv** for dependency management

## Setup Steps

### 1. Install uv

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.sh | iex"
```

### 2. Clone and Install

```bash
# Clone the repository
git clone https://github.com/non-linear-ai/arc
cd arc

# Install with development dependencies
uv sync --dev
```

### 3. Verify Installation

```bash
# Run Arc
uv run arc chat

# Show help
uv run arc --help
```

## Quick Commands

### Using uv

```bash
# Run Arc
uv run arc chat

# Run tests
uv run pytest
uv run pytest --cov

# Format and lint
uv run ruff format .
uv run ruff check . --fix
```

### Using Make

```bash
# Show all available commands
make help

# Development
make run              # Start Arc
make test             # Run tests
make test-watch       # Run tests in watch mode

# Code quality
make format           # Format and fix linting
make lint             # Check code style
make ci               # Run all CI checks locally

# Cleanup
make clean            # Remove build artifacts
```

## Testing

### Run Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run tests in watch mode
uv run pytest-watcher .

# Or use Make
make test
make test-watch
```

### Run Specific Tests

```bash
# Specific module
uv run pytest tests/test_core/

# Specific file
uv run pytest tests/test_core/test_agent.py

# Specific test
uv run pytest tests/test_core/test_agent.py::test_agent_initialization

# Match pattern
uv run pytest -k "test_agent"
```

### Coverage Report

```bash
# Generate HTML coverage report
uv run pytest --cov --cov-report=html

# View report
open htmlcov/index.html  # macOS
```

## Project Structure

```
src/arc/
├── core/           # Agent and client implementation
├── tools/          # Tool implementations and registry
├── database/       # Database layer with services and models
├── ml/             # ML runtime, training, and evaluation
├── graph/          # Arc-Graph schema definitions and validators
├── ui/             # CLI and interactive interface
├── plugins/        # Plugin system for extensibility
├── templates/      # Jinja2 templates for system prompts
├── resources/      # Built-in knowledge files
└── utils/          # Utility functions and helpers

tests/
├── test_core/      # Core agent tests
├── test_tools/     # Tool tests
├── test_ml/        # ML runtime tests
├── test_graph/     # Arc-Graph validator tests
└── ...
```

## Next Steps

- **[Contributing Guidelines](https://github.com/non-linear-ai/arc/blob/main/CONTRIBUTING.md)** - Complete contribution workflow

## Getting Help

- Ask in [GitHub Issues](https://github.com/non-linear-ai/arc/issues)
- Look at existing tests for examples
