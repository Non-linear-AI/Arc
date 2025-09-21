# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Dependencies and Setup
- **Install dependencies**: `uv sync`
- **Install with dev dependencies**: `uv sync --dev`
- **Run the application**: `uv run arc chat`
- **Show help**: `uv run arc --help`

### Testing
- **Run tests**: `uv run pytest`
- **Run tests with coverage**: `uv run pytest --cov`
- **Run tests in watch mode**: `uv run pytest-watcher .`
- **Run specific test**: `uv run pytest tests/test_specific.py`
- **Run tests with verbose output**: `uv run pytest -v`

### Code Quality
- **Lint check**: `uv run ruff check .`
- **Auto-fix linting**: `uv run ruff check . --fix`
- **Format code**: `uv run ruff format .`
- **Run all quality checks**: `make all` (format, lint, test)

### Alternative Commands via Makefile
- `make install` - Install production dependencies
- `make install-dev` - Install development dependencies
- `make test` - Run tests with coverage
- `make lint` - Run linting checks
- `make format` - Format and fix code
- `make clean` - Clean build artifacts

## Architecture Overview

Arc is an AI-native machine learning tool that transforms natural language questions into predictive models using a declarative schema called "Arc-Graph".

### Core Components

**Main Entry Point**: `src/arc/__init__.py:main()` → `src/arc/ui/cli.py:cli()`

**Key Modules**:
- `src/arc/core/` - Core components (ArcAgent, ArcClient, SettingsManager)
- `src/arc/ui/` - User interface (CLI, console, interactive interface)
- `src/arc/database/` - Database management with DuckDB backend
- `src/arc/graph/` - Arc-Graph schema specification and validation
- `src/arc/ml/` - Machine learning runtime and processors
- `src/arc/tools/` - AI agent tools and capabilities
- `src/arc/jobs/` - Background job management
- `src/arc/plugins/` - Plugin system

### Arc-Graph Schema
The declarative ML schema is defined in `src/arc/graph/spec.py`. This is the core abstraction that allows natural language to be converted into portable ML workflows.

### Database Architecture
- **System Database**: Stores Arc metadata, schemas, and system-level data
- **User Database**: Stores user data, models, and ML results
- **DuckDB Backend**: Unified SQL interface for both databases
- **Services**: Database operations abstracted through service container pattern

### Memory-Efficient Prediction System
- **Automatic Streaming**: Automatically uses streaming for datasets >50k rows to prevent OOM errors
- **Transparent Operation**: No user configuration needed - streaming is automatic
- **Optimized Chunking**: Uses 10k row chunks for optimal memory/performance balance
- **Full Compatibility**: Same simple CLI interface for all dataset sizes

### CLI Commands
The application supports both interactive and headless modes:
- **Interactive mode**: `uv run arc chat` (default)
- **Headless mode**: `uv run arc chat --prompt "your prompt"`
- **Change directory**: `uv run arc chat -d /path/to/directory`
- **Set API key**: `uv run arc chat -k your_api_key`
- **Set model**: `uv run arc chat -m gpt-4`

#### Interactive Mode Commands
- **ML commands**: `/ml create-model`, `/ml train`, `/ml predict`, `/ml jobs`
- **SQL commands**: `/sql use [system|user]`, `/sql <query>`
- **System commands**: `/help`, `/config`, `/stats`, `/tree`, `/clear`, `/exit`

#### ML Command Examples
- **Create model**: `/ml create-model --name my_model --schema path/to/schema.yaml`
- **Train model**: `/ml train --model my_model --data training_table`
- **Basic predict**: `/ml predict --model <name> --data <table>`
- **Predict with output**: `/ml predict --model <name> --data <table> --output <table>`
- **List jobs**: `/ml jobs list`
- **Check job status**: `/ml jobs status <job_id>`

### Configuration
- **User settings**: `~/.arc/user-settings.json`
- **Environment variables**: `ARC_API_KEY`, `ARC_BASE_URL`
- **Project config**: `pyproject.toml` (dependencies, tools, testing)

### Key Dependencies
- **OpenAI**: AI model integration
- **DuckDB**: SQL database backend
- **Click**: CLI framework
- **Rich**: Terminal UI enhancements
- **PyTorch**: ML framework support
- **Pandas/NumPy**: Data processing

### Testing
- Test files located in `tests/` directory
- Uses pytest with coverage reporting
- Coverage reports generated in `htmlcov/`
- Configuration in `pyproject.toml` under `[tool.pytest.ini_options]`