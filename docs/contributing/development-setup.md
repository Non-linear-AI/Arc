# Development Setup

This guide will help you set up a development environment for contributing to Arc.

## Prerequisites

- **Python 3.12 or higher** (Python 3.13 also supported)
- **Git** for version control
- **uv** for dependency management

## Installing uv

If you don't have `uv` installed:

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.sh | iex"
```

## Cloning the Repository

```bash
# Clone the repository
git clone https://github.com/non-linear-ai/arc
cd arc
```

## Installing Dependencies

```bash
# Install production dependencies
uv sync

# Install with development dependencies (for contributors)
uv sync --dev
```

This installs all dependencies including:
- Production: openai, torch, duckdb, rich, prompt_toolkit, etc.
- Development: pytest, pytest-cov, ruff, pytest-watcher

## Running Arc from Source

```bash
# Start interactive chat session
uv run arc chat

# Or use Make
make run

# Show available commands
uv run arc --help
```

## Development Commands

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Run tests in watch mode (auto-reruns on file changes)
uv run pytest-watcher .

# Or use Make
make test
make test-watch
```

### Code Quality

```bash
# Run linting checks
uv run ruff check .
# Or: make lint

# Fix linting issues and format code
uv run ruff format .
uv run ruff check . --fix
# Or: make format

# Run all quality checks (format, lint, test)
make all

# Run CI checks locally (non-modifying)
make ci
```

### Cleaning Build Artifacts

```bash
make clean
```

### Make Commands Summary

```bash
# Show all available Make targets
make help

# Install and run
make install-dev    # Install development dependencies
make run            # Start Arc CLI in interactive mode

# Testing and quality
make test           # Run tests with coverage
make test-watch     # Run tests in watch mode
make lint           # Run linting checks
make format         # Format code and fix linting issues
make clean          # Clean build artifacts

# Run all quality checks
make all            # Equivalent to: make format lint test
make ci             # Run CI checks locally (non-modifying)
```

## Development Workflow

### 1. Create a Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Edit code, add features, fix bugs.

### 3. Test Your Changes

```bash
# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/test_your_file.py

# Run specific test
uv run pytest tests/test_your_file.py::test_specific_function

# Watch mode for TDD
uv run pytest-watcher .
```

### 4. Format and Lint

```bash
# Format code
uv run ruff format .

# Check and fix linting issues
uv run ruff check . --fix
```

### 5. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add feature: description of what you added"
```

### 6. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

## Project Structure

Understanding the codebase:

```
src/arc/
├── core/           # Agent and client implementation
│   ├── agent.py    # Main ArcAgent orchestrator
│   └── agents/     # Specialized ML agents
├── tools/          # Tool implementations and registry
├── database/       # Database layer with services and models
├── ml/             # ML runtime, training, and evaluation
├── graph/          # Arc-Graph schema definitions and validators
├── ui/             # CLI and interactive interface
├── plugins/        # Plugin system for extensibility
├── editing/        # File editing utilities
├── jobs/           # Job execution and management
├── templates/      # Jinja2 templates for system prompts
├── resources/      # Built-in knowledge files
└── utils/          # Utility functions and helpers
```

## Testing

### Test Organization

```
tests/
├── test_core/      # Core agent tests
├── test_tools/     # Tool tests
├── test_ml/        # ML runtime tests
├── test_graph/     # Arc-Graph validator tests
└── ...
```

### Writing Tests

```python
import pytest
from arc.core.agent import ArcAgent

def test_agent_initialization():
    """Test that agent initializes correctly."""
    agent = ArcAgent()
    assert agent is not None
    assert agent.max_tool_rounds == 50

@pytest.mark.asyncio
async def test_async_agent_operation():
    """Test async agent operations."""
    agent = ArcAgent()
    result = await agent.process("test message")
    assert result is not None
```

### Running Specific Tests

```bash
# Run tests for a specific module
uv run pytest tests/test_core/

# Run with verbose output
uv run pytest -v

# Run with coverage report
uv run pytest --cov=arc --cov-report=html

# Run only failed tests from last run
uv run pytest --lf

# Run tests matching pattern
uv run pytest -k "test_agent"
```

## Code Style

Arc uses Ruff for linting and formatting:

- **Line length**: 88 characters
- **Target**: Python 3.12
- **Import sorting**: isort style
- **Linting**: pycodestyle, pyflakes, bugbear, comprehensions, pyupgrade

### Configuration

See `pyproject.toml` for full configuration.

### Per-file Ignores

- `tests/**/*.py`: Allow unused args and long lines
- `src/arc/tools/**/*.py`: Allow longer lines (error messages)
- `src/arc/ml/**/*.py`: Allow longer lines (complex expressions)

## Debugging

### Interactive Debugging

```python
# Add breakpoint in code
def my_function():
    breakpoint()  # Execution will stop here
    # ...
```

Run with debugger:
```bash
uv run python -m pdb your_script.py
```

### Logging

Arc uses Python's logging module:

```python
import logging

logger = logging.getLogger(__name__)
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

Enable debug logging:
```bash
ARC_LOG_LEVEL=DEBUG uv run arc chat
```

### Testing with Specific Data

```bash
# Use test database
ARC_USER_DB=~/.arc/test.db uv run arc chat
```

## Common Development Tasks

### Adding a New Tool

1. Create tool file in `src/arc/tools/`
2. Implement tool function with proper type hints
3. Register tool in tool registry
4. Add tests in `tests/test_tools/`
5. Update tool documentation

### Adding a New ML Agent

1. Create agent directory in `src/arc/core/agents/`
2. Implement agent with system prompt template
3. Register agent in agent factory
4. Add tests
5. Update documentation

### Adding Built-in Knowledge

1. Create markdown file in `src/arc/resources/knowledge/`
2. Update metadata.yaml
3. Test knowledge loading
4. Add to documentation

### Adding a New Layer Type

1. Add layer definition in `src/arc/graph/`
2. Update validators
3. Add tests
4. Update Arc-Graph documentation

## Troubleshooting

### Import Errors

If you get import errors:
```bash
# Ensure you're using uv run
uv run python your_script.py

# Or activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python your_script.py
```

### Test Failures

If tests fail:
```bash
# Run with verbose output to see details
uv run pytest -v

# Run specific failing test
uv run pytest tests/test_file.py::test_name -v

# Check coverage to see if new code is tested
uv run pytest --cov --cov-report=html
# Open htmlcov/index.html in browser
```

### Linting Errors

If linting fails:
```bash
# See what's wrong
uv run ruff check .

# Auto-fix what can be fixed
uv run ruff check . --fix

# Format code
uv run ruff format .
```

## Getting Help

- Read the [Architecture Overview](architecture.md)
- Check [Testing Guide](testing.md)
- Ask in [GitHub Discussions](https://github.com/non-linear-ai/arc/discussions)
- Look at existing code for examples

## Next Steps

- **[Architecture Overview](architecture.md)** - Understand the codebase
- **[Testing Guide](testing.md)** - Learn testing practices
- **[Contributing Guidelines](../../CONTRIBUTING.md)** - Contribution workflow
