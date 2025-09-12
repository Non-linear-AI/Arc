# Arc

## Development

### Installation

This project uses `uv` for dependency management. If you don't have `uv` installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project and its dependencies:

```bash
# Clone the repository
cd arc

# Install dependencies
uv sync

# Install in editable mode with development dependencies
uv sync --dev
```

### Running the Application

```bash
uv run arc
```

### Running Tests

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Run tests in watch mode
uv run pytest-watch
```

### Code Quality

```bash
# Run linting
uv run ruff check .

# Fix linting issues
uv run ruff check . --fix

# Format code
uv run ruff format .

```

## Configuration

Project configuration is managed through `pyproject.toml`. Key sections include:

- `[project]` - Project metadata and dependencies
- `[tool.uv]` - Development dependencies
- `[tool.ruff]` - Linting and formatting configuration
- `[tool.pytest.ini_options]` - Test configuration
