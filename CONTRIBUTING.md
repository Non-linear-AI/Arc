# Contributing to Arc

Thank you for your interest in contributing to Arc! This document provides guidelines and workflow for contributors.

## Quick Links to Technical Details

For detailed technical information, see:
- **[Development Setup Guide](docs/contributing/development-setup.md)** - Setup, testing, commands
- **[Publishing Guide](docs/contributing/publishing.md)** - For maintainers

## Getting Started

**Prerequisites**: Python 3.12+, Git, and uv

### Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/arc && cd arc

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --dev
```

See [Development Setup Guide](docs/contributing/development-setup.md) for detailed instructions.

## Development Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

Use prefixes: `feature/`, `fix/`, `docs/`, `refactor/`, or `test/`

### 2. Make Your Changes

Write clean code, add tests, and update documentation as needed.

### 3. Run Tests

```bash
make test              # Run all tests
make test-watch        # Run tests in watch mode
```

See [Development Setup](docs/contributing/development-setup.md#testing) for detailed testing commands.

### 4. Format and Lint

```bash
make format            # Format and fix linting
make lint              # Check code style
```

### 5. Commit Changes

Write clear commit messages with prefix: `Add feature:`, `Fix bug:`, `Docs:`, `Refactor:`, or `Test:`

```bash
git add .
git commit -m "Add feature: support for custom loss functions"
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Create a pull request with clear title, description of changes, and link to relevant issues.

## Code Style

Arc uses **Ruff** (88 char line length, Python 3.12). Configuration in `pyproject.toml`.

```bash
make format            # Format and fix linting
make lint              # Check code style
```

See [Development Setup](docs/contributing/development-setup.md) for details.

## Testing

Place tests in `tests/` directory mirroring source structure. Aim for 80%+ code coverage.

```bash
make test              # Run all tests
make test-watch        # Run tests in watch mode
```

See [Development Setup](docs/contributing/development-setup.md#testing) for details.

## Documentation

When adding features:
- Add docstrings to all public functions/classes (Google-style)
- Update relevant user guides in `docs/guides/`
- Add examples if applicable
- Update API reference if adding commands

## Submitting Changes

### Pull Request Checklist

Before submitting:
- [ ] Code follows style guidelines (Ruff passes)
- [ ] All tests pass
- [ ] New code has tests
- [ ] Documentation is updated
- [ ] Branch is up to date with main

### Pull Request Process

1. Create PR with clear title and description
2. CI checks run automatically
3. Code review by maintainers
4. Address feedback if requested
5. Merge once approved

## Getting Help

- Check the [documentation](docs/)
- Search [GitHub Issues](https://github.com/non-linear-ai/arc/issues)
- Ask in [GitHub Discussions](https://github.com/non-linear-ai/arc/discussions)
