# Contributing to Arc

Thank you for your interest in contributing to Arc! This document provides guidelines and information for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Git
- uv (Python package manager)

### Setup Development Environment

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/arc
   cd arc
   ```

2. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies**:
   ```bash
   uv sync --dev
   ```

4. **Verify installation**:
   ```bash
   uv run arc --help
   ```

For detailed setup instructions, see [Development Setup Guide](docs/contributing/development-setup.md).

## Development Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

Use prefixes:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test improvements

### 2. Make Your Changes

- Write clean, readable code
- Follow existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Or use Make
make test
```

### 4. Format and Lint

```bash
# Format code
uv run ruff format .

# Check and fix linting issues
uv run ruff check . --fix

# Or use Make
make format
```

### 5. Commit Changes

Write clear, concise commit messages:

```bash
git add .
git commit -m "Add feature: clear description of what you added"
```

Good commit message examples:
- `Add feature: support for custom loss functions`
- `Fix bug: handle missing values in feature engineering`
- `Docs: update installation guide with Windows instructions`
- `Refactor: simplify model builder logic`

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear title describing the change
- Description of what changed and why
- Link to relevant issues (if any)
- Screenshots (for UI changes)

## Code Style

Arc uses **Ruff** for linting and formatting:

- **Line length**: 88 characters
- **Target**: Python 3.12
- **Import sorting**: isort style

### Running Code Quality Checks

```bash
# Check code style
uv run ruff check .

# Format code
uv run ruff format .

# Run all quality checks
make all
```

Configuration is in `pyproject.toml`.

## Testing

### Writing Tests

- Place tests in `tests/` directory
- Mirror the source structure (e.g., `src/arc/core/agent.py` → `tests/test_core/test_agent.py`)
- Use descriptive test names
- Test both success and failure cases
- Aim for 80%+ code coverage

Example test:

```python
import pytest
from arc.core.agent import ArcAgent

def test_agent_initialization():
    """Test that agent initializes correctly."""
    agent = ArcAgent()
    assert agent is not None
    assert agent.max_tool_rounds == 50

@pytest.mark.asyncio
async def test_async_operation():
    """Test async agent operations."""
    agent = ArcAgent()
    result = await agent.process("test")
    assert result is not None
```

For detailed testing guidelines, see [Testing Guide](docs/contributing/testing.md).

## Documentation

### Types of Documentation

1. **Code Comments**: Explain complex logic
2. **Docstrings**: Document all public functions/classes
3. **User Guides**: Step-by-step instructions in `docs/guides/`
4. **API Reference**: Command and specification references in `docs/api-reference/`
5. **Examples**: Complete tutorials in `docs/examples/`

### Docstring Style

Use Google-style docstrings:

```python
def train_model(model: Model, data: Dataset, epochs: int = 50) -> TrainingResult:
    """Train a machine learning model.

    Args:
        model: The model to train
        data: Training dataset
        epochs: Number of training epochs (default: 50)

    Returns:
        TrainingResult containing metrics and trained model

    Raises:
        ValueError: If data is empty or invalid
        TrainingError: If training fails
    """
    # Implementation...
```

### Updating Documentation

When adding features:
- Update relevant user guides
- Add examples if applicable
- Update API reference if adding commands
- Update CHANGELOG.md (once released)

## Submitting Changes

### Pull Request Checklist

Before submitting a pull request, ensure:

- [ ] Code follows style guidelines (Ruff passes)
- [ ] All tests pass (`make test`)
- [ ] New code has tests (80%+ coverage)
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

### Pull Request Process

1. **Create PR** with clear title and description
2. **CI checks** will run automatically
3. **Code review** by maintainers
4. **Address feedback** if requested
5. **Merge** once approved

### Review Timeline

- Initial review: Within 3-5 business days
- Follow-up reviews: Within 2 business days

## What to Contribute

### Good First Issues

Look for issues labeled `good-first-issue`:
- Documentation improvements
- Bug fixes
- Test coverage improvements
- Minor feature additions

### High-Priority Areas

- Built-in knowledge expansion
- Data processor implementations
- Model architecture templates
- Integration examples

### Areas Needing Help

- Windows compatibility testing
- Performance optimizations
- Error handling improvements
- User experience enhancements

## Getting Help

- **Documentation**: Check [docs/](docs/) directory
- **Architecture**: See [Architecture Overview](docs/contributing/architecture.md)
- **GitHub Discussions**: Ask questions
- **GitHub Issues**: Report bugs or request features

## Code of Conduct

Be respectful and constructive:
- Use welcoming and inclusive language
- Respect differing viewpoints
- Accept constructive criticism gracefully
- Focus on what's best for the community

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors are recognized in:
- GitHub contributors page
- Release notes (for significant contributions)
- Special thanks in documentation (for major features)

## Additional Resources

- [Development Setup Guide](docs/contributing/development-setup.md) - Detailed setup instructions
- [Architecture Overview](docs/contributing/architecture.md) - Understand the codebase
- [Testing Guide](docs/contributing/testing.md) - Testing practices
- [Publishing Guide](docs/contributing/publishing.md) - For maintainers

## Questions?

If you have questions about contributing:
1. Check the [documentation](docs/)
2. Search [GitHub Issues](https://github.com/non-linear-ai/arc/issues)
3. Ask in [GitHub Discussions](https://github.com/non-linear-ai/arc/discussions)
4. Open a new issue with the `question` label

Thank you for contributing to Arc!
