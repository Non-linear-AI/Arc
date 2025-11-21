# Testing Guide

This guide covers testing practices and guidelines for Arc contributors.

## Test Framework

Arc uses **pytest** for testing with these plugins:
- `pytest-cov`: Coverage reporting
- `pytest-asyncio`: Async test support
- `pytest-watcher`: Auto-rerun tests on file changes

## Running Tests

### Basic Test Commands

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Run tests with HTML coverage report
uv run pytest --cov --cov-report=html
# Open htmlcov/index.html in browser

# Run tests in watch mode (auto-reruns on file changes)
uv run pytest-watcher .

# Or use Make
make test
make test-watch
```

### Running Specific Tests

```bash
# Run tests for a specific module
uv run pytest tests/test_core/

# Run a specific test file
uv run pytest tests/test_core/test_agent.py

# Run a specific test function
uv run pytest tests/test_core/test_agent.py::test_agent_initialization

# Run tests matching a pattern
uv run pytest -k "test_agent"

# Run tests with verbose output
uv run pytest -v

# Run only failed tests from last run
uv run pytest --lf

# Run tests in parallel (faster)
uv run pytest -n auto
```

## Test Organization

Tests are organized by component:

```
tests/
├── test_core/          # Core agent tests
│   ├── test_agent.py
│   └── test_agents/    # Specialized agent tests
├── test_tools/         # Tool tests
│   ├── test_ml_tools.py
│   └── test_core_tools.py
├── test_ml/            # ML runtime tests
│   ├── test_runtime.py
│   ├── test_builder.py
│   ├── test_training.py
│   └── test_evaluator.py
├── test_graph/         # Arc-Graph validator tests
│   ├── test_validators.py
│   └── test_builder.py
├── test_database/      # Database tests
│   ├── test_services.py
│   └── test_models.py
├── test_ui/            # CLI tests
│   └── test_cli.py
└── conftest.py         # Shared fixtures
```

## Writing Tests

### Basic Test Structure

```python
import pytest
from arc.core.agent import ArcAgent

def test_agent_initialization():
    """Test that agent initializes correctly."""
    agent = ArcAgent()
    assert agent is not None
    assert agent.max_tool_rounds == 50

def test_agent_with_custom_rounds():
    """Test agent with custom max_tool_rounds."""
    agent = ArcAgent(max_tool_rounds=10)
    assert agent.max_tool_rounds == 10
```

### Async Tests

```python
import pytest
from arc.core.agent import ArcAgent

@pytest.mark.asyncio
async def test_async_agent_operation():
    """Test async agent operations."""
    agent = ArcAgent()
    result = await agent.process("test message")
    assert result is not None
    assert isinstance(result, dict)
```

### Using Fixtures

```python
import pytest
from arc.database.service_container import ServiceContainer

@pytest.fixture
def service_container():
    """Create a service container for testing."""
    container = ServiceContainer(db_path=":memory:")
    yield container
    container.close()

def test_with_fixture(service_container):
    """Test using a fixture."""
    model_service = service_container.model_service
    assert model_service is not None
```

### Parametrized Tests

```python
import pytest

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_doubling(input, expected):
    """Test that input doubles correctly."""
    assert input * 2 == expected
```

### Testing Exceptions

```python
import pytest
from arc.graph.validators import ValidationError

def test_invalid_spec_raises_error():
    """Test that invalid spec raises ValidationError."""
    with pytest.raises(ValidationError):
        validate_spec({"invalid": "spec"})

def test_error_message():
    """Test that error has correct message."""
    with pytest.raises(ValidationError, match="Missing required field"):
        validate_spec({})
```

### Mocking

```python
from unittest.mock import Mock, patch
import pytest

def test_with_mock():
    """Test with mocked dependency."""
    mock_llm = Mock()
    mock_llm.generate.return_value = "response"

    agent = ArcAgent(llm=mock_llm)
    result = agent.process_message("test")

    mock_llm.generate.assert_called_once()
    assert "response" in result

@patch('arc.core.agent.LLMClient')
def test_with_patch(mock_llm_client):
    """Test with patched class."""
    mock_llm_client.return_value.generate.return_value = "response"

    agent = ArcAgent()
    result = agent.process_message("test")

    assert "response" in result
```

## Test Coverage

### Coverage Configuration

Coverage settings are in `pyproject.toml`:

```toml
[tool.coverage.run]
branch = true
source = ["arc"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]
```

### Checking Coverage

```bash
# Generate coverage report
uv run pytest --cov

# Generate HTML report
uv run pytest --cov --cov-report=html

# View report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Coverage Goals

- **Overall**: Aim for 80%+ coverage
- **New code**: 90%+ coverage
- **Critical paths**: 100% coverage (agents, validators, ML runtime)

## Testing Best Practices

### 1. Test Naming

Use descriptive test names:

```python
# Good
def test_agent_initialization_with_custom_rounds():
    pass

def test_validator_raises_error_on_missing_field():
    pass

# Bad
def test_1():
    pass

def test_agent():
    pass
```

### 2. Test Isolation

Each test should be independent:

```python
# Good
def test_model_training():
    """Each test creates its own model."""
    model = create_test_model()
    result = train_model(model)
    assert result.success

# Bad (tests share state)
global_model = None

def test_model_creation():
    global global_model
    global_model = create_test_model()

def test_model_training():
    # Depends on previous test!
    result = train_model(global_model)
```

### 3. Use Fixtures for Setup

```python
@pytest.fixture
def temp_database():
    """Create temporary database for testing."""
    db = Database(":memory:")
    db.create_tables()
    yield db
    db.close()

def test_with_database(temp_database):
    """Test uses fixture for setup/teardown."""
    result = temp_database.query("SELECT 1")
    assert result is not None
```

### 4. Test Edge Cases

```python
def test_division_by_zero():
    """Test edge case: division by zero."""
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)

def test_empty_input():
    """Test edge case: empty input."""
    result = process_list([])
    assert result == []

def test_none_input():
    """Test edge case: None input."""
    with pytest.raises(ValueError):
        process_data(None)
```

### 5. Keep Tests Fast

```python
# Good: Use in-memory database
@pytest.fixture
def fast_db():
    return Database(":memory:")

# Bad: Use slow file-based database
@pytest.fixture
def slow_db():
    return Database("/tmp/test.db")  # Slower I/O
```

### 6. Test Behavior, Not Implementation

```python
# Good: Test behavior
def test_user_can_login():
    """Test that user can log in with correct credentials."""
    result = login("user", "password")
    assert result.success
    assert result.user_id is not None

# Bad: Test implementation details
def test_login_calls_hash_password():
    """Test internal implementation (fragile)."""
    with patch('auth.hash_password') as mock:
        login("user", "password")
        mock.assert_called_once()  # Too coupled to implementation
```

## Testing Different Components

### Testing Agents

```python
import pytest
from arc.core.agents.ml_model import MLModelAgent

@pytest.mark.asyncio
async def test_ml_model_agent():
    """Test ML model agent generates valid Arc-Graph."""
    agent = MLModelAgent()
    result = await agent.run({
        "instruction": "Build binary classifier",
        "data_table": "test_data"
    })

    assert result.success
    assert "arc_graph" in result.artifacts
    # Validate Arc-Graph spec
    assert result.artifacts["arc_graph"]["inputs"] is not None
```

### Testing Tools

```python
from arc.tools.database_query import database_query

def test_database_query_tool(temp_database):
    """Test database query tool."""
    result = database_query("SELECT 1 as value")

    assert result.success
    assert len(result.data) == 1
    assert result.data[0]["value"] == 1
```

### Testing ML Components

```python
import torch
from arc.ml.builder import ModelBuilder

def test_model_builder():
    """Test building PyTorch model from Arc-Graph."""
    arc_graph = {
        "inputs": {"features": {"dtype": "float32", "shape": [None, 10]}},
        "graph": [
            {"name": "fc", "type": "torch.nn.Linear",
             "params": {"in_features": 10, "out_features": 1},
             "inputs": {"input": "features"}}
        ],
        "outputs": {"prediction": "fc.output"}
    }

    builder = ModelBuilder(arc_graph)
    model = builder.build()

    assert isinstance(model, torch.nn.Module)

    # Test forward pass
    x = torch.randn(32, 10)
    y = model(x)
    assert y.shape == (32, 1)
```

### Testing Validators

```python
from arc.graph.validators import validate_arc_graph

def test_validator_accepts_valid_spec():
    """Test that validator accepts valid spec."""
    valid_spec = {
        "inputs": {...},
        "graph": [...],
        "outputs": {...}
    }

    # Should not raise
    validate_arc_graph(valid_spec)

def test_validator_rejects_missing_inputs():
    """Test that validator rejects spec without inputs."""
    invalid_spec = {
        "graph": [...],
        "outputs": {...}
    }

    with pytest.raises(ValidationError, match="Missing 'inputs'"):
        validate_arc_graph(invalid_spec)
```

## Continuous Integration

Tests run automatically on:
- Every push to GitHub
- Every pull request
- Before merging to main

### Local CI Simulation

Run the same checks as CI:

```bash
# Run all CI checks
make ci

# Or manually:
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov
```

## Debugging Tests

### Run Single Test with Output

```bash
# Show print statements
uv run pytest tests/test_file.py::test_name -s

# Show verbose output
uv run pytest tests/test_file.py::test_name -v

# Show local variables on failure
uv run pytest tests/test_file.py::test_name -l
```

### Use Breakpoints

```python
def test_something():
    """Test with breakpoint."""
    result = complex_function()
    breakpoint()  # Execution stops here
    assert result == expected
```

Run with:
```bash
uv run pytest tests/test_file.py::test_something -s
```

### Show Warnings

```bash
# Show all warnings
uv run pytest -W all

# Turn warnings into errors
uv run pytest -W error
```

## Test Data

### Test Fixtures Location

Store test data in:
```
tests/fixtures/
├── sample_data.csv
├── arc_graph_examples/
│   ├── simple_mlp.yaml
│   └── dcn.yaml
└── test_databases/
    └── sample.db
```

### Loading Test Data

```python
import os
import yaml

def load_test_arc_graph(name):
    """Load test Arc-Graph specification."""
    path = os.path.join(
        os.path.dirname(__file__),
        "fixtures/arc_graph_examples",
        f"{name}.yaml"
    )
    with open(path) as f:
        return yaml.safe_load(f)

def test_with_fixture_data():
    """Test using fixture data."""
    arc_graph = load_test_arc_graph("simple_mlp")
    # ... test with arc_graph
```

## Next Steps

- **[Development Setup](development-setup.md)** - Set up dev environment
- **[Architecture Overview](architecture.md)** - Understand the codebase
- **[Contributing Guidelines](../../CONTRIBUTING.md)** - Contribution workflow
