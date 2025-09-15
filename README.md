```
    ▓▓▓▓▓╗   ▓▓▓▓▓▓╗    ▓▓▓▓▓▓╗
   ▓▓╔══▓▓╗  ▓▓╔══▓▓╗  ▓▓╔════╝
   ▓▓▓▓▓▓▓║  ▓▓▓▓▓▓╔╝  ▓▓║
   ▓▓╔══▓▓║  ▓▓╔══▓▓╗  ▓▓║
   ▓▓║  ▓▓║  ▓▓║  ╚▓▓╗ ╚▓▓▓▓▓▓╗
   ╚═╝  ╚═╝  ╚═╝   ╚═╝  ╚═════╝
   From Question to Prediction
```

Arc is a AI-native machine learning tool to enable machine learning accessible to everyone, from data analysts to seasoned ML engineers. It bridges the gap between natural language questions and predictive models, transforming how you work with data.

**For Business Users & Analysts:** Have you ever wanted to predict customer churn or forecast sales without writing complex code? With Arc, you can. Use plain English to explore data, build models, and get predictions. Arc's AI handles the complexity for you.

**For Machine Learning Engineers & Data Scientists:** Arc streamlines your ML workflow. Instead of boilerplate PyTorch, TensorFlow, or JAX, you use a declarative, AI-native approach. Arc translates your intent into a portable and declarative ML schema, letting you focus on high-level architecture and rapid iteration.

## 💡 How It Works

Arc uses a declarative schema called `Arc-Graph` as its foundation. When you give a command in natural language, Arc's AI engine generates a complete, self-contained model definition in this schema, which is then translated and executed across different ML frameworks and computing environments.

This approach provides the best of both worlds:

  * **Simplicity:** A simple, conversational interface for creating models.
  * **Power & Portability:** A declarative, explicit, and human-readable definition of your entire ML workflow, from data prep to prediction, that can run anywhere.

## ✨ Key Features

  - 🤖 **Natural Language to Model** - Go from a question in plain English to a trained predictive model without writing a single line of ML code.
  - 📜 **Declarative & AI-Native** - Leverages the `Arc-Graph` schema to define models with high-level concepts. The AI handles the implementation; you verify the logic.
  - 🗄️ **Unified Data & ML with SQL** - Connect your data sources via standard SQL. Arc manages your ML assets (models, features, results) in a dedicated database that you can query using standard SQL.
  - ⚡ **End-to-End & Portable** - The `Arc-Graph` files contain your ML workflow, ensuring train/serve parity and making your models easy to version, share, and reproduce.
  - 🎯 **Smart & Interactive** - AI-powered guidance and a user-friendly interactive mode are enabled by default to help you get started quickly.

## 🚀 Quick Start

### Installation

```bash
# Clone and install the project
git clone https://github.com/non-linear-ai/arc
cd arc
```

### Your First Model

TODO: Update along the development progress

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
# Start interactive chat session
uv run arc chat

# Or use the Makefile
make run

# Show available commands
uv run arc --help
```

### Running Tests

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Run tests in watch mode (automatically reruns tests when files change)
uv run pytest-watcher .
```

### Using Make Commands (Alternative)

For convenience, you can also use the provided Make targets:

```bash
# Install and run
make install-dev          # Install development dependencies
make run                  # Start Arc CLI in interactive mode

# Testing and quality
make test                 # Run tests with coverage
make test-watch          # Run tests in watch mode
make lint                # Run linting checks
make format              # Format code and fix linting issues
make clean               # Clean build artifacts

# Run all quality checks
make all                 # Equivalent to: make format lint test
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
