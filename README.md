```
                         ▓▓▓▓▓╗   ▓▓▓▓▓▓╗    ▓▓▓▓▓▓╗
                        ▓▓╔══▓▓╗  ▓▓╔══▓▓╗  ▓▓╔════╝
                        ▓▓▓▓▓▓▓║  ▓▓▓▓▓▓╔╝  ▓▓║
                        ▓▓╔══▓▓║  ▓▓╔══▓▓╗  ▓▓║
                        ▓▓║  ▓▓║  ▓▓║  ╚▓▓╗ ╚▓▓▓▓▓▓╗
                        ╚═╝  ╚═╝  ╚═╝   ╚═╝  ╚═════╝
                         From Question to Prediction
```
---

Arc is an AI-native machine learning tool to enable machine learning accessible to everyone, from data analysts to seasoned ML engineers. It bridges the gap between natural language questions and predictive models, transforming how you work with data.

**For Business Users & Analysts:** Have you ever wanted to predict customer churn or forecast sales without writing complex code? With Arc, you can. Use plain English to explore data, build models, and get predictions. Arc's AI handles the complexity for you.

**For Machine Learning Engineers & Data Scientists:** Arc streamlines your ML workflow. Instead of boilerplate PyTorch, TensorFlow, or JAX, you use a declarative, AI-native approach. Arc translates your intent into a portable and declarative ML schema, letting you focus on high-level architecture and rapid iteration.

## 💡 How It Works

Arc is built on three foundational pillars:

- **Arc-Graph** - Declarative YAML schema for ML model architecture and training configuration
- **Arc-Pipeline** - Declarative YAML schema for feature engineering and data processing pipelines
- **Arc-Knowledge** - Curated best practices and patterns (extendable via `.arc/knowledge/`)

When you give a command in natural language, Arc's AI consults the Arc-Knowledge to generate optimal specifications:

```
Your Question → Arc AI (+ Arc-Knowledge) → Arc-Graph + Arc-Pipeline → Training → Predictions
```

**The Arc-Knowledge includes:**
- Data loading patterns (CSV, Parquet, JSON, S3, Snowflake)
- Feature engineering techniques (normalization, encoding, splits)
- Model architectures (DCN, MMOE, Transformers, etc.)
- Best practices and proven patterns

**Extensibility:** Add your own patterns and project-specific knowledge to `.arc/knowledge/` to guide Arc's AI for your use case.

This approach provides the best of both worlds:

  * **Simplicity:** A conversational interface - just describe what you want
  * **Power & Portability:** Declarative, version-controlled YAML files that run anywhere PyTorch runs
  * **Transparency:** Human-readable specifications you can review, modify, and share
  * **Customizable:** Extend the Arc-Knowledge with your own patterns and practices

## ✨ Key Features

  - 🤖 **Natural Language to Model** - Go from a question in plain English to a trained predictive model without writing a single line of ML code.
  - 📜 **Declarative Schemas (Arc-Graph & Arc-Pipeline)** - Arc's AI generates complete specifications in human-readable YAML. Arc-Graph defines your model architecture, Arc-Pipeline defines your feature engineering workflows. You review and approve; the AI handles the implementation.
  - 🧠 **Extensible Arc-Knowledge** - Built-in curated knowledge of ML best practices, data patterns, and model architectures. Extend it with your own project-specific patterns in `.arc/knowledge/` to customize Arc's AI for your domain.
  - 🗄️ **Unified Data & ML with SQL** - Connect your data sources via standard SQL. Arc manages your ML assets (models, features, results) in a dedicated database that you can query using standard SQL.
  - ⚡ **End-to-End & Portable** - Arc-Graph and Arc-Pipeline files contain your complete ML workflow, ensuring train/serve parity and making your models easy to version, share, and reproduce.
  - 🎯 **Smart & Interactive** - AI-powered guidance and a user-friendly interactive mode are enabled by default to help you get started quickly.

## 🚀 Quick Start

### Installation

```bash
# Clone and install the project
git clone https://github.com/non-linear-ai/arc
cd arc
```

### Your First Model

Let's build a diabetes prediction model in 3 simple steps:

#### 1. Configure Your API Key (One-Time Setup)

Start Arc and configure your API (saved to `~/.arc/`, only needed once):

```bash
uv run arc chat
> /config
```

Example configuration:

```
◇ Configuration
  API Key            ********
  Base URL           https://api.deepseek.com/v1
  Model              deepseek-chat
```

**Note:** Arc works with agentic and OpenAI API-compatible models, such as Gemini, OpenAI GPT models, or Anthropic Sonnet models.

#### 2. Ask Arc to Build Your Model

Simply describe what you want:

```
Download the Pima Indians Diabetes dataset and build a model to predict diabetes from patient health metrics
```

Arc will:
- ✅ Download the dataset automatically
- ✅ Analyze the data and engineer features
- ✅ Generate an Arc-Graph model specification
- ✅ Train and evaluate the model
- ✅ Launch TensorBoard locally to monitor training curves and metrics in real-time
- ✅ Show you predictions and performance metrics

#### 3. Explore Your Results

Your model is trained! Use `/sql SHOW TABLES` and other SQL commands to explore your data and predictions. Check the logs for the TensorBoard URL to view training curves and metrics.

#### What Just Happened?

Arc generated an **Arc-Graph** specification that looks like this:

```yaml
# Arc-Graph: Model Architecture
inputs:
  patient_data:
    dtype: float32
    shape: [null, 8]
    columns: [pregnancies, glucose, blood_pressure, skin_thickness,
              insulin, bmi, diabetes_pedigree, age]

graph:
  - name: classifier
    type: torch.nn.Linear
    params:
      in_features: 8
      out_features: 1
    inputs:
      input: patient_data

  - name: sigmoid
    type: torch.nn.Sigmoid
    inputs:
      input: classifier.output

outputs:
  prediction: sigmoid.output
```

This Arc-Graph specification is:
- **Human-readable** - You can understand and modify it
- **Portable** - Runs anywhere PyTorch runs
- **Versionable** - Track changes in Git
- **Reproducible** - Guarantees train/serve parity

For more details, see the [Arc-Graph Specification Guide](docs/arc-graph.md).

## 📚 Documentation

### The Three Pillars
- **[Arc-Graph Specification](docs/arc-graph.md)** - Define ML model architecture and training in YAML
- **[Arc-Pipeline Specification](docs/arc-pipeline.md)** - Define feature engineering and data processing workflows
- **[Arc-Knowledge](docs/arc-knowledge.md)** - Built-in best practices and how to extend with your own

### Built-in Knowledge Guides
- **[Data Loading Patterns](src/arc/resources/knowledge/data_loading.md)** - CSV, Parquet, JSON, S3, Snowflake
- **[Feature Engineering](src/arc/resources/knowledge/ml_data_preparation.md)** - ML-specific transformations and splits
- **[Model Architectures](src/arc/resources/knowledge/)** - DCN, MMOE, Transformers, and more

### Integrations
- **[S3 Data Loading](docs/s3-setup.md)** - Connect to S3 buckets (public and private)
- **[Snowflake Integration](docs/snowflake-setup.md)** - Query Snowflake data warehouses

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

### S3 Data Loading

Arc supports loading data from S3 buckets (public and private) using DuckDB's S3 extensions. Public buckets and IAM roles work immediately without configuration. Private buckets require credentials.

**Quick example:**

```bash
uv run arc chat
```

```sql
-- Public S3 bucket (no setup)
/sql CREATE TABLE taxi AS
     SELECT * FROM 's3://nyc-tlc/trip data/yellow_tripdata_2023-01.parquet'

-- Private S3 bucket (requires credentials)
/sql CREATE TABLE data AS
     SELECT * FROM 's3://my-private-bucket/data.parquet'
```

**Supported formats**: CSV, Parquet, JSON, Apache Iceberg

**📖 For complete setup instructions, see [docs/s3-setup.md](docs/s3-setup.md)**

This includes:
- Configuration for public buckets, IAM roles, and private buckets
- S3-compatible storage (MinIO, Wasabi)
- Usage patterns and best practices
- Troubleshooting guide

### Snowflake Data Loading

Arc supports loading data from Snowflake data warehouses using DuckDB's Snowflake extension. Query Snowflake tables directly, join them with S3 and local data, and extract data for cost-efficient local feature engineering.

**Quick example:**

```bash
uv run arc chat
```

```sql
-- Query Snowflake directly
/sql SELECT * FROM snowflake.public.customers
     WHERE state = 'CA' LIMIT 10

-- Extract for local feature engineering (recommended)
/sql CREATE TABLE local_customers AS
     SELECT * FROM snowflake.public.customers
     WHERE signup_date >= '2024-01-01'
```

**Supported workflows**: Direct queries, data extraction, cross-database joins

**📖 For complete setup instructions, see [docs/snowflake-setup.md](docs/snowflake-setup.md)**

This includes:
- Credential configuration (settings file and environment variables)
- Library path setup for Linux/macOS/Windows
- Startup scripts for easy launch
- Best practices and troubleshooting
