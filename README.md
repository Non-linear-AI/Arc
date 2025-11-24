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
- **Arc-Knowledge** - Curated best practices and patterns (extendable via `~/.arc/knowledge/`)

When you give a command in natural language, Arc's AI consults the Arc-Knowledge to generate optimal specifications:

```
Your Question → Arc AI (+ Arc-Knowledge) → Arc-Graph + Arc-Pipeline → Training → Predictions
```

**The Arc-Knowledge includes:**
- Data loading patterns (CSV, Parquet, JSON, S3, Snowflake)
- Feature engineering techniques (normalization, encoding, splits)
- Model architectures (DCN, MMOE, Transformers, etc.)
- Best practices and proven patterns

**Extensibility:** Add your own patterns and project-specific knowledge to `~/.arc/knowledge/` to guide Arc's AI for your use case.

This approach provides the best of both worlds:

  * **Simplicity:** A conversational interface - just describe what you want
  * **Power & Portability:** Declarative, version-controlled YAML files that run anywhere PyTorch runs
  * **Transparency:** Human-readable specifications you can review, modify, and share
  * **Customizable:** Extend the Arc-Knowledge with your own patterns and practices

## ✨ Key Features

  - 🤖 **Natural Language to Model** - Go from a question in plain English to a trained predictive model without writing a single line of ML code.
  - 📜 **Declarative Schemas (Arc-Graph & Arc-Pipeline)** - Arc's AI generates complete specifications in human-readable YAML. Arc-Graph defines your model architecture, Arc-Pipeline defines your feature engineering workflows. You review and approve; the AI handles the implementation.
  - 🧠 **Extensible Arc-Knowledge** - Built-in curated knowledge of ML best practices, data patterns, and model architectures. Extend it with your own project-specific patterns in `~/.arc/knowledge/` to customize Arc's AI for your domain.
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

For more details, see the [Arc-Graph Specification Guide](docs/concepts/arc-graph.md).

## 📚 Documentation

**📖 [Complete Documentation](docs/index.md)** - Start here for comprehensive guides, examples, and API reference.

### Quick Links

**Getting Started:**
- **[Installation Guide](docs/getting-started/installation.md)** - Set up Arc
- **[Quick Start Tutorial](docs/getting-started/quickstart.md)** - Build your first model
- **[Configuration Guide](docs/getting-started/configuration.md)** - Configure API keys

**Core Concepts:**
- **[The Three Pillars](docs/concepts/overview.md)** - Understand Arc's architecture
- **[Arc-Graph](docs/concepts/arc-graph.md)** - Model architecture specification
- **[Arc-Pipeline](docs/concepts/arc-pipeline.md)** - Feature engineering workflows
- **[Arc-Knowledge](docs/concepts/arc-knowledge.md)** - ML best practices system

**User Guides:**
- **[Data Loading](docs/guides/data-loading.md)** - Load data from CSV, Parquet, S3, Snowflake
- **[Feature Engineering](docs/guides/feature-engineering.md)** - Transform and prepare data
- **[Model Training](docs/guides/model-training.md)** - Train ML models
- **[Model Evaluation](docs/guides/model-evaluation.md)** - Evaluate performance
- **[Making Predictions](docs/guides/making-predictions.md)** - Use trained models

**Examples:**
- **[Logistic Regression Example](docs/examples/logistic_regression_console/)** - Complete working example

**Integrations:**
- **[AWS S3](docs/integrations/s3.md)** - Load data from S3 buckets
- **[Snowflake](docs/integrations/snowflake.md)** - Query Snowflake warehouses

**For Contributors:**
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute
- **[Development Setup](docs/contributing/development-setup.md)** - Set up dev environment

## Development

Want to contribute? See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Dev Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/non-linear-ai/arc
cd arc
uv sync --dev

# Run tests
uv run pytest

# Format and lint
uv run ruff format .
uv run ruff check . --fix
```

For detailed instructions, see [Development Setup Guide](docs/contributing/development-setup.md).

## Project Configuration

Project settings are in `pyproject.toml`. For API configuration, see [Configuration Guide](docs/getting-started/configuration.md).

For S3 and Snowflake setup, see:
- **[S3 Integration Guide](docs/integrations/s3.md)**
- **[Snowflake Integration Guide](docs/integrations/snowflake.md)**
