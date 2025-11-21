# The Three Pillars of Arc

Arc is built on three foundational concepts that work together to enable AI-native machine learning. Understanding these pillars is key to mastering Arc.

## Overview

```
Your Question → Arc AI (+ Arc-Knowledge) → Arc-Graph + Arc-Pipeline → Training → Predictions
```

When you describe what you want in natural language, Arc's AI agent:
1. Consults **Arc-Knowledge** for ML best practices
2. Generates **Arc-Graph** (model architecture) and **Arc-Pipeline** (data processing) specifications
3. Executes the workflow to train your model and generate predictions

## The Three Pillars

### 1. Arc-Graph: Model Architecture

**Arc-Graph** is a declarative YAML specification for ML model architecture and training configuration.

**Purpose**: Define what your model looks like and how it should be trained.

**Key Components**:
- **inputs**: Define input shapes, types, and feature names
- **graph**: Specify model layers and connections (using PyTorch layers)
- **outputs**: Define model outputs
- **trainer**: Training configuration (optimizer, loss, epochs, batch size)
- **evaluator**: Evaluation metrics

**Example**:
```yaml
inputs:
  user_features:
    dtype: float32
    shape: [null, 10]

graph:
  - name: hidden
    type: torch.nn.Linear
    params:
      in_features: 10
      out_features: 64
    inputs:
      input: user_features

  - name: relu
    type: torch.nn.ReLU
    inputs:
      input: hidden.output

outputs:
  prediction: relu.output
```

[Learn more about Arc-Graph →](arc-graph.md)

### 2. Arc-Pipeline: Feature Engineering

**Arc-Pipeline** is a declarative YAML specification for data processing and feature engineering workflows.

**Purpose**: Transform raw data into ML-ready features.

**Key Capabilities**:
- Load data from multiple sources (CSV, Parquet, S3, Snowflake, databases)
- Transform and normalize features
- Create train/validation/test splits
- Handle missing values and outliers
- Generate derived features

**Example**:
```yaml
name: user_features_pipeline
version: 1.0

sources:
  - name: raw_users
    type: duckdb
    params:
      query: "SELECT * FROM users"

transforms:
  - name: normalize_age
    type: min_max_scaler
    input: raw_users.age
    params:
      feature_range: [0, 1]

outputs:
  - name: processed_users
    type: duckdb_table
```

[Learn more about Arc-Pipeline →](arc-pipeline.md)

### 3. Arc-Knowledge: ML Best Practices

**Arc-Knowledge** is a curated collection of ML best practices, patterns, and domain expertise that guides Arc's AI agent.

**Purpose**: Provide the AI with expert knowledge to generate optimal specifications.

**What's Included**:
- Data loading patterns (CSV, Parquet, S3, Snowflake)
- Feature engineering techniques (normalization, encoding, splits)
- Model architectures (MLP, DCN, MMOE, Transformers)
- Training best practices
- Evaluation strategies

**Extensibility**: You can add your own knowledge to `~/.arc/knowledge/` to customize Arc's behavior for your domain.

**Example Knowledge Structure**:
```
~/.arc/knowledge/
├── metadata.yaml          # Knowledge catalog
├── custom_models.md       # Your model architectures
└── domain_patterns.md     # Your domain-specific patterns
```

[Learn more about Arc-Knowledge →](arc-knowledge.md)

## How They Work Together

### The Workflow

1. **User Describes Goal**
   ```
   "Build a recommendation model using user demographics and purchase history"
   ```

2. **AI Consults Arc-Knowledge**
   - What architecture works best for recommendations? (MMOE, DCN)
   - How to process user demographics? (normalization, encoding)
   - How to handle purchase history? (sequence features, embeddings)

3. **AI Generates Specifications**
   - **Arc-Pipeline**: Load data, normalize features, create embeddings
   - **Arc-Graph**: MMOE architecture with appropriate layer sizes

4. **Arc Executes**
   - Runs the Arc-Pipeline to process data
   - Builds the model from Arc-Graph
   - Trains the model
   - Evaluates performance

5. **Results**
   - Trained model ready for predictions
   - Portable YAML specifications for reproducibility
   - TensorBoard logs for analysis

## Key Benefits

### Declarative & Portable

Arc-Graph and Arc-Pipeline are declarative YAML files:
- **Version controlled**: Track changes in Git
- **Portable**: Run anywhere PyTorch runs
- **Reproducible**: Same specs = same results
- **Human-readable**: Understand and modify easily

### AI-Native

Arc's AI generates specifications using Arc-Knowledge:
- **No ML coding**: Describe what you want in plain English
- **Best practices**: AI applies expert knowledge automatically
- **Iterative**: Easily revise and refine specifications

### Extensible

Customize Arc for your needs:
- **Custom layers**: Use any PyTorch layer in Arc-Graph
- **Custom processors**: Add data processors in Arc-Pipeline
- **Custom knowledge**: Extend Arc-Knowledge with your patterns

### Train/Serve Parity

The same Arc-Graph used for training is used for inference:
- **No translation**: Same spec for train and serve
- **No drift**: Architecture guaranteed to match
- **Easy deployment**: Load spec and model weights

## Design Philosophy

Arc's Three Pillars embody several design principles:

### Separation of Concerns
- **Arc-Graph**: WHAT to compute (model architecture)
- **Arc-Pipeline**: HOW to prepare data (feature engineering)
- **Arc-Knowledge**: WHY these choices (best practices)

### Declarative Over Imperative
- Specify WHAT you want, not HOW to do it
- Let Arc handle implementation details
- Focus on high-level design decisions

### AI-Assisted, Human-Controlled
- AI generates specifications from natural language
- Humans review and approve before execution
- Specifications are transparent and editable

### Composability
- Mix and match components
- Reuse pipelines across projects
- Build libraries of specifications

## Getting Started with the Three Pillars

Ready to dive deeper? Explore each pillar:

1. **[Arc-Graph](arc-graph.md)** - Learn to read and customize model architectures
2. **[Arc-Pipeline](arc-pipeline.md)** - Master feature engineering workflows
3. **[Arc-Knowledge](arc-knowledge.md)** - Extend Arc with your own expertise

Or jump straight to the [Quick Start Tutorial](../getting-started/quickstart.md) to see them in action!

## Common Questions

### Do I need to write Arc-Graph/Arc-Pipeline myself?

**No!** Arc's AI generates these specifications for you from natural language. You only need to review and approve them. However, understanding the format helps you customize and debug.

### Can I edit the generated specifications?

**Yes!** The YAML specifications are human-readable and editable. You can modify them directly if you want fine-grained control.

### What if I need a custom architecture?

You can:
1. Describe it in natural language and let the AI generate it
2. Add it to Arc-Knowledge so the AI knows about it
3. Edit the Arc-Graph YAML directly

### How is this different from writing PyTorch?

| Aspect | PyTorch Code | Arc |
|--------|--------------|-----|
| **Format** | Python code | Declarative YAML |
| **Creation** | Manual coding | AI-generated |
| **Portability** | Code dependencies | Self-contained YAML |
| **Versioning** | Full codebase | Single YAML file |
| **Train/Serve** | Separate implementations | Same specification |
| **Learning Curve** | Deep ML knowledge | Natural language |

Arc complements PyTorch - it generates PyTorch models but from a higher-level specification.

## Next Steps

- **[Arc-Graph Specification](arc-graph.md)** - Deep dive into model architecture specs
- **[Arc-Pipeline Specification](arc-pipeline.md)** - Deep dive into data processing specs
- **[Arc-Knowledge System](arc-knowledge.md)** - Learn to extend Arc's knowledge
- **[Examples](../examples/diabetes-prediction.md)** - See complete end-to-end examples
