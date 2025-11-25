# Arc-Graph Specification

> **Arc-Graph** is Arc's declarative YAML schema for defining machine learning models. It provides a complete, human-readable specification of your model architecture and training configuration that can be version-controlled, shared, and executed anywhere PyTorch runs.

## 1. Overview

**Arc-Graph** allows you to define machine learning models declaratively in YAML. Think of it as "YAML for ML models" - you describe what you want, and Arc handles the PyTorch implementation.

### What is Arc-Graph?

Arc-Graph is a specification that describes:
- **Model Architecture** - Neural network structure (inputs, layers, outputs)
- **Training Configuration** - Optimizer, loss function, hyperparameters
- **Reusable Modules** - Sub-graphs that can be instantiated multiple times

### Why Arc-Graph?

Traditional ML development requires writing imperative PyTorch/TensorFlow code. Arc-Graph takes a declarative approach:

| Traditional Approach | Arc-Graph Approach |
|---------------------|-------------------|
| Write Python classes | Write YAML specs |
| Implement forward() | Declare layers in graph |
| Manual tensor wiring | Automatic from inputs map |
| Hard to version/share | Git-friendly YAML |
| Train/serve mismatch risk | Guaranteed parity |

### Arc-Graph Benefits

- **Human-readable** - Easy to understand and modify
- **Declarative** - Describes what you want, not how to implement it
- **Portable** - Runs anywhere PyTorch runs
- **Versionable** - Track in Git like any other code
- **Reproducible** - Guarantees train/serve parity
- **AI-friendly** - LLMs can generate and modify Arc-Graph specs

### Design Principles

1. **Explicit over implicit** - All model details are clearly defined
2. **Separation of concerns** - Model architecture separated from training configuration
3. **Flexible organization** - Supports both unified YAML and separate model/trainer files
4. **Native PyTorch** - Uses standard PyTorch layer types (`torch.nn.*`, `torch.*`)

## 2. File Structure Options

Arc supports two organizational patterns:

### Option 1: Unified YAML (Recommended for simple models)
```yaml
# diabetes_model.yaml
model:
  inputs: {...}
  graph: [...]
  outputs: {...}

trainer:
  optimizer: {...}
  loss: {...}
  config: {...}
```

### Option 2: Separate Files (Recommended for complex models)
```yaml
# model.yaml
inputs: {...}
graph: [...]
outputs: {...}
```

```yaml
# trainer.yaml
optimizer: {...}
loss: {...}
config: {...}
```

## 3. Model Structure

The `model` section defines the neural network architecture and input/output interface.

### 3.1 Basic Structure

```yaml
model:  # Optional wrapper (omit if using separate files)
  inputs:
    <input_name>:
      dtype: <float32|float64|int|long|bool>
      shape: [<dim1>, <dim2>, ...]  # Use null for batch dimension
      columns: [<col1>, <col2>, ...]  # Maps to table columns

  graph:
    - name: <unique_layer_name>
      type: <layer_type>  # torch.nn.*, torch.*, torch.nn.functional.*, arc.stack
      params:
        <param_name>: <value>
      inputs:
        <input_port>: <source_reference>

  outputs:
    <output_name>: <node_name>.<output_port>
```

### 3.2 Input Specification

Each input defines a tensor expected by the model:

```yaml
inputs:
  # Simple numeric input
  features:
    dtype: float32
    shape: [null, 10]  # [batch_size, num_features]
    columns: [age, income, ...]  # Maps to table columns

  # Categorical input with embedding
  user_id:
    dtype: long
    shape: [null]
    columns: [user_id]
    categorical: true
    vocab_size: 10000
    embedding_dim: 128

  # Multi-input for complex models
  item_features:
    dtype: float32
    shape: [null, 5]
    columns: [price, rating, views, likes, shares]
```

**Field Reference:**

| Field | Required | Description | Example |
|-------|----------|-------------|---------|
| `dtype` | ✅ Yes | Tensor data type | `float32`, `long`, `int`, `bool` |
| `shape` | ✅ Yes | Tensor dimensions (use `null` for batch) | `[null, 10]`, `[null, 28, 28]` |
| `columns` | ✅ Yes | Table columns mapping to this input | `[age, income]` |
| `categorical` | ⭕ No | Whether this is a categorical feature | `true`, `false` (default) |
| `vocab_size` | ⭕ No | Vocabulary size for categorical | `10000` |
| `embedding_dim` | ⭕ No | Embedding dimension for categorical | `128` |

### 3.3 Graph (Layer Definitions)

The `graph` is an ordered list of layers forming a directed acyclic graph (DAG):

```yaml
graph:
  - name: hidden_layer
    type: torch.nn.Linear
    params:
      in_features: 10
      out_features: 64
      bias: true
    inputs:
      input: features  # References input by name

  - name: activation
    type: torch.nn.ReLU
    inputs:
      input: hidden_layer.output  # References previous layer output

  - name: output_layer
    type: torch.nn.Linear
    params:
      in_features: 64
      out_features: 1
    inputs:
      input: activation.output
```

**Input Referencing:**
- `features` - References model input directly
- `layer_name.output` - References output of a previous layer
- `layer_name.hidden_state` - For RNN/LSTM layers
- `layer_name.cell_state` - For LSTM layers

### 3.4 Outputs

Maps user-friendly names to layer outputs:

```yaml
outputs:
  prediction: output_layer.output
  logits: output_layer.output
  probability: sigmoid.output
```

Multiple outputs are supported for multi-task models.

## 4. Trainer Structure

The `trainer` section configures the training process:

```yaml
trainer:  # Optional wrapper (omit if using separate files)
  optimizer:
    type: torch.optim.AdamW
    config:
      learning_rate: 0.001
      weight_decay: 0.01

  loss:
    type: torch.nn.BCELoss  # or torch.nn.CrossEntropyLoss, torch.nn.MSELoss
    inputs:
      pred: model.prediction  # References model output
      target: outcome  # References target column name

  config:
    epochs: 10
    batch_size: 32
    learning_rate: 0.001  # Also set in optimizer config
```

**Loss Input Referencing:**
- `model.<output_name>` - References a model output
- `<column_name>` - References a target column from the training data

## 5. Supported Layer Types

### 5.1 PyTorch Neural Network Layers (`torch.nn.*`)

**Linear Layers:**
```yaml
- name: fc1
  type: torch.nn.Linear
  params:
    in_features: 128
    out_features: 64
    bias: true
```

**Activation Functions:**
- `torch.nn.ReLU`
- `torch.nn.Sigmoid`
- `torch.nn.Tanh`
- `torch.nn.LeakyReLU`
- `torch.nn.GELU`
- `torch.nn.Softmax` (params: `dim`)

**Normalization:**
```yaml
- name: norm
  type: torch.nn.LayerNorm
  params:
    normalized_shape: [64]
```
- Also: `torch.nn.BatchNorm1d`, `torch.nn.GroupNorm`

**Dropout:**
```yaml
- name: dropout
  type: torch.nn.Dropout
  params:
    p: 0.5
```

**Embeddings:**
```yaml
- name: user_embedding
  type: torch.nn.Embedding
  params:
    num_embeddings: 10000
    embedding_dim: 128
```

**Recurrent Layers:**
```yaml
- name: lstm
  type: torch.nn.LSTM
  params:
    input_size: 128
    hidden_size: 256
    num_layers: 2
    batch_first: true
    dropout: 0.2
```
- Also: `torch.nn.GRU`, `torch.nn.RNN`

**Convolutional Layers:**
- `torch.nn.Conv1d`, `torch.nn.Conv2d`, `torch.nn.Conv3d`
- `torch.nn.MaxPool1d`, `torch.nn.MaxPool2d`
- `torch.nn.AvgPool1d`, `torch.nn.AvgPool2d`

**Transformer:**
```yaml
- name: transformer
  type: torch.nn.TransformerEncoderLayer
  params:
    d_model: 512
    nhead: 8
    dim_feedforward: 2048
    dropout: 0.1
```
- Also: `torch.nn.TransformerEncoder`, `torch.nn.MultiheadAttention`

### 5.2 Tensor Operations (`torch.*`)

**Concatenation:**
```yaml
- name: concat
  type: torch.cat
  params:
    dim: 1
  inputs:
    tensors: [user_features.output, item_features.output]
```

**Arithmetic:**
- `torch.add` - Element-wise addition
- `torch.mul` - Element-wise multiplication
- `torch.matmul` - Matrix multiplication

**Shape Operations:**
- `torch.reshape` (params: `shape`)
- `torch.flatten` (params: `start_dim`, `end_dim`)
- `torch.squeeze` (params: `dim`)
- `torch.unsqueeze` (params: `dim`)

**Reductions:**
- `torch.mean` (params: `dim`, `keepdim`)
- `torch.sum` (params: `dim`, `keepdim`)
- `torch.max` (params: `dim`, `keepdim`)

### 5.3 Functional Operations (`torch.nn.functional.*`)

```yaml
- name: activation
  type: torch.nn.functional.relu
  inputs:
    input: hidden.output
```

Supported: `relu`, `sigmoid`, `tanh`, `gelu`, `softmax`, `log_softmax`

### 5.4 Custom Modules (`arc.stack`)

Use `arc.stack` to instantiate custom PyTorch modules:

```yaml
- name: dcn_layer
  type: arc.stack
  params:
    module: modules.dcn_cross  # References the modules section
    in_features: 128
    num_layers: 3
  inputs:
    x0: features.output
    xl: features.output
```

This requires a `modules` section (see Advanced Features below).

## 6. Complete Examples

### 6.1 Simple Binary Classification

```yaml
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
  logits: classifier.output
  prediction: sigmoid.output
```

**Trainer (trainer.yaml):**
```yaml
optimizer:
  type: torch.optim.AdamW
  config:
    learning_rate: 0.001

loss:
  type: torch.nn.BCELoss
  inputs:
    pred: model.prediction
    target: outcome

config:
  epochs: 10
  batch_size: 32
```

### 6.2 Multi-Layer Neural Network

```yaml
inputs:
  features:
    dtype: float32
    shape: [null, 20]
    columns: [feat1, feat2, feat3, ...]  # 20 features

graph:
  - name: hidden1
    type: torch.nn.Linear
    params:
      in_features: 20
      out_features: 128
    inputs:
      input: features

  - name: relu1
    type: torch.nn.ReLU
    inputs:
      input: hidden1.output

  - name: dropout1
    type: torch.nn.Dropout
    params:
      p: 0.3
    inputs:
      input: relu1.output

  - name: hidden2
    type: torch.nn.Linear
    params:
      in_features: 128
      out_features: 64
    inputs:
      input: dropout1.output

  - name: relu2
    type: torch.nn.ReLU
    inputs:
      input: hidden2.output

  - name: output_layer
    type: torch.nn.Linear
    params:
      in_features: 64
      out_features: 1
    inputs:
      input: relu2.output

  - name: sigmoid
    type: torch.nn.Sigmoid
    inputs:
      input: output_layer.output

outputs:
  prediction: sigmoid.output
```

### 6.3 Multi-Input Model (Recommendation System)

```yaml
inputs:
  user_id:
    dtype: long
    shape: [null]
    columns: [user_id]

  movie_id:
    dtype: long
    shape: [null]
    columns: [movie_id]

  user_features:
    dtype: float32
    shape: [null, 3]
    columns: [age, gender, occupation]

graph:
  # User embedding
  - name: user_embedding
    type: torch.nn.Embedding
    params:
      num_embeddings: 1000
      embedding_dim: 32
    inputs:
      input: user_id

  - name: flatten_user
    type: torch.nn.Flatten
    inputs:
      input: user_embedding.output

  # Movie embedding
  - name: movie_embedding
    type: torch.nn.Embedding
    params:
      num_embeddings: 5000
      embedding_dim: 32
    inputs:
      input: movie_id

  - name: flatten_movie
    type: torch.nn.Flatten
    inputs:
      input: movie_embedding.output

  # Concatenate all features
  - name: concat_all
    type: torch.cat
    params:
      dim: 1
    inputs:
      tensors: [flatten_user.output, flatten_movie.output, user_features.output]

  # Prediction layers
  - name: fc1
    type: torch.nn.Linear
    params:
      in_features: 67  # 32 + 32 + 3
      out_features: 128
    inputs:
      input: concat_all.output

  - name: relu
    type: torch.nn.ReLU
    inputs:
      input: fc1.output

  - name: fc2
    type: torch.nn.Linear
    params:
      in_features: 128
      out_features: 1
    inputs:
      input: relu.output

  - name: sigmoid
    type: torch.nn.Sigmoid
    inputs:
      input: fc2.output

outputs:
  rating_prediction: sigmoid.output
```

**Trainer:**
```yaml
optimizer:
  type: torch.optim.Adam
  config:
    learning_rate: 0.001

loss:
  type: torch.nn.MSELoss
  inputs:
    pred: model.rating_prediction
    target: rating

config:
  epochs: 20
  batch_size: 256
```

## 7. Advanced Features

### 7.1 Reusable Modules

Define reusable sub-graphs with the `modules` section:

```yaml
modules:
  dcn_cross:
    inputs: [x0, xl]
    graph:
      - name: cross_product
        type: torch.mul
        inputs:
          input: [x0, xl]

      - name: dense
        type: torch.nn.Linear
        params:
          in_features: 128
          out_features: 128
        inputs:
          input: cross_product.output

      - name: add_residual
        type: torch.add
        inputs:
          input: [dense.output, xl]

    outputs:
      output: add_residual.output

inputs:
  features:
    dtype: float32
    shape: [null, 128]
    columns: [...]

graph:
  - name: dcn_layer1
    type: arc.stack
    params:
      module: modules.dcn_cross
      in_features: 128
      num_layers: 3
    inputs:
      x0: features.output
      xl: features.output

  # Use module output
  - name: output
    type: torch.nn.Linear
    params:
      in_features: 128
      out_features: 1
    inputs:
      input: dcn_layer1.output

outputs:
  prediction: output.output
```

### 7.2 Loss Function in Model YAML

You can optionally include loss configuration directly in the model YAML:

```yaml
inputs: {...}
graph: [...]
outputs: {...}

loss:
  type: torch.nn.BCELoss
  inputs:
    pred: model.prediction
    target: outcome
```

This is useful when the loss is tightly coupled to the model architecture.

### 7.3 Metadata Fields

Optional metadata fields for tracking:

```yaml
name: diabetes_classifier  # Model name
data_table: patients  # Training data table
plan_id: abc123  # ML plan ID for lineage

inputs: {...}
graph: [...]
outputs: {...}
```

These fields are typically injected by Arc's tooling.

## 8. Additional Resources

- **PyTorch Layer Documentation**: https://pytorch.org/docs/stable/nn.html
- **Example Models**: See `examples/` directory
- **Test Fixtures**: See `tests/fixtures/` for validated examples
