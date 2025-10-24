# Multi-Layer Perceptron (MLP) Architecture Guide

## Overview

Multi-Layer Perceptron (MLP) is a feedforward neural network consisting of fully connected layers. It's the foundational architecture for tabular data prediction tasks including classification and regression.

## Architecture Pattern

### Basic Structure

```
Input → Dense Layer(s) → Activation → ... → Output Layer
```

### Key Components

1. **Input Layer**: Accepts tabular features
2. **Hidden Layers**: Fully connected (Dense) layers with non-linear activations
3. **Output Layer**: Task-specific output (classification or regression)
4. **Activations**: ReLU for hidden layers, task-specific for output

## Arc-Graph Implementation Patterns

### 1. Binary Classification MLP

```yaml
inputs:
  features:
    dtype: float32
    shape: [null, N]  # N = number of input features
    columns: [feature1, feature2, ...]

graph:
  - id: hidden1
    type: torch.nn.Linear
    inputs: [features]
    params:
      in_features: N
      out_features: 128

  - id: relu1
    type: torch.nn.ReLU
    inputs: [hidden1]

  - id: dropout1
    type: torch.nn.Dropout
    inputs: [relu1]
    params:
      p: 0.3

  - id: hidden2
    type: torch.nn.Linear
    inputs: [dropout1]
    params:
      in_features: 128
      out_features: 64

  - id: relu2
    type: torch.nn.ReLU
    inputs: [hidden2]

  - id: output
    type: torch.nn.Linear
    inputs: [relu2]
    params:
      in_features: 64
      out_features: 1

outputs:
  prediction:
    node: output

loss:
  type: torch.nn.BCEWithLogitsLoss
```

### 2. Multi-Class Classification MLP

```yaml
inputs:
  features:
    dtype: float32
    shape: [null, N]
    columns: [feature1, feature2, ...]

graph:
  - id: hidden1
    type: torch.nn.Linear
    inputs: [features]
    params:
      in_features: N
      out_features: 256

  - id: relu1
    type: torch.nn.ReLU
    inputs: [hidden1]

  - id: dropout1
    type: torch.nn.Dropout
    inputs: [relu1]
    params:
      p: 0.3

  - id: hidden2
    type: torch.nn.Linear
    inputs: [dropout1]
    params:
      in_features: 256
      out_features: 128

  - id: relu2
    type: torch.nn.ReLU
    inputs: [hidden2]

  - id: hidden3
    type: torch.nn.Linear
    inputs: [relu2]
    params:
      in_features: 128
      out_features: 64

  - id: relu3
    type: torch.nn.ReLU
    inputs: [hidden3]

  - id: output
    type: torch.nn.Linear
    inputs: [relu3]
    params:
      in_features: 64
      out_features: num_classes  # Number of classes

outputs:
  prediction:
    node: output

loss:
  type: torch.nn.CrossEntropyLoss
```

### 3. Regression MLP

```yaml
inputs:
  features:
    dtype: float32
    shape: [null, N]
    columns: [feature1, feature2, ...]

graph:
  - id: hidden1
    type: torch.nn.Linear
    inputs: [features]
    params:
      in_features: N
      out_features: 128

  - id: relu1
    type: torch.nn.ReLU
    inputs: [hidden1]

  - id: hidden2
    type: torch.nn.Linear
    inputs: [relu1]
    params:
      in_features: 128
      out_features: 64

  - id: relu2
    type: torch.nn.ReLU
    inputs: [hidden2]

  - id: output
    type: torch.nn.Linear
    inputs: [relu2]
    params:
      in_features: 64
      out_features: 1

outputs:
  prediction:
    node: output

loss:
  type: torch.nn.MSELoss
```

## Configuration Guidelines

### Hidden Layer Sizing

**Progressive reduction** is a common pattern:
- Start with larger hidden layers (128-256 units)
- Gradually reduce size in deeper layers (64, 32)
- Final layer matches output requirements

Example progression:
- Input (N features) → 256 → 128 → 64 → Output

### Depth Selection

- **Shallow (2-3 layers)**: Simple patterns, small datasets
- **Medium (3-4 layers)**: Most tabular tasks
- **Deep (5+ layers)**: Complex relationships, large datasets

### Regularization

**Dropout** is essential for preventing overfitting:
```yaml
- id: dropout
  type: torch.nn.Dropout
  inputs: [previous_layer]
  params:
    p: 0.3  # Drop 30% of neurons
```

Common dropout rates:
- 0.2-0.3: Standard regularization
- 0.4-0.5: Heavy regularization for overfitting
- 0.1: Light regularization

### Activation Functions

**Hidden layers**: Use ReLU for most cases
```yaml
- id: relu
  type: torch.nn.ReLU
  inputs: [linear_layer]
```

**Output layer**: Task-dependent
- Binary classification: No activation (use BCEWithLogitsLoss)
- Multi-class: No activation (use CrossEntropyLoss)
- Regression: No activation (linear output)

### Loss Functions

**Binary Classification**:
```yaml
loss:
  type: torch.nn.BCEWithLogitsLoss
  target_columns: [target]
```

**Multi-Class Classification**:
```yaml
loss:
  type: torch.nn.CrossEntropyLoss
  target_columns: [target]
```

**Regression**:
```yaml
loss:
  type: torch.nn.MSELoss
  target_columns: [target]
```

## Input Handling

### Feature Columns

All feature columns should be numeric. The input shape `[null, N]` where:
- `null`: Batch dimension (variable)
- `N`: Number of features

```yaml
inputs:
  features:
    dtype: float32
    shape: [null, 10]  # 10 features
    columns:
      - age
      - income
      - credit_score
      - balance
      - num_transactions
      - avg_transaction
      - account_age
      - num_products
      - is_active
      - credit_limit
```

### Categorical Features

Categorical features should be encoded before the model:
- Use one-hot encoding
- Use embedding layers (for high cardinality)
- Include encoded columns in feature list

## Best Practices

1. **Start simple**: Begin with 2-3 layers, expand if needed
2. **Monitor overfitting**: Use dropout and validation metrics
3. **Scale features**: Ensure features are normalized/standardized
4. **Match input size**: `in_features` must match number of columns
5. **Match layer dimensions**: Output of layer N = input of layer N+1
6. **Progressive reduction**: Gradually reduce hidden layer sizes
7. **Appropriate loss**: Match loss function to task type

## Common Patterns

### Pattern 1: Standard Binary Classifier
- 2-3 hidden layers
- ReLU activations
- Dropout (0.3)
- Single output unit
- BCEWithLogitsLoss

### Pattern 2: Deep Multi-Class Classifier
- 3-4 hidden layers
- ReLU activations
- Dropout (0.3-0.4)
- Output units = number of classes
- CrossEntropyLoss

### Pattern 3: Regression Model
- 2-3 hidden layers
- ReLU activations
- Light dropout (0.2)
- Single output unit
- MSELoss

## Example: PIDD Diabetes Prediction

For a diabetes prediction task with 8 features:

```yaml
inputs:
  features:
    dtype: float32
    shape: [null, 8]
    columns: [pregnancies, glucose, blood_pressure, skin_thickness,
              insulin, bmi, diabetes_pedigree, age]

graph:
  - id: hidden1
    type: torch.nn.Linear
    inputs: [features]
    params:
      in_features: 8
      out_features: 64

  - id: relu1
    type: torch.nn.ReLU
    inputs: [hidden1]

  - id: dropout1
    type: torch.nn.Dropout
    inputs: [relu1]
    params:
      p: 0.3

  - id: hidden2
    type: torch.nn.Linear
    inputs: [dropout1]
    params:
      in_features: 64
      out_features: 32

  - id: relu2
    type: torch.nn.ReLU
    inputs: [hidden2]

  - id: output
    type: torch.nn.Linear
    inputs: [relu2]
    params:
      in_features: 32
      out_features: 1

outputs:
  prediction:
    node: output

loss:
  type: torch.nn.BCEWithLogitsLoss
  target_columns: [outcome]
```

## When to Use MLP

**Ideal for**:
- Tabular/structured data
- Classification tasks
- Regression tasks
- Simple to moderate feature interactions

**Consider alternatives for**:
- Explicit feature crossing → DCN
- Sequential data → RNN/Transformer
- Image data → CNN
- Multi-task learning → MMoE
