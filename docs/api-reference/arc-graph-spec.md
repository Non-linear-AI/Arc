# Arc-Graph Specification Reference

Quick reference for Arc-Graph YAML specification format. For complete details, see the [Arc-Graph Concepts documentation](../concepts/arc-graph.md).

## Overview

Arc-Graph is a declarative YAML specification for ML model architecture and training configuration.

## Basic Structure

```yaml
# Required sections
inputs: { }      # Input definitions
graph: [ ]       # Model layers and connections
outputs: { }     # Output definitions

# Optional sections
trainer: { }     # Training configuration
evaluator: { }   # Evaluation metrics
```

## Inputs Section

Define model inputs with shapes and types.

```yaml
inputs:
  input_name:
    dtype: float32              # Data type
    shape: [null, features]     # Shape (null = batch dimension)
    columns: [col1, col2, ...]  # Optional: feature names
```

**Example**:
```yaml
inputs:
  user_features:
    dtype: float32
    shape: [null, 10]
    columns: [age, income, ...]

  item_features:
    dtype: float32
    shape: [null, 5]
```

## Graph Section

Define model layers and connections using PyTorch layers.

```yaml
graph:
  - name: layer_name
    type: torch.nn.LayerType
    params:
      param1: value1
      param2: value2
    inputs:
      input: previous_layer.output
```

**Common Layer Types**:
- `torch.nn.Linear` - Fully connected layer
- `torch.nn.Conv1d/Conv2d` - Convolutional layers
- `torch.nn.LSTM/GRU` - Recurrent layers
- `torch.nn.ReLU/LeakyReLU` - Activation functions
- `torch.nn.Dropout` - Dropout regularization
- `torch.nn.BatchNorm1d/LayerNorm` - Normalization
- `torch.nn.Sigmoid/Softmax` - Output activations

**Example**:
```yaml
graph:
  - name: hidden1
    type: torch.nn.Linear
    params:
      in_features: 10
      out_features: 64
    inputs:
      input: user_features

  - name: relu1
    type: torch.nn.ReLU
    inputs:
      input: hidden1.output

  - name: dropout1
    type: torch.nn.Dropout
    params:
      p: 0.2
    inputs:
      input: relu1.output
```

## Outputs Section

Define model outputs.

```yaml
outputs:
  output_name: layer_name.output
```

**Example**:
```yaml
outputs:
  prediction: final_layer.output
  embedding: hidden_layer.output  # Multi-output
```

## Trainer Section (Optional)

Training configuration.

```yaml
trainer:
  optimizer:
    type: torch.optim.OptimizerType
    params:
      lr: 0.001
      weight_decay: 0.0001
  loss: torch.nn.LossType
  epochs: 50
  batch_size: 32
  validation_split: 0.2
```

**Common Optimizers**:
- `torch.optim.Adam` - Adaptive learning rate
- `torch.optim.SGD` - Stochastic gradient descent
- `torch.optim.AdamW` - Adam with weight decay

**Common Loss Functions**:
- `torch.nn.BCELoss` - Binary cross-entropy
- `torch.nn.CrossEntropyLoss` - Multi-class classification
- `torch.nn.MSELoss` - Mean squared error (regression)
- `torch.nn.L1Loss` - Mean absolute error

**Example**:
```yaml
trainer:
  optimizer:
    type: torch.optim.Adam
    params:
      lr: 0.001
      betas: [0.9, 0.999]
  loss: torch.nn.BCELoss
  epochs: 100
  batch_size: 64
  validation_split: 0.15
```

## Evaluator Section (Optional)

Evaluation metrics.

```yaml
evaluator:
  metrics:
    - accuracy
    - precision
    - recall
    - f1
    - auc
```

**Classification Metrics**:
- `accuracy` - Overall accuracy
- `precision` - Precision score
- `recall` - Recall score
- `f1` - F1 score
- `auc` - Area under ROC curve

**Regression Metrics**:
- `mse` - Mean squared error
- `rmse` - Root mean squared error
- `mae` - Mean absolute error
- `r2` - R-squared score

## Complete Example: Binary Classifier

```yaml
inputs:
  features:
    dtype: float32
    shape: [null, 8]
    columns: [age, bmi, glucose, blood_pressure, insulin, skin_thickness, diabetes_pedigree, pregnancies]

graph:
  # Hidden layer 1
  - name: hidden1
    type: torch.nn.Linear
    params:
      in_features: 8
      out_features: 64
    inputs:
      input: features

  - name: relu1
    type: torch.nn.ReLU
    inputs:
      input: hidden1.output

  - name: dropout1
    type: torch.nn.Dropout
    params:
      p: 0.2
    inputs:
      input: relu1.output

  # Hidden layer 2
  - name: hidden2
    type: torch.nn.Linear
    params:
      in_features: 64
      out_features: 32
    inputs:
      input: dropout1.output

  - name: relu2
    type: torch.nn.ReLU
    inputs:
      input: hidden2.output

  - name: dropout2
    type: torch.nn.Dropout
    params:
      p: 0.2
    inputs:
      input: relu2.output

  # Output layer
  - name: output
    type: torch.nn.Linear
    params:
      in_features: 32
      out_features: 1
    inputs:
      input: dropout2.output

  - name: sigmoid
    type: torch.nn.Sigmoid
    inputs:
      input: output.output

outputs:
  prediction: sigmoid.output

trainer:
  optimizer:
    type: torch.optim.Adam
    params:
      lr: 0.001
  loss: torch.nn.BCELoss
  epochs: 50
  batch_size: 32
  validation_split: 0.2

evaluator:
  metrics:
    - accuracy
    - precision
    - recall
    - f1
    - auc
```

## Complete Example: Multi-Input Model

```yaml
inputs:
  user_features:
    dtype: float32
    shape: [null, 4]

  item_features:
    dtype: float32
    shape: [null, 3]

graph:
  # User branch
  - name: user_fc
    type: torch.nn.Linear
    params:
      in_features: 4
      out_features: 16
    inputs:
      input: user_features

  - name: user_relu
    type: torch.nn.ReLU
    inputs:
      input: user_fc.output

  # Item branch
  - name: item_fc
    type: torch.nn.Linear
    params:
      in_features: 3
      out_features: 16
    inputs:
      input: item_features

  - name: item_relu
    type: torch.nn.ReLU
    inputs:
      input: item_fc.output

  # Concatenate
  - name: concat
    type: torch.cat
    params:
      dim: 1
    inputs:
      tensors: [user_relu.output, item_relu.output]

  # Final prediction
  - name: final
    type: torch.nn.Linear
    params:
      in_features: 32
      out_features: 1
    inputs:
      input: concat.output

outputs:
  rating: final.output

trainer:
  optimizer:
    type: torch.optim.Adam
    params:
      lr: 0.001
  loss: torch.nn.MSELoss
  epochs: 100
  batch_size: 64
```

## Layer Input/Output Patterns

### Single Input
```yaml
- name: layer
  type: torch.nn.Linear
  inputs:
    input: previous_layer.output
```

### Multiple Inputs (Concatenation)
```yaml
- name: concat
  type: torch.cat
  params:
    dim: 1
  inputs:
    tensors: [layer1.output, layer2.output]
```

### Addition (Residual Connection)
```yaml
- name: residual
  type: torch.add
  inputs:
    input1: skip_connection.output
    input2: layer.output
```

## Common Patterns

### Dropout After Activation
```yaml
- name: linear
  type: torch.nn.Linear
  params: {in_features: 64, out_features: 32}
  inputs: {input: prev.output}

- name: relu
  type: torch.nn.ReLU
  inputs: {input: linear.output}

- name: dropout
  type: torch.nn.Dropout
  params: {p: 0.3}
  inputs: {input: relu.output}
```

### Batch Normalization
```yaml
- name: linear
  type: torch.nn.Linear
  params: {in_features: 64, out_features: 32}
  inputs: {input: prev.output}

- name: batch_norm
  type: torch.nn.BatchNorm1d
  params: {num_features: 32}
  inputs: {input: linear.output}

- name: relu
  type: torch.nn.ReLU
  inputs: {input: batch_norm.output}
```

### Residual Block
```yaml
- name: fc1
  type: torch.nn.Linear
  params: {in_features: 64, out_features: 64}
  inputs: {input: prev.output}

- name: relu
  type: torch.nn.ReLU
  inputs: {input: fc1.output}

- name: fc2
  type: torch.nn.Linear
  params: {in_features: 64, out_features: 64}
  inputs: {input: relu.output}

- name: residual
  type: torch.add
  inputs:
    input1: prev.output  # Skip connection
    input2: fc2.output
```

## Validation Rules

Arc validates Arc-Graph specifications:

1. **All referenced layers must exist** - Can't use `layer.output` if `layer` doesn't exist
2. **Types must match** - Input/output shapes must be compatible
3. **Required sections** - Must have `inputs`, `graph`, and `outputs`
4. **Valid layer types** - Only PyTorch layer types supported
5. **Unique layer names** - No duplicate names in graph

## Tips

1. **Start simple** - Begin with basic MLP, add complexity gradually
2. **Name layers clearly** - Use descriptive names: `user_embedding`, not `layer1`
3. **Check shapes** - Ensure layer dimensions match
4. **Use proven patterns** - Stick to well-tested architectures
5. **Test incrementally** - Add layers one at a time, test each addition

## Next Steps

- **[Complete Arc-Graph Documentation](../concepts/arc-graph.md)** - Full specification details
- **[Custom Architecture Examples](../examples/custom-architecture.md)** - Advanced patterns
- **[Model Training Guide](../guides/model-training.md)** - Training best practices

## PyTorch Layer Reference

For complete layer documentation, see:
- [PyTorch Neural Network Layers](https://pytorch.org/docs/stable/nn.html)
- [PyTorch Activation Functions](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
- [PyTorch Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
