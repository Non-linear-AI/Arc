# Deep & Cross Network (DCN) Architecture

## Overview
DCN combines **explicit feature crossing** (cross network) with **implicit pattern learning** (deep network) through a parallel architecture design.

**When to use**: CTR prediction, recommendation systems, when feature interactions are critical.

## Core Architecture Pattern

### Cross Layer Mathematics
Each cross layer computes: `x_{l+1} = x_0 ⊙ (W_l * x_l + b_l) + x_l`

- Preserves input dimension across layers
- Element-wise multiplication with original input
- Residual connection maintains feature information
- Bounded-degree feature interactions

### Parallel Design
1. **Cross Network**: Stacked cross layers for explicit interactions
2. **Deep Network**: Traditional MLP for implicit patterns
3. **Combination**: Concatenate outputs before final prediction layer

## Arc-Graph Implementation

### Required Modules

**Cross Layer Module**:
```yaml
modules:
  cross_layer:
    inputs: [x_current, x_original]
    graph:
      - name: linear_transform
        type: torch.nn.Linear
        params: { in_features: INPUT_DIM, out_features: INPUT_DIM }
        inputs: { input: x_current }

      - name: element_wise_product
        type: torch.mul
        inputs: [x_original, linear_transform.output]

      - name: residual_add
        type: torch.add
        inputs: [element_wise_product.output, x_current]

    outputs:
      cross_output: residual_add.output
```

**Deep Network Module**:
```yaml
  deep_network:
    inputs: [input_features]
    graph:
      - name: dense1
        type: torch.nn.Linear
        params: { in_features: INPUT_DIM, out_features: 512 }
        inputs: { input: input_features }

      - name: relu1
        type: torch.nn.functional.relu
        inputs: { input: dense1.output }

      - name: dense2
        type: torch.nn.Linear
        params: { in_features: 512, out_features: 256 }
        inputs: { input: relu1.output }

      - name: relu2
        type: torch.nn.functional.relu
        inputs: { input: dense2.output }

      - name: dense3
        type: torch.nn.Linear
        params: { in_features: 256, out_features: 128 }
        inputs: { input: relu2.output }

    outputs:
      deep_output: dense3.output
```

### Main Graph Pattern

```yaml
graph:
  # Cross Network - stacked cross layers
  - name: cross_network
    type: arc.stack
    params:
      module: cross_layer
      count: 3  # Typically 3-6 layers
    inputs: [features, features]  # Current and original

  # Deep Network
  - name: deep_branch
    type: module.deep_network
    inputs: [features]

  # Combine outputs
  - name: combined_features
    type: torch.cat
    params: { dim: 1 }
    inputs: [cross_network.output, deep_branch.output]

  # Final prediction
  - name: output_layer
    type: torch.nn.Linear
    params: { in_features: CROSS_DIM + DEEP_DIM, out_features: 1 }
    inputs: { input: combined_features.output }

  # For binary classification, add sigmoid for evaluation
  - name: probabilities
    type: torch.nn.functional.sigmoid
    inputs: { input: output_layer.output }

outputs:
  logits: output_layer.output
  probabilities: probabilities.output

training:
  loss:
    type: torch.nn.functional.binary_cross_entropy_with_logits
    inputs:
      input: logits
      target: target_column
  optimizer:
    type: torch.optim.Adam
    lr: 0.001
  epochs: 50
  batch_size: 64
  validation_split: 0.2
  metrics: [accuracy, auroc]
```

## Configuration Guidelines

### Cross Network Depth
- **2-3 layers**: Simple interactions, small datasets
- **3-4 layers**: Balanced (recommended for most cases)
- **5-6 layers**: Complex interactions, large datasets

### Deep Network Width
- **Wide**: [512, 256, 128] for rich patterns (recommended)
- **Narrow**: [256, 128, 64] for simpler tasks or smaller data

### Dimension Calculation
When combining cross and deep outputs:
- Cross output dimension = input dimension
- Deep output dimension = last layer size
- Combined dimension = cross_dim + deep_dim

Example: If input=100, deep=[512,256,128]:
- Cross output: 100
- Deep output: 128
- Combined: 228
- Final layer: `in_features: 228, out_features: 1`

## Common Use Cases

### CTR Prediction
- Binary classification (click or not)
- Loss: `torch.nn.functional.binary_cross_entropy_with_logits` (in training section)
- Output: Logits and probabilities
- Features: User, item, context (categorical + numerical)
- Example training config:
  ```yaml
  training:
    loss:
      type: torch.nn.functional.binary_cross_entropy_with_logits
      inputs: { input: logits, target: clicked }
    optimizer:
      type: torch.optim.Adam
      lr: 0.001
    epochs: 50
    batch_size: 64
    validation_split: 0.2
    metrics: [accuracy, auroc]
  ```

### Recommendation Scoring
- Regression or ranking
- Loss: `torch.nn.functional.mse_loss` (in training section)
- Output: Score prediction
- Features: User-item interactions, metadata
- Example training config:
  ```yaml
  training:
    loss:
      type: torch.nn.functional.mse_loss
      inputs: { input: prediction, target: rating }
    optimizer:
      type: torch.optim.Adam
      lr: 0.001
    epochs: 50
    batch_size: 64
    validation_split: 0.2
    metrics: [mse, mae]
  ```

## Best Practices

1. **Feature Preprocessing**: Normalize numerical features, embed categoricals before DCN
2. **Cross Layer Count**: Start with 3-4, increase if underfitting
3. **Deep Network**: Match capacity to cross network
4. **Regularization**: Use dropout in deep network if overfitting
5. **Training Configuration**: Include loss inside training section with appropriate optimizer and hyperparameters
6. **Loss Function**: Match to task (BCEWithLogitsLoss for CTR, MSELoss for rating prediction)
7. **Output Structure**: For classification, include both logits (for loss) and probabilities (for evaluation)
