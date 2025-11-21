# Advanced Tutorial: Custom Model Architectures

This tutorial demonstrates how to build advanced model architectures with Arc, including Deep & Cross Networks, multi-input models, and custom layers.

**What You'll Learn**:
- Using advanced architectures (DCN, MMOE)
- Building multi-input models
- Customizing Arc-Graph specifications
- Handling complex data relationships

**Prerequisites**: Complete the [Diabetes Prediction Tutorial](diabetes-prediction.md) first.

## Example 1: Deep & Cross Network (DCN)

Deep & Cross Networks excel at learning feature interactions automatically.

### When to Use DCN

- Recommendation systems
- Click-through rate (CTR) prediction
- Any task where feature interactions matter

### Building a DCN Model

```
/ml model --name recommendation_dcn
          --instruction "
          Build a Deep & Cross Network with:
          - 3 cross layers
          - Deep network: 256, 128, 64 neurons
          - For predicting movie ratings
          "
          --data-table movie_features
```

### The Generated Arc-Graph

Arc generates a DCN spec like:

```yaml
inputs:
  features:
    dtype: float32
    shape: [null, 20]

graph:
  # Cross Network (learns feature interactions)
  - name: cross_1
    type: torch.nn.Linear
    params:
      in_features: 20
      out_features: 20
    inputs:
      input: features

  - name: cross_2
    type: torch.nn.Linear
    params:
      in_features: 20
      out_features: 20
    inputs:
      input: cross_1.output

  - name: cross_3
    type: torch.nn.Linear
    params:
      in_features: 20
      out_features: 20
    inputs:
      input: cross_2.output

  # Deep Network (learns high-level representations)
  - name: deep_1
    type: torch.nn.Linear
    params:
      in_features: 20
      out_features: 256
    inputs:
      input: features

  - name: deep_relu_1
    type: torch.nn.ReLU
    inputs:
      input: deep_1.output

  # ... more deep layers ...

  # Combine cross and deep
  - name: concat
    type: torch.cat
    params:
      dim: 1
    inputs:
      tensors: [cross_3.output, deep_3.output]

  - name: final
    type: torch.nn.Linear
    params:
      in_features: 84  # 20 + 64
      out_features: 1
    inputs:
      input: concat.output

outputs:
  rating: final.output
```

### Key Benefits

- **Automatic feature interactions**: Cross network learns interactions
- **Deep representations**: Deep network captures complex patterns
- **Best of both worlds**: Combines explicit and implicit interactions

## Example 2: Multi-Input Model

Build models that take multiple types of inputs (e.g., user features + item features).

### Use Case: Movie Recommendation

Predict ratings using:
- User features (age, gender, occupation)
- Movie features (genre, year, director)

### Data Preparation

```sql
-- Create user features table
CREATE TABLE user_features AS
SELECT
    user_id,
    age_normalized,
    gender_encoded,
    occupation_encoded
FROM users;

-- Create movie features table
CREATE TABLE movie_features AS
SELECT
    movie_id,
    genre_encoded,
    year_normalized,
    director_encoded
FROM movies;

-- Create ratings table (join key)
CREATE TABLE ratings AS
SELECT
    user_id,
    movie_id,
    rating
FROM raw_ratings;
```

### Feature Engineering

```
/ml data --name recommendation_features
         --instruction "
         Prepare multi-input recommendation data:
         1. Join users, movies, and ratings on user_id and movie_id
         2. Keep user and movie features separate for multi-input model
         3. Create 80/20 train/test split
         "
         --source-tables user_features,movie_features,ratings
```

### Building Multi-Input Model

```
/ml model --name multi_input_recommender
          --instruction "
          Build a multi-input model with:
          - User input branch: 32 -> 16 neurons
          - Movie input branch: 64 -> 32 neurons
          - Concatenate both branches
          - Final layers: 64 -> 32 -> 1
          - Predict rating (regression)
          "
          --data-table recommendation_features_train
```

### The Multi-Input Arc-Graph

```yaml
inputs:
  user_features:
    dtype: float32
    shape: [null, 4]  # age, gender, occupation, etc.

  movie_features:
    dtype: float32
    shape: [null, 3]  # genre, year, director

graph:
  # User branch
  - name: user_fc1
    type: torch.nn.Linear
    params:
      in_features: 4
      out_features: 32
    inputs:
      input: user_features

  - name: user_relu1
    type: torch.nn.ReLU
    inputs:
      input: user_fc1.output

  - name: user_fc2
    type: torch.nn.Linear
    params:
      in_features: 32
      out_features: 16
    inputs:
      input: user_relu1.output

  # Movie branch
  - name: movie_fc1
    type: torch.nn.Linear
    params:
      in_features: 3
      out_features: 64
    inputs:
      input: movie_features

  - name: movie_relu1
    type: torch.nn.ReLU
    inputs:
      input: movie_fc1.output

  - name: movie_fc2
    type: torch.nn.Linear
    params:
      in_features: 64
      out_features: 32
    inputs:
      input: movie_relu1.output

  # Combine branches
  - name: concat
    type: torch.cat
    params:
      dim: 1
    inputs:
      tensors: [user_fc2.output, movie_fc2.output]

  # Final prediction layers
  - name: final_fc1
    type: torch.nn.Linear
    params:
      in_features: 48  # 16 + 32
      out_features: 64
    inputs:
      input: concat.output

  - name: final_relu
    type: torch.nn.ReLU
    inputs:
      input: final_fc1.output

  - name: final_fc2
    type: torch.nn.Linear
    params:
      in_features: 64
      out_features: 1
    inputs:
      input: final_relu.output

outputs:
  rating: final_fc2.output
```

## Example 3: Multi-Task Learning (MMOE)

Train one model for multiple related tasks using Multi-gate Mixture-of-Experts.

### Use Case: E-commerce Prediction

Predict multiple outcomes simultaneously:
- Will user click?
- Will user purchase?
- How much will they spend?

### Building MMOE Model

```
/ml model --name ecommerce_mmoe
          --instruction "
          Build an MMOE model with:
          - 3 expert networks (each: 128 -> 64 neurons)
          - 3 tasks: click prediction, purchase prediction, spend regression
          - Shared bottom: 256 -> 128 neurons
          - Each task has its own tower: 32 -> 16 neurons
          "
          --data-table ecommerce_features_train
```

### Benefits of Multi-Task Learning

- **Shared learning**: Tasks help each other (positive transfer)
- **Efficient**: One model for multiple predictions
- **Better features**: Learns representations useful for all tasks

## Example 4: Attention Mechanism

Build models with attention layers for sequence or text data.

### Use Case: Time Series Forecasting

```
/ml model --name sales_forecaster
          --instruction "
          Build a time series model with attention:
          - Input: last 30 days of sales data
          - Attention mechanism to weight important days
          - LSTM layers: 128 -> 64
          - Attention over LSTM outputs
          - Predict next 7 days
          "
          --data-table sales_time_series
```

### The Attention Arc-Graph (Simplified)

```yaml
inputs:
  sequence:
    dtype: float32
    shape: [null, 30, 5]  # 30 days, 5 features per day

graph:
  - name: lstm
    type: torch.nn.LSTM
    params:
      input_size: 5
      hidden_size: 128
      num_layers: 2
      batch_first: true
    inputs:
      input: sequence

  # Attention mechanism
  - name: attention_weights
    type: torch.nn.Linear
    params:
      in_features: 128
      out_features: 1
    inputs:
      input: lstm.output

  - name: attention_softmax
    type: torch.nn.Softmax
    params:
      dim: 1
    inputs:
      input: attention_weights.output

  # Weighted sum (context vector)
  - name: context
    type: torch.bmm
    inputs:
      input1: attention_softmax.output
      input2: lstm.output

  - name: fc
    type: torch.nn.Linear
    params:
      in_features: 128
      out_features: 7  # Predict 7 days
    inputs:
      input: context.output

outputs:
  forecast: fc.output
```

## Example 5: Residual Connections (ResNet-style)

Build deep networks with skip connections for better gradient flow.

### Building Deep Network with Residuals

```
/ml model --name deep_classifier
          --instruction "
          Build a deep residual network with:
          - 6 blocks, each with residual connection
          - Each block: Linear -> ReLU -> Linear, then add skip connection
          - Gradually reduce dimensions: 256 -> 128 -> 64 -> 32 -> 16 -> 8
          - Final classification layer
          "
          --data-table processed_features
```

### Residual Block in Arc-Graph

```yaml
# Residual Block 1
- name: block1_fc1
  type: torch.nn.Linear
  params:
    in_features: 256
    out_features: 256
  inputs:
    input: input_layer.output

- name: block1_relu1
  type: torch.nn.ReLU
  inputs:
    input: block1_fc1.output

- name: block1_fc2
  type: torch.nn.Linear
  params:
    in_features: 256
    out_features: 256
  inputs:
    input: block1_relu1.output

# Skip connection (add input to output)
- name: block1_residual
  type: torch.add
  inputs:
    input1: input_layer.output
    input2: block1_fc2.output

- name: block1_relu_final
  type: torch.nn.ReLU
  inputs:
    input: block1_residual.output
```

## Customizing Arc-Graph Manually

You can also edit Arc-Graph YAML files directly for fine-grained control.

### Find and Edit Arc-Graph

```bash
# Models are saved in ~/.arc/models/
ls ~/.arc/models/

# Edit the Arc-Graph
nano ~/.arc/models/my_model/arc_graph.yaml
```

### Example Customizations

**Add Layer Normalization**:
```yaml
- name: layer_norm
  type: torch.nn.LayerNorm
  params:
    normalized_shape: [128]
  inputs:
    input: hidden_layer.output
```

**Change Activation Function**:
```yaml
# Instead of ReLU
- name: activation
  type: torch.nn.ReLU

# Try LeakyReLU
- name: activation
  type: torch.nn.LeakyReLU
  params:
    negative_slope: 0.01

# Or ELU
- name: activation
  type: torch.nn.ELU
  params:
    alpha: 1.0
```

**Add Batch Normalization**:
```yaml
- name: batch_norm
  type: torch.nn.BatchNorm1d
  params:
    num_features: 64
  inputs:
    input: hidden_layer.output
```

## Best Practices for Custom Architectures

### 1. Start Simple

Begin with a simple architecture, then add complexity:
1. Basic MLP
2. Add dropout
3. Add batch/layer norm
4. Try residual connections
5. Add attention (if needed)

### 2. Match Architecture to Problem

| Problem Type | Recommended Architecture |
|--------------|-------------------------|
| Tabular data | MLP, DCN |
| Feature interactions | DCN, cross networks |
| Multiple related tasks | MMOE |
| Sequence data | LSTM, attention |
| Very deep networks | ResNet-style residuals |

### 3. Use Proven Components

Stick to well-tested components:
- **Layers**: Linear, Conv, LSTM
- **Activations**: ReLU, LeakyReLU, ELU
- **Regularization**: Dropout, batch norm, layer norm
- **Connections**: Skip/residual connections

### 4. Validate with Simpler Baselines

Always compare custom architecture to:
- Simple MLP
- Standard architecture for your problem
- Ensure complexity is justified by performance gain

## Troubleshooting Custom Architectures

### Gradient Issues

If gradients vanish or explode:
- Add residual connections
- Use LayerNorm or BatchNorm
- Try different activation (LeakyReLU, ELU)
- Reduce depth
- Use gradient clipping

### Memory Issues

If running out of memory:
- Reduce batch size
- Reduce layer sizes
- Remove unnecessary layers
- Use gradient checkpointing

### Slow Training

If training is very slow:
- Check model size (count parameters)
- Simplify architecture
- Increase batch size
- Use fewer layers

## Next Steps

- **[Arc-Graph Specification](../concepts/arc-graph.md)** - Complete Arc-Graph reference
- **[Model Training Guide](../guides/model-training.md)** - Training best practices
- **[PyTorch Documentation](https://pytorch.org/docs/stable/nn.html)** - Available layers

## Need Help?

- Check [GitHub Issues](https://github.com/non-linear-ai/arc/issues)
- Ask in [Discussions](https://github.com/non-linear-ai/arc/discussions)
- Refer to [Arc-Graph examples](../../src/arc/core/agents/shared/examples/)
