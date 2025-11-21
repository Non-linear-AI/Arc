# Model Training Guide

This guide covers how to train machine learning models with Arc. Arc handles the complexity of model implementation while you focus on the high-level architecture and training strategy.

## Overview

Training a model with Arc involves:
1. **Describe your goal** - Tell Arc what you want to predict
2. **Review Arc-Graph** - Arc generates a model architecture specification
3. **Train** - Arc builds and trains the PyTorch model
4. **Monitor** - View training progress in TensorBoard
5. **Evaluate** - Check model performance

## Quick Start

The simplest way to train a model:

```
Build a model to predict diabetes using the processed_diabetes_data table
```

Arc will:
- Analyze your data
- Generate an Arc-Graph specification
- Train the model
- Launch TensorBoard
- Report results

## Using the /ml model Command

For more control, use the `/ml model` command:

```
/ml model --name diabetes_predictor
          --instruction "Build a binary classifier to predict diabetes"
          --data-table processed_diabetes_data
```

### With a Plan

If you've already created an ML plan:

```
/ml model --name diabetes_predictor
          --instruction "Build the model according to the plan"
          --data-table processed_diabetes_data
          --plan-id <plan_id>
```

## Understanding the Training Process

### Step 1: Arc-Graph Generation

Arc generates an Arc-Graph specification defining:

**Inputs**: Feature shapes and types
```yaml
inputs:
  features:
    dtype: float32
    shape: [null, 8]  # 8 features
    columns: [age, bmi, glucose, ...]
```

**Model Architecture**: Layers and connections
```yaml
graph:
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
```

**Training Configuration**: Optimizer, loss, epochs
```yaml
trainer:
  optimizer:
    type: torch.optim.Adam
    params:
      lr: 0.001
  loss: torch.nn.BCELoss
  epochs: 50
  batch_size: 32
```

### Step 2: Model Training

Arc executes the training process:
1. Builds the PyTorch model from Arc-Graph
2. Loads data from the specified table
3. Creates train/validation dataloaders
4. Trains for specified epochs
5. Tracks metrics (loss, accuracy, etc.)
6. Saves checkpoints

### Step 3: TensorBoard Monitoring

Arc automatically launches TensorBoard at `http://localhost:6006` where you can view:
- Training/validation loss curves
- Accuracy metrics over epochs
- Model architecture graph
- Hyperparameter comparisons

### Step 4: Model Saving

Trained models are saved to:
- `~/.arc/models/<model_name>/model.pt` - Model weights
- `~/.arc/models/<model_name>/arc_graph.yaml` - Model specification
- `~/.arc/models/<model_name>/metadata.json` - Training metadata

## Model Types

### Binary Classification

Predict one of two classes (yes/no, true/false):

```
Build a binary classifier to predict customer churn
```

Common use cases:
- Fraud detection
- Click prediction
- Disease diagnosis
- Email spam detection

### Multi-Class Classification

Predict one of many classes:

```
Build a classifier to predict movie genres (action, comedy, drama, etc.)
```

Common use cases:
- Image classification
- Product categorization
- Sentiment analysis (positive/neutral/negative)

### Regression

Predict continuous values:

```
Build a regression model to predict house prices
```

Common use cases:
- Price prediction
- Demand forecasting
- Risk scoring
- Time series forecasting

### Multi-Output Models

Predict multiple targets simultaneously:

```
Build a model to predict both price and demand
```

## Training Configuration

### Adjusting Training Parameters

You can specify training details:

```
Build a model with:
- 3 hidden layers with 128, 64, 32 neurons
- Learning rate: 0.0001
- Batch size: 64
- Train for 100 epochs
- Use Adam optimizer
```

### Key Training Parameters

**Epochs**: Number of complete passes through the training data
- Start with 50-100 epochs
- Use early stopping to prevent overfitting
- Monitor validation loss - stop if it plateaus or increases

**Batch Size**: Number of samples per gradient update
- Smaller (16-32): More stable, slower training
- Larger (128-256): Faster training, less stable
- Default: 32 is usually a good starting point

**Learning Rate**: Step size for optimizer
- Too high: Training unstable, loss oscillates
- Too low: Training very slow, may get stuck
- Default: 0.001 (Adam optimizer) is usually good
- Try: 0.0001 to 0.01 range

**Optimizer**: Algorithm for updating weights
- **Adam**: Good default, adaptive learning rate
- **SGD**: Classic, sometimes better with momentum
- **AdamW**: Adam with weight decay, good for regularization

### Loss Functions

Arc selects appropriate loss functions, but you can specify:

**Classification**:
- `BCELoss`: Binary classification (with sigmoid output)
- `CrossEntropyLoss`: Multi-class classification
- `BCEWithLogitsLoss`: Binary classification (without sigmoid)

**Regression**:
- `MSELoss`: Mean squared error (penalizes large errors)
- `L1Loss`: Mean absolute error (robust to outliers)
- `HuberLoss`: Combination of MSE and L1

## Model Architectures

### Simple Feed-Forward Network (MLP)

Best for: Tabular data, structured features

```
Build a 3-layer feed-forward network for prediction
```

### Deep & Cross Network (DCN)

Best for: Learning feature interactions, recommendation systems

```
Build a Deep & Cross Network to capture feature interactions
```

### Multi-gate Mixture-of-Experts (MMOE)

Best for: Multi-task learning, predicting multiple related outputs

```
Build an MMOE model to predict both click and conversion
```

### Custom Architecture

Specify your own architecture:

```
Build a model with:
- Input layer with 20 features
- Hidden layer 1: 128 neurons with ReLU
- Dropout: 0.3
- Hidden layer 2: 64 neurons with ReLU
- Dropout: 0.2
- Output layer: 1 neuron with sigmoid
```

## Monitoring Training

### Via TensorBoard

TensorBoard launches automatically. View:

**Scalars Tab**:
- `train/loss`: Training loss per epoch
- `val/loss`: Validation loss per epoch
- `train/accuracy`: Training accuracy
- `val/accuracy`: Validation accuracy

**Graphs Tab**:
- Model architecture visualization
- See how layers connect

**Histograms Tab**:
- Weight distributions
- Gradient distributions

### Via Console Output

Arc prints training progress:

```
Epoch 1/50:
  Train Loss: 0.6234  Train Acc: 0.6543
  Val Loss:   0.5987  Val Acc:   0.6721

Epoch 2/50:
  Train Loss: 0.5456  Train Acc: 0.7123
  Val Loss:   0.5234  Val Acc:   0.7234
...
```

### Check Training Jobs

View all training jobs:

```
/ml jobs list
```

View specific job details:

```
/ml jobs status <job_id>
```

## Examples

### Example 1: Simple Binary Classifier

```
/ml model --name spam_detector
          --instruction "Binary classifier for spam detection with 2 hidden layers (64, 32)"
          --data-table processed_emails
```

### Example 2: Regression Model

```
/ml model --name price_predictor
          --instruction "Regression model to predict house prices with:
                        - 3 hidden layers: 128, 64, 32
                        - Dropout: 0.2 after each layer
                        - Learning rate: 0.0001
                        - 100 epochs"
          --data-table housing_features
```

### Example 3: Multi-Class Classifier

```
/ml model --name genre_classifier
          --instruction "Multi-class classifier for movie genre with:
                        - Deep network: 256, 128, 64 neurons
                        - Batch size: 64
                        - Adam optimizer with lr=0.001"
          --data-table movie_features
```

## Next Steps

- **[Model Evaluation Guide](model-evaluation.md)** - Evaluate your trained models
- **[Making Predictions](making-predictions.md)** - Use your model for inference
- **[Arc-Graph Specification](../concepts/arc-graph.md)** - Understand the model spec format

## Related Documentation

- [Feature Engineering Guide](feature-engineering.md) - Prepare data for training
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard) - TensorBoard guide
