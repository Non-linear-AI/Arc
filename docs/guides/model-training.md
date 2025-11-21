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

## Best Practices

### 1. Start Simple

Begin with a simple model:
- Single hidden layer
- Default hyperparameters
- Train for 50 epochs

Then iterate:
- Add complexity if underfitting
- Add regularization if overfitting

### 2. Use Validation Data

Always split your data:
```
Train: 70%, Validation: 15%, Test: 15%
```

- **Train**: Model learns from this
- **Validation**: Monitor overfitting during training
- **Test**: Final evaluation (use once)

### 3. Monitor for Overfitting

Watch for these signs:
- Training loss decreases, validation loss increases
- Training accuracy high, validation accuracy low
- Gap between train and validation metrics grows

Solutions:
- Add dropout layers
- Reduce model complexity
- Use L1/L2 regularization
- Get more training data
- Use data augmentation

### 4. Use Early Stopping

Stop training when validation loss stops improving:

```
Build a model with early stopping (patience=10 epochs)
```

### 5. Experiment with Hyperparameters

Try different configurations:
- Learning rates: [0.0001, 0.001, 0.01]
- Batch sizes: [16, 32, 64, 128]
- Hidden layer sizes: [32, 64, 128, 256]
- Dropout rates: [0.1, 0.2, 0.3, 0.5]

### 6. Save Checkpoints

Arc automatically saves:
- Best model (lowest validation loss)
- Final model (last epoch)
- Training history

## Troubleshooting

### Training Loss Not Decreasing

**Problem**: Loss stays high or random

**Solutions**:
1. Check data quality (missing values, wrong scales)
2. Reduce learning rate (try 0.0001)
3. Increase model capacity (more/larger layers)
4. Check loss function (correct for your task?)
5. Normalize input features

### Loss is NaN or Inf

**Problem**: Loss becomes NaN or infinity

**Solutions**:
1. Reduce learning rate significantly (try 0.00001)
2. Add gradient clipping
3. Check for extreme values in data
4. Ensure features are normalized
5. Check for division by zero in custom layers

### Overfitting

**Problem**: Great train accuracy, poor validation accuracy

**Solutions**:
1. Add dropout layers (start with 0.2-0.3)
2. Reduce model complexity (fewer/smaller layers)
3. Get more training data
4. Use L1/L2 regularization
5. Use early stopping

### Underfitting

**Problem**: Poor performance on both train and validation

**Solutions**:
1. Increase model capacity (more/larger layers)
2. Train for more epochs
3. Add more features
4. Try different architecture
5. Check data quality

### Training Too Slow

**Problem**: Training takes very long

**Solutions**:
1. Increase batch size (64, 128, 256)
2. Use fewer epochs initially
3. Sample your data for experimentation
4. Check if GPU is being used (if available)

### Out of Memory

**Problem**: GPU/RAM runs out of memory

**Solutions**:
1. Reduce batch size (try 16 or 8)
2. Reduce model size (fewer parameters)
3. Use gradient accumulation
4. Train on smaller data subset

## Advanced Topics

### Transfer Learning

Use pre-trained models as starting points:

```
Build a model using pre-trained embeddings from <model_name>
```

### Multi-Task Learning

Train one model for multiple related tasks:

```
Build a multi-task model to predict:
- Click probability
- Conversion probability
- Time spent
```

### Custom Layers

Use custom PyTorch layers:

```
Build a model with a custom attention mechanism
```

### Distributed Training

For very large datasets or models:

```
Train the model using distributed training across multiple GPUs
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
