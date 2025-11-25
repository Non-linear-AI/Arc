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

Arc generates an Arc-Graph specification defining inputs, model architecture, and training configuration. See [Arc-Graph Specification](../concepts/arc-graph.md) for details.

Arc then builds the PyTorch model, loads data, trains for the specified epochs, and tracks metrics.

## Monitoring Training

Arc automatically launches TensorBoard at `http://localhost:6006` to view training progress, loss curves, and accuracy metrics.

Trained models are saved to `~/.arc/models/<model_name>/` with model weights, Arc-Graph specification, and metadata.

## Check Training Jobs

View all training jobs:

```
/ml jobs list
```

View specific job details:

```
/ml jobs status <job_id>
```

## Example

```
/ml model --name spam_detector
          --instruction "Binary classifier for spam detection with 2 hidden layers (64, 32)"
          --data-table processed_emails
```

## Next Steps

- **[Model Evaluation Guide](model-evaluation.md)** - Evaluate your trained models
- **[Making Predictions](making-predictions.md)** - Use your model for inference
- **[Arc-Graph Specification](../concepts/arc-graph.md)** - Understand the model spec format

## Related Documentation

- [Feature Engineering Guide](feature-engineering.md) - Prepare data for training
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard) - TensorBoard guide
