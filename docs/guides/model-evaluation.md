# Model Evaluation Guide

This guide covers how to evaluate your trained models in Arc. Proper evaluation is crucial to understand model performance and identify areas for improvement.

## Overview

Model evaluation in Arc involves:
- **Metrics Calculation**: Compute performance metrics (accuracy, precision, recall, etc.)
- **Visualization**: View results in TensorBoard
- **Comparison**: Compare different models
- **Analysis**: Understand model strengths and weaknesses

## Quick Start

Evaluate a trained model:

```
Evaluate the diabetes_predictor model on the test set
```

Or with the `/ml evaluate` command:

```
/ml evaluate --model-id <model_id> --data-table test_data
```

## Using the /ml evaluate Command

### Basic Evaluation

```
/ml evaluate --model-id <model_id> --data-table test_data
```

### With Custom Metrics

```
/ml evaluate --model-id <model_id>
             --data-table test_data
             --metrics "accuracy,precision,recall,f1,auc"
```

### Find Your Model ID

```
/ml jobs list
```

This shows all models with their IDs.

## Evaluation Metrics

### Classification Metrics

- **Accuracy**: Percentage of correct predictions
- **Precision**: Of predicted positives, how many are actually positive
- **Recall**: Of actual positives, how many did we predict
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve (1.0 = perfect, 0.5 = random)
- **Confusion Matrix**: Shows true/false positives and negatives

### Regression Metrics

- **MSE**: Mean Squared Error - average of squared errors
- **RMSE**: Root Mean Squared Error - MSE in original units
- **MAE**: Mean Absolute Error - average of absolute errors
- **R²**: Proportion of variance explained (1.0 = perfect fit)

## Viewing Evaluation Results

Query results with `/sql SELECT * FROM evaluations WHERE model_id = '<model_id>'`

Visualize metrics in TensorBoard:

```
/ml evaluate --model-id <model_id> --data-table test_data --tensorboard
```

## Model Comparison

Compare models using SQL queries on the `evaluations` table, grouping by `model_id` and comparing metrics.

## Example

```
/ml evaluate --model-id spam_detector_v1
             --data-table email_test
             --metrics "accuracy,precision,recall,f1,auc"
```

## Next Steps

- **[Making Predictions](making-predictions.md)** - Use your model for inference
- **[Model Training Guide](model-training.md)** - Improve your model based on evaluation
- **[Examples](../examples/logistic_regression_console/)** - See complete evaluation examples

## Related Documentation

- [Arc-Graph Specification](../concepts/arc-graph.md) - Understand model architecture
- [CLI Commands Reference](../api-reference/cli-commands.md) - All evaluation commands
