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

#### Accuracy
Percentage of correct predictions:
```
Accuracy = (True Positives + True Negatives) / Total
```

**When to use**: Balanced datasets, overall performance measure

**Limitation**: Misleading for imbalanced datasets

#### Precision
Of predicted positives, how many are actually positive:
```
Precision = True Positives / (True Positives + False Positives)
```

**When to use**: When false positives are costly (spam detection, fraud)

#### Recall (Sensitivity)
Of actual positives, how many did we predict:
```
Recall = True Positives / (True Positives + False Negatives)
```

**When to use**: When false negatives are costly (disease diagnosis, fraud detection)

#### F1 Score
Harmonic mean of precision and recall:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**When to use**: Balance between precision and recall, imbalanced datasets

#### AUC-ROC
Area Under the Receiver Operating Characteristic Curve:

**When to use**: Binary classification, compare models, threshold-independent metric

**Interpretation**:
- 1.0: Perfect classifier
- 0.9-1.0: Excellent
- 0.8-0.9: Good
- 0.7-0.8: Fair
- 0.6-0.7: Poor
- 0.5: Random guessing

#### Confusion Matrix
Shows true positives, false positives, true negatives, false negatives:

```
                Predicted
              Neg    Pos
Actual  Neg   TN     FP
        Pos   FN     TP
```

### Regression Metrics

#### Mean Squared Error (MSE)
Average of squared errors:
```
MSE = Average((Predicted - Actual)²)
```

**When to use**: General regression, penalizes large errors heavily

#### Root Mean Squared Error (RMSE)
Square root of MSE, in same units as target:
```
RMSE = √MSE
```

**When to use**: Interpretable error in original units

#### Mean Absolute Error (MAE)
Average of absolute errors:
```
MAE = Average(|Predicted - Actual|)
```

**When to use**: Robust to outliers, easier to interpret

#### R-squared (R²)
Proportion of variance explained:
```
R² = 1 - (Sum of Squared Errors / Total Variance)
```

**Interpretation**:
- 1.0: Perfect fit
- 0.0: No better than predicting the mean
- < 0.0: Worse than predicting the mean

**When to use**: Compare models, understand explanatory power

## Viewing Evaluation Results

### Via SQL

Query evaluation results:

```sql
-- View latest evaluation
/sql SELECT * FROM evaluations
     WHERE model_id = '<model_id>'
     ORDER BY created_at DESC
     LIMIT 1

-- View all evaluations for a model
/sql SELECT
       model_id,
       created_at,
       accuracy,
       precision,
       recall,
       f1_score
     FROM evaluations
     WHERE model_id = '<model_id>'
     ORDER BY created_at DESC

-- Compare multiple models
/sql SELECT
       model_id,
       MAX(accuracy) as best_accuracy,
       MAX(f1_score) as best_f1
     FROM evaluations
     GROUP BY model_id
     ORDER BY best_accuracy DESC
```

### Via TensorBoard

Arc can visualize evaluation metrics in TensorBoard:

```
/ml evaluate --model-id <model_id>
             --data-table test_data
             --tensorboard
```

View at `http://localhost:6006` to see:
- Metric comparisons
- Confusion matrices
- ROC curves
- Prediction distributions

## Evaluating Different Model Types

### Binary Classification

Key metrics:
```
/ml evaluate --model-id <model_id>
             --data-table test_data
             --metrics "accuracy,precision,recall,f1,auc"
```

Look for:
- AUC > 0.7 (minimum)
- Balanced precision/recall
- Confusion matrix for error patterns

### Multi-Class Classification

Key metrics:
```
/ml evaluate --model-id <model_id>
             --data-table test_data
             --metrics "accuracy,macro_f1,weighted_f1"
```

Look for:
- Per-class precision/recall
- Confusion matrix (which classes confused?)
- Macro vs weighted F1 (performance across all classes)

### Regression

Key metrics:
```
/ml evaluate --model-id <model_id>
             --data-table test_data
             --metrics "mse,rmse,mae,r2"
```

Look for:
- RMSE in context of target range
- R² > 0.5 (minimum for useful model)
- Residual plots (errors vs predicted values)

## Model Comparison

Compare multiple models:

```sql
-- Compare models by accuracy
/sql SELECT
       model_id,
       model_name,
       MAX(accuracy) as best_accuracy,
       MAX(f1_score) as best_f1,
       MAX(auc) as best_auc
     FROM evaluations
     GROUP BY model_id, model_name
     ORDER BY best_accuracy DESC

-- Find best model for your metric
/sql SELECT model_id, model_name, accuracy
     FROM evaluations
     WHERE accuracy = (SELECT MAX(accuracy) FROM evaluations)
```

## Examples

### Example 1: Binary Classification Evaluation

```
/ml evaluate --model-id spam_detector_v1
             --data-table email_test
             --metrics "accuracy,precision,recall,f1,auc"

# Then check results
/sql SELECT * FROM evaluations
     WHERE model_id = 'spam_detector_v1'
     ORDER BY created_at DESC LIMIT 1
```

### Example 2: Regression Evaluation

```
/ml evaluate --model-id price_predictor
             --data-table housing_test
             --metrics "mse,rmse,mae,r2"
```

### Example 3: Compare Multiple Models

```sql
/sql SELECT
       model_name,
       accuracy,
       f1_score,
       auc
     FROM evaluations
     WHERE model_name IN ('model_v1', 'model_v2', 'model_v3')
     ORDER BY f1_score DESC
```

## Next Steps

- **[Making Predictions](making-predictions.md)** - Use your model for inference
- **[Model Training Guide](model-training.md)** - Improve your model based on evaluation
- **[Examples](../examples/logistic_regression_console/)** - See complete evaluation examples

## Related Documentation

- [Arc-Graph Specification](../concepts/arc-graph.md) - Understand model architecture
- [CLI Commands Reference](../api-reference/cli-commands.md) - All evaluation commands
