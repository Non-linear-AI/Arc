# Making Predictions Guide

This guide covers how to use your trained Arc models to make predictions on new data.

## Overview

Once you've trained and evaluated a model, you can use it for inference (making predictions). Arc makes this simple:
1. **Load new data** - Get data you want predictions for
2. **Run prediction** - Apply your trained model
3. **View results** - Access predictions via SQL

## Quick Start

Make predictions on new data:

```
Use the diabetes_predictor model to make predictions on new_patients table
```

Or with the `/ml predict` command:

```
/ml predict --model diabetes_predictor
            --data new_patients
            --output predictions
```

## Using the /ml predict Command

### Basic Prediction

```
/ml predict --model <model_name>
            --data <input_table>
```

Predictions are saved to `<model_name>_predictions` table by default.

### Custom Output Table

```
/ml predict --model diabetes_predictor
            --data new_patients
            --output patient_risk_scores
```

## Prediction Workflow

**Prepare Input Data**: Ensure new data has the same features as training data with identical preprocessing.

**Run Prediction**: `/ml predict --model my_model --data new_data`

**View Predictions**: `/sql SELECT * FROM my_model_predictions LIMIT 10`

## Understanding Prediction Outputs

**Classification**: Returns predictions with probabilities and predicted classes.

**Regression**: Returns predicted values.

## Example: Batch Scoring

```
/ml predict --model churn_predictor
            --data all_customers
            --output churn_scores
```

Query results with `/sql SELECT * FROM churn_scores WHERE churn_probability > 0.7`

## Next Steps

- **[Model Evaluation Guide](model-evaluation.md)** - Evaluate prediction quality
- **[Model Training Guide](model-training.md)** - Retrain with new data
- **[API Reference](../api-reference/cli-commands.md)** - All prediction commands

## Related Documentation

- [Arc-Graph Specification](../concepts/arc-graph.md) - Understand model architecture
- [Data Loading Guide](data-loading.md) - Load new data for prediction
