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

### Batch Prediction

Predict on large datasets:

```
/ml predict --model price_predictor
            --data all_houses
            --batch-size 1000
```

## Prediction Workflow

### Step 1: Prepare Input Data

Your input data must:
- **Have the same features** as training data
- **Be in the same format** (same column names, types)
- **Be preprocessed identically** (same normalization, encoding)

Example:
```sql
-- Check training data format
/sql DESCRIBE training_data

-- Ensure new data matches
/sql DESCRIBE new_data
```

### Step 2: Run Prediction

```
/ml predict --model my_model --data new_data
```

Arc will:
1. Load the trained model
2. Load the Arc-Graph specification
3. Apply preprocessing (if specified)
4. Run inference
5. Save predictions to a table

### Step 3: View Predictions

```sql
-- View all predictions
/sql SELECT * FROM my_model_predictions LIMIT 10

-- Join with original data
/sql SELECT
       n.*,
       p.prediction,
       p.confidence
     FROM new_data n
     JOIN my_model_predictions p
       ON n.id = p.id

-- Get high-confidence predictions
/sql SELECT * FROM my_model_predictions
     WHERE confidence > 0.9
     ORDER BY confidence DESC
```

## Understanding Prediction Outputs

### Classification Models

For binary classification:
```sql
id | prediction | probability | predicted_class
---|------------|-------------|----------------
1  | 0.8234     | 0.8234      | 1
2  | 0.2145     | 0.2145      | 0
3  | 0.9567     | 0.9567      | 1
```

- **prediction**: Raw model output (probability)
- **probability**: Same as prediction for binary classification
- **predicted_class**: 0 or 1 based on threshold (default 0.5)

For multi-class classification:
```sql
id | predicted_class | class_0_prob | class_1_prob | class_2_prob
---|-----------------|--------------|--------------|-------------
1  | 2               | 0.1          | 0.2          | 0.7
2  | 0               | 0.8          | 0.1          | 0.1
```

- **predicted_class**: The class with highest probability
- **class_X_prob**: Probability for each class

### Regression Models

```sql
id | prediction | confidence_interval_low | confidence_interval_high
---|------------|-------------------------|-------------------------
1  | 245000     | 230000                  | 260000
2  | 189000     | 175000                  | 203000
```

- **prediction**: Predicted value
- **confidence_interval**: Range of likely values (if computed)

## Prediction Patterns

### Pattern 1: Batch Scoring

Score all records in a table:

```
/ml predict --model churn_predictor
            --data all_customers
            --output churn_scores
```

Then use the scores:
```sql
-- Find high-risk customers
/sql SELECT customer_id, churn_probability
     FROM churn_scores
     WHERE churn_probability > 0.7
     ORDER BY churn_probability DESC

-- Send to CRM or campaign tool
/sql COPY (
       SELECT customer_id, email, churn_probability
       FROM churn_scores c
       JOIN customers cu ON c.customer_id = cu.id
       WHERE churn_probability > 0.7
     ) TO 'high_risk_customers.csv' (HEADER, DELIMITER ',')
```

### Pattern 2: Real-Time Scoring

For single-record predictions (API/application use):

```sql
-- Create a view with preprocessing
CREATE VIEW customer_features AS
SELECT
    id,
    age,
    income,
    -- Apply same preprocessing as training
    (age - 18.0) / (85.0 - 18.0) as age_normalized,
    (income - 0.0) / (200000.0 - 0.0) as income_normalized
FROM customers;

-- Make prediction on specific customer
/ml predict --model churn_predictor
            --data customer_features
            --filter "id = 12345"
```

### Pattern 3: Scheduled Predictions

Run predictions regularly:

```bash
# In a cron job or scheduled task
arc chat --non-interactive <<EOF
/ml predict --model daily_forecast --data latest_data
/sql COPY (SELECT * FROM daily_forecast_predictions)
     TO 'forecasts_$(date +%Y%m%d).csv'
exit
EOF
```

### Pattern 4: What-If Analysis

Test different scenarios:

```sql
-- Create scenario data
CREATE TABLE scenarios AS
SELECT
    1 as scenario_id,
    'base' as scenario_name,
    current_price as price,
    current_marketing_spend as marketing_spend
FROM products
UNION ALL
SELECT
    2, 'price_increase', current_price * 1.1, current_marketing_spend
FROM products
UNION ALL
SELECT
    3, 'marketing_boost', current_price, current_marketing_spend * 1.5
FROM products;

-- Predict for each scenario
/ml predict --model demand_predictor --data scenarios
```

## Feature Preprocessing for Predictions

### Same Preprocessing as Training

**Critical**: Apply the exact same preprocessing as training.

Arc handles this automatically if you:
1. Use the same Arc-Pipeline for training and prediction data
2. Or include preprocessing in the Arc-Graph

### Manual Preprocessing

If preprocessing manually, ensure consistency:

```sql
-- Training preprocessing (save these values!)
CREATE TABLE feature_stats AS
SELECT
    AVG(age) as age_mean,
    STDDEV(age) as age_std,
    AVG(income) as income_mean,
    STDDEV(income) as income_std
FROM training_data;

-- Apply same preprocessing to new data
CREATE TABLE new_data_processed AS
SELECT
    id,
    (age - (SELECT age_mean FROM feature_stats)) /
    (SELECT age_std FROM feature_stats) as age_normalized,
    (income - (SELECT income_mean FROM feature_stats)) /
    (SELECT income_std FROM feature_stats) as income_normalized
FROM new_data;

-- Now predict
/ml predict --model my_model --data new_data_processed
```

## Prediction Performance

### Optimizing Batch Predictions

For large datasets:

```
/ml predict --model my_model
            --data large_dataset
            --batch-size 5000
            --parallel
```

Options:
- `--batch-size`: Number of records per batch (larger = faster but more memory)
- `--parallel`: Use multiple workers (if available)

### Monitoring Prediction Time

```sql
-- Track prediction timing
/sql SELECT
       model_name,
       COUNT(*) as num_predictions,
       SUM(prediction_time_ms) as total_time_ms,
       AVG(prediction_time_ms) as avg_time_ms
     FROM prediction_logs
     GROUP BY model_name
```

## Handling Special Cases

### Missing Features in New Data

If new data is missing features used during training:

```sql
-- Fill missing values with training defaults
CREATE TABLE new_data_filled AS
SELECT
    id,
    COALESCE(age, (SELECT AVG(age) FROM training_data)) as age,
    COALESCE(income, (SELECT AVG(income) FROM training_data)) as income,
    COALESCE(country, 'Unknown') as country
FROM new_data;
```

### New Categorical Values

If new data has categories not seen in training:

```sql
-- Map unknown categories to a default
CREATE TABLE new_data_mapped AS
SELECT
    id,
    age,
    income,
    CASE
        WHEN country IN ('USA', 'UK', 'Canada')  -- Known values
        THEN country
        ELSE 'Other'  -- Unknown → default
    END as country
FROM new_data;
```

### Out-of-Range Values

If new data has extreme values:

```sql
-- Cap values to training range
CREATE TABLE new_data_capped AS
SELECT
    id,
    CASE
        WHEN age < 18 THEN 18
        WHEN age > 100 THEN 100
        ELSE age
    END as age,
    CASE
        WHEN income < 0 THEN 0
        WHEN income > 1000000 THEN 1000000
        ELSE income
    END as income
FROM new_data;
```

## Prediction Confidence

### Using Prediction Probabilities

For classification, examine prediction confidence:

```sql
-- High confidence predictions (reliable)
/sql SELECT * FROM predictions
     WHERE probability > 0.9 OR probability < 0.1

-- Low confidence predictions (uncertain)
/sql SELECT * FROM predictions
     WHERE probability BETWEEN 0.4 AND 0.6

-- Count by confidence level
/sql SELECT
       CASE
           WHEN probability >= 0.8 THEN 'High confidence'
           WHEN probability >= 0.6 THEN 'Medium confidence'
           ELSE 'Low confidence'
       END as confidence_level,
       COUNT(*) as count
     FROM predictions
     GROUP BY confidence_level
```

### When to Trust Predictions

High confidence predictions when:
- Probability far from 0.5 (classification)
- New data similar to training data
- Model has high accuracy on validation set

Low confidence predictions when:
- Probability near 0.5 (classification)
- New data very different from training
- New data has missing/unusual values

## Exporting Predictions

### To CSV

```sql
/sql COPY (
       SELECT * FROM predictions
     ) TO 'predictions.csv' (HEADER, DELIMITER ',')
```

### To Database

```sql
-- Export to PostgreSQL (example)
/sql COPY predictions TO 'postgres://user:pass@host/db' (SCHEMA 'public', TABLE 'predictions')
```

### To Application

```sql
-- Generate JSON for API
/sql COPY (
       SELECT json_group_array(
           json_object(
               'id', id,
               'prediction', prediction,
               'probability', probability
           )
       )
       FROM predictions
     ) TO 'predictions.json'
```

## Troubleshooting

### Predictions Don't Match Training Accuracy

**Problem**: Model performs well on test set but poorly on new data

**Causes**:
1. **Data drift**: New data different from training data
2. **Preprocessing mismatch**: Different preprocessing for new data
3. **Feature leakage**: Training had features not available in production

**Solutions**:
1. Check new data distribution vs training data
2. Verify preprocessing is identical
3. Retrain model with recent data
4. Monitor prediction distributions

### Model File Not Found

**Problem**: Arc can't find the trained model

**Solution**:
```
/ml jobs list
```
Use the exact model name or ID from the list.

### Feature Mismatch Error

**Problem**: "Feature X not found" or "Expected Y features, got Z"

**Solution**:
Ensure new data has all required features:
```sql
-- Check training features
/sql DESCRIBE training_data

-- Check new data features
/sql DESCRIBE new_data

-- Add missing features with defaults if needed
```

### Slow Predictions

**Problem**: Predictions take too long

**Solutions**:
1. Increase batch size: `--batch-size 10000`
2. Use parallel processing: `--parallel`
3. Simplify model (if possible)
4. Use GPU acceleration (if available)

## Examples

### Example 1: Customer Churn Prediction

```
# Predict churn for all active customers
/ml predict --model churn_model
            --data active_customers
            --output churn_risk

# Find high-risk customers
/sql SELECT c.customer_id, c.name, c.email, p.churn_probability
     FROM churn_risk p
     JOIN customers c ON p.customer_id = c.customer_id
     WHERE p.churn_probability > 0.75
     ORDER BY p.churn_probability DESC

# Export for intervention campaign
/sql COPY (...) TO 'high_churn_risk.csv'
```

### Example 2: House Price Prediction

```
# Predict prices for new listings
/ml predict --model price_model
            --data new_listings
            --output price_estimates

# Compare to asking price
/sql SELECT
       listing_id,
       asking_price,
       predicted_price,
       asking_price - predicted_price as price_difference,
       CASE
           WHEN asking_price > predicted_price * 1.1 THEN 'Overpriced'
           WHEN asking_price < predicted_price * 0.9 THEN 'Underpriced'
           ELSE 'Fair'
       END as assessment
     FROM new_listings n
     JOIN price_estimates p ON n.listing_id = p.listing_id
```

### Example 3: Fraud Detection

```
# Score all transactions in real-time
/ml predict --model fraud_detector
            --data today_transactions
            --output fraud_scores

# Flag suspicious transactions
/sql SELECT t.transaction_id, t.amount, t.merchant, f.fraud_probability
     FROM fraud_scores f
     JOIN transactions t ON f.transaction_id = t.transaction_id
     WHERE f.fraud_probability > 0.8

# Automatic blocking for high-risk
/sql UPDATE transactions
     SET status = 'blocked_for_review'
     WHERE transaction_id IN (
         SELECT transaction_id FROM fraud_scores
         WHERE fraud_probability > 0.95
     )
```

## Next Steps

- **[Model Evaluation Guide](model-evaluation.md)** - Evaluate prediction quality
- **[Model Training Guide](model-training.md)** - Retrain with new data
- **[API Reference](../api-reference/cli-commands.md)** - All prediction commands

## Related Documentation

- [Arc-Graph Specification](../concepts/arc-graph.md) - Understand model architecture
- [Data Loading Guide](data-loading.md) - Load new data for prediction
