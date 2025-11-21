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

**Critical**: Apply the exact same preprocessing as training.

Arc handles this automatically if you:
1. Use the same Arc-Pipeline for training and prediction data
2. Or include preprocessing in the Arc-Graph

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
