# Complete Tutorial: Diabetes Prediction

This comprehensive tutorial walks you through building a complete machine learning model to predict diabetes risk using the Pima Indians Diabetes dataset. You'll learn the entire Arc workflow from data loading to deployment.

**What You'll Learn**:
- Loading and exploring data
- Feature engineering best practices
- Training a binary classifier
- Evaluating model performance
- Making predictions on new data

**Time Required**: ~20 minutes

## Overview

We'll build a model that predicts whether a patient has diabetes based on health metrics:
- Number of pregnancies
- Glucose level
- Blood pressure
- Skin thickness
- Insulin level
- BMI (Body Mass Index)
- Diabetes pedigree function
- Age

## Prerequisites

- Arc installed ([Installation Guide](../getting-started/installation.md))
- API key configured ([Configuration Guide](../getting-started/configuration.md))

## Step 1: Start Arc and Load Data

Launch Arc:

```bash
arc chat
```

Load the Pima Indians Diabetes dataset:

```
Download the Pima Indians Diabetes dataset from UCI Machine Learning Repository and load it into a table called diabetes_raw
```

Arc will download and load the data automatically.

### Verify Data Loading

Check the data:

```sql
-- View table structure
/sql DESCRIBE diabetes_raw

-- Preview data
/sql SELECT * FROM diabetes_raw LIMIT 10

-- Check record count
/sql SELECT COUNT(*) as total_records FROM diabetes_raw

-- Check target distribution
/sql SELECT
       outcome,
       COUNT(*) as count,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as percentage
     FROM diabetes_raw
     GROUP BY outcome
```

Expected output:
- 768 total records
- 8 features + 1 target (outcome)
- ~65% negative (no diabetes), ~35% positive (has diabetes)

## Step 2: Explore the Data

Understanding your data is crucial for building good models.

### Check for Missing Values

```sql
/sql SELECT
       COUNT(*) - COUNT(pregnancies) as pregnancies_missing,
       COUNT(*) - COUNT(glucose) as glucose_missing,
       COUNT(*) - COUNT(blood_pressure) as blood_pressure_missing,
       COUNT(*) - COUNT(bmi) as bmi_missing
     FROM diabetes_raw
```

### Analyze Feature Distributions

```sql
-- Summary statistics
/sql SELECT
       'glucose' as feature,
       MIN(glucose) as min_val,
       MAX(glucose) as max_val,
       AVG(glucose) as mean_val,
       STDDEV(glucose) as std_val
     FROM diabetes_raw
UNION ALL
SELECT
       'bmi',
       MIN(bmi), MAX(bmi), AVG(bmi), STDDEV(bmi)
     FROM diabetes_raw
UNION ALL
SELECT
       'age',
       MIN(age), MAX(age), AVG(age), STDDEV(age)
     FROM diabetes_raw
```

### Identify Data Quality Issues

```sql
-- Check for zeros that might be missing values
/sql SELECT
       SUM(CASE WHEN glucose = 0 THEN 1 ELSE 0 END) as glucose_zeros,
       SUM(CASE WHEN blood_pressure = 0 THEN 1 ELSE 0 END) as bp_zeros,
       SUM(CASE WHEN bmi = 0 THEN 1 ELSE 0 END) as bmi_zeros
     FROM diabetes_raw
```

**Note**: Some zeros might represent actual missing values (e.g., BMI or glucose of 0 is impossible).

## Step 3: Feature Engineering

Now let's prepare the data for machine learning.

```
/ml data --name diabetes_processed
         --instruction "
         Prepare the diabetes data for binary classification:
         1. Remove rows where glucose, blood_pressure, or bmi = 0 (likely missing values)
         2. Normalize all numeric features to 0-1 range
         3. Create a binary feature: is_high_risk (age > 50 or bmi > 35)
         4. Create train/test split (80/20)
         "
         --source-tables diabetes_raw
```

Arc will:
1. Create an Arc-Pipeline specification
2. Execute the transformations
3. Create `diabetes_processed_train` and `diabetes_processed_test` tables

### Verify Feature Engineering

```sql
-- Check train/test split
/sql SELECT
       (SELECT COUNT(*) FROM diabetes_processed_train) as train_count,
       (SELECT COUNT(*) FROM diabetes_processed_test) as test_count

-- Check feature ranges (should be 0-1 after normalization)
/sql SELECT
       MIN(glucose_normalized) as glucose_min,
       MAX(glucose_normalized) as glucose_max,
       MIN(bmi_normalized) as bmi_min,
       MAX(bmi_normalized) as bmi_max
     FROM diabetes_processed_train

-- Check new feature
/sql SELECT
       is_high_risk,
       COUNT(*) as count
     FROM diabetes_processed_train
     GROUP BY is_high_risk
```

## Step 4: Train the Model

Now let's train a model:

```
/ml model --name diabetes_predictor
          --instruction "
          Build a binary classifier for diabetes prediction with:
          - 3 hidden layers: 64, 32, 16 neurons
          - Dropout: 0.2 after each hidden layer
          - Learning rate: 0.001
          - Train for 50 epochs
          - Use Adam optimizer
          - Track accuracy and AUC metrics
          "
          --data-table diabetes_processed_train
```

Arc will:
1. Generate an Arc-Graph specification
2. Show you the spec for review
3. Build the PyTorch model
4. Train for 50 epochs
5. Launch TensorBoard
6. Save the trained model

### Monitor Training

**In Console**: Watch epoch-by-epoch progress
```
Epoch 1/50:
  Train Loss: 0.6234  Train Acc: 0.6543
  Val Loss:   0.5987  Val Acc:   0.6721
...
```

**In TensorBoard**: Open `http://localhost:6006`
- View loss curves (train vs validation)
- Monitor accuracy over time
- Check for overfitting (train/val divergence)

### Understanding the Arc-Graph

The generated Arc-Graph looks like:

```yaml
inputs:
  patient_features:
    dtype: float32
    shape: [null, 9]  # 8 original + 1 derived feature

graph:
  - name: hidden1
    type: torch.nn.Linear
    params:
      in_features: 9
      out_features: 64
    inputs:
      input: patient_features

  - name: relu1
    type: torch.nn.ReLU
    inputs:
      input: hidden1.output

  - name: dropout1
    type: torch.nn.Dropout
    params:
      p: 0.2
    inputs:
      input: relu1.output

  # ... similar for hidden2 and hidden3 ...

  - name: output
    type: torch.nn.Linear
    params:
      in_features: 16
      out_features: 1
    inputs:
      input: dropout3.output

  - name: sigmoid
    type: torch.nn.Sigmoid
    inputs:
      input: output.output

outputs:
  prediction: sigmoid.output

trainer:
  optimizer:
    type: torch.optim.Adam
    params:
      lr: 0.001
  loss: torch.nn.BCELoss
  epochs: 50
  batch_size: 32
```

This specification is:
- Human-readable
- Portable (runs anywhere PyTorch runs)
- Version-controllable (track in Git)
- Reproducible (same spec = same results)

## Step 5: Evaluate the Model

Evaluate on the test set:

```
/ml evaluate --model-id <model_id>
             --data-table diabetes_processed_test
             --metrics "accuracy,precision,recall,f1,auc"
```

Find your model ID with:
```
/ml jobs list
```

### View Evaluation Results

```sql
-- Get evaluation metrics
/sql SELECT * FROM evaluations
     WHERE model_id = '<your_model_id>'
     ORDER BY created_at DESC LIMIT 1

-- Expected results (approximate):
-- accuracy: ~0.75-0.80
-- precision: ~0.70-0.75
-- recall: ~0.65-0.75
-- f1: ~0.70-0.75
-- auc: ~0.80-0.85
```

### Analyze Results

```sql
-- View confusion matrix data
/sql SELECT
       SUM(CASE WHEN actual = 0 AND predicted = 0 THEN 1 ELSE 0 END) as true_negatives,
       SUM(CASE WHEN actual = 0 AND predicted = 1 THEN 1 ELSE 0 END) as false_positives,
       SUM(CASE WHEN actual = 1 AND predicted = 0 THEN 1 ELSE 0 END) as false_negatives,
       SUM(CASE WHEN actual = 1 AND predicted = 1 THEN 1 ELSE 0 END) as true_positives
     FROM predictions
```

### Interpretation

**Good Results**:
- AUC > 0.80: Model has good discriminative power
- Balanced precision/recall: Not biased toward one class
- F1 > 0.70: Good overall performance

**Areas for Improvement**:
- If precision low: Too many false positives (healthy patients flagged as diabetic)
- If recall low: Too many false negatives (diabetic patients missed)
- If both low: Need better features or more data

## Step 6: Make Predictions

Use the model to predict on new patients:

```sql
-- Create new patient data (example)
CREATE TABLE new_patients AS
SELECT
    1 as patient_id,
    2 as pregnancies,
    120 as glucose,
    70 as blood_pressure,
    30 as skin_thickness,
    100 as insulin,
    28.5 as bmi,
    0.5 as diabetes_pedigree,
    35 as age
UNION ALL
SELECT 2, 6, 150, 85, 35, 180, 35.2, 0.8, 55;

-- Preprocess new data (same as training)
CREATE TABLE new_patients_processed AS
SELECT
    patient_id,
    -- Apply same normalization as training
    (glucose - 0) / (200 - 0) as glucose_normalized,
    (blood_pressure - 0) / (122 - 0) as blood_pressure_normalized,
    (bmi - 0) / (67.1 - 0) as bmi_normalized,
    (age - 21) / (81 - 21) as age_normalized,
    -- Apply same feature engineering
    CASE WHEN age > 50 OR bmi > 35 THEN 1 ELSE 0 END as is_high_risk
    -- ... normalize all other features ...
FROM new_patients;
```

Make predictions:

```
/ml predict --model diabetes_predictor
            --data new_patients_processed
            --output patient_risk_scores
```

View predictions:

```sql
/sql SELECT
       n.patient_id,
       n.age,
       n.bmi,
       n.glucose,
       p.prediction as diabetes_probability,
       CASE
           WHEN p.prediction >= 0.7 THEN 'High Risk'
           WHEN p.prediction >= 0.4 THEN 'Medium Risk'
           ELSE 'Low Risk'
       END as risk_category
     FROM new_patients n
     JOIN patient_risk_scores p ON n.patient_id = p.patient_id
     ORDER BY p.prediction DESC
```

## Step 7: Model Improvement (Optional)

If you want to improve the model:

### Try Different Architectures

```
/ml model --name diabetes_predictor_v2
          --instruction "Build a Deep & Cross Network to capture feature interactions"
          --data-table diabetes_processed_train
```

### Add More Features

```
/ml data --name diabetes_processed_v2
         --instruction "
         Create additional features:
         - bmi_age_interaction: bmi * age
         - glucose_category: low/normal/high based on glucose levels
         - compound_risk_score: weighted combination of risk factors
         "
         --source-tables diabetes_raw
```

### Tune Hyperparameters

Try different learning rates, layer sizes, dropout rates:

```
/ml model --name diabetes_predictor_tuned
          --instruction "
          Try different configurations:
          - Learning rates: 0.0001, 0.001, 0.01
          - Layer sizes: [128, 64, 32] vs [64, 32, 16]
          - Dropout: 0.1, 0.2, 0.3
          Find the best combination
          "
          --data-table diabetes_processed_train
```

## Complete Workflow Summary

```bash
# 1. Load data
"Download diabetes dataset"

# 2. Explore
/sql SELECT * FROM diabetes_raw LIMIT 10
/sql SELECT outcome, COUNT(*) FROM diabetes_raw GROUP BY outcome

# 3. Feature engineering
/ml data --name diabetes_processed
         --instruction "Clean, normalize, and split data"
         --source-tables diabetes_raw

# 4. Train model
/ml model --name diabetes_predictor
          --instruction "Build binary classifier"
          --data-table diabetes_processed_train

# 5. Evaluate
/ml evaluate --model-id <id>
             --data-table diabetes_processed_test

# 6. Predict
/ml predict --model diabetes_predictor
            --data new_patients_processed
```

## Key Takeaways

1. **Data Quality Matters**: We removed zeros that represented missing values
2. **Feature Engineering is Critical**: Normalization and derived features improved performance
3. **Monitor Training**: Use TensorBoard to catch overfitting early
4. **Evaluate Thoroughly**: Look beyond accuracy - check precision, recall, F1, AUC
5. **Iterate**: Try different architectures and features to improve

## Next Steps

- **[Feature Engineering Guide](../guides/feature-engineering.md)** - Learn advanced techniques
- **[Model Training Guide](../guides/model-training.md)** - Deep dive into training
- **[Custom Architecture Example](custom-architecture.md)** - Build advanced models

## Code Repository

The complete code for this tutorial is available in the `examples/diabetes_prediction/` directory.

## Need Help?

- Check the [FAQ](../faq.md)
- Ask in [GitHub Discussions](https://github.com/non-linear-ai/arc/discussions)
- Open an [issue](https://github.com/non-linear-ai/arc/issues)
