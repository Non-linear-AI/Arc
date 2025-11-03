# ML Data Preparation Guide

This guide covers best practices for preparing training datasets for machine learning models. Use this guidance when your task involves creating a training dataset for ML model training.

## 1. Output Schema Requirements

Your pipeline creates a training dataset for downstream machine learning model training. The final output table MUST meet these requirements:

### Required Columns

The final output table must contain:

- **Feature columns**: All features must be numeric (DECIMAL, FLOAT, INTEGER). Column names should be lowercase snake_case (e.g., `age`, `glucose_level`, `bmi`).

- **Target column**: Named according to the task (e.g., `outcome`, `target`, `label`). Type depends on problem:
  - Binary/multiclass classification: INTEGER (0, 1, 2, ...)
  - Regression: DECIMAL or FLOAT

- **Split column**: Must be named `split` (lowercase) with TEXT values:
  - `'training'` for training samples
  - `'validation'` for validation samples

## 2. Train/Validation Splits

### Stratified Splitting

When creating train/validation splits, use **stratified splitting** to maintain class distribution:

```sql
-- Stratified 80/20 split maintaining class balance
CASE
  WHEN MOD(ROW_NUMBER() OVER (PARTITION BY target ORDER BY id), 10) < 8
    THEN 'training'
  ELSE 'validation'
END as split
```

**Key points**:
- `PARTITION BY target` ensures each class is split proportionally
- `ORDER BY id` (or another column) ensures reproducibility
- Use MOD with denominator 10 for easy percentage control (8/10 = 80%)

### Data Leakage Prevention

**CRITICAL**: Fit preprocessing parameters (mean, stddev, min, max) on the training set ONLY, then apply to both train and validation.

**Pattern**:
1. Create train/val split FIRST
2. Calculate scaling parameters FROM training samples only (WHERE split = 'training')
3. Join those parameters back to apply transformations to ALL samples

**Example** (StandardScaler for feature `age`):
```sql
WITH split_data AS (
  -- Step 1: Create split first
  SELECT *,
    CASE WHEN MOD(ROW_NUMBER() OVER (PARTITION BY target ORDER BY id), 10) < 8
      THEN 'training' ELSE 'validation' END as split
  FROM "source_table"
),
train_stats AS (
  -- Step 2: Calculate mean/stddev from training set only
  SELECT
    AVG(age) as age_mean,
    STDDEV(age) as age_stddev
  FROM split_data
  WHERE split = 'training'
)
SELECT
  -- Step 3: Apply to both train and validation
  (age - train_stats.age_mean) / NULLIF(train_stats.age_stddev, 0) as age_scaled,
  target,
  split
FROM split_data
CROSS JOIN train_stats
```

## 3. Feature Engineering Patterns

### StandardScaler

Normalize features to zero mean and unit variance:

```sql
(value - mean) / stddev
```

Fit mean/stddev on training set only, then apply to all samples (see data leakage prevention above).

### MinMaxScaler

Scale features to [0, 1] range:

```sql
(value - min) / (max - min)
```

Fit min/max on training set only.

### One-hot Encoding

Use CASE statements to create binary indicator columns:

```sql
CASE WHEN category = 'A' THEN 1 ELSE 0 END as category_a,
CASE WHEN category = 'B' THEN 1 ELSE 0 END as category_b
```

### Missing Value Imputation

Use COALESCE with training-set statistics:

```sql
COALESCE(feature_value, training_median) as feature_imputed
```

## 4. Complete Example: ML Training Dataset Pipeline

This example demonstrates a complete ML data preparation pipeline for binary classification (diabetes prediction):

```yaml
steps:
  - name: drop_old_training_data
    type: execute
    depends_on: []
    sql: DROP TABLE IF EXISTS "diabetes_training_data"

  - name: drop_old_split_data
    type: execute
    depends_on: []
    sql: DROP VIEW IF EXISTS "diabetes_split_data"

  # Step 1: Create stratified train/val split (80/20)
  - name: diabetes_split_data
    type: view
    depends_on: [drop_old_split_data, diabetes]
    sql: |
      SELECT
        *,
        CASE
          WHEN MOD(ROW_NUMBER() OVER (PARTITION BY Outcome ORDER BY id), 10) < 8
            THEN 'training'
          ELSE 'validation'
        END as split
      FROM "diabetes"
      WHERE Pregnancies IS NOT NULL
        AND Glucose IS NOT NULL
        AND BloodPressure IS NOT NULL
        AND Outcome IS NOT NULL

  # Step 2: Calculate scaling parameters from training set ONLY
  - name: drop_old_train_stats
    type: execute
    depends_on: []
    sql: DROP VIEW IF EXISTS "train_stats"

  - name: train_stats
    type: view
    depends_on: [drop_old_train_stats, diabetes_split_data]
    sql: |
      SELECT
        AVG(Pregnancies) as pregnancies_mean,
        STDDEV(Pregnancies) as pregnancies_stddev,
        AVG(Glucose) as glucose_mean,
        STDDEV(Glucose) as glucose_stddev,
        AVG(BloodPressure) as bp_mean,
        STDDEV(BloodPressure) as bp_stddev,
        AVG(SkinThickness) as skin_mean,
        STDDEV(SkinThickness) as skin_stddev,
        AVG(Insulin) as insulin_mean,
        STDDEV(Insulin) as insulin_stddev,
        AVG(BMI) as bmi_mean,
        STDDEV(BMI) as bmi_stddev,
        AVG(DiabetesPedigreeFunction) as dpf_mean,
        STDDEV(DiabetesPedigreeFunction) as dpf_stddev,
        AVG(Age) as age_mean,
        STDDEV(Age) as age_stddev
      FROM "diabetes_split_data"
      WHERE split = 'training'

  # Step 3: Apply StandardScaler to ALL samples using training statistics
  - name: diabetes_training_data
    type: table
    depends_on: [drop_old_training_data, diabetes_split_data, train_stats]
    sql: |
      SELECT
        -- Scaled features (StandardScaler: (x - mean) / stddev)
        (Pregnancies - train_stats.pregnancies_mean) /
          NULLIF(train_stats.pregnancies_stddev, 0) as pregnancies,
        (Glucose - train_stats.glucose_mean) /
          NULLIF(train_stats.glucose_stddev, 0) as glucose,
        (BloodPressure - train_stats.bp_mean) /
          NULLIF(train_stats.bp_stddev, 0) as blood_pressure,
        (SkinThickness - train_stats.skin_mean) /
          NULLIF(train_stats.skin_stddev, 0) as skin_thickness,
        (Insulin - train_stats.insulin_mean) /
          NULLIF(train_stats.insulin_stddev, 0) as insulin,
        (BMI - train_stats.bmi_mean) /
          NULLIF(train_stats.bmi_stddev, 0) as bmi,
        (DiabetesPedigreeFunction - train_stats.dpf_mean) /
          NULLIF(train_stats.dpf_stddev, 0) as diabetes_pedigree,
        (Age - train_stats.age_mean) /
          NULLIF(train_stats.age_stddev, 0) as age,
        -- Target column (INTEGER for binary classification)
        Outcome as outcome,
        -- Split column (TEXT: 'training' or 'validation')
        diabetes_split_data.split as split
      FROM "diabetes_split_data"
      CROSS JOIN "train_stats"

outputs: [diabetes_training_data]
```

**Key features demonstrated**:
- Stratified 80/20 split maintaining Outcome class balance (PARTITION BY Outcome)
- Data leakage prevention: statistics calculated from training set only, then applied to both train and validation
- StandardScaler implementation: `(value - mean) / stddev` with NULLIF for division-by-zero protection
- Proper output schema: numeric features + `outcome` target + `split` column
- Idempotency: DROP statements for all created objects

## Summary

This guide focuses on ML-specific data preparation tasks:

1. **Output Schema Requirements**: Training datasets must have numeric features, a target column, and a `split` column
2. **Train/Validation Splits**: Use stratified splitting to maintain class distribution
3. **Data Leakage Prevention**: Fit preprocessing parameters on training set only, then apply to both sets
4. **Feature Engineering**: StandardScaler, MinMaxScaler, one-hot encoding, and imputation patterns

For **loading external data files** (CSV, Parquet, JSON) into the database, see the separate [Data Loading](data_loading.md) guide.
