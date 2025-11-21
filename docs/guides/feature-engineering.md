# Feature Engineering Guide

This guide covers how to transform and prepare your data for machine learning using Arc. Feature engineering is the process of creating ML-ready features from raw data.

## Overview

Feature engineering in Arc involves:
- **Normalization/Scaling**: Scale numerical features to appropriate ranges
- **Encoding**: Convert categorical features to numerical representations
- **Feature Creation**: Derive new features from existing ones
- **Train/Val/Test Splits**: Split data for proper model evaluation
- **Missing Value Handling**: Deal with incomplete data

## Quick Start

The easiest way to engineer features is to ask Arc in natural language:

```
Process the users table:
- Normalize age and income
- One-hot encode the country column
- Create train/test split (80/20)
Save as processed_users
```

Arc will create an Arc-Pipeline specification and execute it automatically.

## Common Feature Engineering Tasks

### 1. Normalization and Scaling

#### Min-Max Scaling

Scale features to [0, 1] range:

```
Normalize the age and income columns in users table to 0-1 range
```

#### Standardization (Z-score)

Scale to mean=0, std=1:

```
Standardize the numeric features in the data table
```

#### When to Use Each

- **Min-Max**: When you need a fixed range (e.g., neural network inputs)
- **Standardization**: When features have different units or large outliers

### 2. Categorical Encoding

#### One-Hot Encoding

Convert categories to binary columns:

```
One-hot encode the country and category columns
```

Example output:
```
Original: country = "USA"
Encoded:  country_USA = 1, country_UK = 0, country_FR = 0
```

#### Label Encoding

Convert categories to integers:

```
Label encode the priority column (low=0, medium=1, high=2)
```

#### When to Use Each

- **One-Hot**: For unordered categories (country, color, category)
- **Label**: For ordered categories (priority, size) or tree-based models

### 3. Creating Train/Test Splits

Split your data for proper model evaluation:

```
Split the data table into train (80%) and test (20%) sets
```

Or with validation set:

```
Split data into train (70%), validation (15%), and test (15%)
```

Arc will create separate tables:
- `data_train`
- `data_val` (if requested)
- `data_test`

### 4. Handling Missing Values

#### Drop Rows with Missing Values

```
Remove rows with any missing values from the data table
```

#### Fill Missing Values

```
Fill missing values in the age column with the median
Fill missing values in the category column with 'unknown'
```

#### Imputation Strategies

- **Numerical columns**: mean, median, mode, or constant value
- **Categorical columns**: mode or constant value (e.g., 'unknown')

### 5. Feature Creation

Create derived features:

```
Create new features from the data table:
- age_group: categorize age into young/middle/senior
- is_premium: TRUE if purchase_amount > 100
- days_since_signup: difference between today and signup_date
```

### 6. Date/Time Features

Extract useful features from timestamps:

```
Extract date features from the timestamp column:
- year, month, day
- day_of_week
- hour
- is_weekend
```

## Using the /ml data Command

For reproducible feature engineering, use the `/ml data` command:

```
/ml data --name processed_features
         --instruction "Normalize numeric columns and one-hot encode categorical columns"
         --source-tables raw_data
```

This creates an **Arc-Pipeline** specification that:
- Documents your transformations
- Can be versioned in Git
- Can be reused with new data
- Ensures train/serve consistency

## Feature Engineering Patterns

### Pattern 1: Basic Preprocessing

```
/ml data --name processed_data
         --instruction "
         - Drop rows with missing values
         - Normalize age, income, and credit_score to 0-1
         - One-hot encode gender and state
         - Create 80/20 train/test split
         "
         --source-tables customers
```

### Pattern 2: Advanced Feature Creation

```
/ml data --name customer_features
         --instruction "
         - Calculate total_purchases per customer
         - Calculate days_since_last_purchase
         - Create customer_lifetime_value
         - Categorize customers into tiers (bronze/silver/gold)
         - Normalize all numeric features
         "
         --source-tables customers,purchases
```

### Pattern 3: Time Series Features

```
/ml data --name time_series_features
         --instruction "
         - Extract hour, day_of_week, month from timestamp
         - Create is_weekend flag
         - Calculate rolling_avg_7days for value column
         - Create lag features (lag_1, lag_7)
         "
         --source-tables time_series_data
```

### Pattern 4: Text Features

```
/ml data --name text_features
         --instruction "
         - Calculate text_length for description column
         - Count number of words
         - Create has_special_chars flag
         - Extract keywords (if applicable)
         "
         --source-tables products
```

## Checking Your Features

After feature engineering, always validate:

```sql
-- Check the shape
/sql SELECT COUNT(*), COUNT(DISTINCT user_id) FROM processed_data

-- Check for missing values
/sql SELECT
       SUM(CASE WHEN age IS NULL THEN 1 ELSE 0 END) as age_nulls,
       SUM(CASE WHEN income IS NULL THEN 1 ELSE 0 END) as income_nulls
     FROM processed_data

-- Check feature distributions
/sql SELECT
       MIN(age_normalized) as age_min,
       MAX(age_normalized) as age_max,
       AVG(age_normalized) as age_mean
     FROM processed_data

-- Check train/test split
/sql SELECT 'train' as split, COUNT(*) as count FROM processed_data_train
     UNION ALL
     SELECT 'test' as split, COUNT(*) as count FROM processed_data_test
```

## Best Practices

### 1. Understand Your Data First

Before feature engineering:
```sql
-- Check distributions
/sql SELECT
       MIN(age), MAX(age), AVG(age), STDDEV(age)
     FROM users

-- Check categories
/sql SELECT country, COUNT(*) as count
     FROM users
     GROUP BY country
     ORDER BY count DESC

-- Check missing values
/sql SELECT
       COUNT(*) as total,
       COUNT(*) - COUNT(age) as age_missing,
       COUNT(*) - COUNT(income) as income_missing
     FROM users
```

### 2. Keep Raw Data

Always preserve your raw data:
```sql
-- Raw data (never modified)
raw_users

-- Processed data (from feature engineering)
processed_users
```

### 3. Document Transformations

Use Arc-Pipeline specifications to document transformations. They serve as:
- Documentation of your preprocessing steps
- Reproducible workflows
- Version-controlled feature engineering

### 4. Consistent Train/Serve

Feature engineering must be identical for training and serving:
- **Arc-Pipeline ensures this** - same spec used for both
- Calculate statistics (mean, std) only on training data
- Apply same transformations to test/validation/production data

### 5. Handle Outliers

Decide how to handle outliers:
- **Cap/Floor**: Limit to min/max values
- **Remove**: Drop extreme values (carefully!)
- **Transform**: Use log or square root transformations
- **Robust Scaling**: Use median/IQR instead of mean/std

### 6. Feature Selection

Not all features improve models:
- Start with domain knowledge
- Remove highly correlated features
- Use feature importance from initial models
- Iterate based on model performance

## Advanced Topics

### Custom Transformations

For complex transformations, you can write SQL directly:

```sql
/sql CREATE TABLE custom_features AS
SELECT
    user_id,
    age,
    income,
    -- Custom feature: income per year of age
    income / NULLIF(age, 0) as income_per_year,
    -- Custom feature: age category
    CASE
        WHEN age < 25 THEN 'young'
        WHEN age < 50 THEN 'middle'
        ELSE 'senior'
    END as age_category,
    -- Custom feature: is high earner
    income > (SELECT AVG(income) FROM users) as is_high_earner
FROM users
```

### Handling Imbalanced Data

For classification with imbalanced classes:

```
Balance the training data using oversampling of the minority class
```

Or:
```
Create a balanced training set with equal samples from each class
```

### Feature Scaling for Neural Networks

Neural networks typically need features in [0, 1] or [-1, 1]:

```
Prepare data for neural network training:
- Min-max scale all numeric features to 0-1
- One-hot encode categorical features
- Ensure no missing values
```

### Feature Engineering for Different Model Types

Different models have different requirements:

- **Neural Networks**: Require scaled features, one-hot encoding
- **Tree-based (RandomForest, XGBoost)**: Handle raw features well, no scaling needed
- **Linear Models**: Benefit from scaling and regularization
- **SVMs**: Require scaled features

## Debugging Feature Engineering

If your feature engineering fails:

### Check for SQL Errors

```sql
-- Test your transformations step by step
/sql SELECT COUNT(*) FROM source_table  -- Does source exist?
/sql SELECT age, income FROM source_table LIMIT 5  -- Do columns exist?
/sql SELECT MIN(age), MAX(age) FROM source_table  -- Check ranges
```

### Check for Data Quality Issues

```sql
-- Negative values in log transform?
/sql SELECT COUNT(*) FROM table WHERE value <= 0

-- Division by zero?
/sql SELECT COUNT(*) FROM table WHERE denominator = 0

-- Invalid dates?
/sql SELECT COUNT(*) FROM table WHERE date IS NULL OR date < '1900-01-01'
```

### Review the Pipeline Specification

Arc creates an Arc-Pipeline YAML file. Review it to understand what transformations are being applied:

```
Show me the Arc-Pipeline specification for processed_data
```

## Examples

### Complete Example: Customer Churn Prediction

```
/ml data --name churn_features
         --instruction "
         Prepare customer data for churn prediction:
         1. From customers table, select: customer_id, age, account_balance, months_active
         2. From transactions table, calculate: total_transactions, avg_transaction_amount
         3. Join on customer_id
         4. Create features:
            - is_senior: age > 65
            - is_low_balance: account_balance < 1000
            - transaction_frequency: total_transactions / months_active
         5. Normalize age, account_balance, and transaction features
         6. One-hot encode customer_tier
         7. Create 70/15/15 train/val/test split
         "
         --source-tables customers,transactions
```

### Complete Example: Recommendation System

```
/ml data --name recommendation_features
         --instruction "
         Prepare data for movie recommendation:
         1. From users: select user_id, age, gender, occupation
         2. From ratings: calculate avg_rating, num_ratings per user
         3. Join datasets
         4. Normalize age and rating statistics
         5. One-hot encode gender and occupation
         6. Create train/test split (80/20)
         "
         --source-tables users,ratings
```

## Next Steps

- **[Model Training Guide](model-training.md)** - Train models with your engineered features
- **[Model Evaluation Guide](model-evaluation.md)** - Evaluate model performance
- **[Arc-Pipeline Specification](../concepts/arc-pipeline.md)** - Learn the Pipeline spec format

## Related Documentation

- [Arc Knowledge: ML Data Preparation](../../src/arc/resources/knowledge/ml_data_preparation.md) - Technical details
- [Data Loading Guide](data-loading.md) - Load data before feature engineering
