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

Min-Max scaling (0-1 range):

```
Normalize the age and income columns in users table to 0-1 range
```

Standardization (mean=0, std=1):

```
Standardize the numeric features in the data table
```

### 2. Categorical Encoding

One-hot encoding (categories to binary columns):

```
One-hot encode the country and category columns
```

Label encoding (categories to integers):

```
Label encode the priority column (low=0, medium=1, high=2)
```

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

Drop rows or fill with appropriate values:

```
Remove rows with any missing values from the data table
Fill missing values in the age column with the median
Fill missing values in the category column with 'unknown'
```

### 5. Feature Creation

Create derived features and extract date/time information:

```
Create new features from the data table:
- age_group: categorize age into young/middle/senior
- is_premium: TRUE if purchase_amount > 100
- year, month, day, day_of_week, hour, is_weekend from timestamp
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

## Example: Basic Preprocessing

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

## Next Steps

- **[Model Training Guide](model-training.md)** - Train models with your engineered features
- **[Model Evaluation Guide](model-evaluation.md)** - Evaluate model performance
- **[Arc-Pipeline Specification](../concepts/arc-pipeline.md)** - Learn the Pipeline spec format

## Related Documentation

- [Arc Knowledge: ML Data Preparation](../../src/arc/resources/knowledge/ml_data_preparation.md) - Technical details
- [Data Loading Guide](data-loading.md) - Load data before feature engineering
