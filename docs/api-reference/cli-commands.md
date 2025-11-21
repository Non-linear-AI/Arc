# CLI Commands Reference

Complete reference for all Arc commands and options.

## Overview

Arc provides an interactive CLI with several command prefixes:
- `/ml` - Machine learning operations
- `/sql` - SQL queries
- `/config` - Configuration management
- `/report` - Generate reports
- `/help` - Get help
- `/clear` - Clear conversation context

## Starting Arc

```bash
# Interactive mode (recommended)
arc chat

# Non-interactive mode (for scripting)
arc chat --non-interactive

# Show version
arc --version

# Show help
arc --help
```

## Machine Learning Commands (/ml)

### /ml plan

Generate an ML workflow plan.

**Syntax**:
```
/ml plan --name <plan_name>
         --instruction "<description>"
         --source-tables <tables>
```

**Options**:
- `--name` (required): Name for the ML plan
- `--instruction` (required): Description of what you want to build
- `--source-tables` (required): Comma-separated list of source tables

**Example**:
```
/ml plan --name customer_churn
         --instruction "Predict customer churn using demographics and behavior"
         --source-tables customers,transactions,support_tickets
```

### /ml revise-plan

Revise an existing ML plan.

**Syntax**:
```
/ml revise-plan --instruction "<changes>"
```

**Options**:
- `--instruction` (required): What changes to make

**Example**:
```
/ml revise-plan --instruction "Add feature: days_since_last_purchase"
```

**Note**: Operates on the most recent plan in the conversation.

### /ml data

Create a data processing pipeline.

**Syntax**:
```
/ml data --name <processor_name>
         --instruction "<description>"
         --source-tables <tables>
```

**Options**:
- `--name` (required): Name for the data processor
- `--instruction` (required): Data processing instructions
- `--source-tables` (required): Source tables to process

**Example**:
```
/ml data --name processed_features
         --instruction "Normalize numeric features and one-hot encode categories"
         --source-tables raw_customer_data
```

**Output**:
- Creates Arc-Pipeline specification
- Executes the pipeline
- Creates output table(s)

### /ml model

Train a machine learning model.

**Syntax**:
```
/ml model --name <model_name>
          --instruction "<description>"
          --data-table <table>
          [--plan-id <plan_id>]
```

**Options**:
- `--name` (required): Name for the model
- `--instruction` (required): Model training instructions
- `--data-table` (required): Table with training data
- `--plan-id` (optional): Use an existing ML plan

**Examples**:
```
# Basic model training
/ml model --name churn_predictor
          --instruction "Binary classifier for churn prediction"
          --data-table processed_features

# With ML plan
/ml model --name churn_predictor
          --instruction "Follow the plan"
          --data-table processed_features
          --plan-id plan_abc123

# With detailed architecture
/ml model --name price_model
          --instruction "Regression model with 3 layers (128, 64, 32), dropout 0.2, learning rate 0.001"
          --data-table housing_features
```

**Output**:
- Arc-Graph specification
- Trained model
- Training metrics
- TensorBoard logs

### /ml evaluate

Evaluate a trained model.

**Syntax**:
```
/ml evaluate --model-id <model_id>
             --data-table <table>
             [--metrics "<metrics>"]
```

**Options**:
- `--model-id` (required): ID of the trained model
- `--data-table` (required): Evaluation data table
- `--metrics` (optional): Comma-separated list of metrics

**Available Metrics**:
- Classification: `accuracy`, `precision`, `recall`, `f1`, `auc`
- Regression: `mse`, `rmse`, `mae`, `r2`

**Examples**:
```
# Default metrics
/ml evaluate --model-id model_123
             --data-table test_data

# Custom metrics
/ml evaluate --model-id model_123
             --data-table test_data
             --metrics "accuracy,precision,recall,f1,auc"
```

**Output**:
- Evaluation results stored in `evaluations` table
- Optional TensorBoard visualizations

### /ml predict

Make predictions with a trained model.

**Syntax**:
```
/ml predict --model <model_name>
            --data <table>
            [--output <output_table>]
            [--batch-size <size>]
```

**Options**:
- `--model` (required): Model name or ID
- `--data` (required): Input data table
- `--output` (optional): Output table name (default: `<model>_predictions`)
- `--batch-size` (optional): Batch size for prediction (default: 1000)

**Examples**:
```
# Basic prediction
/ml predict --model churn_predictor
            --data new_customers

# Custom output table
/ml predict --model price_model
            --data new_listings
            --output price_estimates

# Large batch
/ml predict --model fraud_detector
            --data all_transactions
            --batch-size 10000
```

### /ml jobs

Manage ML jobs.

**Syntax**:
```
/ml jobs list                    # List all jobs
/ml jobs status <job_id>         # Show job details
```

**Examples**:
```
# List all ML jobs
/ml jobs list

# Show specific job
/ml jobs status job_abc123
```

**Output**:
- Job ID, name, type, status
- Training metrics (if applicable)
- Timestamps

## SQL Commands (/sql)

Execute SQL queries against Arc's DuckDB database.

**Syntax**:
```
/sql <query>
```

**Examples**:
```
# Show all tables
/sql SHOW TABLES

# Describe table structure
/sql DESCRIBE customers

# Select data
/sql SELECT * FROM customers LIMIT 10

# Aggregation
/sql SELECT
       country,
       COUNT(*) as customer_count,
       AVG(age) as avg_age
     FROM customers
     GROUP BY country
     ORDER BY customer_count DESC

# Create table
/sql CREATE TABLE processed AS
     SELECT
         id,
         UPPER(name) as name_upper,
         age * 12 as age_months
     FROM raw_data

# Join tables
/sql SELECT
       c.customer_id,
       c.name,
       COUNT(o.order_id) as order_count
     FROM customers c
     LEFT JOIN orders o ON c.customer_id = o.customer_id
     GROUP BY c.customer_id, c.name

# Export to CSV
/sql COPY (SELECT * FROM results)
     TO 'output.csv' (HEADER, DELIMITER ',')
```

## Configuration Commands (/config)

Manage Arc configuration.

**Syntax**:
```
/config
```

**Interactive Prompts**:
1. API Key
2. Base URL (optional)
3. Model name

**Settings Stored**: `~/.arc/user-settings.json`

**Example Session**:
```
> /config

◇ Configuration
  API Key            ********
  Base URL           https://api.openai.com/v1
  Model              gpt-4

Settings saved to ~/.arc/user-settings.json
```

## Report Commands (/report)

Generate reports on ML workflows.

**Syntax**:
```
/report [--type <report_type>]
```

**Report Types**:
- `summary` (default): Overview of all ML artifacts
- `model`: Detailed model performance
- `data`: Data processing summary

**Examples**:
```
# Generate summary report
/report

# Model-specific report
/report --type model

# Data processing report
/report --type data
```

## Utility Commands

### /help

Show available commands and help information.

**Syntax**:
```
/help
/help <command>
```

**Examples**:
```
# Show all commands
/help

# Help for specific command
/help ml model
```

### /clear

Clear the conversation context (start fresh).

**Syntax**:
```
/clear
```

**Effect**:
- Clears chat history
- Resets conversation state
- Does not delete data or models

### exit

Exit Arc CLI.

**Syntax**:
```
exit
```

## Natural Language Mode

You can also interact with Arc using natural language (without command prefixes):

**Examples**:
```
# Data loading
"Load customers.csv into a table"

# Feature engineering
"Normalize the age and income columns in the customers table"

# Model training
"Build a model to predict customer churn"

# Evaluation
"How well does my churn model perform?"

# Prediction
"Use the churn model to score all active customers"
```

Arc will translate your request into the appropriate commands.

## Environment Variables

Override configuration with environment variables:

```bash
# API configuration
export ARC_API_KEY="your-api-key"
export ARC_BASE_URL="https://api.openai.com/v1"
export ARC_MODEL="gpt-4"

# Database paths (advanced)
export ARC_SYSTEM_DB="~/.arc/system.db"
export ARC_USER_DB="~/.arc/user.db"

# Start Arc
arc chat
```

## Exit Codes

Arc returns standard exit codes:

- `0`: Success
- `1`: General error
- `2`: Configuration error
- `3`: Data error
- `4`: Model error

## Keyboard Shortcuts

- **Ctrl+C**: Cancel current operation
- **Ctrl+D**: Exit Arc
- **Esc**: Interrupt streaming response
- **Up/Down Arrow**: Navigate command history
- **Tab**: Auto-complete (where supported)

## Tips and Tricks

### 1. Command History

Arc saves command history. Use Up/Down arrows to navigate previous commands.

### 2. Multi-line SQL

For complex queries, use natural language to describe them:

```
"Show me the top 10 customers by total purchase amount,
 including their name, email, and number of orders"
```

Arc will generate and execute the SQL.

### 3. Piping Results

You can describe complex workflows in one message:

```
"Load sales.csv, aggregate by month, then create a forecast model"
```

Arc will execute the workflow step-by-step.

### 4. Quick Checks

```sql
-- Quick data check
/sql SELECT COUNT(*), MIN(date), MAX(date) FROM data

-- Quick model list
/ml jobs list

-- Quick eval check
/sql SELECT * FROM evaluations ORDER BY created_at DESC LIMIT 1
```

### 5. Batch Operations

For multiple operations, list them:

```
"I need to:
1. Load customers.csv
2. Normalize the features
3. Train a churn model
4. Evaluate on test set
5. Generate predictions for all active customers"
```

## Common Patterns

### Pattern: Complete ML Workflow

```bash
# 1. Load data
/sql CREATE TABLE data AS SELECT * FROM 'data.csv'

# 2. Feature engineering
/ml data --name processed --instruction "normalize and split" --source-tables data

# 3. Train
/ml model --name my_model --instruction "binary classifier" --data-table processed_train

# 4. Evaluate
/ml evaluate --model-id <id> --data-table processed_test

# 5. Predict
/ml predict --model my_model --data new_data
```

### Pattern: Iterative Model Development

```bash
# Try model v1
/ml model --name model_v1 --instruction "simple MLP" --data-table data

# Evaluate
/ml evaluate --model-id model_v1_id --data-table test

# Try model v2 (improved)
/ml model --name model_v2 --instruction "deeper network with dropout" --data-table data

# Compare
/sql SELECT model_name, accuracy, f1_score FROM evaluations ORDER BY accuracy DESC
```

### Pattern: Data Exploration

```sql
-- Table overview
/sql SHOW TABLES

-- Schema
/sql DESCRIBE my_table

-- Summary stats
/sql SELECT
       COUNT(*) as total,
       COUNT(DISTINCT user_id) as unique_users,
       MIN(date) as earliest,
       MAX(date) as latest
     FROM data

-- Distribution
/sql SELECT
       category,
       COUNT(*) as count,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
     FROM data
     GROUP BY category
     ORDER BY count DESC
```

## Next Steps

- **[Quick Start Tutorial](../getting-started/quickstart.md)** - Learn by doing
- **[User Guides](../guides/data-loading.md)** - In-depth guides
- **[Examples](../examples/diabetes-prediction.md)** - Complete tutorials

## Related Documentation

- [Configuration Guide](../getting-started/configuration.md)
- [Arc-Graph Specification](../concepts/arc-graph.md)
- [Arc-Pipeline Specification](../concepts/arc-pipeline.md)
