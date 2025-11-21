# CLI Commands Reference

Quick reference for Arc commands.

## Command Overview

- `/ml` - Machine learning operations (plan, data, model, evaluate, predict, jobs)
- `/sql` - SQL queries
- `/config` - Configuration
- `/report` - Generate reports
- `/help` - Show help
- `/clear` - Clear context

## Starting Arc

```bash
arc chat                      # Interactive mode
arc chat --non-interactive    # Scripting mode
arc --help                    # Show help
arc --version                 # Show version
```

## ML Commands

### /ml plan

Generate ML workflow plan.

```bash
/ml plan --name <name>
         --instruction "<description>"
         --source-tables <tables>
```

Example:
```bash
/ml plan --name customer_churn
         --instruction "Predict churn"
         --source-tables customers,transactions
```

### /ml revise-plan

Revise existing plan.

```bash
/ml revise-plan --instruction "<changes>"
```

### /ml data

Create data processing pipeline.

```bash
/ml data --name <name>
         --instruction "<description>"
         --source-tables <tables>
```

Example:
```bash
/ml data --name processed_features
         --instruction "Normalize features and split data"
         --source-tables raw_data
```

### /ml model

Train a model.

```bash
/ml model --name <name>
          --instruction "<description>"
          --data-table <table>
          [--plan-id <id>]
```

Example:
```bash
/ml model --name churn_predictor
          --instruction "Binary classifier"
          --data-table processed_features
```

### /ml evaluate

Evaluate trained model.

```bash
/ml evaluate --model-id <id>
             --data-table <table>
             [--metrics "<metrics>"]
```

Available metrics:
- Classification: `accuracy`, `precision`, `recall`, `f1`, `auc`
- Regression: `mse`, `rmse`, `mae`, `r2`

### /ml predict

Make predictions.

```bash
/ml predict --model <name>
            --data <table>
            [--output <table>]
```

### /ml jobs

Manage jobs.

```bash
/ml jobs list              # List all jobs
/ml jobs status <job_id>   # Show job details
```

## SQL Commands

Execute SQL queries:

```bash
/sql <query>
```

Common examples:
```sql
/sql SHOW TABLES
/sql DESCRIBE table_name
/sql SELECT * FROM table_name LIMIT 10
/sql CREATE TABLE new AS SELECT * FROM old WHERE condition
```

## Configuration

```bash
/config
```

Interactive prompts for:
- API Key
- Base URL
- Model name

Settings saved to `~/.arc/user-settings.json`.

Alternative: Use environment variables:
```bash
export ARC_API_KEY="your-key"
export ARC_BASE_URL="https://api.openai.com/v1"
export ARC_MODEL="gpt-4"
```

## Other Commands

```bash
/report              # Generate ML workflow report
/help                # Show available commands
/clear               # Clear conversation context
exit                 # Exit Arc
```

## Natural Language

You can also use plain English instead of commands:

```
"Load customers.csv into a table"
"Build a model to predict churn"
"Evaluate my model on test data"
```

## Keyboard Shortcuts

- **Ctrl+C**: Cancel operation
- **Ctrl+D**: Exit Arc
- **Esc**: Interrupt streaming
- **Up/Down**: Command history

## Quick Examples

### Complete Workflow

```bash
# 1. Load data
/sql CREATE TABLE data AS SELECT * FROM 'data.csv'

# 2. Process
/ml data --name processed --instruction "normalize and split" --source-tables data

# 3. Train
/ml model --name my_model --instruction "classifier" --data-table processed_train

# 4. Evaluate
/ml evaluate --model-id <id> --data-table processed_test

# 5. Predict
/ml predict --model my_model --data new_data
```

### Data Exploration

```sql
/sql SHOW TABLES
/sql DESCRIBE my_table
/sql SELECT COUNT(*), MIN(date), MAX(date) FROM my_table
```

## Need Help?

- Run `/help` in Arc for interactive help
- Run `arc --help` for command-line options
- See [User Guides](../guides/data-loading.md) for detailed workflows
