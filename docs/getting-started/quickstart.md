# Quick Start Tutorial

This tutorial will guide you through building your first machine learning model with Arc. You'll build a diabetes prediction model in just a few minutes - without writing any ML code!

## What You'll Build

In this tutorial, you'll:
- Download the Pima Indians Diabetes dataset
- Let Arc analyze the data and engineer features
- Generate an Arc-Graph model specification
- Train and evaluate the model
- View predictions and performance metrics

**Time required:** ~5 minutes

## Prerequisites

Before starting, make sure you have:
- [Installed Arc](installation.md)
- [Configured your API key](configuration.md)

## Step 1: Start Arc

Launch Arc's interactive chat interface:

```bash
arc chat
```

You should see Arc's welcome message with the ASCII logo.

## Step 2: Describe What You Want

Simply tell Arc what you want to build in plain English:

```
Download the Pima Indians Diabetes dataset and build a model to predict diabetes from patient health metrics
```

Press Enter and watch Arc work its magic!

## What Happens Next

Arc will automatically:

1. **Download the Dataset**
   - Fetches the Pima Indians Diabetes dataset
   - Loads it into Arc's database

2. **Analyze the Data**
   - Examines features: pregnancies, glucose, blood pressure, BMI, age, etc.
   - Determines appropriate preprocessing steps

3. **Engineer Features**
   - Normalizes numerical features
   - Creates train/test splits
   - Generates processed data tables

4. **Generate Arc-Graph Specification**
   - Creates a YAML specification for the model architecture
   - Defines inputs, model layers, and outputs
   - You'll see the spec and can review/approve it

5. **Train the Model**
   - Trains the model with your data
   - Launches TensorBoard for real-time monitoring
   - Tracks metrics (loss, accuracy, etc.)

6. **Evaluate Performance**
   - Computes evaluation metrics
   - Shows predictions vs actual values
   - Displays model performance statistics

## Step 3: Explore Your Results

Once training completes, you can explore your data and results using SQL:

```sql
-- View available tables
/sql SHOW TABLES

-- See predictions
/sql SELECT * FROM predictions LIMIT 10

-- Check model performance
/sql SELECT * FROM evaluations ORDER BY created_at DESC LIMIT 1
```

## Understanding the Arc-Graph

Arc generated an **Arc-Graph** specification for your model. It looks something like this:

```yaml
# Arc-Graph: Model Architecture
inputs:
  patient_data:
    dtype: float32
    shape: [null, 8]
    columns: [pregnancies, glucose, blood_pressure, skin_thickness,
              insulin, bmi, diabetes_pedigree, age]

graph:
  - name: classifier
    type: torch.nn.Linear
    params:
      in_features: 8
      out_features: 1
    inputs:
      input: patient_data

  - name: sigmoid
    type: torch.nn.Sigmoid
    inputs:
      input: classifier.output

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
- **Human-readable** - You can understand and modify it
- **Portable** - Runs anywhere PyTorch runs
- **Versionable** - Track changes in Git
- **Reproducible** - Guarantees train/serve parity

Learn more about Arc-Graph in the [Arc-Graph documentation](../concepts/arc-graph.md).

## Step 4: View Training Progress

Arc automatically launches TensorBoard to visualize training progress. Check your console output for the TensorBoard URL (usually `http://localhost:6006`).

In TensorBoard, you can view:
- Training and validation loss curves
- Accuracy metrics over time
- Model architecture graph
- Hyperparameter comparisons

## What's Next?

Now that you've built your first model, explore more:

- **[Model Training Guide](../guides/model-training.md)** - Learn about training workflows
- **[Feature Engineering](../guides/feature-engineering.md)** - Advanced data preparation
- **[Making Predictions](../guides/making-predictions.md)** - Use your model for inference
- **[Examples](../examples/logistic_regression_console/)** - More detailed examples
