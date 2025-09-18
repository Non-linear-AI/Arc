# Logistic Regression Console Demo

This example shows how to register and train a simple logistic regression model directly from the interactive `arc chat` console.

## Prerequisites

- `uv` and the Arc CLI dependencies installed (`uv sync --dev`).
- An API key available to the CLI (set `ARC_API_KEY` or configure it via `/config`).
- Internet access to download the public Pima Indians Diabetes dataset.

## Manual Walkthrough

1. Start the console from the project root:

   ```bash
   uv run arc chat
   ```

2. Load the training data into the user database:

   ```text
   /sql use user
   /sql CREATE TABLE iris_raw AS SELECT * FROM read_csv_auto('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv');
   /sql CREATE TABLE iris AS SELECT column0 AS pregnancies, column1 AS glucose, column2 AS blood_pressure, column3 AS skin_thickness, column4 AS insulin, column5 AS bmi, column6 AS diabetes_pedigree, column7 AS age, column8 AS outcome FROM iris_raw;
   ```

3. Register the Arc-Graph model (the schema is provided in `model.yaml`):

   ```text
   /ml create-model --name "pima_classifier" --schema "examples/logistic_regression_console/model.yaml"
   ```

4. Launch training:

   ```text
   /ml train --model "pima_classifier" --data "iris"
   ```

5. Inspect job progress:

   ```text
   /ml jobs list
   /ml jobs status <job_id>
   ```

## Automated Demo

A helper script is provided to play the same sequence end-to-end. From the project root run:

```bash
UV_CACHE_DIR=.uv-cache ARC_API_KEY=your_key \
  ./examples/logistic_regression_console/run_logistic_regression_console_demo.sh
```

The script streams the required slash commands into `uv run arc chat`, creating the dataset, registering the model, starting a training job, and printing job status. Override `UV_CACHE_DIR` if your environment restricts access to the default cache path.

## Files

- `model.yaml` – Arc-Graph specification for the logistic regression model.
- `run_logistic_regression_console_demo.sh` – Automation script for the demo.
