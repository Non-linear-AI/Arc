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
   /sql CREATE TABLE iris_raw AS SELECT * FROM read_csv_auto('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv');
   /sql CREATE TABLE iris AS SELECT column0 AS pregnancies, column1 AS glucose, column2 AS blood_pressure, column3 AS skin_thickness, column4 AS insulin, column5 AS bmi, column6 AS diabetes_pedigree, column7 AS age, column8 AS outcome FROM iris_raw;
   ```

3. Use the chat interface to create a model and launch training:

   ```text
   Build a binary classification model using the iris table to predict outcome.
   Use logistic regression with glucose, bmi, and age as features.
   ```

   The agent will guide you through the ML workflow (ml_plan → ml_data → ml_model).

   Alternatively, you can use the low-level CLI commands:
   ```text
   # This creates model + trainer and launches training in one step
   /ml model --name "pima_classifier" --instruction "Binary classification for diabetes prediction" --data-table "iris" --target-column "outcome"
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

## Troubleshooting

### CPR Warning or Arrow Keys Not Working

If you see:
```
WARNING: your terminal doesn't support cursor position requests (CPR).
```

Or arrow keys show as escape sequences (`^[[A`), this is a terminal compatibility issue. The demo script handles this automatically by:
- Auto-accepting prompts with `yes "1"`
- Filtering out CPR warnings

**If the script still doesn't work**, try these alternatives:

**Option 1: Run in a fully interactive terminal**
```bash
uv run arc chat
# Then manually paste the commands from the script
```

**Option 2: Use a different terminal**
Try running the script in:
- Native Terminal.app (macOS)
- iTerm2 (macOS)
- gnome-terminal or konsole (Linux)
- Windows Terminal (Windows)

**Option 3: Force non-interactive mode**
```bash
# Bypass TTY detection entirely
yes "1" | bash run_logistic_regression_console_demo.sh < /dev/null
```

### Interactive Prompts During Automated Demo

Arc may show interactive prompts asking you to review generated specifications. The updated script automatically accepts these with option "1" (Accept). If you want to review specs manually, run the commands interactively in `arc chat` instead of using the automated script.

## Files

- `model.yaml` – Arc-Graph specification for the logistic regression model.
- `run_logistic_regression_console_demo.sh` – Automation script for the demo.
