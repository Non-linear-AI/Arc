# Arc Training Test Scripts

This directory contains utility scripts for testing and debugging Arc training functionality.

## Scripts

### 1. `test_training.py` - Direct Training Test

Launch training jobs directly with a saved model, without needing to regenerate the YAML every time.

**Usage:**
```bash
python scripts/test_training.py <model_id> <train_table> --target-column <column> [OPTIONS]
```

**Examples:**
```bash
# Test with specific model version
python scripts/test_training.py test-v8 pidd --target-column outcome

# Test with latest version of a model (auto-resolves to test-v8 if that's latest)
python scripts/test_training.py test pidd --target-column outcome

# Include validation table
python scripts/test_training.py test-v8 pidd --target-column outcome --validation-table pidd_val

# Monitor job until completion
python scripts/test_training.py test-v8 pidd --target-column outcome --monitor

# Launch TensorBoard for live metrics visualization
python scripts/test_training.py test-v8 pidd --target-column outcome --tensorboard --monitor

# Verbose logging
python scripts/test_training.py test-v8 pidd --target-column outcome --monitor --verbose
```

**Options:**
- `--target-column, -t`: Target column for prediction (required)
- `--validation-table, -v`: Optional validation table name
- `--system-db`: Path to Arc system database (default: `~/.arc/arc_system.db`)
- `--user-db`: Path to Arc user database (default: `~/.arc/arc_user.db`)
- `--artifacts-dir`: Directory for training artifacts (default: `~/.arc/artifacts`)
- `--monitor`: Monitor job status until completion
- `--tensorboard`: Launch TensorBoard after job submission
- `--tensorboard-port`: TensorBoard port (default: 6006)
- `--verbose, -V`: Enable verbose logging

### 2. `extract_model_yaml.py` - Extract Model YAML

Extract and view the YAML specification from a saved model.

**Usage:**
```bash
python scripts/extract_model_yaml.py <model_id> [--output FILE]
```

**Examples:**
```bash
# Print to stdout
python scripts/extract_model_yaml.py test-v8

# Save to file
python scripts/extract_model_yaml.py test-v8 --output test-v8.yaml

# Get latest version of a model
python scripts/extract_model_yaml.py test --output test-latest.yaml

# Use with other tools (e.g., validate YAML)
python scripts/extract_model_yaml.py test-v8 | python -m yaml
```

**Options:**
- `--output, -o`: Output file path (default: print to stdout)
- `--system-db`: Path to Arc system database (default: `~/.arc/arc_system.db`)

## Common Workflows

### Test Training with Saved Model

1. **Generate and save a model** (via Arc CLI):
   ```bash
   arc
   > /ml model --name test --instruction "predict outcome in pidd" --data-table pidd --target-column outcome
   ```

2. **Extract YAML to inspect** (optional):
   ```bash
   python scripts/extract_model_yaml.py test-v8 --output test-v8.yaml
   cat test-v8.yaml
   ```

3. **Launch training**:
   ```bash
   python scripts/test_training.py test-v8 pidd --target-column outcome --monitor
   ```

### Debug Training Issues

1. **Enable verbose logging**:
   ```bash
   python scripts/test_training.py test-v8 pidd -t outcome --verbose --monitor
   ```

2. **Check job status** (if not using --monitor):
   ```bash
   arc
   > /ml jobs status <job_id>
   > /ml jobs logs <job_id>
   ```

### Iterate on Training Configuration

1. **Save current YAML**:
   ```bash
   python scripts/extract_model_yaml.py test-v8 > current.yaml
   ```

2. **Edit YAML manually**:
   ```bash
   vim current.yaml
   # Modify training parameters, etc.
   ```

3. **Re-register model** (via Arc CLI):
   ```bash
   arc
   > /ml model --name test-v9 --spec current.yaml --train-table pidd --target-column outcome
   ```

4. **Test new version**:
   ```bash
   python scripts/test_training.py test-v9 pidd -t outcome --monitor
   ```

## Notes

- Both scripts automatically resolve model names to their latest version if no specific version is provided
- The scripts use the same database and configuration as the Arc CLI
- Training artifacts are saved to `~/.arc/artifacts/` by default
- TensorBoard logs are automatically generated during training (if enabled)

## Troubleshooting

**Script not found:**
```bash
# Make sure you're in the project root
cd /path/to/Arc

# Run with python -m if needed
python -m scripts.test_training test-v8 pidd -t outcome
```

**Database not found:**
```bash
# Check database path
ls ~/.arc/arc_system.db

# Or specify custom paths
python scripts/test_training.py test-v8 pidd -t outcome --system-db /custom/path/system.db --user-db /custom/path/user.db
```

**Import errors:**
```bash
# Make sure you're using the project virtual environment
source .venv/bin/activate

# Or use uv
uv run python scripts/test_training.py test-v8 pidd -t outcome
```
