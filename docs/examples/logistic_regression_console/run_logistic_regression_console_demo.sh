#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
MODEL_PATH="$SCRIPT_DIR/model.yaml"

echo "📟 Running logistic regression console demo..."

COMMAND_FILE=$(mktemp)
trap 'rm -f "$COMMAND_FILE"' EXIT

cat > "$COMMAND_FILE" <<EOF
Load and prepare the Pima Indians diabetes dataset from https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv into a table called diabetes with columns: pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age, outcome
/ml model --name "pima_classifier" --instruction "Binary classification model for diabetes prediction using logistic regression" --data-table "diabetes" --target-column "outcome"
/ml jobs list
/exit
EOF

echo "------------------------------------------------------------"
cat "$COMMAND_FILE"
echo "------------------------------------------------------------"

UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_ROOT/.uv-cache}"
mkdir -p "$UV_CACHE_DIR"

echo "🚀 Streaming commands through arc chat"
# Use 'yes' to auto-accept any interactive prompts (answer "1" = Accept)
# Redirect stderr to /dev/null to suppress CPR warnings
yes "1" 2>/dev/null | UV_CACHE_DIR="$UV_CACHE_DIR" uv run arc chat < "$COMMAND_FILE" 2>&1 | grep -v "CPR" || true

echo "✅ Demo complete. Review the output above for training status and job ID."
