#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
MODEL_PATH="$SCRIPT_DIR/model.yaml"

echo "📟 Running logistic regression console demo..."

COMMAND_FILE=$(mktemp)
trap 'rm -f "$COMMAND_FILE"' EXIT

cat > "$COMMAND_FILE" <<EOF
/sql use user
/sql CREATE TABLE iris_raw AS SELECT * FROM read_csv_auto('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv');
/sql CREATE TABLE iris AS SELECT column0 as pregnancies, column1 as glucose, column2 as blood_pressure, column3 as skin_thickness, column4 as insulin, column5 as bmi, column6 as diabetes_pedigree, column7 as age, column8 as outcome FROM iris_raw;
/ml create-model --name "pima_classifier" --schema "$MODEL_PATH"
/ml train --model "pima_classifier" --data "iris"
/ml jobs list
/exit
EOF

echo "------------------------------------------------------------"
cat "$COMMAND_FILE"
echo "------------------------------------------------------------"

UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_ROOT/.uv-cache}"
mkdir -p "$UV_CACHE_DIR"

echo "🚀 Streaming commands through arc chat"
UV_CACHE_DIR="$UV_CACHE_DIR" uv run arc chat < "$COMMAND_FILE"

echo "✅ Demo complete. Review the output above for training status and job ID."
