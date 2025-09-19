#!/bin/bash

# Test script for enhanced schema generation using CLI
# Tests the pidd table logistic regression schema generation

echo "🚀 Testing Enhanced Schema Generation via CLI"
echo "=============================================="
echo

echo "📋 Test Parameters:"
echo "   Model: test_pidd_logistic"
echo "   Context: Logistic regression for diabetes prediction"
echo "   Table: pidd"
echo "   Output: test_pidd_schema.yaml"
echo

echo "🔍 Running schema generation with enhanced validation..."
echo "   (May take multiple attempts if validation fails)"
echo

# Run the schema generation command
ARC_API_KEY=test_key uv run python -m arc.core.agents.example_script \
    --name "test_pidd_logistic" \
    --context "Create a logistic regression model to predict diabetes outcome based on patient health metrics. Use only the columns that exist in the dataset." \
    --table "pidd" \
    --output "test_pidd_schema.yaml"

echo
echo "📊 Checking results..."

if [ -f "test_pidd_schema.yaml" ]; then
    echo "✅ Schema file generated successfully!"
    echo
    echo "📄 Generated Schema Preview (first 30 lines):"
    echo "----------------------------------------"
    head -30 test_pidd_schema.yaml | cat -n
    echo "----------------------------------------"
    echo

    echo "🔎 Quick Analysis:"
    echo "   Feature columns: $(grep -A 10 'feature_columns:' test_pidd_schema.yaml | grep '    -' | wc -l | tr -d ' ') found"
    echo "   Processors: $(grep -c 'name:' test_pidd_schema.yaml) found"
    echo "   Uses only valid processors: $(grep -c 'transform\.\(impute\|scale\)' test_pidd_schema.yaml) invalid processors found"

    if [ $(grep -c 'transform\.\(impute\|scale\)' test_pidd_schema.yaml) -eq 0 ]; then
        echo "   ✅ No invalid processors detected"
    else
        echo "   ❌ Invalid processors still present"
    fi

else
    echo "❌ Schema file not generated"
    echo "   This could mean validation is working and rejecting invalid schemas"
fi

echo
echo "🧪 Testing validation manually..."

# Test with intentionally bad schema to verify validation works
cat > test_bad_schema.yaml << 'EOF'
features:
  feature_columns: [nonexistent_column, another_fake_column]
  target_columns: [fake_target]
  processors:
    - name: bad_processor
      op: transform.nonexistent
      inputs: { columns: feature_columns }
      outputs: { tensors.features: output }

model:
  inputs:
    features:
      dtype: float32
      shape: [null, 4]
  graph:
    - name: classifier
      type: core.Linear
      params:
        in_features: 4
        out_features: 1
      inputs:
        input: features

trainer:
  optimizer:
    type: adam
  loss:
    type: binary_cross_entropy
    inputs:
      predictions: model.classifier.output
      targets: target_columns.fake_target
EOF

echo "📝 Created intentionally bad schema for validation test"
echo "   Bad schema should be rejected by validation"

echo
echo "🏁 Test completed!"
echo
echo "💡 To run the full Python test:"
echo "   python test_schema_generation.py"
echo
echo "💡 To manually test generate-schema:"
echo "   uv run arc generate-schema --help"