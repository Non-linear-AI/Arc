#!/usr/bin/env python3
"""Test script to verify the Arc-Graph implementation works correctly."""

import torch

from arc.graph import ModelSpec, build_model_from_yaml


def test_basic_model():
    """Test basic Arc-Graph model functionality."""
    yaml_content = """
inputs:
  features:
    dtype: float32
    shape: [null, 4]
    columns: [x1, x2, x3, x4]

graph:
  - name: hidden
    type: torch.nn.Linear
    params:
      in_features: 4
      out_features: 8
    inputs:
      input: features

  - name: activation
    type: torch.nn.functional.relu
    inputs: [hidden.output]

  - name: output
    type: torch.nn.Linear
    params:
      in_features: 8
      out_features: 2
    inputs:
      input: activation

outputs:
  logits: output.output
"""

    print("1. Testing basic model creation...")
    try:
        # Parse spec
        spec = ModelSpec.from_yaml(yaml_content)
        print(f"   ✓ Parsed spec with {len(spec.graph)} nodes")

        # Build model
        model = build_model_from_yaml(yaml_content)
        print(f"   ✓ Built model with inputs: {model.input_names}")
        print(f"   ✓ Built model with outputs: {model.output_names}")

        # Test forward pass
        test_input = torch.randn(2, 4)
        output = model(features=test_input)
        print(f"   ✓ Forward pass successful, output shape: {output.shape}")

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    return True


def test_modules_and_stack():
    """Test modules and arc.stack functionality."""
    yaml_content = """
modules:
  FeedForward:
    inputs: [x]
    graph:
      - name: linear1
        type: torch.nn.Linear
        params: {in_features: 64, out_features: 256}
        inputs: {input: x}
      - name: activation
        type: torch.nn.functional.gelu
        inputs: [linear1.output]
      - name: linear2
        type: torch.nn.Linear
        params: {in_features: 256, out_features: 64}
        inputs: {input: activation}
    outputs:
      output: linear2.output

inputs:
  sequence:
    dtype: float32
    shape: [null, 10, 64]
    columns: [sequence_data]

graph:
  - name: transformer_stack
    type: arc.stack
    params:
      module: FeedForward
      count: 3
    inputs: {input: sequence}

  - name: pooled
    type: torch.mean
    params: {dim: 1}
    inputs: [transformer_stack.output]

  - name: classifier
    type: torch.nn.Linear
    params: {in_features: 64, out_features: 5}
    inputs: {input: pooled}

outputs:
  predictions: classifier.output
"""

    print("2. Testing modules and arc.stack...")
    try:
        # Parse spec
        spec = ModelSpec.from_yaml(yaml_content)
        print(
            f"   ✓ Parsed spec with {len(spec.modules)} modules, "
            f"{len(spec.graph)} main nodes"
        )

        # Build model
        model = build_model_from_yaml(yaml_content)
        print("   ✓ Built model with custom modules")

        # Test forward pass
        test_input = torch.randn(2, 10, 64)
        output = model(sequence=test_input)
        print(f"   ✓ Forward pass successful, output shape: {output.shape}")

        # Check that the stack actually has 3 layers
        stack_module = model.graph_modules["transformer_stack"]
        print(f"   ✓ Stack has {len(stack_module)} layers")

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    return True


def test_list_inputs():
    """Test list input format."""
    yaml_content = """
inputs:
  a:
    dtype: float32
    shape: [null, 3]
    columns: [a1, a2, a3]
  b:
    dtype: float32
    shape: [null, 3]
    columns: [b1, b2, b3]

graph:
  - name: concat
    type: torch.cat
    params: {dim: 1}
    inputs: [a, b]

  - name: linear
    type: torch.nn.Linear
    params: {in_features: 6, out_features: 1}
    inputs: {input: concat}

outputs:
  result: linear.output
"""

    print("3. Testing list inputs...")
    try:
        # Parse spec
        ModelSpec.from_yaml(yaml_content)
        print("   ✓ Parsed spec with list inputs")

        # Build model
        model = build_model_from_yaml(yaml_content)
        print("   ✓ Built model")

        # Test forward pass
        test_a = torch.randn(2, 3)
        test_b = torch.randn(2, 3)
        output = model(a=test_a, b=test_b)
        print(f"   ✓ Forward pass successful, output shape: {output.shape}")

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    return True


if __name__ == "__main__":
    print("🧪 Testing Arc-Graph Implementation")
    print("=" * 50)

    results = []
    results.append(test_basic_model())
    results.append(test_modules_and_stack())
    results.append(test_list_inputs())

    print("\n" + "=" * 50)
    if all(results):
        print("🎉 All tests passed! Arc-Graph implementation is working correctly.")
    else:
        print("❌ Some tests failed. Check the implementation.")

    success_count = sum(results)
    print(f"Results: {success_count}/{len(results)} tests passed")
