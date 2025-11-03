#!/usr/bin/env python3
"""Validation script for binary operations and multi-input layer handling.

This script demonstrates that CustomModuleWrapper and FunctionalWrapper
correctly handle different types of multi-input operations:
- Binary operations (torch.add, torch.mul) that need positional args
- Tuple operations (torch.cat, torch.stack) that need tuple as first arg
"""

import torch

from arc.graph import GraphNode, ModelInput, ModelSpec, ModuleDefinition
from arc.ml.builder import ModelBuilder


def validate_torch_add():
    """Validate torch.add binary operation."""
    print("=" * 60)
    print("Test 1: torch.add (Binary Operation)")
    print("=" * 60)

    # Create a model that adds two inputs
    model_spec = ModelSpec(
        inputs={
            "x": ModelInput(dtype="float32", shape=[None, 4]),
            "y": ModelInput(dtype="float32", shape=[None, 4]),
        },
        graph=[
            GraphNode(
                name="add_op",
                type="torch.add",
                params={},
                inputs=["x", "y"],
            )
        ],
        outputs={"result": "add_op.output"},
    )

    builder = ModelBuilder(enable_shape_validation=False)
    model = builder.build_model(model_spec)

    # Test forward pass
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    y = torch.tensor([[10.0, 20.0, 30.0, 40.0]])

    result = model({"x": x, "y": y})
    expected = torch.tensor([[11.0, 22.0, 33.0, 44.0]])

    print(f"Input x: {x.tolist()}")
    print(f"Input y: {y.tolist()}")
    print(f"Result: {result['result'].tolist()}")
    print(f"Expected: {expected.tolist()}")

    assert torch.allclose(result["result"], expected)
    print("✓ torch.add works correctly\n")


def validate_torch_mul():
    """Validate torch.mul binary operation."""
    print("=" * 60)
    print("Test 2: torch.mul (Binary Operation)")
    print("=" * 60)

    model_spec = ModelSpec(
        inputs={
            "a": ModelInput(dtype="float32", shape=[None, 3]),
            "b": ModelInput(dtype="float32", shape=[None, 3]),
        },
        graph=[
            GraphNode(
                name="mul_op",
                type="torch.mul",
                params={},
                inputs=["a", "b"],
            )
        ],
        outputs={"product": "mul_op.output"},
    )

    builder = ModelBuilder(enable_shape_validation=False)
    model = builder.build_model(model_spec)

    a = torch.tensor([[2.0, 3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0, 7.0]])

    result = model({"a": a, "b": b})
    expected = torch.tensor([[10.0, 18.0, 28.0]])

    print(f"Input a: {a.tolist()}")
    print(f"Input b: {b.tolist()}")
    print(f"Result: {result['product'].tolist()}")
    print(f"Expected: {expected.tolist()}")

    assert torch.allclose(result["product"], expected)
    print("✓ torch.mul works correctly\n")


def validate_torch_cat_two_inputs():
    """Validate torch.cat with 2 inputs (tuple operation)."""
    print("=" * 60)
    print("Test 3: torch.cat with 2 Inputs (Tuple Operation)")
    print("=" * 60)

    model_spec = ModelSpec(
        inputs={
            "tensor1": ModelInput(dtype="float32", shape=[None, 2]),
            "tensor2": ModelInput(dtype="float32", shape=[None, 2]),
        },
        graph=[
            GraphNode(
                name="cat_op",
                type="torch.cat",
                params={"dim": 1},
                inputs=["tensor1", "tensor2"],
            )
        ],
        outputs={"concatenated": "cat_op.output"},
    )

    builder = ModelBuilder(enable_shape_validation=False)
    model = builder.build_model(model_spec)

    t1 = torch.tensor([[1.0, 2.0]])
    t2 = torch.tensor([[3.0, 4.0]])

    result = model({"tensor1": t1, "tensor2": t2})
    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    print(f"Input tensor1: {t1.tolist()}")
    print(f"Input tensor2: {t2.tolist()}")
    print(f"Result (concatenated): {result['concatenated'].tolist()}")
    print(f"Expected: {expected.tolist()}")

    assert torch.allclose(result["concatenated"], expected)
    print("✓ torch.cat with 2 inputs works correctly\n")


def validate_torch_cat_three_inputs():
    """Validate torch.cat with 3+ inputs (critical for tuple handling)."""
    print("=" * 60)
    print("Test 4: torch.cat with 3+ Inputs (Critical Test)")
    print("=" * 60)

    model_spec = ModelSpec(
        inputs={
            "t1": ModelInput(dtype="float32", shape=[None, 1]),
            "t2": ModelInput(dtype="float32", shape=[None, 1]),
            "t3": ModelInput(dtype="float32", shape=[None, 1]),
        },
        graph=[
            GraphNode(
                name="cat_op",
                type="torch.cat",
                params={"dim": 1},
                inputs=["t1", "t2", "t3"],
            )
        ],
        outputs={"result": "cat_op.output"},
    )

    builder = ModelBuilder(enable_shape_validation=False)
    model = builder.build_model(model_spec)

    t1 = torch.tensor([[1.0]])
    t2 = torch.tensor([[2.0]])
    t3 = torch.tensor([[3.0]])

    result = model({"t1": t1, "t2": t2, "t3": t3})
    expected = torch.tensor([[1.0, 2.0, 3.0]])

    print(f"Input t1: {t1.tolist()}")
    print(f"Input t2: {t2.tolist()}")
    print(f"Input t3: {t3.tolist()}")
    print(f"Result: {result['result'].tolist()}")
    print(f"Expected: {expected.tolist()}")

    assert torch.allclose(result["result"], expected)
    print("✓ torch.cat with 3+ inputs works correctly\n")


def validate_torch_stack():
    """Validate torch.stack (another tuple operation)."""
    print("=" * 60)
    print("Test 5: torch.stack (Tuple Operation)")
    print("=" * 60)

    model_spec = ModelSpec(
        inputs={
            "a": ModelInput(dtype="float32", shape=[2]),
            "b": ModelInput(dtype="float32", shape=[2]),
            "c": ModelInput(dtype="float32", shape=[2]),
        },
        graph=[
            GraphNode(
                name="stack_op",
                type="torch.stack",
                params={"dim": 0},
                inputs=["a", "b", "c"],
            )
        ],
        outputs={"stacked": "stack_op.output"},
    )

    builder = ModelBuilder(enable_shape_validation=False)
    model = builder.build_model(model_spec)

    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    c = torch.tensor([5.0, 6.0])

    result = model({"a": a, "b": b, "c": c})
    expected = torch.stack([a, b, c], dim=0)

    print(f"Input a: {a.tolist()}")
    print(f"Input b: {b.tolist()}")
    print(f"Input c: {c.tolist()}")
    print(f"Result shape: {result['stacked'].shape}")
    print(f"Result:\n{result['stacked']}")
    print(f"Expected:\n{expected}")

    assert torch.allclose(result["stacked"], expected)
    print("✓ torch.stack works correctly\n")


def validate_mixed_operations():
    """Validate a model with both binary ops and tuple ops."""
    print("=" * 60)
    print("Test 6: Mixed Binary and Tuple Operations")
    print("=" * 60)

    # Model: add(x, y), mul(z, w), then cat results
    model_spec = ModelSpec(
        inputs={
            "x": ModelInput(dtype="float32", shape=[None, 2]),
            "y": ModelInput(dtype="float32", shape=[None, 2]),
            "z": ModelInput(dtype="float32", shape=[None, 2]),
            "w": ModelInput(dtype="float32", shape=[None, 2]),
        },
        graph=[
            GraphNode(
                name="add_op",
                type="torch.add",
                params={},
                inputs=["x", "y"],
            ),
            GraphNode(
                name="mul_op",
                type="torch.mul",
                params={},
                inputs=["z", "w"],
            ),
            GraphNode(
                name="cat_op",
                type="torch.cat",
                params={"dim": 1},
                inputs=["add_op.output", "mul_op.output"],
            ),
        ],
        outputs={"result": "cat_op.output"},
    )

    builder = ModelBuilder(enable_shape_validation=False)
    model = builder.build_model(model_spec)

    x = torch.tensor([[1.0, 2.0]])
    y = torch.tensor([[3.0, 4.0]])
    z = torch.tensor([[2.0, 3.0]])
    w = torch.tensor([[4.0, 5.0]])

    result = model({"x": x, "y": y, "z": z, "w": w})

    # x + y = [4.0, 6.0]
    # z * w = [8.0, 15.0]
    # cat([x+y, z*w]) = [4.0, 6.0, 8.0, 15.0]
    expected = torch.tensor([[4.0, 6.0, 8.0, 15.0]])

    print(f"x + y = {(x + y).tolist()}")
    print(f"z * w = {(z * w).tolist()}")
    print(f"cat([x+y, z*w]) = {result['result'].tolist()}")
    print(f"Expected: {expected.tolist()}")

    assert torch.allclose(result["result"], expected)
    print("✓ Mixed operations work correctly\n")


def validate_dcn_cross_layer():
    """Validate DCN cross layer pattern (real-world usage)."""
    print("=" * 60)
    print("Test 7: DCN Cross Layer Pattern (Real-World)")
    print("=" * 60)

    # DCN cross layer: x_l+1 = x_0 * (w_l^T x_l) + x_l
    cross_layer = ModuleDefinition(
        inputs=["x_current", "x_original"],
        outputs={"cross_output": "residual_add.output"},
        graph=[
            GraphNode(
                name="linear_transform",
                type="torch.nn.Linear",
                params={"in_features": 4, "out_features": 4},
                inputs={"input": "x_current"},
            ),
            GraphNode(
                name="element_wise_product",
                type="torch.mul",
                params={},
                inputs=["x_original", "linear_transform.output"],
            ),
            GraphNode(
                name="residual_add",
                type="torch.add",
                params={},
                inputs=["element_wise_product.output", "x_current"],
            ),
        ],
    )

    model_spec = ModelSpec(
        inputs={
            "x_current": ModelInput(dtype="float32", shape=[None, 4]),
            "x_original": ModelInput(dtype="float32", shape=[None, 4]),
        },
        graph=[
            GraphNode(
                name="cross1",
                type="module.cross_layer",
                params={},
                inputs=["x_current", "x_original"],
            )
        ],
        outputs={"output": "cross1.cross_output"},
        modules={"cross_layer": cross_layer},
    )

    builder = ModelBuilder(enable_shape_validation=False)
    model = builder.build_model(model_spec)

    x_current = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    x_original = torch.tensor([[0.5, 1.0, 1.5, 2.0]])

    result = model({"x_current": x_current, "x_original": x_original})

    print(f"x_current: {x_current.tolist()}")
    print(f"x_original: {x_original.tolist()}")
    print(f"Output shape: {result['output'].shape}")
    print(f"Output: {result['output'].tolist()}")

    # Just verify shape and no errors - actual values depend on Linear weights
    assert result["output"].shape == (1, 4)
    print("✓ DCN cross layer works correctly\n")


def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("BINARY OPERATIONS VALIDATION SUITE")
    print("=" * 60)
    print()

    try:
        validate_torch_add()
        validate_torch_mul()
        validate_torch_cat_two_inputs()
        validate_torch_cat_three_inputs()
        validate_torch_stack()
        validate_mixed_operations()
        validate_dcn_cross_layer()

        print("=" * 60)
        print("ALL VALIDATIONS PASSED ✓")
        print("=" * 60)
        print("\nBinary and multi-input operations work correctly:")
        print("  • Binary operations (torch.add, torch.mul)")
        print("  • Tuple operations (torch.cat, torch.stack)")
        print("  • torch.cat with 2 inputs")
        print("  • torch.cat with 3+ inputs")
        print("  • Mixed operation graphs")
        print("  • Real-world DCN cross layer pattern")

    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
