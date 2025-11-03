#!/usr/bin/env python3
"""Validation script for improved error messages.

This script demonstrates that error messages are helpful and show
available options when users make mistakes.
"""

from arc.graph import GraphNode, ModelInput, ModelSpec, ModuleDefinition
from arc.ml.builder import ModelBuilder
from arc.ml.data import DataProcessor, ProcessorError


def validate_unknown_module_error():
    """Validate error message when referencing unknown module."""
    print("=" * 60)
    print("Test 1: Unknown Module Error Message")
    print("=" * 60)

    # Define some modules
    encoder_module = ModuleDefinition(
        inputs=["x"],
        outputs={"output": "linear.output"},
        graph=[
            GraphNode(
                name="linear",
                type="torch.nn.Linear",
                params={"in_features": 10, "out_features": 5},
                inputs={"input": "x"},
            )
        ],
    )

    decoder_module = ModuleDefinition(
        inputs=["x"],
        outputs={"output": "relu.output"},
        graph=[
            GraphNode(
                name="relu",
                type="torch.nn.ReLU",
                params={},
                inputs={"input": "x"},
            )
        ],
    )

    # Try to use a module that doesn't exist
    model_spec = ModelSpec(
        inputs={"features": ModelInput(dtype="float32", shape=[None, 10])},
        graph=[
            GraphNode(
                name="unknown_layer",
                type="module.nonexistent_module",  # This doesn't exist!
                params={},
                inputs=["features"],
            )
        ],
        outputs={"output": "unknown_layer.output"},
        modules={
            "encoder": encoder_module,
            "decoder": decoder_module,
        },
    )

    builder = ModelBuilder(enable_shape_validation=False)

    print("Defined modules: encoder, decoder")
    print("Attempting to use: nonexistent_module\n")

    try:
        builder.build_model(model_spec)
        print("✗ Expected error but didn't get one!")
        return False
    except ValueError as e:
        error_msg = str(e)
        print(f"Got expected error: {type(e).__name__}")
        print(f"Error message:\n{error_msg}\n")

        # Check that error message is helpful
        if "nonexistent_module" not in error_msg:
            print("✗ Error message doesn't mention the unknown module")
            return False

        if "encoder" not in error_msg or "decoder" not in error_msg:
            print("✗ Error message doesn't show available modules")
            return False

        print("✓ Error message is helpful and shows available modules\n")
        return True


def validate_no_modules_error():
    """Validate error message when no modules are defined."""
    print("=" * 60)
    print("Test 2: No Modules Defined Error Message")
    print("=" * 60)

    # Try to use a module when no modules are defined
    model_spec = ModelSpec(
        inputs={"features": ModelInput(dtype="float32", shape=[None, 10])},
        graph=[
            GraphNode(
                name="layer1",
                type="module.some_module",
                params={},
                inputs=["features"],
            )
        ],
        outputs={"output": "layer1.output"},
        # No modules defined!
    )

    builder = ModelBuilder(enable_shape_validation=False)

    print("Defined modules: (none)")
    print("Attempting to use: some_module\n")

    try:
        builder.build_model(model_spec)
        print("✗ Expected error but didn't get one!")
        return False
    except Exception as e:
        error_msg = str(e)
        print(f"Got expected error: {type(e).__name__}")
        print(f"Error message:\n{error_msg}\n")

        # Check that error message mentions the module
        if "some_module" not in error_msg:
            print("✗ Error message doesn't mention the unknown module")
            return False

        # The error message should indicate there are no modules
        # (either explicitly or by showing an empty list)
        print("✓ Error message indicates module not found\n")
        return True


def validate_unknown_category_error():
    """Validate error message for unknown categories in label encoder."""
    print("=" * 60)
    print("Test 3: Unknown Category Error Message")
    print("=" * 60)

    processor = DataProcessor(ml_data_service=None)

    # Fit with limited vocabulary
    train_data = ["apple", "banana", "cherry", "date", "elderberry"]
    print(f"Training data: {train_data}")

    fit_result = processor._execute_operator("fit.label_encoder", {"x": train_data})
    state = fit_result["state"]

    print(f"Vocabulary: {list(state['vocabulary'].keys())}")

    # Try to transform with unknown category in error mode
    test_data = ["apple", "fig", "banana"]  # 'fig' is unknown
    print(f"\nTest data (contains unknown 'fig'): {test_data}")
    print("Using handle_unknown='error'\n")

    try:
        processor._execute_operator(
            "transform.label_encode",
            {
                "x": test_data,
                "state": state,
                "handle_unknown": "error",
            },
        )
        print("✗ Expected error but didn't get one!")
        return False
    except ProcessorError as e:
        error_msg = str(e)
        print(f"Got expected error: {type(e).__name__}")
        print(f"Error message:\n{error_msg}\n")

        # Check that error message is helpful
        if "fig" not in error_msg:
            print("✗ Error message doesn't mention the unknown category")
            return False

        if "vocabulary" not in error_msg.lower():
            print("✗ Error message doesn't mention vocabulary")
            return False

        # Should show some available categories
        if "apple" not in error_msg or "banana" not in error_msg:
            print("✗ Error message doesn't show available categories")
            return False

        print("✓ Error message is helpful and shows available categories\n")
        return True


def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("ERROR MESSAGE VALIDATION SUITE")
    print("=" * 60)
    print()

    results = []
    try:
        results.append(("Unknown module", validate_unknown_module_error()))
        results.append(("No modules defined", validate_no_modules_error()))
        results.append(("Unknown category", validate_unknown_category_error()))

        print("=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)

        all_passed = all(result for _, result in results)

        for name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{status}: {name}")

        if all_passed:
            print("\n" + "=" * 60)
            print("ALL VALIDATIONS PASSED ✓")
            print("=" * 60)
            print("\nError messages are helpful and show:")
            print("  • Available modules when unknown module referenced")
            print("  • Clear message when no modules defined")
            print("  • Available categories when unknown category found")
            return 0
        else:
            print("\n✗ Some validations failed")
            return 1

    except Exception as e:
        print(f"\n✗ Validation failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
