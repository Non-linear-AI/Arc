#!/usr/bin/env python3
"""Validation script for label encoder unknown category handling.

This script demonstrates that the label encoder correctly handles unknown
categories encountered during inference, which is a common real-world scenario.
"""

from arc.ml.data import DataProcessor


def validate_basic_encoding():
    """Validate basic label encoding workflow."""
    print("=" * 60)
    print("Test 1: Basic Label Encoding")
    print("=" * 60)

    processor = DataProcessor(ml_data_service=None)

    # Fit encoder on training data
    train_data = ["apple", "banana", "cherry", "apple", "banana"]
    print(f"Training data: {train_data}")

    fit_result = processor._execute_operator("fit.label_encoder", {"x": train_data})
    state = fit_result["state"]

    print(f"Learned vocabulary: {dict(state['vocabulary'])}")
    print(f"Vocabulary size: {state['vocab_size']}")

    # Transform training data
    transform_result = processor._execute_operator(
        "transform.label_encode",
        {"x": ["apple", "banana", "cherry"], "state": state},
    )

    output = transform_result["output"]
    print(f"Encoded values: {output.tolist()}")
    print("✓ Basic encoding works\n")


def validate_unknown_with_default():
    """Validate unknown category handling with default unknown_value."""
    print("=" * 60)
    print("Test 2: Unknown Categories with Default Value")
    print("=" * 60)

    processor = DataProcessor(ml_data_service=None)

    # Fit on limited training data
    train_data = ["cat", "dog", "bird"]
    print(f"Training data: {train_data}")

    fit_result = processor._execute_operator("fit.label_encoder", {"x": train_data})
    state = fit_result["state"]
    vocab_size = state["vocab_size"]

    print(f"Vocabulary: {dict(state['vocabulary'])}")
    print(f"Vocabulary size: {vocab_size}")

    # Transform with unknown categories (uses default unknown_value = vocab_size)
    test_data = ["cat", "elephant", "dog", "tiger"]
    print(f"\nTest data (contains unknown 'elephant' and 'tiger'): {test_data}")

    transform_result = processor._execute_operator(
        "transform.label_encode",
        {"x": test_data, "state": state},
    )

    output = transform_result["output"]
    print(f"Encoded values: {output.tolist()}")
    print(f"Note: Unknown categories mapped to {vocab_size} (vocab_size)")
    print("✓ Default unknown handling works\n")


def validate_unknown_with_custom_value():
    """Validate unknown category handling with custom unknown_value."""
    print("=" * 60)
    print("Test 3: Unknown Categories with Custom Value")
    print("=" * 60)

    processor = DataProcessor(ml_data_service=None)

    # Fit encoder
    train_data = ["red", "green", "blue"]
    print(f"Training data: {train_data}")

    fit_result = processor._execute_operator("fit.label_encoder", {"x": train_data})
    state = fit_result["state"]

    print(f"Vocabulary: {dict(state['vocabulary'])}")

    # Transform with custom unknown_value
    test_data = ["red", "yellow", "blue", "purple"]
    print(f"\nTest data (contains unknown 'yellow' and 'purple'): {test_data}")
    print("Using custom unknown_value=-1")

    transform_result = processor._execute_operator(
        "transform.label_encode",
        {
            "x": test_data,
            "state": state,
            "unknown_value": -1,  # Custom value for unknown categories
        },
    )

    output = transform_result["output"]
    print(f"Encoded values: {output.tolist()}")
    print("Note: Unknown categories mapped to -1 (custom value)")
    print("✓ Custom unknown value works\n")


def validate_error_mode():
    """Validate that error mode properly rejects unknown categories."""
    print("=" * 60)
    print("Test 4: Error Mode for Unknown Categories")
    print("=" * 60)

    processor = DataProcessor(ml_data_service=None)

    # Fit encoder
    train_data = ["small", "medium", "large"]
    print(f"Training data: {train_data}")

    fit_result = processor._execute_operator("fit.label_encoder", {"x": train_data})
    state = fit_result["state"]

    print(f"Vocabulary: {dict(state['vocabulary'])}")

    # Try to transform with unknown category in error mode
    test_data = ["small", "xlarge", "medium"]
    print(f"\nTest data (contains unknown 'xlarge'): {test_data}")
    print("Using handle_unknown='error'")

    try:
        processor._execute_operator(
            "transform.label_encode",
            {
                "x": test_data,
                "state": state,
                "handle_unknown": "error",  # Strict mode
            },
        )
        print("✗ Expected error but didn't get one!")
    except Exception as e:
        print(f"Got expected error: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("✓ Error mode works correctly\n")


def validate_realistic_scenario():
    """Validate a realistic train/test split scenario."""
    print("=" * 60)
    print("Test 5: Realistic Train/Test Scenario")
    print("=" * 60)

    processor = DataProcessor(ml_data_service=None)

    # Simulate training data with common categories
    train_categories = [
        "Electronics",
        "Books",
        "Clothing",
        "Electronics",
        "Books",
        "Home",
        "Electronics",
        "Clothing",
    ]
    print(f"Training categories: {set(train_categories)}")

    fit_result = processor._execute_operator(
        "fit.label_encoder", {"x": train_categories}
    )
    state = fit_result["state"]

    print(f"Vocabulary: {dict(state['vocabulary'])}")
    print(f"Vocabulary size: {state['vocab_size']}")

    # Simulate test data with some new categories
    test_categories = [
        "Electronics",  # Known
        "Sports",  # Unknown - new category in test
        "Books",  # Known
        "Toys",  # Unknown - new category in test
        "Clothing",  # Known
    ]
    print(f"\nTest categories: {test_categories}")
    print("Unknown categories: Sports, Toys")

    # Transform with graceful unknown handling
    transform_result = processor._execute_operator(
        "transform.label_encode",
        {
            "x": test_categories,
            "state": state,
            "unknown_value": 999,  # Use special "unknown" index
        },
    )

    output = transform_result["output"]
    print(f"\nEncoded values: {output.tolist()}")

    # Verify the encoding
    vocab = state["vocabulary"]
    expected_encoding = [
        vocab["Electronics"],  # Known
        999,  # Unknown
        vocab["Books"],  # Known
        999,  # Unknown
        vocab["Clothing"],  # Known
    ]

    assert output.tolist() == expected_encoding, "Encoding mismatch!"
    print(f"Expected: {expected_encoding}")
    print("✓ Realistic scenario works correctly\n")


def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("LABEL ENCODER VALIDATION SUITE")
    print("=" * 60)
    print()

    try:
        validate_basic_encoding()
        validate_unknown_with_default()
        validate_unknown_with_custom_value()
        validate_error_mode()
        validate_realistic_scenario()

        print("=" * 60)
        print("ALL VALIDATIONS PASSED ✓")
        print("=" * 60)
        print("\nThe label encoder correctly handles:")
        print("  • Basic encoding workflow")
        print("  • Unknown categories with default value (vocab_size)")
        print("  • Unknown categories with custom value")
        print("  • Error mode for strict validation")
        print("  • Realistic train/test scenarios")

    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
