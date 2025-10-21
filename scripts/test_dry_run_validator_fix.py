"""Test script to verify the dry-run validator tuple unpacking bug fix.

This script simulates the error that occurred when using functional losses
in the dry-run validator. The bug was that get_component_class_or_function
returns a tuple (component, component_kind) but the validator wasn't unpacking it.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def test_get_component_returns_tuple():
    """Verify get_component_class_or_function returns a tuple."""
    from arc.graph.model.components import get_component_class_or_function
    import torch.nn.functional as F

    # Test functional loss (the one that was failing)
    result = get_component_class_or_function(
        "torch.nn.functional.binary_cross_entropy_with_logits"
    )

    # Verify it returns a tuple
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected tuple of length 2, got {len(result)}"

    component, component_kind = result

    # Verify the component is the function
    assert (
        component is F.binary_cross_entropy_with_logits
    ), f"Expected binary_cross_entropy_with_logits, got {component}"

    # Verify the kind is 'function'
    assert component_kind == "function", f"Expected 'function', got {component_kind}"

    print("✓ get_component_class_or_function correctly returns tuple")


def test_loss_function_instantiation():
    """Verify the fixed validator code pattern works."""
    from arc.graph.model.components import get_component_class_or_function
    import torch

    # Simulate what the validator does (fixed version)
    loss_type = "torch.nn.functional.binary_cross_entropy_with_logits"
    loss_params = {}

    # This is the FIX - unpack the tuple
    loss_fn_class, component_kind = get_component_class_or_function(loss_type)

    # Verify component_kind is correct
    assert component_kind == "function"

    # Handle based on component kind (fixed logic)
    if component_kind == "function":
        loss_fn = loss_fn_class  # Functional loss - use directly
    else:
        loss_fn = loss_fn_class(**loss_params)  # Class loss - instantiate

    # Test that the loss function is callable
    assert callable(loss_fn), "Loss function should be callable"

    # Test with dummy tensors
    output = torch.randn(4, 1)
    target = torch.rand(4, 1)

    try:
        loss = loss_fn(output, target)
        assert isinstance(loss, torch.Tensor), f"Expected Tensor, got {type(loss)}"
        print(f"✓ Loss function works: {loss.item():.4f}")
    except Exception as e:
        print(f"✗ Loss function failed: {e}")
        raise


def test_class_based_loss():
    """Verify class-based losses also work with the fix."""
    from arc.graph.model.components import get_component_class_or_function
    import torch

    # Test with class-based loss
    loss_type = "torch.nn.BCEWithLogitsLoss"
    loss_params = {}

    # Unpack tuple
    loss_fn_class, component_kind = get_component_class_or_function(loss_type)

    # Verify component_kind is correct
    assert component_kind == "module", f"Expected 'module', got {component_kind}"

    # Handle based on component kind
    if component_kind == "function":
        loss_fn = loss_fn_class
    else:
        loss_fn = loss_fn_class(**loss_params)  # Instantiate class

    # Test with dummy tensors
    output = torch.randn(4, 1)
    target = torch.rand(4, 1)

    try:
        loss = loss_fn(output, target)
        assert isinstance(loss, torch.Tensor), f"Expected Tensor, got {type(loss)}"
        print(f"✓ Class-based loss works: {loss.item():.4f}")
    except Exception as e:
        print(f"✗ Class-based loss failed: {e}")
        raise


def test_old_buggy_code_pattern():
    """Show that the old buggy pattern fails."""
    from arc.graph.model.components import get_component_class_or_function
    import torch

    print("\n--- Testing OLD BUGGY pattern ---")
    loss_type = "torch.nn.functional.binary_cross_entropy_with_logits"

    # This is the BUG - not unpacking the tuple
    loss_fn_class = get_component_class_or_function(loss_type)  # Returns tuple!

    print(f"loss_fn_class type: {type(loss_fn_class)}")
    print(f"loss_fn_class value: {loss_fn_class}")

    # The buggy code checked hasattr on a tuple
    has_self = hasattr(loss_fn_class, "__self__")
    print(f"hasattr(tuple, '__self__'): {has_self}")

    # This would fail because we're trying to use a tuple as a function
    try:
        output = torch.randn(4, 1)
        target = torch.rand(4, 1)
        loss = loss_fn_class(output, target)  # This calls the tuple!
        print(f"✗ Unexpected success: {loss}")
    except TypeError as e:
        if "'tuple' object is not callable" in str(e):
            print(f"✓ Confirmed bug: {e}")
        else:
            print(f"✗ Different error: {e}")
            raise


if __name__ == "__main__":
    print("Testing dry-run validator tuple unpacking fix\n")

    test_get_component_returns_tuple()
    test_loss_function_instantiation()
    test_class_based_loss()
    test_old_buggy_code_pattern()

    print("\n✅ All tests passed! The fix is working correctly.")
