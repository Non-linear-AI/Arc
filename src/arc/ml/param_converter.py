"""Automatic parameter type conversion for PyTorch modules and loss functions.

Handles cases where YAML-parsed parameters need type conversion (e.g., float -> Tensor)
to match PyTorch's expected parameter types.
"""

from __future__ import annotations

import inspect
from typing import Any, get_args, get_origin

import torch


def convert_params_for_pytorch_module(
    module_class: type, params: dict[str, Any]
) -> dict[str, Any]:
    """Automatically convert parameters based on PyTorch module's type annotations.

    Uses inspect.signature to detect which parameters expect torch.Tensor types
    and automatically converts numeric values (int, float) to Tensors.

    Args:
        module_class: PyTorch module or loss function class
        params: Raw parameter dictionary from YAML

    Returns:
        Converted parameter dictionary with proper types

    Examples:
        >>> # BCEWithLogitsLoss has pos_weight: Optional[torch.Tensor]
        >>> params = {'pos_weight': 1.87, 'reduction': 'mean'}
        >>> converted = convert_params_for_pytorch_module(nn.BCEWithLogitsLoss, params)
        >>> # Returns: {'pos_weight': tensor(1.87), 'reduction': 'mean'}
    """
    if not params:
        return {}

    try:
        sig = inspect.signature(module_class.__init__)
    except (ValueError, TypeError):
        # Can't inspect signature, return params as-is
        return dict(params)

    converted = {}

    for param_name, param_value in params.items():
        # Skip if parameter not in signature
        if param_name not in sig.parameters:
            converted[param_name] = param_value
            continue

        param_info = sig.parameters[param_name]
        annotation = param_info.annotation

        # Skip if no type annotation
        if annotation == inspect.Parameter.empty:
            converted[param_name] = param_value
            continue

        # Check if parameter expects a Tensor
        if _expects_tensor(annotation):
            # Convert numeric values to Tensors
            if isinstance(param_value, (int, float)) and not isinstance(
                param_value, bool
            ):
                converted[param_name] = torch.tensor(float(param_value))
            elif isinstance(param_value, list):
                # Handle list of numbers -> Tensor
                converted[param_name] = torch.tensor(param_value)
            else:
                # Already a Tensor or incompatible type
                converted[param_name] = param_value
        else:
            # No conversion needed
            converted[param_name] = param_value

    return converted


def _expects_tensor(annotation: Any) -> bool:
    """Check if a type annotation expects a torch.Tensor.

    Handles:
    - torch.Tensor
    - Optional[torch.Tensor]
    - Union[torch.Tensor, ...]
    - Tensor | None (Python 3.10+)

    Args:
        annotation: Type annotation from inspect.signature

    Returns:
        True if annotation expects Tensor type
    """
    # Direct tensor type
    if annotation is torch.Tensor:
        return True

    # Handle Optional[Tensor] and Union[Tensor, ...]
    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        # Check if any of the union args is torch.Tensor
        return any(arg is torch.Tensor for arg in args)

    return False


def safe_instantiate_with_conversion(module_class: type, params: dict[str, Any]) -> Any:
    """Safely instantiate PyTorch module with automatic parameter conversion.

    Tries to instantiate with original params first. If that fails with a type
    error related to Tensors, automatically converts parameters and retries.

    This provides a fallback for cases where type annotations aren't available
    or are incomplete.

    Args:
        module_class: PyTorch module or loss function class
        params: Raw parameter dictionary

    Returns:
        Instantiated module

    Raises:
        Original exception if conversion doesn't help
    """
    # Try with annotation-based conversion first
    converted_params = convert_params_for_pytorch_module(module_class, params)

    try:
        return module_class(**converted_params)
    except (TypeError, RuntimeError) as e:
        error_msg = str(e)

        # Check if error is related to Tensor type requirements
        if "Tensor" in error_msg or "buffer" in error_msg:
            # Try converting ALL numeric params to Tensors as a fallback
            fallback_params = {}
            for key, value in params.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    fallback_params[key] = torch.tensor(float(value))
                elif isinstance(value, list):
                    try:
                        fallback_params[key] = torch.tensor(value)
                    except (TypeError, ValueError):
                        fallback_params[key] = value
                else:
                    fallback_params[key] = value

            # Retry with fallback conversion
            return module_class(**fallback_params)
        else:
            # Not a Tensor-related error, re-raise original
            raise
