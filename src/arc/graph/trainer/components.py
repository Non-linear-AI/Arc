"""Core PyTorch trainer components with torch.optim and torch.nn prefixes for
Arc-Graph."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
import torch.optim as optim

# Direct PyTorch optimizer mapping with torch.optim prefix
CORE_OPTIMIZERS = {
    "torch.optim.SGD": optim.SGD,
    "torch.optim.Adam": optim.Adam,
    "torch.optim.AdamW": optim.AdamW,
    "torch.optim.RMSprop": optim.RMSprop,
    "torch.optim.Adagrad": optim.Adagrad,
    "torch.optim.Adadelta": optim.Adadelta,
    "torch.optim.Adamax": optim.Adamax,
    "torch.optim.ASGD": optim.ASGD,
    "torch.optim.NAdam": optim.NAdam,
    "torch.optim.RAdam": optim.RAdam,
    "torch.optim.LBFGS": optim.LBFGS,
    "torch.optim.Rprop": optim.Rprop,
    "torch.optim.SparseAdam": optim.SparseAdam,
}

# Direct PyTorch loss function mapping with torch.nn prefix
CORE_LOSSES = {
    # Classification losses
    "torch.nn.CrossEntropyLoss": nn.CrossEntropyLoss,
    "torch.nn.NLLLoss": nn.NLLLoss,
    "torch.nn.BCELoss": nn.BCELoss,
    "torch.nn.BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    "torch.nn.MultiLabelMarginLoss": nn.MultiLabelMarginLoss,
    "torch.nn.MultiLabelSoftMarginLoss": nn.MultiLabelSoftMarginLoss,
    "torch.nn.MultiMarginLoss": nn.MultiMarginLoss,
    # Regression losses
    "torch.nn.MSELoss": nn.MSELoss,
    "torch.nn.L1Loss": nn.L1Loss,
    "torch.nn.SmoothL1Loss": nn.SmoothL1Loss,
    "torch.nn.HuberLoss": nn.HuberLoss,
    "torch.nn.PoissonNLLLoss": nn.PoissonNLLLoss,
    # Ranking and embedding losses
    "torch.nn.MarginRankingLoss": nn.MarginRankingLoss,
    "torch.nn.HingeEmbeddingLoss": nn.HingeEmbeddingLoss,
    "torch.nn.TripletMarginLoss": nn.TripletMarginLoss,
    "torch.nn.TripletMarginWithDistanceLoss": nn.TripletMarginWithDistanceLoss,
    "torch.nn.CosineEmbeddingLoss": nn.CosineEmbeddingLoss,
    # Distributional losses
    "torch.nn.KLDivLoss": nn.KLDivLoss,
    "torch.nn.GaussianNLLLoss": nn.GaussianNLLLoss,
    "torch.nn.CTCLoss": nn.CTCLoss,
}


def get_optimizer_class(optimizer_type: str) -> type[optim.Optimizer]:
    """Get PyTorch optimizer class by type name.

    Args:
        optimizer_type: Optimizer type name (e.g., "pytorch.Adam")

    Returns:
        PyTorch optimizer class

    Raises:
        ValueError: If optimizer type is not supported
    """
    if optimizer_type not in CORE_OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    return CORE_OPTIMIZERS[optimizer_type]


def get_loss_class(loss_type: str) -> type[nn.Module]:
    """Get PyTorch loss function class by type name.

    Args:
        loss_type: Loss function type name (e.g., "pytorch.CrossEntropyLoss")

    Returns:
        PyTorch loss function class

    Raises:
        ValueError: If loss type is not supported
    """
    if loss_type not in CORE_LOSSES:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    return CORE_LOSSES[loss_type]


def validate_optimizer_params(optimizer_type: str, params: dict[str, Any]) -> bool:
    """Validate optimizer parameters for a given optimizer type.

    Args:
        optimizer_type: Optimizer type name (e.g., "pytorch.Adam")
        params: Parameters dictionary

    Returns:
        True if parameters are valid

    Raises:
        ValueError: If parameters are invalid
    """
    # Learning rate is required for all optimizers
    if "lr" not in params:
        raise ValueError(f"Missing required parameter 'lr' for {optimizer_type}")

    lr = params["lr"]
    if not isinstance(lr, (int, float)) or lr <= 0:
        raise ValueError(f"Learning rate must be a positive number, got: {lr}")

    # Optimizer-specific validation
    if optimizer_type == "torch.optim.SGD":
        if "momentum" in params:
            momentum = params["momentum"]
            if not isinstance(momentum, (int, float)) or not (0.0 <= momentum <= 1.0):
                raise ValueError(
                    f"SGD momentum must be between 0 and 1, got: {momentum}"
                )

    elif optimizer_type == "torch.optim.Adam":
        if "betas" in params:
            betas = params["betas"]
            if not isinstance(betas, (list, tuple)) or len(betas) != 2:
                raise ValueError(
                    f"Adam betas must be a tuple of 2 values, got: {betas}"
                )
            if not all(0.0 <= b < 1.0 for b in betas):
                raise ValueError(f"Adam betas must be between 0 and 1, got: {betas}")

    elif optimizer_type == "torch.optim.RMSprop" and "alpha" in params:
        alpha = params["alpha"]
        if not isinstance(alpha, (int, float)) or alpha <= 0:
            raise ValueError(f"RMSprop alpha must be positive, got: {alpha}")

    return True


def validate_loss_params(loss_type: str, params: dict[str, Any]) -> bool:
    """Validate loss function parameters for a given loss type.

    Args:
        loss_type: Loss function type name (e.g., "pytorch.CrossEntropyLoss")
        params: Parameters dictionary

    Returns:
        True if parameters are valid

    Raises:
        ValueError: If parameters are invalid
    """
    # Most loss functions can be instantiated without parameters
    # Add specific validation as needed

    if loss_type == "torch.nn.CrossEntropyLoss":
        if "ignore_index" in params:
            ignore_index = params["ignore_index"]
            if not isinstance(ignore_index, int):
                raise ValueError(
                    f"CrossEntropyLoss ignore_index must be an integer, "
                    f"got: {ignore_index}"
                )

    elif loss_type == "torch.nn.SmoothL1Loss":
        if "beta" in params:
            beta = params["beta"]
            if not isinstance(beta, (int, float)) or beta <= 0:
                raise ValueError(f"SmoothL1Loss beta must be positive, got: {beta}")

    elif loss_type == "torch.nn.HuberLoss" and "delta" in params:
        delta = params["delta"]
        if not isinstance(delta, (int, float)) or delta <= 0:
            raise ValueError(f"HuberLoss delta must be positive, got: {delta}")

    return True


def get_supported_optimizer_types() -> list[str]:
    """Get list of all supported optimizer types.

    Returns:
        List of supported optimizer type names
    """
    return list(CORE_OPTIMIZERS.keys())


def get_supported_loss_types() -> list[str]:
    """Get list of all supported loss function types.

    Returns:
        List of supported loss function type names
    """
    return list(CORE_LOSSES.keys())
