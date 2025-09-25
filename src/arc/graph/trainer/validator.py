"""Validation for trainer specifications."""

from __future__ import annotations

from typing import Any

from arc.graph.trainer.components import (
    get_loss_class,
    get_optimizer_class,
    validate_loss_params,
    validate_optimizer_params,
)


class TrainerValidationError(ValueError):
    """Exception raised for trainer validation errors."""

    pass


def _require(obj: dict[str, Any], key: str, msg: str | None = None) -> Any:
    """Helper function for requiring dictionary keys."""
    if key not in obj:
        raise TrainerValidationError(msg or f"Missing required field: {key}")
    return obj[key]


def validate_trainer_dict(data: dict[str, Any]) -> None:
    """Validate a parsed YAML dict for trainer specification.

    Args:
        data: Dictionary containing trainer specification

    Raises:
        TrainerValidationError: If validation fails
    """
    # Validate optimizer
    optimizer = _require(data, "optimizer", "trainer.optimizer required")
    if not isinstance(optimizer, dict):
        raise TrainerValidationError("trainer.optimizer must be a mapping")

    optimizer_type = _require(optimizer, "type", "trainer.optimizer.type required")

    # Validate optimizer type is supported
    try:
        get_optimizer_class(optimizer_type)
    except ValueError as e:
        raise TrainerValidationError(f"trainer.optimizer: {e}") from e

    # Validate learning rate
    lr = optimizer.get("lr", 0.001)
    if not isinstance(lr, (int, float)) or lr <= 0:
        raise TrainerValidationError(
            f"trainer.optimizer.lr must be a positive number, got: {lr}"
        )

    # Validate optimizer parameters if present
    if "params" in optimizer and optimizer["params"] is not None:
        optimizer_params = dict(optimizer["params"])
        optimizer_params["lr"] = lr  # Add lr to params for validation
        try:
            validate_optimizer_params(optimizer_type, optimizer_params)
        except ValueError as e:
            raise TrainerValidationError(f"trainer.optimizer.params: {e}") from e

    # Validate loss
    loss = _require(data, "loss", "trainer.loss required")
    if not isinstance(loss, dict):
        raise TrainerValidationError("trainer.loss must be a mapping")

    loss_type = _require(loss, "type", "trainer.loss.type required")

    # Validate loss type is supported
    try:
        get_loss_class(loss_type)
    except ValueError as e:
        raise TrainerValidationError(f"trainer.loss: {e}") from e

    # Validate loss inputs if present
    if "inputs" in loss:
        inputs = loss["inputs"]
        if inputs is not None and not isinstance(inputs, dict):
            raise TrainerValidationError("trainer.loss.inputs must be a mapping")

    # Validate loss parameters if present
    if "params" in loss and loss["params"] is not None:
        try:
            validate_loss_params(loss_type, loss["params"])
        except ValueError as e:
            raise TrainerValidationError(f"trainer.loss.params: {e}") from e

    # Validate config (optional)
    if "config" in data and data["config"] is not None:
        config = data["config"]
        if not isinstance(config, dict):
            raise TrainerValidationError("trainer.config must be a mapping")

        # Validate config parameters
        if "epochs" in config:
            epochs = config["epochs"]
            if not isinstance(epochs, int) or epochs <= 0:
                raise TrainerValidationError(
                    f"trainer.config.epochs must be a positive integer, got: {epochs}"
                )

        if "batch_size" in config:
            batch_size = config["batch_size"]
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise TrainerValidationError(
                    f"trainer.config.batch_size must be a positive integer, "
                    f"got: {batch_size}"
                )

        if "validation_split" in config:
            val_split = config["validation_split"]
            if not isinstance(val_split, (int, float)) or not (0.0 <= val_split < 1.0):
                raise TrainerValidationError(
                    f"trainer.config.validation_split must be between 0 and 1, "
                    f"got: {val_split}"
                )

        if "device" in config:
            device = config["device"]
            if not isinstance(device, str):
                raise TrainerValidationError(
                    f"trainer.config.device must be a string, got: {device}"
                )
            if device not in ("auto", "cpu", "cuda", "mps"):
                raise TrainerValidationError(
                    f"trainer.config.device must be one of 'auto', 'cpu', 'cuda', "
                    f"'mps', got: {device}"
                )

        if "early_stopping_patience" in config:
            patience = config["early_stopping_patience"]
            if patience is not None and (
                not isinstance(patience, int) or patience <= 0
            ):
                raise TrainerValidationError(
                    f"trainer.config.early_stopping_patience must be a positive "
                    f"integer or null, got: {patience}"
                )


def validate_trainer_components(data: dict[str, Any]) -> list[str]:
    """Validate trainer components against available types.

    Args:
        data: Trainer specification dictionary

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Validate optimizer type
    if "optimizer" in data and isinstance(data["optimizer"], dict):
        optimizer_type = data["optimizer"].get("type")
        if optimizer_type:
            try:
                get_optimizer_class(optimizer_type)
            except ValueError:
                errors.append(f"Unsupported optimizer type: {optimizer_type}")

    # Validate loss type
    if "loss" in data and isinstance(data["loss"], dict):
        loss_type = data["loss"].get("type")
        if loss_type:
            try:
                get_loss_class(loss_type)
            except ValueError:
                errors.append(f"Unsupported loss type: {loss_type}")

    return errors


def validate_loss_inputs_against_model(
    loss_inputs: dict[str, str], model_outputs: dict[str, str]
) -> list[str]:
    """Validate loss inputs reference valid model outputs.

    Args:
        loss_inputs: Loss input mapping from trainer spec
        model_outputs: Model output mapping from model spec

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    for input_name, source_ref in loss_inputs.items():
        if source_ref.startswith("model."):
            # Reference to model output
            output_name = source_ref[6:]  # Remove "model." prefix
            if output_name not in model_outputs:
                errors.append(
                    f"Loss input '{input_name}' references undefined model output: "
                    f"{source_ref}"
                )
        # Other references (e.g., to target columns) are validated elsewhere

    return errors
