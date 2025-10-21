"""Test script to verify trainer loss function fix.

This script tests that the trainer now supports both functional and class-based
losses using the component registry system.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def test_functional_loss():
    """Test that functional losses work with the new setup."""
    from arc.ml.trainer import ArcTrainer
    import torch

    # Create a mock config with functional loss
    config = SimpleNamespace(
        loss_function="torch.nn.functional.binary_cross_entropy_with_logits",
        loss_params={},
        device="cpu",
        optimizer="torch.optim.Adam",
        optimizer_params={},
        learning_rate=0.001,
        epochs=1,
        batch_size=4,
        validation_split=0.2,
        early_stopping_patience=None,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        checkpoint_every=10,
        save_best_only=False,
        save_dir=None,
        early_stopping_min_delta=0.0001,
        early_stopping_monitor="val_loss",
        early_stopping_mode="min",
        log_every=10,
        verbose=False,
        gradient_clip_val=None,
        gradient_clip_norm=None,
        accumulate_grad_batches=1,
        seed=None,
        reshape_targets=False,
        target_output_key=None,
    )

    trainer = ArcTrainer(config)

    # Test loss function setup
    loss_fn = trainer._setup_loss_function()

    # Verify it's callable
    assert callable(loss_fn), "Loss function should be callable"

    # Test with dummy tensors
    output = torch.randn(4, 1)
    target = torch.rand(4, 1)

    loss = loss_fn(output, target)
    assert isinstance(loss, torch.Tensor), f"Expected Tensor, got {type(loss)}"

    print(f"✓ Functional loss works: {loss.item():.4f}")


def test_class_based_loss():
    """Test that class-based losses work with the new setup."""
    from arc.ml.trainer import ArcTrainer
    import torch

    # Create a mock config with class-based loss
    config = SimpleNamespace(
        loss_function="torch.nn.BCEWithLogitsLoss",
        loss_params={},
        device="cpu",
        optimizer="torch.optim.Adam",
        optimizer_params={},
        learning_rate=0.001,
        epochs=1,
        batch_size=4,
        validation_split=0.2,
        early_stopping_patience=None,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        checkpoint_every=10,
        save_best_only=False,
        save_dir=None,
        early_stopping_min_delta=0.0001,
        early_stopping_monitor="val_loss",
        early_stopping_mode="min",
        log_every=10,
        verbose=False,
        gradient_clip_val=None,
        gradient_clip_norm=None,
        accumulate_grad_batches=1,
        seed=None,
        reshape_targets=False,
        target_output_key=None,
    )

    trainer = ArcTrainer(config)

    # Test loss function setup
    loss_fn = trainer._setup_loss_function()

    # Verify it's callable (should be an instance of BCEWithLogitsLoss)
    assert callable(loss_fn), "Loss function should be callable"
    assert hasattr(loss_fn, "__call__"), "Loss function should have __call__ method"

    # Test with dummy tensors
    output = torch.randn(4, 1)
    target = torch.rand(4, 1)

    loss = loss_fn(output, target)
    assert isinstance(loss, torch.Tensor), f"Expected Tensor, got {type(loss)}"

    print(f"✓ Class-based loss works: {loss.item():.4f}")


def test_class_based_loss_with_params():
    """Test that class-based losses with params work."""
    from arc.ml.trainer import ArcTrainer
    import torch

    # Create a mock config with class-based loss and params
    config = SimpleNamespace(
        loss_function="torch.nn.BCEWithLogitsLoss",
        loss_params={"reduction": "sum"},  # Test with params
        device="cpu",
        optimizer="torch.optim.Adam",
        optimizer_params={},
        learning_rate=0.001,
        epochs=1,
        batch_size=4,
        validation_split=0.2,
        early_stopping_patience=None,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        checkpoint_every=10,
        save_best_only=False,
        save_dir=None,
        early_stopping_min_delta=0.0001,
        early_stopping_monitor="val_loss",
        early_stopping_mode="min",
        log_every=10,
        verbose=False,
        gradient_clip_val=None,
        gradient_clip_norm=None,
        accumulate_grad_batches=1,
        seed=None,
        reshape_targets=False,
        target_output_key=None,
    )

    trainer = ArcTrainer(config)

    # Test loss function setup
    loss_fn = trainer._setup_loss_function()

    # Test with dummy tensors
    output = torch.randn(4, 1)
    target = torch.rand(4, 1)

    loss = loss_fn(output, target)
    assert isinstance(loss, torch.Tensor), f"Expected Tensor, got {type(loss)}"

    # With reduction='sum', the loss should be larger than with reduction='mean'
    print(f"✓ Class-based loss with params works: {loss.item():.4f}")


def test_mse_functional_loss():
    """Test MSE functional loss."""
    from arc.ml.trainer import ArcTrainer
    import torch

    config = SimpleNamespace(
        loss_function="torch.nn.functional.mse_loss",
        loss_params={},
        device="cpu",
        optimizer="torch.optim.Adam",
        optimizer_params={},
        learning_rate=0.001,
        epochs=1,
        batch_size=4,
        validation_split=0.2,
        early_stopping_patience=None,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        checkpoint_every=10,
        save_best_only=False,
        save_dir=None,
        early_stopping_min_delta=0.0001,
        early_stopping_monitor="val_loss",
        early_stopping_mode="min",
        log_every=10,
        verbose=False,
        gradient_clip_val=None,
        gradient_clip_norm=None,
        accumulate_grad_batches=1,
        seed=None,
        reshape_targets=False,
        target_output_key=None,
    )

    trainer = ArcTrainer(config)

    loss_fn = trainer._setup_loss_function()

    output = torch.randn(4, 3)
    target = torch.randn(4, 3)

    loss = loss_fn(output, target)
    assert isinstance(loss, torch.Tensor), f"Expected Tensor, got {type(loss)}"

    print(f"✓ MSE functional loss works: {loss.item():.4f}")


if __name__ == "__main__":
    print("Testing trainer loss function fix\n")

    test_functional_loss()
    test_class_based_loss()
    test_class_based_loss_with_params()
    test_mse_functional_loss()

    print("\n✅ All tests passed! The trainer now supports both functional and class-based losses.")
