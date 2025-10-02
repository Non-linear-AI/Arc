#!/usr/bin/env python3
"""Test script for ML Plan workflow."""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import just the ml_plan module directly
import arc.core.ml_plan as ml_plan_module

MLPlan = ml_plan_module.MLPlan
compute_plan_diff = ml_plan_module.compute_plan_diff


def test_ml_plan_creation():
    """Test creating an ML plan from analysis."""
    print("=" * 60)
    print("Test 1: ML Plan Creation")
    print("=" * 60)

    analysis = {
        "summary": "Binary classification task predicting diabetes risk from 8 tabular features using a simple MLP architecture.",
        "selected_components": ["mlp"],
        "feature_engineering": "Normalize all numerical features using StandardScaler. The 8 features are moderate in number and don't require complex interactions.",
        "model_architecture_and_loss": "Use a simple MLP with 2-3 hidden layers with dimensions [64, 32]. Binary cross-entropy loss is the standard choice.",
        "training_configuration": "Use Adam optimizer with learning rate 0.001. Apply dropout (0.2-0.3) between layers. Train for 50-100 epochs with early stopping.",
        "evaluation": "Primary metric: AUC-ROC. Secondary metrics: F1 score for balance between precision/recall.",
        "confirmation_question": "Does this approach look good to you?",
    }

    plan = MLPlan.from_analysis(analysis, version=1, stage="initial")

    print("\nCreated Plan:")
    print(plan.format_for_display())

    print("\n✓ Test 1 passed\n")
    return plan


def test_ml_plan_revision():
    """Test revising an ML plan with diff."""
    print("=" * 60)
    print("Test 2: ML Plan Revision with Diff")
    print("=" * 60)

    # Create initial plan
    initial_analysis = {
        "summary": "Binary classification task predicting diabetes risk from 8 tabular features using a simple MLP architecture.",
        "selected_components": ["mlp"],
        "feature_engineering": "Normalize all numerical features using StandardScaler.",
        "model_architecture_and_loss": "Use a simple MLP with hidden layers [64, 32]. Binary cross-entropy loss.",
        "training_configuration": "Adam optimizer with learning rate 0.001. Dropout 0.2-0.3.",
        "evaluation": "Primary metric: AUC-ROC.",
        "confirmation_question": "Does this approach look good?",
    }

    plan_v1 = MLPlan.from_analysis(initial_analysis, version=1, stage="initial")

    # Create revised plan
    revised_analysis = {
        "summary": "Binary classification task predicting diabetes risk - increased model capacity based on training results.",
        "selected_components": ["mlp", "dcn"],  # Added DCN
        "feature_engineering": "Normalize all numerical features using StandardScaler. Add polynomial features for interactions.",  # Enhanced
        "model_architecture_and_loss": "Use MLP with hidden layers [128, 64, 32] for increased capacity. Binary cross-entropy loss.",  # Increased capacity
        "training_configuration": "Adam optimizer with learning rate 0.0005 (reduced). Dropout 0.3. Add L2 regularization.",  # Adjusted
        "evaluation": "Primary metric: AUC-ROC. Secondary: F1 score, precision, recall.",  # Enhanced
        "confirmation_question": "Approve these capacity and regularization improvements?",
    }

    plan_v2 = MLPlan.from_analysis(
        revised_analysis,
        version=2,
        stage="post_training",
        reason="Training showed underfitting (train_loss=0.45, val_loss=0.48)",
    )

    # Compute diff
    diff = compute_plan_diff(plan_v1, plan_v2)

    print("\nInitial Plan (v1):")
    print(plan_v1.format_for_display(show_metadata=False))

    print("\n" + "=" * 60)
    print("\nRevised Plan with Diff:")
    print(diff.format_for_display(plan_v2))

    print("\n✓ Test 2 passed\n")


def test_plan_serialization():
    """Test plan serialization and deserialization."""
    print("=" * 60)
    print("Test 3: Plan Serialization")
    print("=" * 60)

    analysis = {
        "summary": "Test summary",
        "selected_components": ["mlp"],
        "feature_engineering": "Test FE",
        "model_architecture_and_loss": "Test arch",
        "training_configuration": "Test train",
        "evaluation": "Test eval",
        "confirmation_question": "Test confirm?",
    }

    plan = MLPlan.from_analysis(analysis, version=1)

    # Serialize
    plan_dict = plan.to_dict()
    print("\nSerialized plan:")
    print(json.dumps(plan_dict, indent=2))

    # Deserialize
    restored_plan = MLPlan.from_dict(plan_dict)

    # Verify
    assert restored_plan.summary == plan.summary
    assert restored_plan.version == plan.version
    assert restored_plan.selected_components == plan.selected_components

    print("\n✓ Test 3 passed - serialization works correctly\n")


def test_generation_context():
    """Test converting plan to generation context."""
    print("=" * 60)
    print("Test 4: Generation Context")
    print("=" * 60)

    analysis = {
        "summary": "Test summary",
        "selected_components": ["mlp"],
        "feature_engineering": "Test FE",
        "model_architecture_and_loss": "Test arch",
        "training_configuration": "Test train",
        "evaluation": "Test eval",
        "confirmation_question": "Test confirm?",
    }

    plan = MLPlan.from_analysis(analysis, version=2, stage="post_training")

    # Convert to generation context
    context = plan.to_generation_context()

    print("\nGeneration context:")
    print(json.dumps(context, indent=2))

    # Verify it doesn't include metadata
    assert "version" not in context
    assert "stage" not in context
    assert "summary" in context
    assert "selected_components" in context

    print("\n✓ Test 4 passed - generation context excludes metadata\n")


if __name__ == "__main__":
    print("\n" + "🧪 ML Plan Tests" + "\n")

    # Run tests
    test_ml_plan_creation()
    test_ml_plan_revision()
    test_plan_serialization()
    test_generation_context()

    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
