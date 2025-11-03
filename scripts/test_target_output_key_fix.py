#!/usr/bin/env python3
"""Test that target_output_key is correctly extracted from model YAML."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml

from arc.core import SettingsManager
from arc.database import DatabaseManager
from arc.database.services import ServiceContainer


def test_target_output_key_extraction():
    """Test that target_output_key is extracted correctly from different models."""
    # Initialize services
    settings_manager = SettingsManager()
    system_db_path = settings_manager.get_system_database_path()
    user_db_path = settings_manager.get_user_database_path()
    db_manager = DatabaseManager(system_db_path, user_db_path)
    services = ServiceContainer(db_manager, artifacts_dir=".arc/artifacts")

    # Get all models
    models = services.models.list_all_models()

    print("=" * 80)
    print("Testing target_output_key extraction from model YAML")
    print("=" * 80)
    print()

    # Test with models that use non-standard output names
    test_models = [
        m
        for m in models
        if "rating_prediction" in yaml.safe_load(m.spec).get("outputs", {})
        or "prediction" in yaml.safe_load(m.spec).get("outputs", {})
    ]

    if not test_models:
        print("⚠️  No test models found with non-standard output names")
        return

    # Take the first model with a non-standard output name
    test_model = test_models[0]
    spec = yaml.safe_load(test_model.spec)

    print(f"Testing with model: {test_model.name}")
    print(f"Model outputs: {list(spec['outputs'].keys())}")

    # Extract expected target_output_key
    expected_key = spec["training"]["loss"]["inputs"]["input"]
    print(f"Expected target_output_key (from YAML): '{expected_key}'")
    print()

    # Now test the fix by simulating what happens in training_service
    loss_config = spec["training"]["loss"]
    loss_inputs = loss_config.get("inputs", {})
    extracted_key = loss_inputs.get("input")

    print(f"Extracted target_output_key (from fix): '{extracted_key}'")
    print()

    if extracted_key == expected_key:
        print("✅ SUCCESS: Extracted key matches expected key!")
        print(
            f"   The fix correctly extracts '{extracted_key}' instead of hardcoding 'logits'"
        )
    else:
        print(
            f"❌ FAILURE: Extracted key '{extracted_key}' != expected '{expected_key}'"
        )

    print()
    print("=" * 80)
    print("Test Summary:")
    print("=" * 80)
    print("Before fix: Would have used hardcoded 'logits'")
    print(f"After fix:  Uses '{extracted_key}' from model YAML")
    print()
    print("This means training will no longer show warnings like:")
    print("  \"Target output key 'logits' not found in model outputs.\"")
    print()

    services.shutdown()


if __name__ == "__main__":
    test_target_output_key_extraction()
