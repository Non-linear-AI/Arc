"""Script to verify model generation works correctly after the fix.

This script tests that:
1. Model generation produces valid YAML
2. YAML validates against Arc-Graph schema
3. Model can be built from the generated YAML
"""

import asyncio
import sys

from arc.core import SettingsManager
from arc.database.manager import DatabaseManager
from arc.database.services import ServiceContainer


async def test_model_generation():
    """Test model generation with a simple example."""
    print("=" * 80)
    print("Testing Model Generation")
    print("=" * 80)
    print()

    # Initialize settings and database
    settings_manager = SettingsManager()
    system_db_path = settings_manager.get_system_database_path()
    user_db_path = settings_manager.get_user_database_path()
    db_manager = DatabaseManager(system_db_path, user_db_path)
    services = ServiceContainer(db_manager, artifacts_dir=".arc/artifacts")

    # Get API key from settings or environment
    api_key = settings_manager.get_api_key()
    if not api_key:
        print(
            "❌ API key not configured. Set ARC_API_KEY env var or run: arc set api-key"
        )
        return False

    # Get other settings
    base_url = settings_manager.get_base_url()
    model = settings_manager.get_current_model()

    try:
        # Create model generator agent
        from arc.core.agents.ml_model import MLModelAgent

        agent = MLModelAgent(
            services=services,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

        # Test with diabetes dataset
        print("Generating model for diabetes_processed dataset...")
        print("-" * 80)

        model_spec, model_yaml, conversation_history = await agent.generate_model(
            name="test-diabetes-model",
            user_context="Create a simple neural network for predicting diabetes progression",
            table_name="diabetes_processed",
            target_column="target",
            # Note: No model_plan parameter - testing that default None works
            knowledge_references=None,
        )

        print("\n✅ Model generation succeeded!")
        print(f"\nGenerated YAML ({len(model_yaml)} characters):")
        print("-" * 80)
        print(model_yaml)
        print("-" * 80)

        # Validate the YAML has all required sections
        import yaml

        parsed = yaml.safe_load(model_yaml)

        print("\n📋 Validating required sections...")
        required_sections = ["inputs", "graph", "outputs", "training"]
        missing = [s for s in required_sections if s not in parsed]

        if missing:
            print(f"❌ Missing sections: {missing}")
            return False

        print("✅ All required sections present")

        # Check training section has loss
        if "loss" not in parsed["training"]:
            print("❌ Training section missing 'loss' field")
            return False

        print("✅ Training section has loss configuration")

        # Try to build the model
        print("\n🔨 Building model from YAML...")
        from arc.graph.model import build_model_from_yaml

        # Extract model-only YAML (without training section)
        model_only = {k: v for k, v in parsed.items() if k != "training"}
        model_yaml_only = yaml.dump(
            model_only, default_flow_style=False, sort_keys=False
        )

        model = build_model_from_yaml(model_yaml_only)
        print(f"✅ Model built successfully: {type(model).__name__}")

        # Print model summary
        print("\n📊 Model Summary:")
        print(f"  - Inputs: {list(model.input_names)}")
        print(f"  - Outputs: {list(model.output_names)}")

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED - Model generation is working correctly!")
        print("=" * 80)

        return True

    except Exception as e:
        print("\n❌ Model generation failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        services.shutdown()
        db_manager.close()


if __name__ == "__main__":
    success = asyncio.run(test_model_generation())
    sys.exit(0 if success else 1)
