"""Test prompt template rendering to verify model_plan handling.

This script verifies that:
1. When model_plan is None (default), the ML PLAN GUIDANCE section is not included
2. When model_plan is provided, the ML PLAN GUIDANCE section is included
3. The template renders correctly in both cases
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def test_prompt_template():
    """Test the prompt template rendering with different model_plan values."""
    print("=" * 80)
    print("Testing Prompt Template Rendering")
    print("=" * 80)
    print()

    # Load the template
    template_dir = Path("src/arc/core/agents/ml_model/templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("prompt.j2")

    # Create mock data profile
    mock_data_profile = {
        "feature_columns": [
            {"name": "age", "type": "INTEGER"},
            {"name": "bmi", "type": "FLOAT"},
            {"name": "bp", "type": "FLOAT"},
        ],
        "feature_count": 3,
        "target_analysis": {
            "column_name": "target",
            "data_type": "FLOAT",
            "is_numeric": True,
            "unique_values": 100,
        },
    }

    # Create mock available components
    mock_components = {
        "node_types": ["torch.nn.Linear", "torch.nn.ReLU"],
        "description": "PyTorch components",
    }

    # Test 1: model_plan is None (should NOT include ML PLAN GUIDANCE section)
    print("Test 1: Rendering template with model_plan=None")
    print("-" * 80)

    context_without_plan = {
        "model_name": "test-model",
        "user_intent": "Create a simple model",
        "data_profile": mock_data_profile,
        "available_components": mock_components,
        "model_plan": None,
        "existing_yaml": None,
        "editing_instructions": None,
        "is_editing": False,
        "data_processing_context": None,
    }

    rendered_without_plan = template.render(context_without_plan)

    if "ML PLAN GUIDANCE" in rendered_without_plan:
        print(
            "❌ FAILED: ML PLAN GUIDANCE section should NOT be present when model_plan=None"
        )
        print("\nFound ML PLAN GUIDANCE in rendered template:")
        # Find and print the relevant section
        lines = rendered_without_plan.split("\n")
        for i, line in enumerate(lines):
            if "ML PLAN" in line:
                start = max(0, i - 2)
                end = min(len(lines), i + 10)
                print("\n".join(lines[start:end]))
                break
        return False
    else:
        print("✅ PASSED: ML PLAN GUIDANCE section correctly omitted")

    print()

    # Test 2: model_plan is provided (should include ML PLAN GUIDANCE section)
    print("Test 2: Rendering template with model_plan='Use 3-layer MLP...'")
    print("-" * 80)

    context_with_plan = context_without_plan.copy()
    context_with_plan["model_plan"] = (
        "Use a 3-layer MLP architecture with ReLU activations."
    )

    rendered_with_plan = template.render(context_with_plan)

    if "ML PLAN GUIDANCE" not in rendered_with_plan:
        print(
            "❌ FAILED: ML PLAN GUIDANCE section should be present when model_plan is provided"
        )
        return False
    else:
        print("✅ PASSED: ML PLAN GUIDANCE section correctly included")

    # Verify the plan content is in the rendered template
    if context_with_plan["model_plan"] not in rendered_with_plan:
        print("❌ FAILED: model_plan content not found in rendered template")
        return False
    else:
        print("✅ PASSED: model_plan content correctly rendered")

    print()

    # Test 3: Empty string model_plan (should NOT include section due to Jinja truthiness)
    print("Test 3: Rendering template with model_plan='' (empty string)")
    print("-" * 80)

    context_with_empty_plan = context_without_plan.copy()
    context_with_empty_plan["model_plan"] = ""

    rendered_with_empty = template.render(context_with_empty_plan)

    if "ML PLAN GUIDANCE" in rendered_with_empty:
        print(
            "❌ FAILED: ML PLAN GUIDANCE section should NOT be present for empty string"
        )
        return False
    else:
        print("✅ PASSED: Empty string correctly treated as falsy (section omitted)")

    print()
    print("=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print()
    print("Summary:")
    print("  - When model_plan is None: ML PLAN GUIDANCE section is omitted ✓")
    print("  - When model_plan has content: ML PLAN GUIDANCE section is included ✓")
    print("  - When model_plan is empty string: ML PLAN GUIDANCE section is omitted ✓")
    print()
    print(
        "This confirms the fix is correct: by NOT passing model_plan=None explicitly,"
    )
    print(
        "the template properly checks for truthiness and omits the section when appropriate."
    )

    return True


if __name__ == "__main__":
    import sys

    success = test_prompt_template()
    sys.exit(0 if success else 1)
