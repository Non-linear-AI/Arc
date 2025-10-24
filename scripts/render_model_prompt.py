#!/usr/bin/env python3
"""Script to render and print the ML model agent prompt.

This script demonstrates what the actual prompt looks like when rendered
with real parameters, helping to visualize the full system message sent to the LLM.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc.core.agents.ml_model.ml_model import MLModelAgent
from arc.database.manager import DatabaseManager
from arc.database.services.container import ServiceContainer


def create_sample_table(db_manager):
    """Create sample PIDD table for demonstration."""
    # Create PIDD table (Pima Indians Diabetes Dataset)
    db_manager.user_execute("""
        CREATE TABLE IF NOT EXISTS pidd (
            pregnancies INTEGER,
            glucose INTEGER,
            blood_pressure INTEGER,
            skin_thickness INTEGER,
            insulin INTEGER,
            bmi DECIMAL(5,2),
            diabetes_pedigree DECIMAL(5,3),
            age INTEGER,
            outcome INTEGER
        )
    """)

    # Insert sample data
    db_manager.user_execute("""
        INSERT INTO pidd VALUES
        (6, 148, 72, 35, 0, 33.6, 0.627, 50, 1),
        (1, 85, 66, 29, 0, 26.6, 0.351, 31, 0),
        (8, 183, 64, 0, 0, 23.3, 0.672, 32, 1),
        (1, 89, 66, 23, 94, 28.1, 0.167, 21, 0),
        (0, 137, 40, 35, 168, 43.1, 2.288, 33, 1)
    """)


async def render_model_prompt():
    """Render prompt for model generation."""
    print("=" * 80)
    print("ML MODEL GENERATION PROMPT")
    print("=" * 80)
    print()

    # Initialize database and services (use in-memory for rendering)
    db_manager = DatabaseManager(system_db_path=":memory:", user_db_path=":memory:")
    services = ServiceContainer(db_manager)

    # Create sample table
    create_sample_table(db_manager)

    # Create agent (dummy API key since we're just rendering the prompt)
    agent = MLModelAgent(
        services=services,
        api_key="dummy-key-for-rendering",
        model="gpt-4",
    )

    # Sample parameters matching the user's command:
    # /ml model --instruction "predict outcome in pidd" --name test --data-table pidd --target-column outcome
    name = "test"
    user_context = "predict outcome in pidd"
    table_name = "pidd"
    target_column = "outcome"

    # Get data profile
    data_profile = await agent._get_unified_data_profile(table_name, target_column)

    # Get available components
    available_components = agent._get_model_components()

    # Render the template with context
    system_message = agent._render_template(
        agent.get_template_name(),
        {
            "model_name": name,
            "user_intent": user_context,
            "data_profile": data_profile,
            "available_components": available_components,
            "ml_plan_architecture": None,
            "recommended_knowledge": "\n\n# Architecture Knowledge: mlp\n\n[MLP knowledge content would be here]",
            "existing_yaml": None,
            "editing_instructions": None,
            "is_editing": False,
        },
    )

    print(system_message)
    print()
    print("=" * 80)
    print(f"Prompt length: {len(system_message)} characters")
    print("=" * 80)


async def main():
    """Render model generation prompt."""
    try:
        await render_model_prompt()

        print()
        print("✓ Successfully rendered prompt")
        print()
        print("TIP: Redirect output to a file to review in detail:")
        print("  python scripts/render_model_prompt.py > model_prompt_output.txt")

    except Exception as e:
        print(f"Error rendering prompt: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
