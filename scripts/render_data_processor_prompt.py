#!/usr/bin/env python3
"""Script to render and print the data processor agent prompt.

This script demonstrates what the actual prompt looks like when rendered
with real parameters, helping to visualize the full system message sent to the LLM.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc.core.agents.ml_data.ml_data import MLDataAgent
from arc.database.manager import DatabaseManager
from arc.database.services.container import ServiceContainer


def create_sample_tables(db_manager):
    """Create sample tables with realistic schema for demonstration."""
    # Create transactions table
    db_manager.user_execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id INTEGER,
            user_id VARCHAR,
            amount DECIMAL(10,2),
            transaction_date DATE,
            category VARCHAR,
            status VARCHAR
        )
    """)

    # Insert sample data
    db_manager.user_execute("""
        INSERT INTO transactions VALUES
        (1, 'user_1', 99.99, '2024-01-15', 'ELECTRONICS', 'completed'),
        (2, 'user_1', 45.50, '2024-02-20', 'BOOKS', 'completed'),
        (3, 'user_2', 299.99, '2024-01-10', 'ELECTRONICS', 'completed'),
        (4, 'user_2', NULL, '2024-03-05', 'CLOTHING', 'pending')
    """)

    # Create sessions table
    db_manager.user_execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id INTEGER,
            user_id VARCHAR,
            session_date DATE,
            duration_seconds INTEGER,
            page_views INTEGER
        )
    """)

    db_manager.user_execute("""
        INSERT INTO sessions VALUES
        (1, 'user_1', '2024-01-15', 3600, 25),
        (2, 'user_1', '2024-02-20', 1800, 12),
        (3, 'user_2', '2024-01-10', 7200, 45),
        (4, 'user_2', '2024-03-05', -100, 5)
    """)

    # Create users table
    db_manager.user_execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id VARCHAR NOT NULL,
            email VARCHAR,
            signup_date DATE,
            country VARCHAR
        )
    """)

    db_manager.user_execute("""
        INSERT INTO users VALUES
        ('user_1', 'user1@example.com', '2023-12-01', 'US'),
        ('user_2', 'user2@example.com', '2023-11-15', 'UK'),
        ('user_3', NULL, '2024-01-01', 'CA')
    """)


async def render_generation_prompt():
    """Render prompt for generation mode (no existing YAML)."""
    print("=" * 80)
    print("DATA PROCESSOR GENERATION PROMPT (New Pipeline)")
    print("=" * 80)
    print()

    # Initialize database and services (use in-memory for rendering)
    db_manager = DatabaseManager(system_db_path=":memory:", user_db_path=":memory:")
    services = ServiceContainer(db_manager)

    # Create sample tables with realistic data
    create_sample_tables(db_manager)

    # Create agent (dummy API key since we're just rendering the prompt)
    agent = MLDataAgent(
        services=services,
        api_key="dummy-key-for-rendering",
        model="gpt-4",
    )

    # Sample instruction from main agent
    instruction = """Create user engagement features from the transactions \
and sessions tables.

Requirements:
- Filter data to last 6 months (transactions after 2023-06-01)
- Calculate transaction metrics: count, total spent, average amount
- Calculate session metrics: count, average duration, last session date
- Include recency features (days since last transaction/session)
- Handle NULL values appropriately
- Filter out users with less than 3 transactions

Data insights from exploration:
- transactions.amount has 0.5% NULL values (use COALESCE)
- transactions.user_id has no NULLs (safe to use)
- sessions.duration has negative values in 0.2% of rows (filter out)"""

    source_tables = ["transactions", "sessions", "users"]
    database = "user"

    # Get schema context (this will show real schema if tables exist)
    schema_info = await agent._get_schema_context(source_tables, database)

    # Render the prompt
    system_prompt = await agent._render_system_prompt(
        instruction=instruction,
        schema_info=schema_info,
        source_tables=source_tables,
        existing_yaml=None,  # Generation mode
    )

    print(system_prompt)
    print()
    print("=" * 80)
    print(f"Prompt length: {len(system_prompt)} characters")
    print("=" * 80)


async def render_editing_prompt():
    """Render prompt for editing mode (with existing YAML)."""
    print()
    print()
    print("=" * 80)
    print("DATA PROCESSOR EDITING PROMPT (Modify Existing Pipeline)")
    print("=" * 80)
    print()

    # Initialize database and services (use in-memory for rendering)
    db_manager = DatabaseManager(system_db_path=":memory:", user_db_path=":memory:")
    services = ServiceContainer(db_manager)

    # Create sample tables with realistic data
    create_sample_tables(db_manager)

    # Create agent
    agent = MLDataAgent(
        services=services,
        api_key="dummy-key-for-rendering",
        model="gpt-4",
    )

    # User feedback for editing
    instruction = (
        "Add a new feature to calculate the user's favorite product category. "
        "Use the category field from transactions and find the most frequent "
        "category per user."
    )

    source_tables = ["transactions"]
    database = "user"

    # Existing YAML that will be edited
    existing_yaml = """name: user_transaction_features
description: Basic transaction features per user

steps:
  - name: drop_old_features
    type: execute
    depends_on: []
    sql: DROP TABLE IF EXISTS user_features

  - name: user_features
    type: table
    depends_on: [drop_old_features, transactions]
    sql: |
      SELECT
        user_id,
        COUNT(*) as transaction_count,
        SUM(amount) as total_spent,
        AVG(amount) as avg_amount
      FROM transactions
      WHERE user_id IS NOT NULL
      GROUP BY user_id

outputs: [user_features]"""

    # Get schema context
    schema_info = await agent._get_schema_context(source_tables, database)

    # Render the prompt
    system_prompt = await agent._render_system_prompt(
        instruction=instruction,
        schema_info=schema_info,
        source_tables=source_tables,
        existing_yaml=existing_yaml,  # Editing mode
    )

    print(system_prompt)
    print()
    print("=" * 80)
    print(f"Prompt length: {len(system_prompt)} characters")
    print("=" * 80)


async def main():
    """Render both generation and editing prompts."""
    try:
        # Render generation prompt
        await render_generation_prompt()

        # Render editing prompt
        await render_editing_prompt()

        print()
        print("✓ Successfully rendered both prompts")
        print()
        print("TIP: Redirect output to a file to review in detail:")
        print("  python scripts/render_data_processor_prompt.py > prompt_output.txt")

    except Exception as e:
        print(f"Error rendering prompts: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
