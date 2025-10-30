#!/usr/bin/env python3
"""Test script for ml_plan tool (ML planning).

This script allows you to test ML plan generation directly
without going through the agent.

Usage:
    python scripts/test_ml_plan.py <name> <instruction> <source_tables> [options]

Examples:
    # Create new plan
    python scripts/test_ml_plan.py diabetes-classifier "Build binary classifier for diabetes prediction" pidd

    # Create plan with data insights
    python scripts/test_ml_plan.py diabetes-classifier "Build classifier. Data insights: 768 samples, binary outcome (500/268), 8 numeric features, no missing values" pidd

    # Revise existing plan
    python scripts/test_ml_plan.py diabetes-classifier "Change model to use 4 layers instead of 3" pidd
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from arc.database import DatabaseManager
from arc.database.services import ServiceContainer
from arc.tools.ml import MLPlanTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test ml_plan tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "name",
        help="Name for the ML plan (e.g., 'diabetes-classifier')"
    )

    parser.add_argument(
        "instruction",
        help="Description of ML problem/goals OR changes to make to existing plan"
    )

    parser.add_argument(
        "source_tables",
        help="Comma-separated list of source tables (e.g., 'pidd' or 'users,transactions')"
    )

    parser.add_argument(
        "--api-key",
        help="API key for LLM calls (default: from ARC_API_KEY env var)"
    )

    parser.add_argument(
        "--base-url",
        help="Base URL for LLM API (default: from ARC_BASE_URL env var)"
    )

    parser.add_argument(
        "--model",
        help="Model name for LLM (default: from settings or claude-sonnet-4)"
    )

    parser.add_argument(
        "--system-db",
        default="~/.arc/arc_system.db",
        help="Path to Arc system database (default: ~/.arc/arc_system.db)"
    )

    parser.add_argument(
        "--user-db",
        default="~/.arc/arc_user.db",
        help="Path to Arc user database (default: ~/.arc/arc_user.db)"
    )

    parser.add_argument(
        "--auto-confirm",
        action="store_true",
        help="Skip interactive confirmation workflow"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("arc").setLevel(logging.DEBUG)

    try:
        # Get API configuration
        import os
        api_key = args.api_key or os.getenv("ARC_API_KEY", "")
        base_url = args.base_url or os.getenv("ARC_BASE_URL")
        model = args.model or os.getenv("ARC_MODEL", "claude-sonnet-4")

        if not api_key:
            raise ValueError(
                "API key required. Set ARC_API_KEY environment variable or use --api-key"
            )

        # Initialize database
        system_db_path = Path(args.system_db).expanduser()
        user_db_path = Path(args.user_db).expanduser()
        logger.info(f"System database: {system_db_path}")
        logger.info(f"User database: {user_db_path}")
        db_manager = DatabaseManager(str(system_db_path), str(user_db_path))

        # Initialize service container
        services = ServiceContainer(db_manager)

        # Display test info
        logger.info("=" * 80)
        logger.info(f"Name: {args.name}")
        logger.info(f"Instruction: {args.instruction}")
        logger.info(f"Source tables: {args.source_tables}")
        logger.info(f"Auto-confirm: {args.auto_confirm}")
        logger.info("=" * 80)

        # Initialize ml_plan tool
        # Note: agent parameter is optional for testing
        tool = MLPlanTool(
            services=services,
            api_key=api_key,
            base_url=base_url,
            model=model,
            ui_interface=None,  # No UI in test mode
            agent=None,  # No agent needed for testing
        )

        # Execute ml_plan
        logger.info("\nExecuting ml_plan...")
        result = await tool.execute(
            name=args.name,
            instruction=args.instruction,
            source_tables=args.source_tables,
            auto_confirm=args.auto_confirm,
        )

        # Display results
        logger.info("\n" + "=" * 80)
        logger.info("RESULT")
        logger.info("=" * 80)
        logger.info(f"Success: {result.success}")
        logger.info(f"Output: {result.output}")

        if result.metadata:
            logger.info("\nMetadata:")
            for key, value in result.metadata.items():
                if key != "yaml_content":  # Skip large YAML content
                    logger.info(f"  {key}: {value}")

        if result.success and result.metadata:
            plan_id = result.metadata.get("plan_id")
            recommended_knowledge = result.metadata.get("recommended_knowledge_ids", [])

            if plan_id:
                logger.info(f"\n✓ ML Plan created: {plan_id}")

                if recommended_knowledge:
                    logger.info(f"\nRecommended knowledge IDs:")
                    for kid in recommended_knowledge:
                        logger.info(f"  • {kid}")

                logger.info("\nUse this plan with other ML tools:")
                logger.info(f"  ml_data --plan-id {plan_id}")
                logger.info(f"  ml_model --plan-id {plan_id}")

            return 0
        else:
            logger.error("\n✗ ML plan generation failed!")
            return 1

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        return 1
    finally:
        # Clean up
        if 'db_manager' in locals():
            db_manager.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
