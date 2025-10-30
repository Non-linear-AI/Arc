#!/usr/bin/env python3
"""Test script for ml_data tool (data processing pipeline).

This script allows you to test data processing execution directly
without going through the agent.

Usage:
    python scripts/test_ml_data.py <name> <instruction> <source_tables> [options]

Examples:
    # Basic test with single source table
    python scripts/test_ml_data.py diabetes_processed "Split into train/validation with 80/20 split" pidd

    # Test with multiple source tables
    python scripts/test_ml_data.py merged_data "Join users and transactions" users,transactions

    # Test with plan_id
    python scripts/test_ml_data.py diabetes_features "Create features" pidd --plan-id diabetes-plan-v1
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from arc.database import DatabaseManager  # noqa: E402
from arc.database.services import ServiceContainer  # noqa: E402
from arc.tools.ml_data import MLDataTool  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test ml_data tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("name", help="Name for the data processor")

    parser.add_argument(
        "instruction",
        help="Natural language description of data processing requirements",
    )

    parser.add_argument(
        "source_tables",
        help="Comma-separated list of source tables (e.g., 'pidd' or 'users,transactions')",
    )

    parser.add_argument(
        "--database",
        "-d",
        choices=["system", "user"],
        default="user",
        help="Target database (default: user)",
    )

    parser.add_argument(
        "--plan-id", "-p", help="Optional ML plan ID (e.g., 'diabetes-plan-v1')"
    )

    parser.add_argument(
        "--api-key", help="API key for LLM calls (default: from ARC_API_KEY env var)"
    )

    parser.add_argument(
        "--base-url", help="Base URL for LLM API (default: from ARC_BASE_URL env var)"
    )

    parser.add_argument(
        "--model", help="Model name for LLM (default: from settings or claude-sonnet-4)"
    )

    parser.add_argument(
        "--system-db",
        default="~/.arc/arc_system.db",
        help="Path to Arc system database (default: ~/.arc/arc_system.db)",
    )

    parser.add_argument(
        "--user-db",
        default="~/.arc/arc_user.db",
        help="Path to Arc user database (default: ~/.arc/arc_user.db)",
    )

    parser.add_argument(
        "--auto-confirm",
        action="store_true",
        help="Skip interactive confirmation workflow",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
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

        # Parse source tables
        source_tables = [t.strip() for t in args.source_tables.split(",")]

        # Display test info
        logger.info("=" * 80)
        logger.info(f"Name: {args.name}")
        logger.info(f"Instruction: {args.instruction}")
        logger.info(f"Source tables: {source_tables}")
        logger.info(f"Database: {args.database}")
        if args.plan_id:
            logger.info(f"Plan ID: {args.plan_id}")
        logger.info(f"Auto-confirm: {args.auto_confirm}")
        logger.info("=" * 80)

        # Initialize ml_data tool
        tool = MLDataTool(
            services=services,
            api_key=api_key,
            base_url=base_url,
            model=model,
            ui_interface=None,  # No UI in test mode
        )

        # Execute ml_data
        logger.info("\nExecuting ml_data...")
        result = await tool.generate(
            name=args.name,
            instruction=args.instruction,
            source_tables=source_tables,
            database=args.database,
            auto_confirm=args.auto_confirm,
            plan_id=args.plan_id,
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
                logger.info(f"  {key}: {value}")

        if result.success:
            logger.info("\n✓ Data processing completed successfully!")

            # Extract data_processing_id if available
            if result.metadata and "data_processing_id" in result.metadata:
                data_processing_id = result.metadata["data_processing_id"]
                logger.info(f"\nData Processing ID: {data_processing_id}")
                logger.info("Use this ID with ml_model to load processing context:")
                logger.info(f"  --data-processing-id {data_processing_id}")

            return 0
        else:
            logger.error("\n✗ Data processing failed!")
            return 1

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        return 1
    finally:
        # Clean up
        if "db_manager" in locals():
            db_manager.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
