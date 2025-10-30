#!/usr/bin/env python3
"""Test script for ml_model tool (unified model generation + training).

This script allows you to test model generation and training execution directly
without going through the agent.

Usage:
    python scripts/test_ml_model.py <name> <instruction> <data_table> <target_column> [options]

Examples:
    # Basic test
    python scripts/test_ml_model.py diabetes_model "Create binary classifier" pidd outcome

    # Test with plan_id
    python scripts/test_ml_model.py diabetes_model "Create model" pidd outcome --plan-id diabetes-plan-v1

    # Test with data_processing_id
    python scripts/test_ml_model.py diabetes_model "Create model" diabetes_processed_data outcome --data-processing-id data_abc123

    # Test with separate train table
    python scripts/test_ml_model.py diabetes_model "Create model" pidd outcome --train-table diabetes_train_data
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from arc.database import DatabaseManager  # noqa: E402
from arc.database.services import ServiceContainer  # noqa: E402
from arc.ml import TensorBoardManager  # noqa: E402
from arc.tools.ml import MLModelTool  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test ml_model tool (unified model + training)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "name", help="Experiment name (used for both model and trainer)"
    )

    parser.add_argument(
        "instruction",
        help="Description of model architecture and training requirements",
    )

    parser.add_argument("data_table", help="Database table to profile for generation")

    parser.add_argument("target_column", help="Target column for prediction")

    parser.add_argument(
        "--train-table",
        help="Training data table (defaults to data_table if not provided)",
    )

    parser.add_argument(
        "--plan-id", "-p", help="Optional ML plan ID (e.g., 'diabetes-plan-v1')"
    )

    parser.add_argument(
        "--data-processing-id",
        help="Optional data processing execution ID from ml_data tool",
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
        "--artifacts-dir",
        default=".arc/artifacts",
        help="Directory for training artifacts (default: .arc/artifacts)",
    )

    parser.add_argument(
        "--auto-confirm",
        action="store_true",
        help="Skip interactive confirmation workflow",
    )

    parser.add_argument(
        "--monitor", action="store_true", help="Monitor training job until completion"
    )

    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Launch TensorBoard after job submission",
    )

    parser.add_argument(
        "--tensorboard-port",
        type=int,
        default=6006,
        help="TensorBoard port (default: 6006)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args()


def monitor_job(services, job_id: str, poll_interval: int = 5):
    """Monitor job status until completion."""
    logger.info(f"Monitoring job {job_id}...")
    start_time = time.time()
    last_status = None

    try:
        while True:
            job_info = services.jobs.get_job_by_id(job_id)

            if job_info is None:
                logger.error(f"Job {job_id} not found")
                break

            status = job_info.status.value
            message = job_info.message

            if status != last_status:
                elapsed = time.time() - start_time
                logger.info(f"[{elapsed:.1f}s] Status: {status} - {message}")
                last_status = status

            if status in ("completed", "failed", "cancelled"):
                elapsed = time.time() - start_time
                logger.info(f"Job finished in {elapsed:.1f}s with status: {status}")
                return status == "completed"

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        logger.info("\nMonitoring interrupted by user")
        return False


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

        # Initialize artifacts directory
        artifacts_path = Path(args.artifacts_dir)
        if str(artifacts_path).startswith("~"):
            artifacts_dir = artifacts_path.expanduser()
        else:
            artifacts_dir = artifacts_path
        logger.info(f"Artifacts directory: {artifacts_dir}")

        # Initialize service container
        services = ServiceContainer(db_manager, artifacts_dir=str(artifacts_dir))

        # Initialize TensorBoard manager if needed
        tensorboard_manager = None
        if args.tensorboard:
            try:
                tensorboard_manager = TensorBoardManager()
            except Exception as e:
                logger.warning(f"Failed to initialize TensorBoard manager: {e}")

        # Display test info
        logger.info("=" * 80)
        logger.info(f"Name: {args.name}")
        logger.info(f"Instruction: {args.instruction}")
        logger.info(f"Data table: {args.data_table}")
        logger.info(f"Target column: {args.target_column}")
        if args.train_table:
            logger.info(f"Train table: {args.train_table}")
        if args.plan_id:
            logger.info(f"Plan ID: {args.plan_id}")
        if args.data_processing_id:
            logger.info(f"Data processing ID: {args.data_processing_id}")
        logger.info(f"Auto-confirm: {args.auto_confirm}")
        logger.info("=" * 80)

        # Initialize ml_model tool
        tool = MLModelTool(
            services=services,
            runtime=services.ml_runtime,
            api_key=api_key,
            base_url=base_url,
            model=model,
            ui_interface=None,  # No UI in test mode
            tensorboard_manager=tensorboard_manager,
        )

        # Execute ml_model
        logger.info("\nExecuting ml_model...")
        result = await tool.execute(
            name=args.name,
            instruction=args.instruction,
            data_table=args.data_table,
            train_table=args.train_table,
            target_column=args.target_column,
            plan_id=args.plan_id,
            data_processing_id=args.data_processing_id,
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
            job_id = result.metadata.get("job_id")
            model_id = result.metadata.get("model_id")

            if model_id:
                logger.info(f"\n✓ Model created: {model_id}")

            if job_id:
                logger.info(f"✓ Training job submitted: {job_id}")
                logger.info("\nMonitor training progress:")
                logger.info(f"  • Status: /ml jobs status {job_id}")
                logger.info(f"  • Logs: /ml jobs logs {job_id}")

                # Monitor if requested
                if args.monitor:
                    success = monitor_job(services, job_id)
                    return 0 if success else 1

            return 0
        else:
            logger.error("\n✗ Model generation/training failed!")
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
