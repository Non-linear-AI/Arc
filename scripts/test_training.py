#!/usr/bin/env python3
"""Test script for directly launching training with a saved model.

This script allows you to test training execution with a saved model YAML
without needing to regenerate the YAML every time.

Usage:
    python scripts/test_training.py <model_id> <train_table> [--target-column COLUMN]

Examples:
    # Test with model ID and table
    python scripts/test_training.py test-v8 pidd --target-column outcome

    # Test with latest version of a model
    python scripts/test_training.py test pidd --target-column outcome
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

from arc.database import get_database_manager
from arc.ml.runtime import MLRuntime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test training with a saved model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "model_name",
        help="Model name or model ID (e.g., 'test' or 'test-v8')"
    )

    parser.add_argument(
        "train_table",
        help="Training table name"
    )

    parser.add_argument(
        "--target-column",
        "-t",
        required=True,
        help="Target column for prediction"
    )

    parser.add_argument(
        "--validation-table",
        "-v",
        help="Optional validation table name"
    )

    parser.add_argument(
        "--db-path",
        default="~/.arc/arc_system.db",
        help="Path to Arc database (default: ~/.arc/arc_system.db)"
    )

    parser.add_argument(
        "--artifacts-dir",
        default="~/.arc/artifacts",
        help="Directory for training artifacts (default: ~/.arc/artifacts)"
    )

    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Monitor job status until completion"
    )

    parser.add_argument(
        "--verbose",
        "-V",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def get_model_yaml(db_manager, model_name: str) -> tuple[str, str, str]:
    """Get model YAML from database.

    Args:
        db_manager: Database manager instance
        model_name: Model name or model ID

    Returns:
        Tuple of (model_id, model_yaml, description)
    """
    # Try to get model by ID first
    model = db_manager.services.models.get_model_by_id(model_name)

    # If not found, try to get latest version by name
    if model is None:
        model = db_manager.services.models.get_latest_model_by_name(model_name)

    if model is None:
        raise ValueError(f"Model not found: {model_name}")

    logger.info(f"Found model: {model.id} (version {model.version})")
    logger.info(f"Description: {model.description}")

    return model.id, model.spec, model.description or "Test training"


def monitor_job(runtime: MLRuntime, job_id: str, poll_interval: int = 5):
    """Monitor job status until completion.

    Args:
        runtime: ML runtime instance
        job_id: Job ID to monitor
        poll_interval: Seconds between status checks
    """
    logger.info(f"Monitoring job {job_id}...")
    logger.info(f"Poll interval: {poll_interval}s")

    start_time = time.time()
    last_status = None

    try:
        while True:
            # Get job status
            job_info = runtime.job_service.get_job_by_id(job_id)

            if job_info is None:
                logger.error(f"Job {job_id} not found")
                break

            status = job_info.status.value
            message = job_info.message

            # Print status if changed
            if status != last_status:
                elapsed = time.time() - start_time
                logger.info(f"[{elapsed:.1f}s] Status: {status} - {message}")
                last_status = status

            # Check if completed
            if status in ("completed", "failed", "cancelled"):
                elapsed = time.time() - start_time
                logger.info(f"Job finished in {elapsed:.1f}s with status: {status}")

                if status == "failed":
                    logger.error(f"Job failed: {message}")
                    return False
                elif status == "cancelled":
                    logger.warning(f"Job cancelled: {message}")
                    return False
                else:
                    logger.info(f"Job completed successfully!")
                    return True

            # Wait before next check
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        logger.info("\nMonitoring interrupted by user")
        return False


def main():
    """Main entry point."""
    args = parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("arc").setLevel(logging.DEBUG)

    try:
        # Initialize database
        db_path = Path(args.db_path).expanduser()
        logger.info(f"Opening database: {db_path}")
        db_manager = get_database_manager(str(db_path))

        # Get model YAML
        model_id, model_yaml, description = get_model_yaml(db_manager, args.model_name)

        # Display model info
        logger.info("=" * 80)
        logger.info(f"Model ID: {model_id}")
        logger.info(f"Description: {description}")
        logger.info(f"Training table: {args.train_table}")
        logger.info(f"Target column: {args.target_column}")
        if args.validation_table:
            logger.info(f"Validation table: {args.validation_table}")
        logger.info("=" * 80)

        # Initialize ML Runtime
        artifacts_dir = Path(args.artifacts_dir).expanduser()
        logger.info(f"Artifacts directory: {artifacts_dir}")

        runtime = MLRuntime(
            db_manager=db_manager,
            artifacts_dir=str(artifacts_dir)
        )

        # Submit training job
        logger.info("\nSubmitting training job...")
        job_id = runtime.train_model(
            model_name=args.model_name,
            train_table=args.train_table,
            validation_table=args.validation_table,
        )

        logger.info(f"✓ Training job submitted successfully!")
        logger.info(f"Job ID: {job_id}")
        logger.info("")
        logger.info("Monitor training progress:")
        logger.info(f"  • Status: /ml jobs status {job_id}")
        logger.info(f"  • Logs: /ml jobs logs {job_id}")
        logger.info("")

        # Monitor if requested
        if args.monitor:
            success = monitor_job(runtime, job_id)
            return 0 if success else 1

        return 0

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
