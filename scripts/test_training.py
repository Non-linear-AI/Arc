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

from arc.core import SettingsManager
from arc.database import DatabaseManager
from arc.database.services import ModelService, ServiceContainer
from arc.ml import TensorBoardManager
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
        "--artifacts-dir",
        default=".arc/artifacts",
        help="Directory for training artifacts (default: .arc/artifacts, project-local)"
    )

    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Monitor job status until completion"
    )

    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Launch TensorBoard after job submission"
    )

    parser.add_argument(
        "--tensorboard-port",
        type=int,
        default=6006,
        help="TensorBoard port (default: 6006)"
    )

    parser.add_argument(
        "--verbose",
        "-V",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def get_model_yaml(model_service: ModelService, model_name: str) -> tuple[str, str, str]:
    """Get model YAML from database.

    Args:
        model_service: Model service instance
        model_name: Model name or model ID

    Returns:
        Tuple of (model_id, model_yaml, description)
    """
    # Try to get model by ID first
    model = model_service.get_model_by_id(model_name)

    # If not found, try to get latest version by name
    if model is None:
        model = model_service.get_latest_model_by_name(model_name)

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
        system_db_path = Path(args.system_db).expanduser()
        user_db_path = Path(args.user_db).expanduser()
        logger.info(f"System database: {system_db_path}")
        logger.info(f"User database: {user_db_path}")
        db_manager = DatabaseManager(str(system_db_path), str(user_db_path))

        # Initialize artifacts directory (expand ~ if present, otherwise keep as-is)
        artifacts_path = Path(args.artifacts_dir)
        if str(artifacts_path).startswith('~'):
            artifacts_dir = artifacts_path.expanduser()
        else:
            artifacts_dir = artifacts_path
        logger.info(f"Artifacts directory: {artifacts_dir}")

        # Initialize service container
        services = ServiceContainer(db_manager, artifacts_dir=str(artifacts_dir))

        # Get model from database
        model_service = services.models
        model = model_service.get_model_by_id(args.model_name)

        # If not found, try to get latest version by name
        if model is None:
            model = model_service.get_latest_model_by_name(args.model_name)

        if model is None:
            raise ValueError(f"Model not found: {args.model_name}")

        # Display model info
        logger.info("=" * 80)
        logger.info(f"Model ID: {model.id}")
        logger.info(f"Model Name: {model.name}")
        logger.info(f"Version: {model.version}")
        logger.info(f"Description: {model.description}")
        logger.info(f"Training table: {args.train_table}")
        logger.info(f"Target column: {args.target_column}")
        if args.validation_table:
            logger.info(f"Validation table: {args.validation_table}")
        else:
            logger.info("Validation: Using 20% of training data for validation")
        logger.info("=" * 80)

        # Initialize ML Runtime
        runtime = MLRuntime(services, artifacts_dir=artifacts_dir)

        # Submit training job using the model's name (not ID)
        logger.info("\nSubmitting training job...")
        job_id = runtime.train_model(
            model_name=model.name,
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

        # Launch TensorBoard if requested
        if args.tensorboard:
            try:
                tensorboard_manager = TensorBoardManager()

                # Get actual TensorBoard log directory from training run database
                tensorboard_logdir = None
                try:
                    run = services.training_tracking.get_run_by_job_id(job_id)
                    if run and run.tensorboard_log_dir:
                        tensorboard_logdir = Path(run.tensorboard_log_dir)
                except Exception:
                    pass  # Fall through to default

                # Fallback to default if not found
                if tensorboard_logdir is None:
                    tensorboard_logdir = Path(f"tensorboard/run_{job_id}")

                url, pid = tensorboard_manager.launch(job_id, tensorboard_logdir, port=args.tensorboard_port)
                logger.info("✓ TensorBoard launched")
                logger.info(f"  • URL: {url}")
                logger.info(f"  • Process ID: {pid}")
                logger.info(f"  • Logs: {tensorboard_logdir}")
                logger.info("")
            except Exception as e:
                logger.warning(f"Failed to launch TensorBoard: {e}")
                logger.info("You can launch TensorBoard manually:")
                logger.info(f"  tensorboard --logdir {tensorboard_logdir if tensorboard_logdir else f'tensorboard/run_{job_id}'}")
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
