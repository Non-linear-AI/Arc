#!/usr/bin/env python3
"""Test script for directly launching model evaluation.

This script allows you to test evaluation execution with a trained model
without using the interactive CLI.

Usage:
    python scripts/test_evaluation.py <model_id> <test_table> [--output-table TABLE]

Examples:
    # Evaluate model on test dataset
    python scripts/test_evaluation.py test-v8 pidd_test --output-table pidd_predictions

    # Evaluate without saving predictions
    python scripts/test_evaluation.py test-v8 pidd_test

    # With monitoring and TensorBoard
    python scripts/test_evaluation.py test-v8 pidd_test --monitor --tensorboard
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
from arc.database.services import ServiceContainer
from arc.ml import TensorBoardManager
from arc.ml.runtime import MLRuntime
from arc.tools.ml import MLEvaluateTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test model evaluation with a trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "model_id",
        help="Model ID to evaluate (e.g., 'test-v8')"
    )

    parser.add_argument(
        "test_table",
        help="Test dataset table name"
    )

    parser.add_argument(
        "--output-table",
        "-o",
        help="Optional table name to save predictions"
    )

    parser.add_argument(
        "--metrics",
        "-m",
        help="Comma-separated list of metrics (e.g., 'accuracy,precision,recall')"
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
        help="Directory for artifacts (default: .arc/artifacts, project-local)"
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

                    # Show evaluation metrics if available
                    from arc.database.services import EvaluationTrackingService
                    import json

                    eval_tracking = EvaluationTrackingService(runtime.services.models.db_manager)
                    eval_runs = eval_tracking.list_runs(limit=100)
                    eval_run = next((r for r in eval_runs if r.job_id == job_id), None)

                    if eval_run and eval_run.metrics_result:
                        try:
                            metrics = json.loads(eval_run.metrics_result)
                            logger.info("\nEvaluation Metrics:")
                            logger.info("=" * 60)
                            for metric_name, metric_value in metrics.items():
                                if isinstance(metric_value, (int, float)):
                                    logger.info(f"  {metric_name}: {metric_value:.6f}")
                                else:
                                    logger.info(f"  {metric_name}: {metric_value}")
                            logger.info("=" * 60)
                        except Exception:
                            pass

                    return True

            # Wait before next check
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        logger.info("\nMonitoring interrupted by user")
        return False


# Simple UI stub for non-interactive execution
class SimpleUI:
    """Simple UI stub for non-interactive script execution."""

    def __init__(self):
        self._printer = self

    def show_system_error(self, message):
        logger.error(message)

    def show_system_success(self, message):
        logger.info(message)

    def show_info(self, message):
        logger.info(message)


async def main():
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
        model = model_service.get_model_by_id(args.model_id)

        if model is None:
            raise ValueError(f"Model not found: {args.model_id}")

        # Display model info
        logger.info("=" * 80)
        logger.info(f"Model ID: {model.id}")
        logger.info(f"Model Name: {model.name}")
        logger.info(f"Version: {model.version}")
        logger.info(f"Description: {model.description}")
        logger.info(f"Test table: {args.test_table}")
        if args.output_table:
            logger.info(f"Output table: {args.output_table}")
        if args.metrics:
            logger.info(f"Metrics: {args.metrics}")
        logger.info("=" * 80)

        # Initialize ML Runtime
        runtime = MLRuntime(services, artifacts_dir=artifacts_dir)

        # Initialize TensorBoard manager
        tensorboard_manager = None
        try:
            tensorboard_manager = TensorBoardManager()
        except Exception as e:
            logger.warning(f"Failed to initialize TensorBoard manager: {e}")

        # Parse metrics if provided
        metrics = None
        if args.metrics:
            metrics = [m.strip() for m in args.metrics.split(",")]

        # Create simple UI stub
        ui = SimpleUI()

        # Create evaluation tool
        tool = MLEvaluateTool(
            services,
            runtime,
            ui,
            tensorboard_manager
        )

        # Submit evaluation job
        logger.info("\nSubmitting evaluation job...")
        result = await tool.execute(
            model_id=args.model_id,
            data_table=args.test_table,
            metrics=metrics,
            output_table=args.output_table,
            auto_confirm=True,  # Skip interactive confirmation
        )

        if not result.success:
            logger.error(f"Evaluation failed: {result.error}")
            return 1

        # Extract job ID from result metadata
        job_id = result.metadata.get("job_id")
        if not job_id:
            logger.error("No job ID returned from evaluation")
            return 1

        logger.info(f"✓ Evaluation job submitted successfully!")
        logger.info(f"Job ID: {job_id}")
        logger.info("")
        logger.info("Monitor evaluation progress:")
        logger.info(f"  • Status: /ml jobs status {job_id}")
        logger.info(f"  • Logs: /ml jobs logs {job_id}")
        logger.info("")

        # Launch TensorBoard if requested
        if args.tensorboard and tensorboard_manager:
            try:
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
                logger.info(f"  tensorboard --logdir tensorboard/run_{job_id}")
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
    sys.exit(asyncio.run(main()))
