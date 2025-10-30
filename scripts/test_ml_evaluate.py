#!/usr/bin/env python3
"""Test script for ml_evaluate tool (model evaluation).

This script allows you to test model evaluation directly without going
through the agent. Uses the simplified workflow with no LLM generation.

Usage:
    python scripts/test_ml_evaluate.py <model_id> <dataset> [options]

Examples:
    # Basic evaluation
    python scripts/test_ml_evaluate.py diabetes_model-v1 pidd_test_data

    # Evaluation with custom metrics
    python scripts/test_ml_evaluate.py diabetes_model-v1 pidd_test_data --metrics accuracy precision recall f1_score

    # Evaluation with prediction saving
    python scripts/test_ml_evaluate.py diabetes_model-v1 pidd_test_data --output-table diabetes_predictions

    # Monitor evaluation until completion
    python scripts/test_ml_evaluate.py diabetes_model-v1 pidd_test_data --monitor
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

from arc.database import DatabaseManager
from arc.database.services import ServiceContainer
from arc.ml import TensorBoardManager
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
        description="Test ml_evaluate tool (simplified evaluation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "model_id",
        help="Model ID with version (e.g., 'diabetes_model-v1')"
    )

    parser.add_argument(
        "dataset",
        help="Test dataset table name"
    )

    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        help="Optional list of metrics to compute (e.g., accuracy precision recall)"
    )

    parser.add_argument(
        "--output-table",
        "-o",
        help="Optional table name to save predictions"
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
        help="Directory for model artifacts (default: .arc/artifacts)"
    )

    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Monitor evaluation job until completion"
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
        "-v",
        action="store_true",
        help="Enable verbose logging"
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

                # If completed, fetch and display evaluation results
                if status == "completed":
                    try:
                        runs = services.evaluation_tracking.list_runs(job_id=job_id)
                        if runs:
                            run = runs[0]
                            logger.info("\n" + "=" * 80)
                            logger.info("EVALUATION RESULTS")
                            logger.info("=" * 80)
                            if run.metrics_result:
                                import json
                                metrics = json.loads(run.metrics_result)
                                for metric_name, value in metrics.items():
                                    logger.info(f"  {metric_name}: {value:.4f}")
                            if run.prediction_table:
                                logger.info(f"\nPredictions saved to: {run.prediction_table}")
                    except Exception as e:
                        logger.warning(f"Failed to fetch results: {e}")

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
        # Initialize database
        system_db_path = Path(args.system_db).expanduser()
        user_db_path = Path(args.user_db).expanduser()
        logger.info(f"System database: {system_db_path}")
        logger.info(f"User database: {user_db_path}")
        db_manager = DatabaseManager(str(system_db_path), str(user_db_path))

        # Initialize artifacts directory
        artifacts_path = Path(args.artifacts_dir)
        if str(artifacts_path).startswith('~'):
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
        logger.info(f"Model ID: {args.model_id}")
        logger.info(f"Dataset: {args.dataset}")
        if args.metrics:
            logger.info(f"Metrics: {', '.join(args.metrics)}")
        else:
            logger.info("Metrics: Auto-inferred from model's loss function")
        if args.output_table:
            logger.info(f"Output table: {args.output_table}")
        logger.info("=" * 80)

        # Initialize ml_evaluate tool
        tool = MLEvaluateTool(
            services=services,
            runtime=services.ml_runtime,
            ui_interface=None,  # No UI in test mode
            tensorboard_manager=tensorboard_manager,
        )

        # Execute ml_evaluate
        logger.info("\nExecuting ml_evaluate...")
        result = await tool.execute(
            model_id=args.model_id,
            dataset=args.dataset,
            metrics=args.metrics,
            output_table=args.output_table,
            auto_confirm=True,  # Always auto-confirm in test mode
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

        if result.success and result.metadata:
            job_id = result.metadata.get("job_id")
            run_id = result.metadata.get("run_id")
            evaluator_id = result.metadata.get("evaluator_id")

            if evaluator_id:
                logger.info(f"\n✓ Evaluator created: {evaluator_id}")

            if job_id:
                logger.info(f"✓ Evaluation job submitted: {job_id}")
                logger.info(f"✓ Run ID: {run_id}")
                logger.info("\nMonitor evaluation progress:")
                logger.info(f"  • Status: /ml jobs status {job_id}")
                logger.info(f"  • Logs: /ml jobs logs {job_id}")
                logger.info(f"  • TensorBoard: tensorboard --logdir tensorboard/run_{job_id}")

                # Monitor if requested
                if args.monitor:
                    success = monitor_job(services, job_id)
                    return 0 if success else 1

            return 0
        else:
            logger.error("\n✗ Evaluation failed!")
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
