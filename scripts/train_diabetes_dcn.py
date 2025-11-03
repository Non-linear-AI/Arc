#!/usr/bin/env python3
"""Train the diabetes DCN model with the processed data."""

import time
from pathlib import Path

from arc.core import SettingsManager
from arc.database import DatabaseManager
from arc.database.services import ServiceContainer
from arc.ml.runtime import MLRuntime


def main():
    """Train the diabetes DCN model."""
    print("=" * 80)
    print("Training Diabetes DCN Model")
    print("=" * 80)
    print()

    # Initialize database and services
    settings_manager = SettingsManager()
    system_db_path = settings_manager.get_system_database_path()
    user_db_path = settings_manager.get_user_database_path()
    db_manager = DatabaseManager(system_db_path, user_db_path)
    services = ServiceContainer(db_manager, artifacts_dir=".arc/artifacts")

    # Create ML runtime
    runtime = MLRuntime(services, artifacts_dir=Path(".arc/artifacts"))

    # Model configuration
    model_name = "diabetes-dcn-model"  # Model name without version
    data_table = "diabetes_processed"

    print(f"Model: {model_name}")
    print(f"Data table: {data_table}")
    print()

    # Training configuration
    epochs = 50
    batch_size = 32

    print("Training configuration:")
    print(f"  epochs: {epochs}")
    print(f"  batch_size: {batch_size}")
    print("  validation_split: 0.0 (using existing splits)")
    print()

    try:
        # Start training
        print("Starting training...")
        print("-" * 80)
        print("DEBUG: About to call runtime.train_model...")
        import sys

        sys.stdout.flush()

        # Submit training job
        job_id = runtime.train_model(
            model_name=model_name,
            train_table=data_table,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.0,  # Don't split - use existing data
        )

        print(f"DEBUG: train_model returned, job_id={job_id}")
        sys.stdout.flush()

        print(f"Training job submitted: {job_id}")
        print()

        # Wait for training to complete and monitor progress
        print("Monitoring training progress...")
        print()

        last_message = ""
        while True:
            job_status = runtime.training_service.get_job_status(job_id)
            status = job_status["status"]
            message = job_status["message"]

            # Print message if it changed
            if message != last_message:
                print(f"  {message}")
                last_message = message

            if status in ["COMPLETED", "FAILED", "CANCELLED"]:
                break

            time.sleep(2)  # Poll every 2 seconds

        # Wait for job to finish and get result
        result = runtime.training_service.wait_for_job(job_id, timeout=None)

        # Display results
        print()
        print("=" * 80)

        if result and result.success:
            print("Training Complete!")
            print("=" * 80)
            print()

            print(f"Job ID: {job_id}")
            print(f"Training time: {result.training_time:.2f}s")
            print(f"Total epochs: {result.total_epochs}")

            if result.best_epoch is not None:
                print(f"Best epoch: {result.best_epoch}")

            # Display metrics
            if result.final_metrics:
                print("\nFinal Metrics:")
                for metric, value in result.final_metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")

            if result.best_metrics:
                print("\nBest Metrics:")
                for metric, value in result.best_metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")

            if result.model_path:
                print(f"\nModel saved to: {result.model_path}")

            print()
        else:
            print("Training Failed!")
            print("=" * 80)
            print()
            if result:
                print(f"Error: {result.error_message}")
            else:
                print("Error: No result returned")
            print()

    except Exception as e:
        print()
        print("=" * 80)
        print("Training Failed!")
        print("=" * 80)
        print()
        print(f"Error: {e}")
        print()
        import traceback

        traceback.print_exc()
        return 1
    finally:
        # Clean up resources
        try:
            runtime.shutdown()
            services.shutdown()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    exit(main())
