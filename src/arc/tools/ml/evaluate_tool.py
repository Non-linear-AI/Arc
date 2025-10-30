"""ML Evaluate tool for evaluating trained models on test datasets."""

from __future__ import annotations

import asyncio
from pathlib import Path

import yaml

from arc.ml.runtime import MLRuntime
from arc.tools.base import BaseTool, ToolResult


class MLEvaluateTool(BaseTool):
    """Tool for evaluating trained models on test datasets.

    This tool provides:
    1. Automatic target column inference from model specification
    2. Direct evaluator creation (no LLM generation needed)
    3. Auto-registration to database
    4. Async evaluation launch (returns immediately with job_id)

    The evaluation runs in the background and results can be monitored via:
    - /ml jobs status {job_id}
    - /ml jobs logs {job_id}
    - TensorBoard for metrics visualization
    """

    def __init__(
        self,
        services,
        runtime: MLRuntime,
        ui_interface,
        tensorboard_manager=None,
    ) -> None:
        self.services = services
        self.runtime = runtime
        self.ui = ui_interface
        self.tensorboard_manager = tensorboard_manager

    def _infer_target_column_from_model(self, model_spec_yaml: str) -> str | None:
        """Infer target column from model spec's loss inputs.

        Args:
            model_spec_yaml: Model specification YAML string

        Returns:
            Target column name if found in loss spec, None otherwise
        """
        try:
            import yaml

            # Parse YAML directly to check both locations
            spec_dict = yaml.safe_load(model_spec_yaml)

            # Check model-level loss first (new format)
            if "loss" in spec_dict:
                loss_spec = spec_dict["loss"]
                if isinstance(loss_spec, dict) and "inputs" in loss_spec:
                    inputs = loss_spec["inputs"]
                    if isinstance(inputs, dict) and "target" in inputs:
                        return inputs["target"]

            # Fall back to training.loss (old format or if not found at model level)
            if "training" in spec_dict:
                training = spec_dict["training"]
                if isinstance(training, dict) and "loss" in training:
                    loss_spec = training["loss"]
                    if isinstance(loss_spec, dict) and "inputs" in loss_spec:
                        inputs = loss_spec["inputs"]
                        if isinstance(inputs, dict) and "target" in inputs:
                            return inputs["target"]

            return None
        except Exception:
            return None

    async def execute(
        self,
        *,
        model_id: str | None = None,
        data_table: str | None = None,
        metrics: list[str] | None = None,
        output_table: str | None = None,
        auto_confirm: bool = False,
    ) -> ToolResult:
        """Evaluate a trained model on test dataset.

        Args:
            model_id: Model ID with version (e.g., 'my-model-v1')
            data_table: Test dataset table name
            metrics: Optional list of metrics to compute (inferred from model if not provided)
            output_table: Optional table to save predictions
            auto_confirm: Skip confirmation workflows (for testing only)

        Note on async execution:
            This tool returns immediately after launching the evaluation job.
            The evaluation runs in the background and results can be monitored via
            job status, logs, and TensorBoard.

        Returns:
            ToolResult with job_id for monitoring async evaluation
        """
        # Use context manager for section printing
        with self._section_printer(self.ui, "ML Evaluate") as printer:
            # Helper to show error and return
            def _error_in_section(message: str) -> ToolResult:
                if printer:
                    printer.print("")
                    printer.print(f"✗ {message}")
                return ToolResult(
                    success=False,
                    output=message,
                    metadata={"error_shown": True, "error_message": message},
                )

            # Validate services
            if not self.services:
                return _error_in_section(
                    "Evaluation service unavailable. Database services not initialized."
                )

            # Validate required parameters
            if not model_id or not data_table:
                return _error_in_section(
                    "Parameters 'model_id' and 'data_table' are required."
                )

            # Get the registered model
            try:
                model_record = self.services.models.get_model_by_id(str(model_id))
                if not model_record:
                    return _error_in_section(
                        f"Model '{model_id}' not found in registry. "
                        "Train a model first using /ml model"
                    )
            except Exception as exc:
                return _error_in_section(f"Failed to retrieve model '{model_id}': {exc}")

            # Infer target column from model spec
            try:
                target_column = self._infer_target_column_from_model(model_record.spec)
                if not target_column:
                    return _error_in_section(
                        "Cannot infer target column from model spec. "
                        "Ensure model's loss spec includes a 'target' input."
                    )
            except Exception as exc:
                return _error_in_section(f"Failed to infer target column: {exc}")

            # Check if target column exists in data_table
            target_column_exists = False
            try:
                schema_info = self.services.schema.get_schema_info(target_db="user")
                columns = schema_info.get_column_names(str(data_table))
                target_column_exists = str(target_column) in columns
            except Exception:
                # If schema check fails, default to assuming target exists
                target_column_exists = True

            # Create evaluator spec directly (no LLM generation needed)
            from arc.graph.evaluator import EvaluatorSpec

            # Generate evaluator name from model_id if not provided
            evaluator_name = f"{model_id}_evaluator"

            evaluator_spec = EvaluatorSpec(
                name=evaluator_name,
                trainer_ref=str(model_id),  # Use model_id as trainer reference
                dataset=str(data_table),
                target_column=str(target_column),
                metrics=metrics,  # None = infer from model's loss function
                version=None,  # Use latest training run
                output_name=None,  # Auto-detect from model outputs
            )

            # Convert to YAML for database storage
            evaluator_yaml = evaluator_spec.to_yaml()

            if printer:
                printer.print("")
                printer.print(
                    f"[dim]✓ Evaluator created for model '{model_id}' "
                    f"on dataset '{data_table}'[/dim]"
                )

            # Auto-register evaluator to database (or reuse existing)
            try:
                from datetime import UTC, datetime

                from arc.database.models.evaluator import Evaluator

                # Check if evaluator with same spec already exists
                existing_evaluator = self.services.evaluators.get_latest_evaluator_by_name(
                    evaluator_name
                )
                evaluator_record = None

                # Use semantic YAML comparison instead of string comparison
                spec_matches = False
                if existing_evaluator:
                    try:
                        existing_spec_dict = yaml.safe_load(existing_evaluator.spec)
                        new_spec_dict = yaml.safe_load(evaluator_yaml)
                        spec_matches = existing_spec_dict == new_spec_dict
                    except yaml.YAMLError:
                        # If YAML parsing fails, fall back to string comparison
                        spec_matches = (
                            existing_evaluator.spec.strip() == evaluator_yaml.strip()
                        )

                if spec_matches:
                    # Reuse existing evaluator (same spec)
                    evaluator_record = existing_evaluator
                    if printer:
                        printer.print(
                            f"[dim]✓ Using existing evaluator "
                            f"({evaluator_record.id})[/dim]"
                        )
                else:
                    # Create new version (spec changed or first time)
                    next_version = self.services.evaluators.get_next_version_for_name(
                        evaluator_name
                    )
                    evaluator_id = f"{evaluator_name}-v{next_version}"

                    evaluator_record = Evaluator(
                        id=evaluator_id,
                        name=evaluator_name,
                        version=next_version,
                        model_id=model_record.id,
                        model_version=model_record.version,
                        spec=evaluator_yaml,
                        description=f"Evaluator for {model_id} on {data_table}",
                        created_at=datetime.now(UTC),
                        updated_at=datetime.now(UTC),
                    )

                    self.services.evaluators.create_evaluator(evaluator_record)

                    if printer:
                        printer.print(
                            f"[dim]✓ Evaluator registered ({evaluator_record.id})[/dim]"
                        )
            except Exception as exc:
                return _error_in_section(f"Failed to register evaluator: {exc}")

            # Build simple output for ToolResult
            lines = [
                f"Evaluator '{evaluator_name}' registered as {evaluator_record.id}"
            ]

            # Launch evaluation as background job (async pattern)
            if printer:
                printer.print("")
                printer.print("→ Launching evaluation...")

            # Create job record for this evaluation
            from arc.jobs.models import Job, JobType

            job = Job.create(
                job_type=JobType.EVALUATE_MODEL,
                model_id=None,  # Not using legacy model_id
                message=f"Evaluating {evaluator_record.id} on {data_table}",
            )
            self.services.jobs.create_job(job)

            # Create evaluation run record
            from arc.database.models.evaluation import EvaluationStatus
            from arc.database.services import EvaluationTrackingService

            tracking_service = EvaluationTrackingService(
                self.services.models.db_manager
            )

            try:
                eval_run = tracking_service.create_run(
                    evaluator_id=evaluator_record.id,
                    model_id=model_record.id,
                    dataset=str(data_table),
                    target_column=str(target_column),
                    job_id=job.job_id,
                )

                # Update job and run status to running
                job.start(f"Running evaluation: {evaluator_record.id}")
                self.services.jobs.update_job(job)

                tracking_service.update_run_status(
                    eval_run.run_id,
                    EvaluationStatus.RUNNING,
                    timestamp_field="started_at",
                )

                # Load evaluator from model
                from arc.ml.evaluator import ArcEvaluator

                evaluator = ArcEvaluator.load_from_model(
                    artifact_manager=self.runtime.artifact_manager,
                    model_service=self.services.models,
                    evaluator_spec=evaluator_spec,
                    device="cpu",
                )

                # Use provided output_table or None (don't save predictions by default)
                prediction_table = output_table if output_table else None

                # Setup TensorBoard logging directory
                from pathlib import Path

                tensorboard_log_dir = Path(f"tensorboard/run_{job.job_id}")

                # Launch evaluation in background and return immediately
                async def run_evaluation_background():
                    """Background task to run evaluation without blocking."""
                    try:
                        result = await asyncio.to_thread(
                            evaluator.evaluate,
                            self.services.ml_data,
                            prediction_table,
                            tensorboard_log_dir,
                        )

                        # Update run with results
                        tracking_service.update_run_result(
                            eval_run.run_id,
                            metrics_result=result.metrics,
                            prediction_table=prediction_table,
                        )

                        # Mark as completed
                        tracking_service.update_run_status(
                            eval_run.run_id,
                            EvaluationStatus.COMPLETED,
                            timestamp_field="completed_at",
                        )

                        # Update job status
                        metrics_summary = ", ".join(
                            f"{k}={v:.4f}" for k, v in list(result.metrics.items())[:3]
                        )
                        job.complete(f"Evaluation completed: {metrics_summary}")
                        self.services.jobs.update_job(job)

                    except Exception as exc:
                        # Update run with error
                        try:
                            tracking_service.update_run_error(eval_run.run_id, str(exc))
                            tracking_service.update_run_status(
                                eval_run.run_id,
                                EvaluationStatus.FAILED,
                            )
                        except Exception:  # noqa: S110
                            pass

                        # Update job with error
                        try:
                            job.fail(f"Evaluation failed: {str(exc)[:200]}")
                            self.services.jobs.update_job(job)
                        except Exception:  # noqa: S110
                            pass

                # Launch background task (fire-and-forget)
                asyncio.create_task(run_evaluation_background())

                # Return immediately with job info
                lines.append("")
                lines.append("✓ Evaluation job submitted successfully.")
                lines.append(f"  Evaluator: {evaluator_record.id}")
                lines.append(f"  Dataset: {data_table}")
                lines.append(f"  Job ID: {job.job_id}")
                lines.append(f"  Run ID: {eval_run.run_id}")

                # Show job monitoring instructions in section
                if printer:
                    printer.print("")
                    printer.print(
                        "[dim][cyan]ℹ Monitor evaluation progress:[/cyan][/dim]"
                    )
                    printer.print(
                        f"[dim]  • Status: /ml jobs status {job.job_id}[/dim]"
                    )
                    printer.print(f"[dim]  • Logs: /ml jobs logs {job.job_id}[/dim]")

                # Handle TensorBoard launch for monitoring
                if not auto_confirm and self.ui:
                    if self.tensorboard_manager:
                        try:
                            await self._handle_tensorboard_launch(job.job_id, printer)
                        except (OSError, RuntimeError) as e:
                            # Known TensorBoard launch failures
                            if printer:
                                printer.print(
                                    f"[yellow]⚠️  TensorBoard setup failed: {e}[/yellow]"
                                )
                            self._show_manual_tensorboard_instructions(
                                job.job_id, printer
                            )
                        except Exception as e:
                            # Log unexpected errors with full traceback
                            import logging

                            logging.exception(
                                "Unexpected error during TensorBoard launch"
                            )
                            error_msg = f"{e.__class__.__name__}: {e}"
                            if printer:
                                printer.print(
                                    f"[yellow]⚠️  TensorBoard setup failed: "
                                    f"{error_msg}[/yellow]"
                                )
                            self._show_manual_tensorboard_instructions(
                                job.job_id, printer
                            )
                    else:
                        # No TensorBoard manager available
                        if printer:
                            printer.print(
                                "[dim]ℹ TensorBoard auto-launch not available "
                                "(restart arc chat to enable)[/dim]"
                            )
                        self._show_manual_tensorboard_instructions(job.job_id, printer)

                # Evaluation launched successfully - job status can be
                # checked separately

            except Exception as exc:
                # Evaluator was successfully registered but evaluation launch failed
                lines.append("")
                lines.append("⚠️  Evaluation launch failed but evaluator is registered.")
                lines.append(f"Error: {exc}")

                return ToolResult(
                    success=True,  # Evaluator registration succeeded
                    output="\n".join(lines),
                    metadata={
                        "evaluator_id": evaluator_record.id,
                        "evaluator_name": evaluator_name,
                        "model_id": model_record.id,
                        "evaluation_launched": False,
                        "evaluation_error": str(exc),
                    },
                )

            # Build result metadata
            result_metadata = {
                "evaluator_id": evaluator_record.id,
                "evaluator_name": evaluator_name,
                "model_id": model_record.id,
                "evaluation_launched": True,
                "job_id": job.job_id,
                "run_id": eval_run.run_id,
            }

            return ToolResult(
                success=True,
                output="\n".join(lines),
                metadata=result_metadata,
            )

    async def _handle_tensorboard_launch(self, job_id: str, section_printer=None):
        """Handle TensorBoard launch based on user preference.

        Args:
            job_id: Evaluation job identifier
            section_printer: Section printer for indented output
        """
        from arc.core.config import SettingsManager
        from arc.utils.tensorboard_workflow import prompt_tensorboard_preference

        settings = SettingsManager()
        mode = settings.get_tensorboard_mode()

        # First time - no preference set, show combined dialog
        if mode is None:
            mode, should_launch = await prompt_tensorboard_preference(self.ui)
            settings.set_tensorboard_mode(mode)
            if section_printer:
                section_printer.print("")
                section_printer.print(
                    f"[green]✓ TensorBoard preference saved: {mode}[/green]"
                )
            else:
                self.ui._printer.console.print()
                self.ui._printer.console.print(
                    f"[green]✓ TensorBoard preference saved: {mode}[/green]"
                )

            # Launch immediately if user chose to
            if should_launch:
                await self._launch_tensorboard(job_id, section_printer)
            else:
                self._show_manual_tensorboard_instructions(job_id, section_printer)

        # Subsequent times - respect saved preference
        elif mode == "always":
            await self._launch_tensorboard(job_id, section_printer)
        elif mode == "ask":
            if section_printer:
                section_printer.print("")
                section_printer.print(
                    "[cyan]Launch TensorBoard? (http://localhost:6006)[/cyan]"
                )
            else:
                self.ui._printer.console.print()
                self.ui._printer.console.print(
                    "[cyan]Launch TensorBoard? (http://localhost:6006)[/cyan]"
                )
            choice = await self.ui._printer.get_choice_async(
                options=[
                    ("yes", "Yes, launch now"),
                    ("always", "Always launch automatically"),
                    ("no", "No, skip"),
                ],
                default="yes",
            )

            # Handle the choice
            if choice == "always":
                # Update preference to always
                settings.set_tensorboard_mode("always")
                if section_printer:
                    section_printer.print("")
                    section_printer.print(
                        "[green]✓ TensorBoard preference updated: always[/green]"
                    )
                else:
                    self.ui._printer.console.print()
                    self.ui._printer.console.print(
                        "[green]✓ TensorBoard preference updated: always[/green]"
                    )
                await self._launch_tensorboard(job_id, section_printer)
            elif choice == "yes":
                await self._launch_tensorboard(job_id, section_printer)
            else:  # "no"
                self._show_manual_tensorboard_instructions(job_id, section_printer)
        else:  # "never"
            self._show_manual_tensorboard_instructions(job_id, section_printer)

    async def _launch_tensorboard(self, job_id: str, section_printer=None):
        """Launch TensorBoard and show info.

        Args:
            job_id: Evaluation job identifier
            section_printer: Section printer for indented output
        """

        from arc.core.config import SettingsManager

        logdir = Path(f"tensorboard/run_{job_id}")

        try:
            settings = SettingsManager()
            port = settings.get_tensorboard_port()

            url, pid = self.tensorboard_manager.launch(job_id, logdir, port)

            if section_printer:
                section_printer.print("")
                section_printer.print("[green]→ Launching TensorBoard...[/green]")
                section_printer.print(f"  • URL: [bold]{url}[/bold]")
                section_printer.print(f"[dim]  • Process ID: {pid}[/dim]")
                section_printer.print(f"[dim]  • Logs: {logdir}[/dim]")
            else:
                self.ui._printer.console.print()
                self.ui._printer.console.print(
                    "[green]→ Launching TensorBoard...[/green]"
                )
                self.ui._printer.console.print(f"  • URL: [bold]{url}[/bold]")
                self.ui._printer.console.print(f"  • Process ID: {pid}")
                self.ui._printer.console.print(f"  • Logs: {logdir}")
        except (OSError, RuntimeError) as e:
            # Known TensorBoard launch failures
            if section_printer:
                section_printer.print(
                    f"[yellow]⚠️  Failed to launch TensorBoard: {e}[/yellow]"
                )
            else:
                self.ui._printer.console.print(
                    f"[yellow]⚠️  Failed to launch TensorBoard: {e}[/yellow]"
                )
            self._show_manual_tensorboard_instructions(job_id, section_printer)
        except Exception as e:
            # Log unexpected errors with full traceback
            import logging

            logging.exception("Unexpected error during TensorBoard launch")
            error_msg = f"{e.__class__.__name__}: {e}"
            if section_printer:
                section_printer.print(
                    f"[yellow]⚠️  Failed to launch TensorBoard: {error_msg}[/yellow]"
                )
            else:
                self.ui._printer.console.print(
                    f"[yellow]⚠️  Failed to launch TensorBoard: {error_msg}[/yellow]"
                )
            self._show_manual_tensorboard_instructions(job_id, section_printer)

    def _show_manual_tensorboard_instructions(self, job_id: str, section_printer=None):
        """Show manual TensorBoard instructions.

        Args:
            job_id: Evaluation job identifier
            section_printer: Section printer for indented output
        """
        logdir = f"tensorboard/run_{job_id}"
        if section_printer:
            section_printer.print("")
            section_printer.print("[dim][cyan]ℹ View evaluation results:[/cyan][/dim]")
            section_printer.print(f"[dim]  • Status: /ml jobs status {job_id}[/dim]")
            section_printer.print(
                f"[dim]  • TensorBoard: tensorboard --logdir {logdir}[/dim]"
            )
            section_printer.print("")
            section_printer.print(
                "[dim]TensorBoard will show PR curves, ROC curves, "
                "confusion matrix, and more![/dim]"
            )
        else:
            self.ui._printer.console.print()
            self.ui._printer.console.print("[cyan]ℹ View evaluation results:[/cyan]")
            self.ui._printer.console.print(f"  • Status: /ml jobs status {job_id}")
            self.ui._printer.console.print(
                f"  • TensorBoard: tensorboard --logdir {logdir}"
            )
            self.ui._printer.console.print()
            self.ui._printer.console.print(
                "[dim]TensorBoard will show PR curves, ROC curves, "
                "confusion matrix, and more![/dim]"
            )
