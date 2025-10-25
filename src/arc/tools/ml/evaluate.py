"""ML Evaluator specification generation and evaluation execution tool."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from arc.ml.runtime import MLRuntime

from arc.core.agents.ml_evaluate import MLEvaluateAgent
from arc.graph.evaluator import EvaluatorValidationError, validate_evaluator_dict
from arc.tools.base import BaseTool, ToolResult
from arc.utils.yaml_workflow import YamlConfirmationWorkflow


class MLEvaluateTool(BaseTool):
    """Unified tool for generating evaluator specs and launching evaluation.

    This tool combines evaluator generation with evaluation execution in a single
    workflow, similar to the MLTrainTool pattern. It provides:
    1. Evaluator spec generation via LLM
    2. Interactive confirmation workflow for evaluator spec
    3. Auto-registration to database
    4. Async evaluation launch (returns immediately with job_id)

    The evaluation runs in the background and results can be monitored via:
    - /ml jobs status {job_id}
    - /ml jobs logs {job_id}
    - TensorBoard for metrics visualization

    This replaces the separate generate-evaluator command with a unified workflow.
    """

    def __init__(
        self,
        services,
        runtime: MLRuntime,
        api_key: str | None,
        base_url: str | None,
        model: str | None,
        ui_interface,
        tensorboard_manager=None,
    ) -> None:
        self.services = services
        self.runtime = runtime
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
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
            from arc.graph import ModelSpec

            model_spec = ModelSpec.from_yaml(model_spec_yaml)

            # Check if model has loss specification
            if not model_spec.loss:
                return None

            # Look for 'target' input in loss specification
            if model_spec.loss.inputs and "target" in model_spec.loss.inputs:
                return model_spec.loss.inputs["target"]

            return None
        except Exception:
            return None

    async def execute(
        self,
        *,
        name: str | None = None,
        instruction: str | None = None,
        trainer_id: str | None = None,
        evaluate_table: str | None = None,
        auto_confirm: bool = False,
        ml_plan: dict | None = None,
    ) -> ToolResult:
        """Generate evaluator spec, register it, and launch evaluation.

        Args:
            name: Evaluator name
            instruction: User's instruction for evaluation setup (PRIMARY driver)
            trainer_id: Trainer ID with version (e.g., 'my-trainer-v1')
            evaluate_table: Test dataset table name
            auto_confirm: Skip confirmation workflows (for testing only)
            ml_plan: Optional ML plan dict containing evaluation expectations
                (SECONDARY baseline, automatically injected by the main agent)

        Note on instruction vs ml_plan precedence:
            - instruction: PRIMARY driver - user's immediate, specific request
            - ml_plan: SECONDARY baseline - background guidance and context
            - When there's a conflict, instruction takes precedence
            - Example: If instruction says "compute F1 score" but plan says
              "compute accuracy only", the evaluator should compute F1 score
              (instruction wins)
            - The LLM agent should use ml_plan as baseline context and augment/override
              it with specifics from instruction

        Note on async execution:
            This tool returns immediately after launching the evaluation job.
            The evaluation runs in the background and results can be monitored via
            job status, logs, and TensorBoard.

        Returns:
            ToolResult with job_id for monitoring async evaluation
        """
        # Show section title first, before any validation
        # Keep the section printer reference to use for all messages including errors
        ml_evaluator_section_printer = None
        if self.ui:
            self._ml_evaluator_section = self.ui._printer.section(
                color="magenta", add_dot=True
            )
            ml_evaluator_section_printer = self._ml_evaluator_section.__enter__()
            ml_evaluator_section_printer.print("ML Evaluator")

        # Helper to close section and return error
        def _error_in_section(message: str) -> ToolResult:
            if ml_evaluator_section_printer:
                ml_evaluator_section_printer.print("")
                ml_evaluator_section_printer.print(f" {message}")
            if self.ui and hasattr(self, "_ml_evaluator_section"):
                self._ml_evaluator_section.__exit__(None, None, None)
            return ToolResult(success=False, output="", metadata={"error_shown": True})

        # Validate API key and services
        if not self.api_key:
            return _error_in_section(
                "API key required for evaluator generation. "
                "Set ARC_API_KEY or configure an API key before using this tool."
            )

        if not self.services:
            return _error_in_section(
                "Evaluator generation service unavailable. "
                "Database services not initialized."
            )

        # Validate required parameters
        if not name or not instruction or not trainer_id or not evaluate_table:
            return _error_in_section(
                "Parameters 'name', 'instruction', 'trainer_id', and 'evaluate_table' "
                "are required."
            )

        # Get the registered trainer
        try:
            trainer_record = self.services.trainers.get_trainer_by_id(str(trainer_id))
            if not trainer_record:
                return _error_in_section(
                    f"Trainer '{trainer_id}' not found in registry. "
                    "Please train a model first using /ml train"
                )
        except Exception as exc:
            return _error_in_section(
                f"Failed to retrieve trainer '{trainer_id}': {exc}"
            )

        # Get model spec and infer target column
        try:
            model_record = self.services.models.get_model_by_id(trainer_record.model_id)
            if not model_record:
                return _error_in_section(
                    f"Model '{trainer_record.model_id}' not found in registry"
                )

            # Infer target column from model spec
            target_column = self._infer_target_column_from_model(model_record.spec)
            if not target_column:
                return _error_in_section(
                    "Cannot infer target column from model spec. "
                    "Ensure model's loss spec includes a 'target' input."
                )

        except Exception as exc:
            return _error_in_section(f"Failed to retrieve model for trainer: {exc}")

        # Check if target column exists in evaluate table
        target_column_exists = False
        try:
            schema_info = self.services.schema.get_schema_info(target_db="user")
            columns = schema_info.get_column_names(str(evaluate_table))
            target_column_exists = str(target_column) in columns
        except Exception:
            # If schema check fails, default to assuming target exists
            target_column_exists = True

        # Show generation status
        if ml_evaluator_section_printer:
            ml_evaluator_section_printer.print(
                "[dim]Generating Arc-Graph evaluator specification...[/dim]"
            )

        # Extract evaluation guidance from ML plan if provided (Phase 6)
        ml_plan_evaluation = None
        if ml_plan:
            from arc.core.ml_plan import MLPlan

            plan = MLPlan.from_dict(ml_plan)
            ml_plan_evaluation = plan.evaluation

        # Extract knowledge IDs from instruction and ML Plan
        from arc.core.agents.shared.knowledge_selector import (
            extract_knowledge_ids_from_text,
        )

        recommended_knowledge_ids = extract_knowledge_ids_from_text(
            instruction=instruction,
            ml_plan_architecture=ml_plan_evaluation,
        )

        # Generate evaluator spec via LLM
        agent = MLEvaluateAgent(
            self.services,
            self.api_key,
            self.base_url,
            self.model,
        )

        try:
            (
                evaluator_spec,
                evaluator_yaml,
                conversation_history,
            ) = await agent.generate_evaluator(
                name=str(name),
                instruction=str(instruction),
                trainer_ref=str(trainer_id),
                trainer_spec_yaml=trainer_record.spec,
                dataset=str(evaluate_table),
                target_column=str(target_column),
                target_column_exists=target_column_exists,
                ml_plan_evaluation=ml_plan_evaluation,
                recommended_knowledge_ids=recommended_knowledge_ids,
            )
        except Exception as exc:
            from arc.core.agents.ml_evaluate import MLEvaluateError

            if isinstance(exc, MLEvaluateError):
                return ToolResult.error_result(str(exc))
            return ToolResult.error_result(
                f"Unexpected error during evaluator generation: {exc}"
            )

        # Validate the generated evaluator
        try:
            evaluator_dict = yaml.safe_load(evaluator_yaml)
            validate_evaluator_dict(evaluator_dict)
        except yaml.YAMLError as exc:
            return ToolResult.error_result(
                f"Generated evaluator contains invalid YAML: {exc}"
            )
        except EvaluatorValidationError as exc:
            return ToolResult.error_result(
                f"Generated evaluator failed validation: {exc}"
            )
        except Exception as exc:
            # Log unexpected errors with full traceback
            import logging

            logging.exception("Unexpected error during evaluator validation")
            return ToolResult.error_result(
                f"Unexpected validation error: {exc.__class__.__name__}: {exc}"
            )

        # Interactive confirmation workflow (unless auto_confirm is True)
        if not auto_confirm:
            workflow = YamlConfirmationWorkflow(
                validator_func=self._create_validator(),
                editor_func=self._create_editor(
                    instruction, trainer_id, trainer_record, target_column_exists
                ),
                ui_interface=self.ui,
                yaml_type_name="evaluator",
                yaml_suffix=".arc-evaluator.yaml",
            )

            context_dict = {
                "evaluator_name": str(name),
                "instruction": str(instruction),
                "trainer_id": str(trainer_id),
                "trainer_ref": str(trainer_id),
                "trainer_spec_yaml": trainer_record.spec,
                "dataset": str(evaluate_table),
                "target_column": str(target_column),
                "target_column_exists": target_column_exists,
            }

            try:
                proceed, final_yaml = await workflow.run_workflow(
                    evaluator_yaml,
                    context_dict,
                    None,  # No output path - we register to DB
                    conversation_history,  # Pass conversation history for editing
                )
                if not proceed:
                    # Close the section before returning
                    if self.ui and hasattr(self, "_ml_evaluator_section"):
                        self._ml_evaluator_section.__exit__(None, None, None)
                    return ToolResult(
                        success=True,
                        output=" Evaluator cancelled.",
                        metadata={"cancelled": True},
                    )
                evaluator_yaml = final_yaml
            finally:
                workflow.cleanup()

        # Auto-register evaluator to database (or reuse existing)
        try:
            from datetime import UTC, datetime

            from arc.database.models.evaluator import Evaluator

            # Check if evaluator with same spec already exists
            existing_evaluator = self.services.evaluators.get_latest_evaluator_by_name(
                str(name)
            )
            evaluator_record = None

            # Use semantic YAML comparison instead of string comparison
            # This handles whitespace differences and formatting variations
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
                # Display "using existing" message in the ML Evaluator section
                if ml_evaluator_section_printer:
                    ml_evaluator_section_printer.print(
                        ""
                    )  # Empty line before confirmation
                    ml_evaluator_section_printer.print(
                        f"[dim]✓ Using existing evaluator '{name}' "
                        f"({evaluator_record.id} • "
                        f"Trainer: {evaluator_spec.trainer_ref} • "
                        f"Dataset: {evaluator_spec.dataset})[/dim]"
                    )
            else:
                # Create new version (spec changed or first time)
                next_version = self.services.evaluators.get_next_version_for_name(
                    str(name)
                )
                evaluator_id = f"{name}-v{next_version}"

                evaluator_record = Evaluator(
                    id=evaluator_id,
                    name=str(name),
                    version=next_version,
                    trainer_id=trainer_record.id,
                    trainer_version=trainer_record.version,
                    spec=evaluator_yaml,
                    description=f"Generated evaluator for trainer {trainer_id}",
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )

                self.services.evaluators.create_evaluator(evaluator_record)

                # Display registration confirmation in the ML Evaluator section
                if ml_evaluator_section_printer:
                    ml_evaluator_section_printer.print(
                        ""
                    )  # Empty line before confirmation
                    ml_evaluator_section_printer.print(
                        f"[dim]✓ Evaluator '{name}' registered to database "
                        f"({evaluator_record.id} • "
                        f"Trainer: {evaluator_spec.trainer_ref} • "
                        f"Dataset: {evaluator_spec.dataset})[/dim]"
                    )
        except Exception as exc:
            return ToolResult.error_result(f"Failed to register evaluator: {exc}")

        # Build simple output for ToolResult (detailed output already shown in UI)
        lines = [f"Evaluator '{name}' registered successfully as {evaluator_record.id}"]

        # Launch evaluation as background job (async pattern)
        if ml_evaluator_section_printer:
            ml_evaluator_section_printer.print("")
            ml_evaluator_section_printer.print(
                f"→ Launching evaluation with '{name}'..."
            )

        # Create job record for this evaluation
        from arc.jobs.models import Job, JobType

        job = Job.create(
            job_type=JobType.EVALUATE_MODEL,
            model_id=None,  # Not using legacy model_id
            message=f"Evaluating {evaluator_record.id} on {evaluate_table}",
        )
        self.services.jobs.create_job(job)

        # Create evaluation run record
        from arc.database.models.evaluation import EvaluationStatus
        from arc.database.services import EvaluationTrackingService

        tracking_service = EvaluationTrackingService(self.services.trainers.db_manager)

        try:
            eval_run = tracking_service.create_run(
                evaluator_id=evaluator_record.id,
                trainer_id=trainer_record.id,
                dataset=str(evaluate_table),
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

            # Load evaluator from trainer
            from arc.ml.evaluator import ArcEvaluator

            evaluator = ArcEvaluator.load_from_trainer(
                artifact_manager=self.runtime.artifact_manager,
                trainer_service=self.services.trainers,
                evaluator_spec=evaluator_spec,
                device="cpu",
            )

            # Derive output table name from evaluator ID
            output_table = f"{evaluator_record.id}_predictions"

            # Setup TensorBoard logging directory
            tensorboard_log_dir = Path(f"tensorboard/run_{job.job_id}")

            # Launch evaluation in background and return immediately
            async def run_evaluation_background():
                """Background task to run evaluation without blocking."""
                try:
                    result = await asyncio.to_thread(
                        evaluator.evaluate,
                        self.services.ml_data,
                        output_table,
                        tensorboard_log_dir,
                    )

                    # Update run with results
                    tracking_service.update_run_result(
                        eval_run.run_id,
                        metrics_result=result.metrics,
                        prediction_table=output_table,
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
            lines.append(" Evaluation job submitted successfully.")
            lines.append(f"  Evaluator: {evaluator_record.id}")
            lines.append(f"  Dataset: {evaluate_table}")
            lines.append(f"  Job ID: {job.job_id}")
            lines.append(f"  Run ID: {eval_run.run_id}")

            # Show job monitoring instructions in section
            if ml_evaluator_section_printer:
                ml_evaluator_section_printer.print("")
                ml_evaluator_section_printer.print(
                    "[dim][cyan]ℹ Monitor evaluation progress:[/cyan][/dim]"
                )
                ml_evaluator_section_printer.print(
                    f"[dim]  • Status: /ml jobs status {job.job_id}[/dim]"
                )
                ml_evaluator_section_printer.print(
                    f"[dim]  • Logs: /ml jobs logs {job.job_id}[/dim]"
                )

            # Handle TensorBoard launch for monitoring
            if not auto_confirm and self.ui:
                if self.tensorboard_manager:
                    try:
                        await self._handle_tensorboard_launch(
                            job.job_id, ml_evaluator_section_printer
                        )
                    except (OSError, RuntimeError) as e:
                        # Known TensorBoard launch failures
                        if ml_evaluator_section_printer:
                            ml_evaluator_section_printer.print(
                                f"[yellow]⚠️  TensorBoard setup failed: {e}[/yellow]"
                            )
                        self._show_manual_tensorboard_instructions(
                            job.job_id, ml_evaluator_section_printer
                        )
                    except Exception as e:
                        # Log unexpected errors with full traceback
                        import logging

                        logging.exception("Unexpected error during TensorBoard launch")
                        error_msg = f"{e.__class__.__name__}: {e}"
                        if ml_evaluator_section_printer:
                            ml_evaluator_section_printer.print(
                                f"[yellow]⚠️  TensorBoard setup failed: "
                                f"{error_msg}[/yellow]"
                            )
                        self._show_manual_tensorboard_instructions(
                            job.job_id, ml_evaluator_section_printer
                        )
                else:
                    # No TensorBoard manager available
                    if ml_evaluator_section_printer:
                        ml_evaluator_section_printer.print(
                            "[dim]9  TensorBoard auto-launch not available "
                            "(restart arc chat to enable)[/dim]"
                        )
                    self._show_manual_tensorboard_instructions(
                        job.job_id, ml_evaluator_section_printer
                    )

            # Evaluation launched successfully - job status can be checked separately

        except Exception as exc:
            # Evaluator was successfully registered but evaluation launch failed
            lines.append("")
            lines.append("⚠️  Evaluation launch failed but evaluator is registered.")
            lines.append(f"Error: {exc}")
            lines.append("")
            retry_cmd = f"/ml evaluate --evaluator {name} --data {evaluate_table}"
            lines.append(f"Retry evaluation with: {retry_cmd}")

            # Close the section before returning
            if self.ui and hasattr(self, "_ml_evaluator_section"):
                self._ml_evaluator_section.__exit__(None, None, None)

            return ToolResult(
                success=True,  # Evaluator registration succeeded
                output="\n".join(lines),
                metadata={
                    "evaluator_id": evaluator_record.id,
                    "evaluator_name": name,
                    "trainer_id": trainer_record.id,
                    "yaml_content": evaluator_yaml,
                    "evaluation_launched": False,
                    "evaluation_error": str(exc),
                    "from_ml_plan": ml_plan is not None,
                },
            )

        # Close the ML Evaluator section
        if self.ui and hasattr(self, "_ml_evaluator_section"):
            self._ml_evaluator_section.__exit__(None, None, None)

        # Build result metadata
        result_metadata = {
            "evaluator_id": evaluator_record.id,
            "evaluator_name": name,
            "trainer_id": trainer_record.id,
            "yaml_content": evaluator_yaml,
            "evaluation_launched": True,
            "job_id": job.job_id,
            "run_id": eval_run.run_id,
            "from_ml_plan": ml_plan is not None,
        }

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata=result_metadata,
        )

    def _create_validator(self):
        """Create validator function for the workflow."""

        def validate(yaml_str: str) -> list[str]:
            try:
                evaluator_dict = yaml.safe_load(yaml_str)
                validate_evaluator_dict(evaluator_dict)
                return []  # No errors
            except yaml.YAMLError as e:
                return [f"Invalid YAML: {e}"]
            except EvaluatorValidationError as e:
                return [f"Validation error: {e}"]
            except Exception as e:
                return [f"Unexpected error: {e}"]

        return validate

    def _create_editor(
        self,
        _user_instruction: str,
        trainer_id: str,
        trainer_record,
        target_column_exists: bool,
    ):
        """Create editor function for AI-assisted editing with conversation history."""

        async def edit(
            yaml_content: str,
            feedback: str,
            context: dict[str, Any],
            conversation_history: list[dict[str, str]] | None = None,
        ) -> tuple[str | None, list[dict[str, str]] | None]:
            # Extract knowledge IDs from feedback
            from arc.core.agents.shared.knowledge_selector import (
                extract_knowledge_ids_from_text,
            )

            recommended_knowledge_ids = extract_knowledge_ids_from_text(
                instruction=feedback,
                ml_plan_architecture=None,
            )

            agent = MLEvaluateAgent(
                self.services,
                self.api_key,
                self.base_url,
                self.model,
            )

            try:
                (
                    _evaluator_spec,
                    edited_yaml,
                    updated_history,
                ) = await agent.generate_evaluator(
                    name=context["evaluator_name"],
                    instruction=feedback,
                    trainer_ref=trainer_id,
                    trainer_spec_yaml=trainer_record.spec,
                    dataset=context["dataset"],
                    target_column=context["target_column"],
                    target_column_exists=target_column_exists,
                    existing_yaml=yaml_content,
                    recommended_knowledge_ids=recommended_knowledge_ids,
                    conversation_history=conversation_history,
                )
                return edited_yaml, updated_history
            except Exception as e:
                if self.ui:
                    self.ui.show_system_error(f"❌ Edit failed: {e}")
                return None, None

        return edit

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
                    f"[green]✓ TensorBoard preference saved: {mode}[/green]"
                )
            else:
                self.ui._printer.console.print()
                self.ui._printer.console.print(
                    f"[green]✓ TensorBoard preference saved: {mode}[/green]"
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
                        "[green]✓ TensorBoard preference updated: always[/green]"
                    )
                else:
                    self.ui._printer.console.print()
                    self.ui._printer.console.print(
                        "[green]✓ TensorBoard preference updated: always[/green]"
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
                section_printer.print("")
                section_printer.print(
                    f"[dim]  To stop: /ml tensorboard stop {job_id}[/dim]"
                )
            else:
                self.ui._printer.console.print()
                self.ui._printer.console.print(
                    "[green]→ Launching TensorBoard...[/green]"
                )
                self.ui._printer.console.print(f"  • URL: [bold]{url}[/bold]")
                self.ui._printer.console.print(f"  • Process ID: {pid}")
                self.ui._printer.console.print(f"  • Logs: {logdir}")
                self.ui._printer.console.print()
                self.ui._printer.console.print(
                    f"  To stop: /ml tensorboard stop {job_id}"
                )
        except (OSError, RuntimeError) as e:
            # Known TensorBoard launch failures
            if section_printer:
                section_printer.print(
                    f"[yellow]⚠️  Failed to launch TensorBoard: {e}[/yellow]"
                )
            else:
                self.ui._printer.console.print(
                    f"[yellow]⚠️  Failed to launch TensorBoard: {e}[/yellow]"
                )
            self._show_manual_tensorboard_instructions(job_id, section_printer)
        except Exception as e:
            # Log unexpected errors with full traceback
            import logging

            logging.exception("Unexpected error during TensorBoard launch")
            error_msg = f"{e.__class__.__name__}: {e}"
            if section_printer:
                section_printer.print(
                    f"[yellow]⚠️  Failed to launch TensorBoard: {error_msg}[/yellow]"
                )
            else:
                self.ui._printer.console.print(
                    f"[yellow]⚠️  Failed to launch TensorBoard: {error_msg}[/yellow]"
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
