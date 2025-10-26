"""Machine learning tool implementations."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from arc.database.models.model import Model

from arc.core.agents.ml_model import MLModelAgent
from arc.core.agents.ml_plan import MLPlanAgent
from arc.core.agents.ml_train import (
    MLTrainAgent,
)
from arc.graph.model import ModelValidationError, validate_model_dict
from arc.graph.trainer import TrainerValidationError, validate_trainer_dict
from arc.ml.runtime import MLRuntime, MLRuntimeError
from arc.tools.base import BaseTool, ToolResult
from arc.utils.yaml_workflow import YamlConfirmationWorkflow


def _as_optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _as_optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc


def _as_string_list(value: Any, field_name: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or None
    if isinstance(value, Sequence):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned or None
    raise ValueError(f"{field_name} must be an array of strings or comma-separated")


def _load_ml_plan(services, plan_id: str) -> tuple[dict, Any] | tuple[None, str]:
    """Load ML plan from database.

    Args:
        services: ServiceContainer with ml_plans service
        plan_id: ML plan ID to load

    Returns:
        Tuple of (ml_plan_dict, MLPlan_object) on success,
        or (None, error_message) on failure
    """
    try:
        from arc.core.ml_plan import MLPlan

        ml_plan = services.ml_plans.get_plan_content(plan_id)
        plan = MLPlan.from_dict(ml_plan)
        return (ml_plan, plan)
    except ValueError as e:
        return (None, f"Failed to load ML plan '{plan_id}': {e}")
    except Exception as e:
        return (None, f"Unexpected error loading ML plan '{plan_id}': {e}")


class MLModelTool(BaseTool):
    """Tool for generating Arc-Graph model specifications via LLM."""

    def __init__(
        self,
        services,
        api_key: str | None,
        base_url: str | None,
        model: str | None,
        ui_interface,
    ) -> None:
        self.services = services
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.ui = ui_interface

    async def execute(
        self,
        *,
        name: str | None = None,
        instruction: str | None = None,
        data_table: str | None = None,
        target_column: str | None = None,
        auto_confirm: bool = False,
        plan_id: str | None = None,
        recommended_knowledge_ids: list[str] | None = None,
    ) -> ToolResult:
        """Generate Arc-Graph model specification via LLM.

        Args:
            name: Model name
            instruction: User's instruction for model architecture (PRIMARY driver)
            data_table: Database table to profile for generation
            target_column: Target column for prediction
            auto_confirm: Skip confirmation workflows (for testing only)
            plan_id: Optional ML plan ID (e.g., 'pidd-plan-v1') containing
                model_architecture_and_loss guidance (SECONDARY baseline)

        Note on instruction vs plan_id precedence:
            - instruction: PRIMARY driver - user's immediate, specific
              architecture request
            - plan_id: SECONDARY baseline - loads plan from DB for
              architectural guidance
            - When there's a conflict, instruction takes precedence
            - Example: If instruction says "use 3 hidden layers" but plan says
              "use 2 layers", the model spec should use 3 hidden layers
              (instruction wins)
            - The LLM agent should use plan as baseline architectural guidance
              and augment/override it with specifics from instruction

        Returns:
            ToolResult with model registration details
        """
        # Show section title first, before any validation
        # Build metadata for section title
        metadata_parts = []
        if plan_id:
            metadata_parts.append(plan_id)
        if recommended_knowledge_ids:
            metadata_parts.extend(recommended_knowledge_ids)

        # Use context manager for section printing
        with self._section_printer(
            self.ui, "ML Model", metadata=metadata_parts
        ) as printer:
            # Show task description
            if printer:
                printer.print(f"[dim]Task: {instruction}[/dim]")

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

            # Validate API key and services
            if not self.api_key:
                return _error_in_section(
                    "API key required for model generation. "
                    "Set ARC_API_KEY or configure an API key before using this tool."
                )

            if not self.services:
                return _error_in_section(
                    "Model generation service unavailable. "
                    "Database services not initialized."
                )

            # Validate: either plan_id or instruction must be provided
            if not plan_id and not instruction:
                return _error_in_section(
                    "Either 'plan_id' or 'instruction' must be provided. "
                    "ML plan is recommended for full ML workflows."
                )

            # Load plan from database if plan_id is provided
            ml_plan_architecture = None
            if plan_id:
                ml_plan, plan = _load_ml_plan(self.services, plan_id)
                if ml_plan is None:
                    # plan contains error message
                    return _error_in_section(plan)

                # Use plan data if parameters not explicitly provided
                if not instruction:
                    instruction = plan.summary
                if not data_table:
                    data_table = ml_plan.get("data_table")
                if not target_column:
                    target_column = ml_plan.get("target_column")

                # CRITICAL: Extract architecture guidance from ML plan
                ml_plan_architecture = plan.model_architecture_and_loss

            # Validate required parameters
            if not name or not data_table or not target_column:
                return _error_in_section(
                    "Parameters 'name', 'data_table', and 'target_column' are required "
                    "to generate a model specification."
                )

            # Generate model using agent
            agent = MLModelAgent(
                self.services,
                self.api_key,
                self.base_url,
                self.model,
            )

            # Set progress callback for this invocation
            if printer:
                agent.progress_callback = printer.print
            else:
                agent.progress_callback = None

            # Agent will discover relevant knowledge using list_knowledge and
            # read_knowledge tools based on task context and descriptions
            try:
                model_spec, model_yaml, conversation_history = await agent.generate_model(
                    name=str(name),
                    user_context=instruction,  # Use instruction as user_context
                    table_name=str(data_table),
                    target_column=target_column,
                    ml_plan_architecture=ml_plan_architecture,
                    recommended_knowledge_ids=None,  # Let agent discover via tools
                )

                # Show completion message
                if printer:
                    printer.print("[dim]✓ Model generated successfully[/dim]")

            except Exception as exc:
                # Import here to avoid circular imports
                from arc.core.agents.ml_model import MLModelError

                if isinstance(exc, MLModelError):
                    return _error_in_section(str(exc))
                return _error_in_section(
                    f"Unexpected error during model generation: {exc}"
                )

            # Validate the generated model using Arc-Graph validator
            try:
                model_dict = yaml.safe_load(model_yaml)
                validate_model_dict(model_dict)
            except (yaml.YAMLError, ModelValidationError) as exc:
                return _error_in_section(f"Model validation failed: {exc}")
            except Exception as exc:
                # Log unexpected errors with full traceback
                import logging

                logging.exception("Unexpected error during model validation")
                return _error_in_section(
                    f"Unexpected validation error: {exc.__class__.__name__}: {exc}"
                )

            # Interactive confirmation workflow (unless auto_confirm is True)
            if not auto_confirm:
                workflow = YamlConfirmationWorkflow(
                    validator_func=self._create_validator(),
                    editor_func=self._create_editor(instruction),
                    ui_interface=self.ui,
                    yaml_type_name="model",
                    yaml_suffix=".arc-model.yaml",
                )

                context_dict = {
                    "model_name": str(name),
                    "table_name": str(data_table),
                    "target_column": target_column,
                }

                try:
                    proceed, final_yaml = await workflow.run_workflow(
                        model_yaml,
                        context_dict,
                        None,  # No file path
                        conversation_history,  # Pass conversation history for editing
                    )
                    if not proceed:
                        # Show cancellation message before closing section
                        if printer:
                            printer.print("")  # Empty line
                            printer.print(
                                "[dim]✗ Model generation cancelled by user.[/dim]"
                            )
                        return ToolResult(
                            success=True,
                            output="Model generation cancelled by user.",
                            metadata={"cancelled": True},
                        )
                    model_yaml = final_yaml
                finally:
                    workflow.cleanup()

            # Save model to DB with plan_id if using ML plan
            try:
                plan_id = ml_plan.get("plan_id") if ml_plan else None
                model = self._save_model_to_db(
                    name=str(name),
                    yaml_content=model_yaml,
                    description=instruction[:200] if instruction else "Generated model",
                    plan_id=plan_id,
                )
                model_id = model.id
            except Exception as exc:
                return _error_in_section(f"Failed to save model to DB: {exc}")

            # Display registration confirmation in the ML Model section
            if printer:
                printer.print("")  # Empty line before confirmation
                printer.print(
                    f"[dim]✓ Model '{name}' registered to database "
                    f"({model_id} • {len(model_spec.inputs)} inputs • "
                    f"{len(model_spec.graph)} nodes • "
                    f"{len(model_spec.outputs)} outputs)[/dim]"
                )


            # Build simple output for ToolResult (detailed output already shown in UI)
            lines = [f"Model '{name}' registered successfully as {model_id}"]

            # Build simplified metadata
            result_metadata = {
                "model_id": model_id,
                "model_name": name,
                "yaml_content": model_yaml,
                "from_ml_plan": ml_plan is not None,
            }

            return ToolResult(
                success=True,
                output="\n".join(lines),
                metadata=result_metadata,
            )

    def _create_validator(self):
        """Create validator function for the workflow.

        Returns:
            Function that validates YAML and returns list of error strings
        """

        def validate(yaml_str: str) -> list[str]:
            try:
                model_dict = yaml.safe_load(yaml_str)
                validate_model_dict(model_dict)
                return []  # No errors
            except yaml.YAMLError as e:
                return [f"Invalid YAML: {e}"]
            except ModelValidationError as e:
                return [f"Validation error: {e}"]
            except Exception as e:
                return [f"Unexpected error: {e}"]

        return validate

    def _create_editor(self, user_context: str | None = None):
        """Create editor function for AI-assisted editing in the workflow.

        Args:
            user_context: User context description

        Returns:
            Async function that edits YAML based on user feedback and returns
            tuple of (edited_yaml, updated_conversation_history)
        """

        async def edit(
            yaml_content: str,
            feedback: str,
            context: dict[str, Any],
            conversation_history: list[dict[str, str]] | None = None,
        ) -> tuple[str | None, list[dict[str, str]] | None]:
            agent = MLModelAgent(
                self.services,
                self.api_key,
                self.base_url,
                self.model,
            )

            try:
                _model_spec, edited_yaml, updated_history = await agent.generate_model(
                    name=context["model_name"],
                    user_context=user_context or "",
                    table_name=context["table_name"],
                    target_column=context.get("target_column"),
                    existing_yaml=yaml_content,
                    editing_instructions=feedback,
                    conversation_history=conversation_history,
                )

                return edited_yaml, updated_history
            except Exception as e:
                if self.ui:
                    self.ui.show_system_error(f"❌ AI editing failed: {str(e)}")
                return None, None

        return edit

    def _save_model_to_db(
        self,
        name: str,
        yaml_content: str,
        description: str,
        plan_id: str | None = None,
    ) -> Model:
        """Save generated model directly to DB (no file needed).

        Args:
            name: Model name
            yaml_content: YAML specification as string
            description: Model description
            plan_id: Optional ML plan ID that guided this model generation

        Returns:
            Created Model object with model_id

        Raises:
            ValueError: If YAML is invalid or DB save fails
        """
        from datetime import UTC, datetime

        from arc.database.models.model import Model
        from arc.graph.model import ModelSpec
        from arc.ml.runtime import _slugify_name

        # Validate YAML first
        try:
            model_spec = ModelSpec.from_yaml(yaml_content)
            _ = model_spec.get_input_names()
            _ = model_spec.get_output_names()
        except Exception as exc:
            raise ValueError(f"Invalid model YAML: {exc}") from exc

        # Get next version
        latest = self.services.models.get_latest_model_by_name(name)
        version = 1 if latest is None else latest.version + 1

        # Create model ID
        base_slug = _slugify_name(name)
        model_id = f"{base_slug}-v{version}"

        # Create model object
        now = datetime.now(UTC)
        model = Model(
            id=model_id,
            type="ml.model_spec",
            name=name,
            version=version,
            description=description,
            spec=yaml_content,
            created_at=now,
            updated_at=now,
            plan_id=plan_id,  # Link to ML plan if provided
        )

        # Save to DB
        self.services.models.create_model(model)
        return model


class MLTrainTool(BaseTool):
    """Unified tool for generating trainer specs and launching training.

    This tool combines trainer generation with training execution in a single
    workflow, similar to the model generator pattern. It provides:
    1. Trainer spec generation via LLM
    2. Interactive confirmation workflow for trainer spec
    3. Auto-registration to database
    4. Interactive confirmation workflow for training launch
    5. Training execution

    All training configuration (epochs, batch_size, learning_rate, etc.) must
    be defined in the trainer YAML - no runtime overrides are supported.
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

    async def execute(
        self,
        *,
        name: str | None = None,
        instruction: str | None = None,
        model_id: str | None = None,
        train_table: str | None = None,
        auto_confirm: bool = False,
        plan_id: str | None = None,
        recommended_knowledge_ids: list[str] | None = None,
    ) -> ToolResult:
        """Generate trainer spec, register it, and launch training with confirmation.

        Args:
            name: Trainer name
            instruction: User's instruction for training configuration (PRIMARY driver)
            model_id: Model ID (with version, e.g., 'my_model-v1')
            train_table: Training data table
            auto_confirm: Skip confirmation workflows (for testing only)
            plan_id: Optional ML plan ID (e.g., 'pidd-plan-v1') containing
                training_configuration guidance (SECONDARY baseline)

        Note on instruction vs plan_id precedence:
            - instruction: PRIMARY driver - user's immediate, specific request
            - plan_id: SECONDARY baseline - loads plan from DB for guidance and context
            - When there's a conflict, instruction takes precedence
            - Example: If instruction says "use 10 epochs" but plan says "use 5 epochs",
              the trainer spec should use 10 epochs (instruction wins)
            - The LLM agent should use plan as baseline context and augment/override
              it with specifics from instruction

        Note on training configuration:
            All training configuration (epochs, batch_size, learning_rate, validation,
            etc.) must be specified in the trainer YAML generated by the LLM.
            No runtime overrides are supported.
        """
        # Show section title first, before any validation
        # Build metadata for section title
        metadata_parts = []
        if plan_id:
            metadata_parts.append(plan_id)
        if recommended_knowledge_ids:
            metadata_parts.extend(recommended_knowledge_ids)

        # Use context manager for section printing
        with self._section_printer(
            self.ui, "ML Trainer", metadata=metadata_parts
        ) as printer:
            # Show task description
            if printer:
                printer.print(f"[dim]Task: {instruction}[/dim]")

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

            # Validate API key and services
            if not self.api_key:
                return _error_in_section(
                    "API key required for trainer generation. "
                    "Set ARC_API_KEY or configure an API key before using this tool."
                )

            if not self.services:
                return _error_in_section(
                    "Trainer generation service unavailable. "
                    "Database services not initialized."
                )

            # Validate required parameters
            if not name or not instruction or not model_id or not train_table:
                return _error_in_section(
                    "Parameters 'name', 'instruction', 'model_id', and 'train_table' "
                    "are required."
                )

            # Get the registered model by ID
            try:
                model_record = self.services.models.get_model_by_id(str(model_id))
                if not model_record:
                    return _error_in_section(
                        f"Model '{model_id}' not found in registry. "
                        "Please check the model ID or register the model first."
                    )
            except Exception as exc:
                return _error_in_section(f"Failed to retrieve model '{model_id}': {exc}")

            # Load plan from database if plan_id is provided
            ml_plan_training_config = None
            if plan_id:
                ml_plan, plan = _load_ml_plan(self.services, plan_id)
                if ml_plan is None:
                    # plan contains error message
                    return _error_in_section(plan)
                ml_plan_training_config = plan.training_configuration

            # Generate trainer spec via LLM
            # Agent will discover relevant knowledge using tools
            agent = MLTrainAgent(
                self.services,
                self.api_key,
                self.base_url,
                self.model,
            )

            # Set progress callback for this invocation
            if printer:
                agent.progress_callback = printer.print
            else:
                agent.progress_callback = None

            try:
                (
                    trainer_spec,
                    trainer_yaml,
                    conversation_history,
                ) = await agent.generate_trainer(
                    name=str(name),
                    instruction=str(instruction),
                    model_id=model_record.id,
                    model_spec_yaml=model_record.spec,
                    ml_plan_training_config=ml_plan_training_config,
                    recommended_knowledge_ids=None,  # Let agent discover via tools
                )

                # Show completion message
                if printer:
                    printer.print("[dim]✓ Trainer generated successfully[/dim]")

            except Exception as exc:
                from arc.core.agents.ml_train import MLTrainError

                # Print error within the ML Trainer section
                if printer:
                    printer.print("")
                    printer.print(f"✗ {str(exc)}")

                if isinstance(exc, MLTrainError):
                    error_msg = str(exc)
                    return ToolResult(
                        success=False,
                        output=error_msg,
                        metadata={"error_shown": True, "error_message": error_msg},
                    )
                error_msg = f"Unexpected error during trainer generation: {exc}"
                return ToolResult(
                    success=False,
                    output=error_msg,
                    metadata={"error_shown": True, "error_message": error_msg},
                )

            # Validate the generated trainer
            try:
                trainer_dict = yaml.safe_load(trainer_yaml)
                validate_trainer_dict(trainer_dict)
            except (yaml.YAMLError, TrainerValidationError) as exc:
                # Print error within the ML Trainer section
                error_msg = f"Trainer validation failed: {exc}"
                if printer:
                    printer.print("")
                    printer.print(f"✗ {error_msg}")
                return ToolResult(
                    success=False,
                    output=error_msg,
                    metadata={"error_shown": True, "error_message": error_msg},
                )
            except Exception as exc:
                import logging

                logging.exception("Unexpected error during trainer validation")
                # Print error within the ML Trainer section
                error_msg = f"Unexpected validation error: {exc.__class__.__name__}: {exc}"
                if printer:
                    printer.print("")
                    printer.print(f"✗ {error_msg}")
                return ToolResult(
                    success=False,
                    output=error_msg,
                    metadata={"error_shown": True, "error_message": error_msg},
                )

            # Interactive confirmation workflow (unless auto_confirm is True)
            if not auto_confirm:
                workflow = YamlConfirmationWorkflow(
                    validator_func=self._create_validator(),
                    editor_func=self._create_editor(instruction, model_record),
                    ui_interface=self.ui,
                    yaml_type_name="trainer",
                    yaml_suffix=".arc-trainer.yaml",
                )

                context_dict = {
                    "trainer_name": str(name),
                    "instruction": str(instruction),
                    "model_id": model_record.id,
                    "model_spec_yaml": model_record.spec,
                }

                try:
                    proceed, final_yaml = await workflow.run_workflow(
                        trainer_yaml,
                        context_dict,
                        None,  # No output path - we register to DB
                        conversation_history,  # Pass conversation history for editing
                    )
                    if not proceed:
                        return ToolResult(
                            success=True,
                            output="✗ Trainer generation cancelled.",
                            metadata={"cancelled": True},
                        )
                    trainer_yaml = final_yaml
                finally:
                    workflow.cleanup()

            # Auto-register trainer to database
            try:
                # Extract plan_id from ml_plan if provided
                plan_id = ml_plan.get("plan_id") if ml_plan else None

                trainer_record = self.runtime.create_trainer(
                    name=str(name),
                    model_id=str(model_id),
                    schema_yaml=trainer_yaml,
                    description=f"Generated trainer for model {model_id}",
                    plan_id=plan_id,
                )

                # Display registration confirmation in the ML Trainer section
                if printer:
                    printer.print("")  # Empty line before confirmation
                    printer.print(
                        f"[dim]✓ Trainer '{name}' registered to database "
                        f"({trainer_record.id} • Model: {trainer_spec.model_ref} • "
                        f"Optimizer: {trainer_spec.optimizer.type})[/dim]"
                    )
            except Exception as exc:
                return _error_in_section(f"Failed to register trainer: {exc}")

            # Build simple output for ToolResult (detailed output already shown in UI)
            lines = [f"Trainer '{name}' registered successfully as {trainer_record.id}"]

            # Build result metadata (before auto-launch to use in error handling)
            result_metadata = {
                "trainer_id": trainer_record.id,
                "trainer_name": name,
                "model_id": model_record.id,
                "yaml_content": trainer_yaml,
                "training_launched": False,  # Will update if training launches
            }

            # Auto-launch training after trainer is accepted
            job_id = None

            if True:  # Always train when trainer is accepted
                if printer:
                    printer.print("")  # Empty line before training
                    printer.print(
                        f"→ Launching training with trainer '{name}'..."
                    )

                try:
                    job_id = await asyncio.to_thread(
                        self.runtime.train_with_trainer,
                        trainer_name=str(name),
                        train_table=str(train_table),
                    )

                    lines.append("")
                    lines.append("✓ Training job submitted successfully.")
                    lines.append(f"Training table: {train_table}")
                    lines.append(f"Job ID: {job_id}")

                    # Show training success message in section
                    if printer:
                        printer.print("")
                        printer.print(
                            "[dim]✓ Training job submitted successfully.[/dim]"
                        )
                        printer.print(
                            f"[dim]Training table: {train_table}[/dim]"
                        )
                        printer.print(f"[dim]Job ID: {job_id}[/dim]")

                    # Show job monitoring instructions in section
                    if printer:
                        printer.print("")
                        printer.print(
                            "[dim][cyan]ℹ Monitor training progress:[/cyan][/dim]"
                        )
                        printer.print(
                            f"[dim]  • Status: /ml jobs status {job_id}[/dim]"
                        )
                        printer.print(
                            f"[dim]  • Logs: /ml jobs logs {job_id}[/dim]"
                        )

                    # Handle TensorBoard launch
                    if not auto_confirm and self.ui:
                        if self.tensorboard_manager:
                            try:
                                await self._handle_tensorboard_launch(
                                    job_id, printer
                                )
                            except (OSError, RuntimeError) as e:
                                # Known TensorBoard launch failures
                                if printer:
                                    printer.print(
                                        f"[yellow]⚠️  TensorBoard setup failed: {e}[/yellow]"
                                    )
                                self._show_manual_tensorboard_instructions(
                                    job_id, printer
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
                                        "[yellow]⚠️  TensorBoard setup failed:[/yellow]"
                                    )
                                    printer.print(
                                        f"[yellow]{error_msg}[/yellow]"
                                    )
                                self._show_manual_tensorboard_instructions(
                                    job_id, printer
                                )
                        else:
                            # No TensorBoard manager available
                            if printer:
                                printer.print(
                                    "[dim]ℹ️  TensorBoard auto-launch not available "
                                    "(restart arc chat to enable)[/dim]"
                                )
                            self._show_manual_tensorboard_instructions(
                                job_id, printer
                            )

                    # Training job launched successfully - job status can be
                    # checked separately. The agent will monitor job status and
                    # analyze results when training completes

                except MLRuntimeError as exc:
                    # Trainer was successfully registered but training launch failed
                    # Display error in section and return success with warning since
                    # trainer is usable
                    if printer:
                        printer.print("⚠ Training Validation Failed")
                        printer.print("")
                        printer.print(f"[red]{exc}[/red]")
                        printer.print("")
                        printer.print(
                            f"[dim]Note: Trainer '{name}' was registered successfully[/dim]"
                        )
                        retry_cmd = f"/ml jobs submit --trainer {name} --data {train_table}"
                        printer.print(f"[dim]Retry: {retry_cmd}[/dim]")

                    lines.append("")
                    lines.append("⚠ Training Validation Failed")
                    lines.append("")
                    lines.append(f"{exc}")
                    lines.append("")
                    lines.append(f"Note: Trainer '{name}' was registered successfully")
                    retry_cmd = f"/ml jobs submit --trainer {name} --data {train_table}"
                    lines.append(f"Retry: {retry_cmd}")

                    # Extract validation report if available for agent debugging
                    validation_report = getattr(exc, "validation_report", None)

                    metadata = {
                        **result_metadata,
                        "training_launch_failed": True,
                        "training_error": str(exc),
                    }

                    # Include detailed validation report for agent debugging
                    if validation_report:
                        metadata["validation_report"] = validation_report

                    return ToolResult(
                        success=True,  # Trainer registration succeeded
                        output="\n".join(lines),
                        metadata=metadata,
                    )
                except Exception as exc:
                    # Log unexpected errors with full traceback
                    import logging

                    logging.exception("Unexpected error during training launch")

                    # Trainer was successfully registered but training launch failed
                    # Display error in section
                    if printer:
                        printer.print("⚠ Training Validation Failed")
                        printer.print("")
                        printer.print(
                            f"[red]{exc.__class__.__name__}: {exc}[/red]"
                        )
                        printer.print("")
                        printer.print(
                            f"[dim]Note: Trainer '{name}' was registered successfully[/dim]"
                        )
                        retry_cmd = f"/ml jobs submit --trainer {name} --data {train_table}"
                        printer.print(f"[dim]Retry: {retry_cmd}[/dim]")

                    lines.append("")
                    lines.append("⚠ Training Validation Failed")
                    lines.append("")
                    lines.append(f"{exc.__class__.__name__}: {exc}")
                    lines.append("")
                    lines.append(f"Note: Trainer '{name}' was registered successfully")
                    retry_cmd = f"/ml jobs submit --trainer {name} --data {train_table}"
                    lines.append(f"Retry: {retry_cmd}")

                    # Extract validation report if available (might not be present)
                    validation_report = getattr(exc, "validation_report", None)

                    metadata = {
                        **result_metadata,
                        "training_launch_failed": True,
                        "training_error": f"{exc.__class__.__name__}: {exc}",
                    }

                    if validation_report:
                        metadata["validation_report"] = validation_report

                    return ToolResult(
                        success=True,  # Trainer registration succeeded
                        output="\n".join(lines),
                        metadata=metadata,
                    )

            # Build result metadata
            result_metadata = {
                "trainer_id": trainer_record.id,
                "trainer_name": name,
                "model_id": model_record.id,
                "yaml_content": trainer_yaml,
                "training_launched": job_id is not None,
                "from_ml_plan": ml_plan is not None,
            }

            if job_id:
                result_metadata["job_id"] = job_id


            return ToolResult(
                success=True,
                output="\n".join(lines),
                metadata=result_metadata,
            )

    async def _handle_tensorboard_launch(self, job_id: str, section_printer=None):
        """Handle TensorBoard launch based on user preference.

        Args:
            job_id: Training job identifier
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
            job_id: Training job identifier
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
            job_id: Training job identifier
            section_printer: Section printer for indented output
        """
        logdir = f"tensorboard/run_{job_id}"
        if section_printer:
            section_printer.print("")
            section_printer.print("[dim][cyan]ℹ Track training:[/cyan][/dim]")
            section_printer.print(f"[dim]  • Status: /ml jobs status {job_id}[/dim]")
            section_printer.print(
                f"[dim]  • TensorBoard: tensorboard --logdir {logdir}[/dim]"
            )
        else:
            self.ui._printer.console.print()
            self.ui._printer.console.print("[cyan]ℹ Track training:[/cyan]")
            self.ui._printer.console.print(f"  • Status: /ml jobs status {job_id}")
            self.ui._printer.console.print(
                f"  • TensorBoard: tensorboard --logdir {logdir}"
            )

    def _create_validator(self):
        """Create validator function for the workflow."""

        def validate(yaml_str: str) -> list[str]:
            try:
                trainer_dict = yaml.safe_load(yaml_str)
                validate_trainer_dict(trainer_dict)
                return []  # No errors
            except yaml.YAMLError as e:
                return [f"Invalid YAML: {e}"]
            except TrainerValidationError as e:
                return [f"Validation error: {e}"]
            except Exception as e:
                return [f"Unexpected error: {e}"]

        return validate

    def _create_editor(self, _user_instruction: str, model_record):
        """Create editor function for AI-assisted editing with conversation history."""

        async def edit(
            yaml_content: str,
            feedback: str,
            context: dict[str, Any],
            conversation_history: list[dict[str, str]] | None = None,
        ) -> tuple[str | None, list[dict[str, str]] | None]:
            # Agent will discover relevant knowledge using tools
            agent = MLTrainAgent(
                self.services,
                self.api_key,
                self.base_url,
                self.model,
            )

            try:
                (
                    _trainer_spec,
                    edited_yaml,
                    updated_history,
                ) = await agent.generate_trainer(
                    name=context["trainer_name"],
                    instruction=feedback,
                    model_id=model_record.id,
                    model_spec_yaml=model_record.spec,
                    existing_yaml=yaml_content,
                    recommended_knowledge_ids=None,  # Let agent discover via tools
                    conversation_history=conversation_history,
                )
                return edited_yaml, updated_history
            except Exception as e:
                if self.ui:
                    self.ui.show_system_error(f"❌ Edit failed: {e}")
                return None, None

        return edit


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
        plan_id: str | None = None,
        recommended_knowledge_ids: list[str] | None = None,
    ) -> ToolResult:
        """Generate evaluator spec, register it, and launch evaluation.

        Args:
            name: Evaluator name
            instruction: User's instruction for evaluation setup (PRIMARY driver)
            trainer_id: Trainer ID with version (e.g., 'my-trainer-v1')
            evaluate_table: Test dataset table name
            auto_confirm: Skip confirmation workflows (for testing only)
            plan_id: Optional ML plan ID (e.g., 'pidd-plan-v1') containing
                evaluation expectations (SECONDARY baseline)

        Note on instruction vs plan_id precedence:
            - instruction: PRIMARY driver - user's immediate, specific request
            - plan_id: SECONDARY baseline - loads plan from DB for guidance and context
            - When there's a conflict, instruction takes precedence
            - Example: If instruction says "compute F1 score" but plan says
              "compute accuracy only", the evaluator should compute F1 score
              (instruction wins)
            - The LLM agent should use plan as baseline context and augment/override
              it with specifics from instruction

        Note on async execution:
            This tool returns immediately after launching the evaluation job.
            The evaluation runs in the background and results can be monitored via
            job status, logs, and TensorBoard.

        Returns:
            ToolResult with job_id for monitoring async evaluation
        """
        # Show section title first, before any validation
        # Build metadata for section title
        metadata_parts = []
        if plan_id:
            metadata_parts.append(plan_id)
        if recommended_knowledge_ids:
            metadata_parts.extend(recommended_knowledge_ids)

        # Use context manager for section printing
        with self._section_printer(
            self.ui, "ML Evaluator", metadata=metadata_parts
        ) as printer:
            # Show task description
            if printer:
                printer.print(f"[dim]Task: {instruction}[/dim]")

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

            # Load plan from database if plan_id is provided
            ml_plan_evaluation = None
            if plan_id:
                ml_plan, plan = _load_ml_plan(self.services, plan_id)
                if ml_plan is None:
                    # plan contains error message
                    return _error_in_section(plan)
                ml_plan_evaluation = plan.evaluation

            # Generate evaluator spec via LLM
            # Agent will discover relevant knowledge using tools
            from arc.core.agents.ml_evaluate import MLEvaluateAgent

            agent = MLEvaluateAgent(
                self.services,
                self.api_key,
                self.base_url,
                self.model,
            )

            # Set progress callback for this invocation
            if printer:
                agent.progress_callback = printer.print
            else:
                agent.progress_callback = None

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
                    recommended_knowledge_ids=None,  # Let agent discover via tools
                )

                # Show completion message
                if printer:
                    printer.print("[dim]✓ Evaluator generated successfully[/dim]")

            except Exception as exc:
                from arc.core.agents.ml_evaluate import MLEvaluateError

                if isinstance(exc, MLEvaluateError):
                    return _error_in_section(str(exc))
                return _error_in_section(
                    f"Unexpected error during evaluator generation: {exc}"
                )

            # Validate the generated evaluator
            try:
                from arc.graph.evaluator import (
                    EvaluatorValidationError,
                    validate_evaluator_dict,
                )

                evaluator_dict = yaml.safe_load(evaluator_yaml)
                validate_evaluator_dict(evaluator_dict)
            except yaml.YAMLError as exc:
                return _error_in_section(
                    f"Generated evaluator contains invalid YAML: {exc}"
                )
            except EvaluatorValidationError as exc:
                return _error_in_section(
                    f"Generated evaluator failed validation: {exc}"
                )
            except Exception as exc:
                # Log unexpected errors with full traceback
                import logging

                logging.exception("Unexpected error during evaluator validation")
                return _error_in_section(
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
                        return ToolResult(
                            success=True,
                            output="✗ Evaluator cancelled.",
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
                    if printer:
                        printer.print(
                            ""
                        )  # Empty line before confirmation
                        printer.print(
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
                    if printer:
                        printer.print(
                            ""
                        )  # Empty line before confirmation
                        printer.print(
                            f"[dim]✓ Evaluator '{name}' registered to database "
                            f"({evaluator_record.id} • "
                            f"Trainer: {evaluator_spec.trainer_ref} • "
                            f"Dataset: {evaluator_spec.dataset})[/dim]"
                        )
            except Exception as exc:
                return _error_in_section(f"Failed to register evaluator: {exc}")

            # Build simple output for ToolResult (detailed output already shown in UI)
            lines = [f"Evaluator '{name}' registered successfully as {evaluator_record.id}"]

            # Launch evaluation as background job (async pattern)
            if printer:
                printer.print("")
                printer.print(
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
                from pathlib import Path

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
                lines.append("✓ Evaluation job submitted successfully.")
                lines.append(f"  Evaluator: {evaluator_record.id}")
                lines.append(f"  Dataset: {evaluate_table}")
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
                    printer.print(
                        f"[dim]  • Logs: /ml jobs logs {job.job_id}[/dim]"
                    )

                # Handle TensorBoard launch for monitoring
                if not auto_confirm and self.ui:
                    if self.tensorboard_manager:
                        try:
                            await self._handle_tensorboard_launch(
                                job.job_id, printer
                            )
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

                            logging.exception("Unexpected error during TensorBoard launch")
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
                                "[dim]ℹ️  TensorBoard auto-launch not available "
                                "(restart arc chat to enable)[/dim]"
                            )
                        self._show_manual_tensorboard_instructions(
                            job.job_id, printer
                        )

                # Evaluation launched successfully - job status can be checked separately

            except Exception as exc:
                # Evaluator was successfully registered but evaluation launch failed
                lines.append("")
                lines.append("⚠️  Evaluation launch failed but evaluator is registered.")
                lines.append(f"Error: {exc}")
                lines.append("")
                retry_cmd = f"/ml evaluate --evaluator {name} --data {evaluate_table}"
                lines.append(f"Retry evaluation with: {retry_cmd}")


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
                from arc.graph.evaluator import (
                    EvaluatorValidationError,
                    validate_evaluator_dict,
                )

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
            # Agent will discover relevant knowledge using tools
            from arc.core.agents.ml_evaluate import MLEvaluateAgent

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
                    recommended_knowledge_ids=None,  # Let agent discover via tools
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




class MLPlanTool(BaseTool):
    """Tool for creating and revising ML plans with technical decisions."""

    def __init__(
        self,
        services,
        api_key: str | None,
        base_url: str | None,
        model: str | None,
        ui_interface=None,
        agent=None,
    ) -> None:
        self.services = services
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.ui = ui_interface
        self.agent = agent  # Reference to parent agent for auto_accept flag

    async def execute(
        self,
        *,
        instruction: str | None = None,
        source_tables: str | None = None,
        previous_plan: dict | None = None,
        section_to_update: str | None = None,
        conversation_history: str | None = None,  # noqa: ARG002
        verbose: bool = False,
        skip_data_profiling: bool = False,
    ) -> ToolResult:
        # Early validation for common errors (before any section printing)
        if not self.api_key:
            return ToolResult.error_result(
                "API key required for ML planning. "
                "Set ARC_API_KEY or configure an API key before using this tool."
            )

        if not self.services:
            return ToolResult.error_result(
                "ML planning service unavailable. Database services not initialized."
            )

        if not instruction or not source_tables:
            return ToolResult.error_result(
                "Parameters 'instruction' and 'source_tables' "
                "are required for ML planning."
            )

        # Handle section update mode (different workflow - no UI section needed)
        if section_to_update:
            # Section update mode requires previous_plan and instruction
            if not previous_plan:
                return ToolResult.error_result(
                    "Parameter 'previous_plan' is required when updating a section."
                )
            if not instruction:
                return ToolResult.error_result(
                    "Parameter 'instruction' is required when updating a section."
                )

            # Extract the original section content
            section_content = previous_plan.get(section_to_update)
            if section_content is None:
                return ToolResult.error_result(
                    f"Section '{section_to_update}' not found in previous plan."
                )

            # Create agent and update section
            agent = MLPlanAgent(
                self.services,
                self.api_key,
                self.base_url,
                self.model,
                verbose=verbose,
            )

            try:
                updated_section = await agent.update_section(
                    section_name=section_to_update,
                    original_section=str(section_content),
                    feedback_content=str(instruction),
                )

                # Update the plan with new section
                updated_plan = previous_plan.copy()
                updated_plan[section_to_update] = updated_section

                return ToolResult(
                    success=True,
                    output=f"Section '{section_to_update}' updated successfully.",
                    metadata={
                        "ml_plan": updated_plan,
                        "section_updated": section_to_update,
                        "is_revision": True,
                    },
                )

            except Exception as exc:
                from arc.core.ml_plan import MLPlanError

                if isinstance(exc, MLPlanError):
                    return ToolResult.error_result(str(exc))
                return ToolResult.error_result(
                    f"Unexpected error updating section: {exc}"
                )

        # Full plan generation mode
        # Use context manager for section printing
        with self._section_printer(
            self.ui, "ML Plan", color="cyan"
        ) as printer:
            # Show task description
            if printer:
                printer.print(f"[dim]Task: {instruction}[/dim]")

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

            # Helper to print progress messages within the section
            def _progress_callback(message: str):
                if printer:
                    printer.print(message)

            agent = MLPlanAgent(
                self.services,
                self.api_key,
                self.base_url,
                self.model,
                progress_callback=_progress_callback if printer else None,
                verbose=verbose,
            )

            try:
                # Import MLPlan for plan management
                from arc.core.ml_plan import MLPlan

                # Check if auto-accept is enabled
                if self.agent and self.agent.ml_plan_auto_accept:
                    # Auto-accept mode - skip workflow
                    pass  # Continue to generate plan but skip confirmation

                # Internal loop for handling user instruction and revision feedback
                current_instruction = instruction

                # Get version from database to avoid conflicts
                latest_plan = self.services.ml_plans.get_latest_plan_for_tables(
                    str(source_tables)
                )
                version = latest_plan.version + 1 if latest_plan else 1

                while True:
                    try:
                        # Generate the plan (pass source_tables as comma-separated string)
                        analysis = await agent.analyze_problem(
                            user_context=str(current_instruction),
                            source_tables=str(source_tables),
                            instruction=current_instruction
                            if current_instruction != instruction
                            else None,
                            stream=False,
                            skip_data_profiling=skip_data_profiling,
                        )

                        # Show completion message
                        if printer:
                            printer.print(
                                "[dim]✓ Plan generated successfully[/dim]"
                            )

                        # Determine stage
                        if previous_plan:
                            stage = previous_plan.get("stage", "initial")
                            instruction_lower = str(current_instruction).lower()
                            if (
                                current_instruction != instruction
                                and "training" in instruction_lower
                            ):
                                stage = "post_training"
                            elif (
                                current_instruction != instruction
                                and "evaluation" in instruction_lower
                            ):
                                stage = "post_evaluation"
                            reason = (
                                f"Revised based on instruction: "
                                f"{current_instruction[:100]}..."
                                if current_instruction != instruction
                                else "Plan revision"
                            )
                        else:
                            stage = "initial"
                            reason = None

                        plan = MLPlan.from_analysis(
                            analysis, version=version, stage=stage, reason=reason
                        )

                        # Save plan to database immediately with "draft" status
                        # This allows other tools to reference it even before user confirms
                        try:
                            from datetime import UTC, datetime

                            from arc.database.models.ml_plan import (
                                MLPlan as MLPlanModel,
                            )
                            from arc.ml.runtime import _slugify_name

                            # Convert plan to dict for storage
                            plan_dict = plan.to_dict()
                            plan_dict["source_tables"] = str(source_tables)

                            # Convert plan to YAML format for better readability
                            plan_yaml = yaml.dump(
                                plan_dict, default_flow_style=False, sort_keys=False
                            )

                            # Create database model - use first table for plan ID
                            first_table = source_tables.split(",")[0].strip()
                            base_slug = _slugify_name(f"{first_table}-plan")
                            plan_id = f"{base_slug}-v{version}"

                            now = datetime.now(UTC)
                            db_plan = MLPlanModel(
                                plan_id=plan_id,
                                version=version,
                                user_context=str(instruction),
                                source_tables=str(source_tables),
                                plan_yaml=plan_yaml,  # Store as YAML string
                                status="draft",  # Initially draft until user confirms
                                created_at=now,
                                updated_at=now,
                            )

                            # Save to database
                            self.services.ml_plans.create_plan(db_plan)

                            # Add plan_id to plan_dict for use in confirmation workflow
                            plan_dict["plan_id"] = plan_id

                        except Exception as e:
                            # Log error but continue - plan still works in memory
                            if printer:
                                printer.print(
                                    f"[dim yellow]⚠ Could not save plan to "
                                    f"database: {e}[/dim yellow]"
                                )
                            plan_id = None  # Track that we don't have a DB plan

                    except Exception as e:
                        # Handle errors during plan generation
                        error_msg = f"Failed to generate ML plan: {str(e)}"
                        if printer:
                            printer.print("")
                            printer.print(f"✗ {error_msg}")
                        return ToolResult(
                            success=False,
                            output="",
                            metadata={"error_shown": True},
                        )

                    # If auto-accept is enabled, skip workflow
                    if self.agent and self.agent.ml_plan_auto_accept:
                        output_message = (
                            f"Plan '{plan_id}' automatically accepted (auto-accept enabled)"
                        )
                        break

                    # Display plan and run confirmation workflow
                    if self.ui:
                        from arc.utils.ml_plan_workflow import (
                            MLPlanConfirmationWorkflow,
                        )

                        try:
                            workflow = MLPlanConfirmationWorkflow(self.ui)
                            result = await workflow.run_workflow(
                                plan, previous_plan is not None
                            )
                            choice = result.get("choice")
                        except Exception as e:
                            # Handle workflow errors
                            error_msg = f"Workflow execution failed: {str(e)}"
                            self.ui._printer.console.print(f"[red]❌ {error_msg}[/red]")
                            return ToolResult.error_result(error_msg)

                        if choice == "accept":
                            # Update status to confirmed in database
                            if plan_id:
                                try:
                                    self.services.ml_plans.update_status(
                                        plan_id, "confirmed"
                                    )
                                except Exception as e:
                                    # Log but don't fail
                                    if printer:
                                        printer.print(
                                            f"[dim yellow]⚠ Could not update plan "
                                            f"status: {e}[/dim yellow]"
                                        )
                            output_message = (
                                f"Plan '{plan_id}' accepted. "
                                f"Ready to proceed with implementation."
                            )
                            break
                        elif choice == "accept_all":
                            # Update status to confirmed in database
                            if plan_id:
                                try:
                                    self.services.ml_plans.update_status(
                                        plan_id, "confirmed"
                                    )
                                except Exception as e:
                                    # Log but don't fail
                                    if printer:
                                        printer.print(
                                            f"[dim yellow]⚠ Could not update plan "
                                            f"status: {e}[/dim yellow]"
                                        )
                            # Enable auto-accept for this session
                            if self.agent:
                                self.agent.ml_plan_auto_accept = True
                            output_message = (
                                f"Plan '{plan_id}' accepted. "
                                f"Auto-accept enabled for this session."
                            )
                            break
                        elif choice == "feedback":
                            # Mark current plan as rejected since user wants to revise
                            if plan_id:
                                with contextlib.suppress(Exception):
                                    self.services.ml_plans.update_status(
                                        plan_id, "rejected"
                                    )
                            # Get instruction and loop to revise
                            current_instruction = result.get("feedback", "")
                            version += 1
                            # Continue loop to generate revised plan
                            continue
                        elif choice == "cancel":
                            # Mark plan as rejected
                            if plan_id:
                                with contextlib.suppress(Exception):
                                    self.services.ml_plans.update_status(
                                        plan_id, "rejected"
                                    )
                            # Print cancellation message inside section
                            if printer:
                                printer.print(
                                    "ML plan cancelled. What would you like to do instead?"
                                )
                            # Return to main agent with context message
                            # (Message already displayed in section,
                            # but agent needs context)
                            return ToolResult(
                                success=True,
                                output=(
                                    "ML plan cancelled. What would you like to do instead?"
                                ),
                                metadata={"cancelled": True, "suppress_output": True},
                            )
                    else:
                        # Headless mode - auto-accept
                        # Update status to confirmed since it's auto-accepted
                        if plan_id:
                            with contextlib.suppress(Exception):
                                self.services.ml_plans.update_status(plan_id, "confirmed")
                        formatted_result = plan.format_for_display()
                        output_message = (
                            f"Plan '{plan_id}' created successfully.\n\n{formatted_result}"
                        )
                        break

                # Display registration confirmation in the ML Plan section
                if printer and plan_id:
                    printer.print("")  # Empty line before confirmation
                    table_count = len(source_tables.split(","))
                    printer.print(
                        f"[dim]✓ Plan '{plan_id}' saved to database "
                        f"(v{version} • {stage} • {table_count} tables)[/dim]"
                    )


                return ToolResult(
                    success=True,
                    output=output_message,
                    metadata={
                        "ml_plan": plan_dict,
                        "plan_id": plan_id,  # Top-level for easy LLM access
                        "is_revision": previous_plan is not None,
                        "recommended_knowledge_ids": plan.recommended_knowledge_ids,
                    },
                )

            except Exception as exc:
                from arc.core.ml_plan import MLPlanError

                if isinstance(exc, MLPlanError):
                    return _error_in_section(str(exc))
                return _error_in_section(
                    f"Unexpected error during ML planning: {exc}"
                )
