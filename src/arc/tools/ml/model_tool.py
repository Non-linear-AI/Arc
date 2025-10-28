"""ML Model tool for generating Arc-Graph model + training specifications and launching training."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from arc.database.models.model import Model

from arc.core.agents.ml_model import MLModelAgent
from arc.graph.model import ModelValidationError, validate_model_dict
from arc.graph.trainer import TrainerSpec
from arc.ml.runtime import MLRuntime, MLRuntimeError
from arc.tools.base import BaseTool, ToolResult
from arc.tools.ml._utils import _load_ml_plan
from arc.utils.yaml_workflow import YamlConfirmationWorkflow


class MLModelTool(BaseTool):
    """Tool for generating unified model + training specs and launching training."""

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
        data_table: str | None = None,
        train_table: str | None = None,
        target_column: str | None = None,
        auto_confirm: bool = False,
        plan_id: str | None = None,
    ) -> ToolResult:
        """Generate unified model + training specification and launch training.

        Args:
            name: Model/experiment name
            instruction: User's instruction for model architecture + training (PRIMARY driver)
            data_table: Database table to profile for generation
            train_table: Training data table (defaults to data_table if not provided)
            target_column: Target column for prediction
            auto_confirm: Skip confirmation workflows (for testing only)
            plan_id: Optional ML plan ID containing unified model_plan guidance

        Note: This tool now generates BOTH model architecture AND training configuration
        in a single unified YAML, then immediately launches training.

        Returns:
            ToolResult with model registration and training job details
        """
        # Default train_table to data_table if not provided
        if not train_table:
            train_table = data_table

        # Show section title first, before any validation
        # Build metadata for section title
        metadata_parts = []
        if plan_id:
            metadata_parts.append(plan_id)

        # Use context manager for section printing
        with self._section_printer(
            self.ui, "ML Model + Training", metadata=metadata_parts
        ) as printer:
            # Show task description
            if printer:
                printer.print(f"[dim]Task: {instruction}[/dim]")
                printer.print("")  # Empty line after task

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
            model_plan = None
            recommended_knowledge_ids = None
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
                if not train_table:
                    train_table = ml_plan.get("train_table") or data_table

                # CRITICAL: Extract unified model plan (architecture + training) and knowledge
                model_plan = plan.model_plan

                # Extract stage-specific knowledge IDs from plan
                recommended_knowledge_ids = plan.knowledge.get("model", [])

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

            # Preload stage-specific knowledge from plan
            preloaded_knowledge = None
            if recommended_knowledge_ids:
                preloaded_knowledge = agent.knowledge_loader.load_multiple(
                    recommended_knowledge_ids, phase="model"
                )

            # Generate unified model + training specification
            try:
                (
                    model_spec,
                    unified_yaml,
                    conversation_history,
                ) = await agent.generate_model(
                    name=str(name),
                    user_context=instruction,  # Use instruction as user_context
                    table_name=str(data_table),
                    target_column=target_column,
                    model_plan=model_plan,
                    preloaded_knowledge=preloaded_knowledge,
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

            # Parse unified YAML to extract model and training sections
            try:
                full_spec = yaml.safe_load(unified_yaml)

                # Extract training config
                training_config = full_spec.pop("training", None)
                if not training_config:
                    return _error_in_section(
                        "Generated YAML missing required 'training' section. "
                        "The unified specification must include both model and training config."
                    )

                # Validate model portion
                validate_model_dict(full_spec)

                # Convert back to YAML for model-only storage
                model_yaml = yaml.dump(full_spec, default_flow_style=False, sort_keys=False)

            except (yaml.YAMLError, ModelValidationError) as exc:
                return _error_in_section(f"Specification validation failed: {exc}")
            except Exception as exc:
                # Log unexpected errors with full traceback
                import logging

                logging.exception("Unexpected error during specification validation")
                return _error_in_section(
                    f"Unexpected validation error: {exc.__class__.__name__}: {exc}"
                )

            # Interactive confirmation workflow (unless auto_confirm is True)
            if not auto_confirm:
                workflow = YamlConfirmationWorkflow(
                    validator_func=self._create_validator(),
                    editor_func=self._create_editor(instruction),
                    ui_interface=self.ui,
                    yaml_type_name="experiment",
                    yaml_suffix=".arc-experiment.yaml",
                )

                context_dict = {
                    "model_name": str(name),
                    "table_name": str(data_table),
                    "train_table": str(train_table),
                    "target_column": target_column,
                }

                try:
                    proceed, final_unified_yaml = await workflow.run_workflow(
                        unified_yaml,
                        context_dict,
                        None,  # No file path
                        conversation_history,  # Pass conversation history for editing
                    )
                    if not proceed:
                        # Show cancellation message before closing section
                        if printer:
                            printer.print("")  # Empty line
                            printer.print(
                                "[dim]✗ Experiment cancelled by user.[/dim]"
                            )
                        return ToolResult(
                            success=True,
                            output="✗ Experiment cancelled by user.",
                            metadata={"cancelled": True},
                        )

                    # Re-parse the edited YAML
                    full_spec = yaml.safe_load(final_unified_yaml)
                    training_config = full_spec.pop("training", None)
                    if not training_config:
                        return _error_in_section(
                            "Edited YAML missing 'training' section"
                        )
                    model_yaml = yaml.dump(full_spec, default_flow_style=False, sort_keys=False)
                    unified_yaml = final_unified_yaml
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

            # Display registration confirmation
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
                "unified_yaml": unified_yaml,
                "from_ml_plan": ml_plan is not None,
                "training_launched": False,  # Will update if training launches
            }

            # Create trainer spec and launch training
            trainer_name = name  # Use same name for trainer
            try:
                # Build trainer YAML from training config + model reference
                trainer_dict = {
                    "model_ref": model_id,
                    **training_config,
                }
                trainer_yaml = yaml.dump(trainer_dict, default_flow_style=False, sort_keys=False)

                # Create trainer record in database
                trainer_record = self.runtime.create_trainer(
                    name=trainer_name,
                    spec_yaml=trainer_yaml,
                    plan_id=plan_id,
                )
                trainer_id = trainer_record.name

                if printer:
                    printer.print("")
                    printer.print(f"[dim]✓ Trainer '{trainer_name}' created[/dim]")

                result_metadata["trainer_id"] = trainer_id
                result_metadata["trainer_yaml"] = trainer_yaml

            except Exception as exc:
                return _error_in_section(f"Failed to create trainer: {exc}")

            # Launch training
            if printer:
                printer.print("")
                printer.print(f"→ Launching training with trainer '{trainer_name}'...")

            try:
                job_id = await asyncio.to_thread(
                    self.runtime.train_with_trainer,
                    trainer_name=trainer_name,
                    train_table=str(train_table),
                )

                lines.append("")
                lines.append("✓ Training job submitted successfully.")
                lines.append(f"Training table: {train_table}")
                lines.append(f"Job ID: {job_id}")

                # Show training success message
                if printer:
                    printer.print("")
                    printer.print("[dim]✓ Training job submitted successfully.[/dim]")
                    printer.print(f"[dim]Training table: {train_table}[/dim]")
                    printer.print(f"[dim]Job ID: {job_id}[/dim]")

                # Show job monitoring instructions
                if printer:
                    printer.print("")
                    printer.print("[dim][cyan]ℹ Monitor training progress:[/cyan][/dim]")
                    printer.print(f"[dim]  • Status: /ml jobs status {job_id}[/dim]")
                    printer.print(f"[dim]  • Logs: /ml jobs logs {job_id}[/dim]")

                result_metadata["training_launched"] = True
                result_metadata["job_id"] = job_id

                # Handle TensorBoard launch
                if not auto_confirm and self.ui:
                    if self.tensorboard_manager:
                        try:
                            self.tensorboard_manager.launch_for_job(job_id)
                            if printer:
                                printer.print("")
                                printer.print("[dim][green]✓ TensorBoard launched[/green][/dim]")
                        except Exception:
                            if printer:
                                printer.print("")
                                printer.print(
                                    "[dim]ℹ️  TensorBoard auto-launch not available[/dim]"
                                )

            except MLRuntimeError as exc:
                # Training launch failed but model + trainer were created
                if printer:
                    printer.print("⚠ Training Validation Failed")
                    printer.print("")
                    printer.print(f"[red]{exc}[/red]")
                    printer.print("")
                    printer.print(f"[dim]Note: Model and trainer were registered successfully[/dim]")

                lines.append("")
                lines.append("⚠ Training Validation Failed")
                lines.append("")
                lines.append(f"{exc}")

                result_metadata["training_launch_failed"] = True
                result_metadata["training_error"] = str(exc)

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
