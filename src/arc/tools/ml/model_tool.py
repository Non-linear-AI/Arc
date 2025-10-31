"""ML Model tool for generating Arc-Graph model + training specifications and launching training."""

from __future__ import annotations

import asyncio
import copy
import json
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from arc.database.models.model import Model

from arc.core.agents.ml_model import MLModelAgent
from arc.graph.model import ModelValidationError, validate_model_dict
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

    def _build_model_result(
        self,
        status: str,
        model_id: str,
        model_spec_dict: dict | None = None,
        training_job_id: str | None = None,
        train_table: str | None = None,
        training_status: str = "not_started",
        training_error: str | None = None,
    ) -> str:
        """Build structured JSON result for ML model tool.

        Args:
            status: "accepted" or "cancelled"
            model_id: Model identifier
            model_spec_dict: Parsed model specification as dict
            training_job_id: Training job ID if launched
            train_table: Training table name
            training_status: "submitted", "failed", or "not_started"
            training_error: Error message if training failed

        Returns:
            JSON string with structured model result
        """
        result = {
            "status": status,
            "model_id": model_id,
            "model_spec": model_spec_dict or {},
            "training": {
                "status": training_status,
            },
        }

        # Add training details if available
        if training_job_id:
            result["training"]["job_id"] = training_job_id
        if train_table:
            result["training"]["train_table"] = train_table
        if training_error:
            result["training"]["error"] = training_error

        return json.dumps(result)

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
        data_processing_id: str | None = None,
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
            data_processing_id: Optional execution ID to load data processing context

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

            # Validate: if plan_id provided, data_processing_id must be provided
            if plan_id and not data_processing_id:
                return _error_in_section(
                    "Parameter 'data_processing_id' is required when 'plan_id' is provided. "
                    "This links the model to the specific data processing execution. "
                    "Extract 'data_processing_id' from the ml_data tool result JSON."
                )

            # Load plan from database if plan_id is provided
            ml_plan = None
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

            # Validate train_table is provided (required for training launch)
            if not train_table:
                return _error_in_section(
                    "Parameter 'train_table' is required for launching training. "
                    "Specify the table containing training data."
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
                    recommended_knowledge_ids
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
                    data_processing_id=data_processing_id,
                )

                # Inject metadata fields into unified YAML
                # Parse YAML, inject fields, and convert back to YAML string
                full_spec = yaml.safe_load(unified_yaml)
                full_spec["name"] = name
                full_spec["data_table"] = data_table
                if plan_id:
                    full_spec["plan_id"] = plan_id

                # Reconstruct YAML with metadata fields at the top
                # Build dict with specific field order for clean YAML output
                ordered_spec = {}
                ordered_spec["name"] = full_spec.pop("name")
                ordered_spec["data_table"] = full_spec.pop("data_table")
                if "plan_id" in full_spec:
                    ordered_spec["plan_id"] = full_spec.pop("plan_id")
                # Add remaining fields
                ordered_spec.update(full_spec)

                unified_yaml = yaml.dump(
                    ordered_spec, default_flow_style=False, sort_keys=False
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

                # Extract training config (which contains loss)
                training_config = full_spec.pop("training", None)
                if not training_config:
                    return _error_in_section(
                        "Generated YAML missing required 'training' section. "
                        "The unified specification must include both model and training config."
                    )

                # Extract loss from within training
                loss_config = training_config.pop("loss", None)
                if not loss_config:
                    return _error_in_section(
                        "Generated YAML missing required 'training.loss' section. "
                        "The training configuration must include a loss function."
                    )

                # Validate model portion (without loss - loss is in training config)
                validate_model_dict(full_spec)

                # Add loss to model spec level for easy access by other tools (e.g., evaluate)
                full_spec["loss"] = loss_config

                # Reconstruct unified YAML with loss in both places:
                # 1. At model spec level (for evaluation and inference)
                # 2. In training section (for training execution)
                # Use deepcopy to avoid YAML anchors when serializing
                training_config["loss"] = copy.deepcopy(
                    loss_config
                )  # Add back to training
                full_spec["training"] = training_config  # Add training back to spec

                # Rebuild with proper field ordering
                ordered_spec = {}
                if "name" in full_spec:
                    ordered_spec["name"] = full_spec.pop("name")
                if "data_table" in full_spec:
                    ordered_spec["data_table"] = full_spec.pop("data_table")
                if "plan_id" in full_spec:
                    ordered_spec["plan_id"] = full_spec.pop("plan_id")
                ordered_spec["inputs"] = full_spec.pop("inputs")
                if "modules" in full_spec:
                    ordered_spec["modules"] = full_spec.pop("modules")
                ordered_spec["graph"] = full_spec.pop("graph")
                ordered_spec["outputs"] = full_spec.pop("outputs")
                ordered_spec["loss"] = full_spec.pop("loss")
                ordered_spec["training"] = full_spec.pop("training")

                # Update unified_yaml with the reconstructed version
                unified_yaml = yaml.dump(
                    ordered_spec, default_flow_style=False, sort_keys=False
                )

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
                    yaml_type_name="training configuration",
                    yaml_suffix=".arc-training.yaml",
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
                            printer.print("[dim]✗ Training cancelled by user.[/dim]")

                        # Parse the spec dict for JSON output (before we save to DB)
                        full_spec_dict = yaml.safe_load(unified_yaml)

                        # Generate placeholder model_id for cancelled state
                        from arc.ml.runtime import _slugify_name

                        base_slug = _slugify_name(str(name))
                        cancelled_model_id = f"{base_slug}-cancelled"

                        # Return structured JSON with cancelled status
                        output_json = self._build_model_result(
                            status="cancelled",
                            model_id=cancelled_model_id,
                            model_spec_dict=full_spec_dict,
                            training_status="not_started",
                        )

                        return ToolResult(
                            success=True,
                            output=output_json,
                            metadata={"cancelled": True},
                        )

                    # Re-parse the edited YAML and re-inject metadata
                    full_spec = yaml.safe_load(final_unified_yaml)

                    # Re-inject metadata fields (in case they were removed during editing)
                    full_spec["name"] = name
                    full_spec["data_table"] = data_table
                    if plan_id:
                        full_spec["plan_id"] = plan_id

                    # Reconstruct YAML with metadata fields at the top
                    ordered_spec = {}
                    ordered_spec["name"] = full_spec.pop("name")
                    ordered_spec["data_table"] = full_spec.pop("data_table")
                    if "plan_id" in full_spec:
                        ordered_spec["plan_id"] = full_spec.pop("plan_id")
                    ordered_spec.update(full_spec)

                    unified_yaml = yaml.dump(
                        ordered_spec, default_flow_style=False, sort_keys=False
                    )

                    # Validate training section still exists
                    training_config = full_spec.get("training")
                    if not training_config:
                        return _error_in_section(
                            "Edited YAML missing 'training' section"
                        )
                    loss_config = training_config.get("loss")
                    if not loss_config:
                        return _error_in_section(
                            "Edited YAML missing 'training.loss' section"
                        )
                finally:
                    workflow.cleanup()

            # Save model to DB with plan_id if using ML plan
            # IMPORTANT: Save unified_yaml (with training config) not model_yaml (without training)
            try:
                plan_id = ml_plan.get("plan_id") if ml_plan else None
                model = self._save_model_to_db(
                    name=str(name),
                    yaml_content=unified_yaml,
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
                "yaml_content": unified_yaml,  # Store unified YAML (with training)
                "from_ml_plan": ml_plan is not None,
                "training_launched": False,  # Will update if training launches
            }

            # Launch training directly with the model (no trainer needed)
            # The training config is embedded in the unified model YAML
            if printer:
                printer.print("")
                printer.print(f"→ Launching training for model '{name}'...")

            try:
                job_id = await asyncio.to_thread(
                    self.runtime.train_model,
                    model_name=name,
                    train_table=str(train_table),
                )

                lines.append("")
                lines.append("✓ Training job submitted successfully.")
                lines.append(f"Training table: {train_table}")
                lines.append(f"Job ID: {job_id}")

                # Show job monitoring instructions
                if printer:
                    printer.print("")
                    printer.print(
                        "[dim][cyan]ℹ Monitor training progress:[/cyan][/dim]"
                    )
                    printer.print(f"[dim]  • Status: /ml jobs status {job_id}[/dim]")
                    printer.print(f"[dim]  • Logs: /ml jobs logs {job_id}[/dim]")

                result_metadata["training_launched"] = True
                result_metadata["job_id"] = job_id

                # Handle TensorBoard launch
                if not auto_confirm and self.ui:
                    if self.tensorboard_manager:
                        try:
                            await self._handle_tensorboard_launch(job_id, printer)
                        except (OSError, RuntimeError) as e:
                            # Known TensorBoard launch failures
                            if printer:
                                printer.print(
                                    f"[yellow]⚠️  TensorBoard setup failed: {e}[/yellow]"
                                )
                            self._show_manual_tensorboard_instructions(job_id, printer)
                        except Exception as e:
                            # Log unexpected errors with full traceback
                            import logging

                            logging.exception(
                                "Unexpected error during TensorBoard launch"
                            )
                            error_msg = f"{e.__class__.__name__}: {e}"
                            if printer:
                                printer.print(
                                    f"[yellow]⚠️  TensorBoard setup failed: {error_msg}[/yellow]"
                                )
                            self._show_manual_tensorboard_instructions(job_id, printer)
                    else:
                        # No TensorBoard manager available
                        if printer:
                            printer.print(
                                "[dim]ℹ TensorBoard auto-launch not available "
                                "(restart arc chat to enable)[/dim]"
                            )
                        self._show_manual_tensorboard_instructions(job_id, printer)

            except MLRuntimeError as exc:
                # Training launch failed but model was created
                if printer:
                    printer.print("⚠ Training Validation Failed")
                    printer.print("")
                    printer.print(f"[red]{exc}[/red]")
                    printer.print("")
                    printer.print("[dim]Note: Model was registered successfully[/dim]")

                lines.append("")
                lines.append("⚠ Training Validation Failed")
                lines.append("")
                lines.append(f"{exc}")

                result_metadata["training_launch_failed"] = True
                result_metadata["training_error"] = str(exc)

            # Parse the unified YAML to get model spec dict for JSON output
            full_spec_dict = yaml.safe_load(unified_yaml)

            # Build structured JSON output
            if result_metadata.get("training_launched"):
                # Training launched successfully
                output_json = self._build_model_result(
                    status="accepted",
                    model_id=model_id,
                    model_spec_dict=full_spec_dict,
                    training_job_id=result_metadata.get("job_id"),
                    train_table=train_table,
                    training_status="submitted",
                )
            elif result_metadata.get("training_launch_failed"):
                # Training launch failed
                output_json = self._build_model_result(
                    status="accepted",
                    model_id=model_id,
                    model_spec_dict=full_spec_dict,
                    train_table=train_table,
                    training_status="failed",
                    training_error=result_metadata.get("training_error"),
                )
            else:
                # Shouldn't happen, but handle gracefully
                output_json = self._build_model_result(
                    status="accepted",
                    model_id=model_id,
                    model_spec_dict=full_spec_dict,
                    training_status="not_started",
                )

            return ToolResult(
                success=True,
                output=output_json,
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
            yaml_content: Unified YAML specification (includes training section)
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

        # Validate YAML first - need to separate model and training sections
        try:
            full_spec = yaml.safe_load(yaml_content)
            # Remove training section for validation
            training_section = full_spec.pop("training", None)
            model_yaml_only = yaml.dump(
                full_spec, default_flow_style=False, sort_keys=False
            )

            model_spec = ModelSpec.from_yaml(model_yaml_only)
            _ = model_spec.get_input_names()
            _ = model_spec.get_output_names()

            # Verify training section exists
            if not training_section:
                raise ValueError("Unified YAML must include 'training' section")
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
        from pathlib import Path

        from arc.core.config import SettingsManager

        # Get actual TensorBoard log directory from training run database
        logdir = None
        if self.services and hasattr(self.services, "training_tracking"):
            try:
                run = self.services.training_tracking.get_run_by_job_id(job_id)
                if run and run.tensorboard_log_dir:
                    logdir = Path(run.tensorboard_log_dir)
            except Exception:
                pass  # Fall through to default

        # Fallback to default if not found in database
        if logdir is None:
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

        # Get actual TensorBoard log directory from training run database
        logdir = None
        if self.services and hasattr(self.services, "training_tracking"):
            try:
                run = self.services.training_tracking.get_run_by_job_id(job_id)
                if run and run.tensorboard_log_dir:
                    logdir = run.tensorboard_log_dir
            except Exception:
                pass  # Fall through to default

        # Fallback to default if not found in database
        if logdir is None:
            logdir = f"tensorboard/run_{job_id}"
        if section_printer:
            section_printer.print("")
            section_printer.print("[dim][cyan]ℹ View training results:[/cyan][/dim]")
            section_printer.print(f"[dim]  • Status: /ml jobs status {job_id}[/dim]")
            section_printer.print(
                f"[dim]  • TensorBoard: tensorboard --logdir {logdir}[/dim]"
            )
        else:
            self.ui._printer.console.print()
            self.ui._printer.console.print("[cyan]ℹ View training results:[/cyan]")
            self.ui._printer.console.print(f"  • Status: /ml jobs status {job_id}")
            self.ui._printer.console.print(
                f"  • TensorBoard: tensorboard --logdir {logdir}"
            )
