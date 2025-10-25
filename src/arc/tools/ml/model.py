"""ML Model specification generation tool."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from arc.database.models.model import Model

from arc.core.agents.ml_model import MLModelAgent
from arc.graph.model import ModelValidationError, validate_model_dict
from arc.tools.base import BaseTool, ToolResult
from arc.utils.yaml_workflow import YamlConfirmationWorkflow


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
        ml_plan: dict | None = None,
    ) -> ToolResult:
        """Generate Arc-Graph model specification via LLM.

        Args:
            name: Model name
            instruction: User's instruction for model architecture (PRIMARY driver)
            data_table: Database table to profile for generation
            target_column: Target column for prediction
            auto_confirm: Skip confirmation workflows (for testing only)
            ml_plan: Optional ML plan dict containing model_architecture_and_loss
                guidance (SECONDARY baseline, automatically injected by the main agent)

        Note on instruction vs ml_plan precedence:
            - instruction: PRIMARY driver - user's immediate, specific
              architecture request
            - ml_plan: SECONDARY baseline - background architectural guidance
            - When there's a conflict, instruction takes precedence
            - Example: If instruction says "use 3 hidden layers" but plan says
              "use 2 layers", the model spec should use 3 hidden layers
              (instruction wins)
            - The LLM agent should use ml_plan as baseline architectural guidance
              and augment/override it with specifics from instruction

        Returns:
            ToolResult with model registration details
        """
        # Show section title first, before any validation
        # Keep the section printer reference to use for all messages including errors
        ml_model_section_printer = None
        if self.ui:
            self._ml_model_section = self.ui._printer.section(
                color="magenta", add_dot=True
            )
            ml_model_section_printer = self._ml_model_section.__enter__()
            ml_model_section_printer.print("ML Model")

        # Helper to close section and return error
        def _error_in_section(message: str) -> ToolResult:
            if ml_model_section_printer:
                ml_model_section_printer.print("")
                ml_model_section_printer.print(f" {message}")
            if self.ui and hasattr(self, "_ml_model_section"):
                self._ml_model_section.__exit__(None, None, None)
            return ToolResult(success=False, output="", metadata={"error_shown": True})

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

        # Validate: either ml_plan or instruction must be provided
        if not ml_plan and not instruction:
            return _error_in_section(
                "Either 'ml_plan' or 'instruction' must be provided. "
                "ML plan is recommended for full ML workflows."
            )

        # Extract from ML plan if provided (ml_plan is PRIMARY source)
        ml_plan_architecture = None
        if ml_plan:
            from arc.core.ml_plan import MLPlan

            plan = MLPlan.from_dict(ml_plan)

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

        # Show generation status
        if ml_model_section_printer:
            ml_model_section_printer.print(
                "[dim]Generating Arc-Graph model specification...[/dim]"
            )

        # Generate model using agent
        agent = MLModelAgent(
            self.services,
            self.api_key,
            self.base_url,
            self.model,
        )

        # Extract knowledge IDs from instruction and ML Plan
        from arc.core.agents.shared.knowledge_selector import (
            extract_knowledge_ids_from_text,
        )

        recommended_knowledge_ids = extract_knowledge_ids_from_text(
            instruction=instruction,
            ml_plan_architecture=ml_plan_architecture,
        )

        try:
            model_spec, model_yaml, conversation_history = await agent.generate_model(
                name=str(name),
                user_context=instruction,  # Use instruction as user_context
                table_name=str(data_table),
                target_column=target_column,
                ml_plan_architecture=ml_plan_architecture,
                recommended_knowledge_ids=recommended_knowledge_ids,
            )
        except Exception as exc:
            # Import here to avoid circular imports
            from arc.core.agents.ml_model import MLModelError

            if isinstance(exc, MLModelError):
                return ToolResult.error_result(str(exc))
            return ToolResult.error_result(
                f"Unexpected error during model generation: {exc}"
            )

        # Validate the generated model using Arc-Graph validator
        try:
            model_dict = yaml.safe_load(model_yaml)
            validate_model_dict(model_dict)
        except (yaml.YAMLError, ModelValidationError) as exc:
            return ToolResult.error_result(f"Model validation failed: {exc}")
        except Exception as exc:
            # Log unexpected errors with full traceback
            import logging

            logging.exception("Unexpected error during model validation")
            return ToolResult.error_result(
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
                "recommended_knowledge_ids": recommended_knowledge_ids,
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
                    if ml_model_section_printer:
                        ml_model_section_printer.print("")  # Empty line
                        ml_model_section_printer.print(
                            "[dim] Model generation cancelled by user.[/dim]"
                        )
                    # Close the section before returning
                    if self.ui and hasattr(self, "_ml_model_section"):
                        self._ml_model_section.__exit__(None, None, None)
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
            return ToolResult.error_result(f"Failed to save model to DB: {exc}")

        # Display registration confirmation in the ML Model section
        if ml_model_section_printer:
            ml_model_section_printer.print("")  # Empty line before confirmation
            ml_model_section_printer.print(
                f"[dim] Model '{name}' registered to database "
                f"({model_id} • {len(model_spec.inputs)} inputs • "
                f"{len(model_spec.graph)} nodes • "
                f"{len(model_spec.outputs)} outputs)[/dim]"
            )

        # Close the ML Model section
        if self.ui and hasattr(self, "_ml_model_section"):
            self._ml_model_section.__exit__(None, None, None)

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
