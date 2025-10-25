"""ML Evaluator specification generation tool."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import yaml

from arc.tools.base import BaseTool, ToolResult
from arc.utils.yaml_workflow import YamlConfirmationWorkflow


class MLEvaluatorGeneratorTool(BaseTool):
    """Tool for generating Arc-Graph evaluator specifications via LLM."""

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
        context: str | None = None,
        trainer_id: str | None = None,
        data_table: str | None = None,
        target_column: str | None = None,
        auto_confirm: bool = False,
    ) -> ToolResult:
        if not self.api_key:
            return ToolResult.error_result(
                "API key required for evaluator generation. "
                "Set ARC_API_KEY or configure an API key before using this tool."
            )

        if not self.services:
            return ToolResult.error_result(
                "Evaluator generation service unavailable. "
                "Database services not initialized."
            )

        if (
            not name
            or not context
            or not trainer_id
            or not data_table
            or not target_column
        ):
            return ToolResult.error_result(
                "Parameters 'name', 'context', 'trainer_id', 'data_table', and "
                "'target_column' are required to generate an evaluator specification."
            )

        # Get the registered trainer
        try:
            trainer_record = self.services.trainers.get_trainer_by_id(str(trainer_id))
            if not trainer_record:
                return ToolResult.error_result(
                    f"Trainer '{trainer_id}' not found in registry. "
                    "Please train a model first using /ml train"
                )
        except Exception as exc:
            return ToolResult.error_result(
                f"Failed to retrieve trainer '{trainer_id}': {exc}"
            )

        # Check if target column exists in data table
        target_column_exists = False
        try:
            schema_info = self.services.schema.get_schema_info(target_db="user")
            columns = schema_info.get_column_names(str(data_table))
            target_column_exists = str(target_column) in columns

            if self.ui:
                if target_column_exists:
                    self.ui.show_info(
                        f"ℹ Target column '{target_column}' found in data - "
                        "will include metrics"
                    )
                else:
                    self.ui.show_info(
                        f"ℹ Target column '{target_column}' not in data - "
                        "prediction mode"
                    )
        except Exception as exc:
            # If schema check fails, default to assuming target exists
            if self.ui:
                self.ui.show_info(f" Could not check if target column exists: {exc}")
            target_column_exists = True

        # Show UI feedback if UI is available
        if self.ui:
            self.ui.show_info(f"ℹ Using registered trainer: {trainer_record.id}")
            self.ui.show_info(f"> Generating evaluator specification for '{name}'...")

        # Extract knowledge IDs from context
        from arc.core.agents.shared.knowledge_selector import (
            extract_knowledge_ids_from_text,
        )

        recommended_knowledge_ids = extract_knowledge_ids_from_text(
            instruction=context,
            ml_plan_architecture=None,
        )

        from arc.core.agents.ml_evaluate import MLEvaluateAgent

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
                instruction=str(context),
                trainer_ref=str(trainer_id),
                trainer_spec_yaml=trainer_record.spec,
                dataset=str(data_table),
                target_column=str(target_column),
                target_column_exists=target_column_exists,
                recommended_knowledge_ids=recommended_knowledge_ids,
            )
        except Exception as exc:
            # Import here to avoid circular imports
            from arc.core.agents.ml_evaluate import MLEvaluateError

            if isinstance(exc, MLEvaluateError):
                return ToolResult.error_result(str(exc))
            return ToolResult.error_result(
                f"Unexpected error during evaluator generation: {exc}"
            )

        # Validate the generated evaluator using Arc-Graph validator
        try:
            from arc.graph.evaluator import EvaluatorValidationError

            evaluator_dict = yaml.safe_load(evaluator_yaml)
            from arc.graph.evaluator import validate_evaluator_dict

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
                editor_func=self._create_editor(),
                ui_interface=self.ui,
                yaml_type_name="evaluator",
                yaml_suffix=".arc-evaluator.yaml",
            )

            context_dict = {
                "evaluator_name": str(name),
                "context": str(context),
                "trainer_id": str(trainer_id),
                "trainer_ref": str(trainer_id),
                "trainer_spec_yaml": trainer_record.spec,
                "dataset": str(data_table),
                "target_column": str(target_column),
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
                        output=" Evaluator generation cancelled.",
                        metadata={"cancelled": True},
                    )
                evaluator_yaml = final_yaml
            finally:
                workflow.cleanup()

        # Auto-register evaluator to database
        try:
            from arc.database.models.evaluator import Evaluator

            # Generate unique ID for evaluator
            evaluator_id = f"{name}-v{evaluator_spec.version or 1}"

            # Get next version number
            next_version = self.services.evaluators.get_next_version_for_name(str(name))

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

            if self.ui:
                self.ui.show_system_success(
                    f" Evaluator registered: {evaluator_record.id}"
                )
        except Exception as exc:
            return ToolResult.error_result(f"Failed to register evaluator: {exc}")

        summary = (
            f"Trainer: {evaluator_spec.trainer_ref} • "
            f"Dataset: {evaluator_spec.dataset} • "
            f"Target: {evaluator_spec.target_column}"
        )

        lines = [
            f" Evaluator '{evaluator_record.id}' created and registered.",
            summary,
        ]

        if auto_confirm:
            lines.append("\n  YAML:")
            lines.append(evaluator_yaml.strip())
        else:
            lines.append(" Evaluator approved and ready for evaluation.")

        return ToolResult.success_result("\n".join(lines))

    def _create_validator(self):
        """Create validator function for the workflow.

        Returns:
            Function that validates YAML and returns list of error strings
        """

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

    def _create_editor(self):
        """Create editor function for AI-assisted editing with conversation history.

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
            # Extract knowledge IDs from feedback
            from arc.core.agents.shared.knowledge_selector import (
                extract_knowledge_ids_from_text,
            )

            recommended_knowledge_ids = extract_knowledge_ids_from_text(
                instruction=feedback,
                ml_plan_architecture=None,
            )

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
                    trainer_ref=context["trainer_ref"],
                    trainer_spec_yaml=context["trainer_spec_yaml"],
                    dataset=context["dataset"],
                    target_column=context["target_column"],
                    target_column_exists=context.get("target_column_exists", True),
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
