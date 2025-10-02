"""Machine learning tool implementations."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from arc.core.agents.ml_plan import MLPlanAgent
from arc.core.agents.model_generator import ModelGeneratorAgent
from arc.core.agents.predictor_generator import (
    PredictorGeneratorAgent,
)
from arc.core.agents.trainer_generator import (
    TrainerGeneratorAgent,
)
from arc.graph.model import ModelValidationError, validate_model_dict
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


class MLCreateModelTool(BaseTool):
    """Tool for registering new Arc-Graph models."""

    def __init__(self, runtime: MLRuntime):
        self.runtime = runtime

    async def execute(
        self,
        *,
        name: str | None = None,
        schema_path: str | None = None,
        description: str | None = None,
        model_type: str | None = None,
    ) -> ToolResult:
        if not name or not schema_path:
            return ToolResult.error_result(
                "Parameters 'name' and 'schema_path' are required to create a model."
            )

        schema_path_obj = Path(schema_path)

        try:
            model = await asyncio.to_thread(
                self.runtime.create_model,
                name=str(name),
                schema_path=schema_path_obj,
                description=str(description) if description else None,
                model_type=str(model_type) if model_type else None,
            )
        except MLRuntimeError as exc:
            return ToolResult.error_result(str(exc))
        except Exception as exc:  # noqa: BLE001
            return ToolResult.error_result(f"Unexpected error creating model: {exc}")

        message_lines = [
            f"Model '{model.name}' registered.",
            f"ID: {model.id}",
            f"Version: {model.version}",
        ]
        if model.description:
            message_lines.append(f"Description: {model.description}")

        return ToolResult.success_result("\n".join(message_lines))


class MLTrainTool(BaseTool):
    """Tool for launching training jobs."""

    def __init__(self, runtime: MLRuntime):
        self.runtime = runtime

    async def execute(
        self,
        *,
        model_name: str | None = None,
        train_table: str | None = None,
        target_column: str | None = None,
        validation_table: str | None = None,
        validation_split: float | int | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | int | None = None,
        checkpoint_dir: str | None = None,
        description: str | None = None,
        tags: Sequence[str] | str | None = None,
    ) -> ToolResult:
        if not model_name or not train_table:
            return ToolResult.error_result(
                "Parameters 'model_name' and 'train_table' are required "
                "to train a model."
            )

        try:
            parsed_epochs = _as_optional_int(epochs, "epochs")
            parsed_batch_size = _as_optional_int(batch_size, "batch_size")
            parsed_learning_rate = _as_optional_float(learning_rate, "learning_rate")
            parsed_validation_split = _as_optional_float(
                validation_split, "validation_split"
            )
            parsed_tags = _as_string_list(tags, "tags")
        except ValueError as exc:
            return ToolResult.error_result(str(exc))

        try:
            job_id = await asyncio.to_thread(
                self.runtime.train_model,
                model_name=str(model_name),
                train_table=str(train_table),
                target_column=str(target_column) if target_column else None,
                validation_table=str(validation_table) if validation_table else None,
                validation_split=parsed_validation_split,
                epochs=parsed_epochs,
                batch_size=parsed_batch_size,
                learning_rate=parsed_learning_rate,
                checkpoint_dir=str(checkpoint_dir) if checkpoint_dir else None,
                description=str(description) if description else None,
                tags=parsed_tags,
            )
        except MLRuntimeError as exc:
            return ToolResult.error_result(str(exc))
        except Exception as exc:  # noqa: BLE001
            return ToolResult.error_result(
                f"Unexpected error launching training: {exc}"
            )

        lines = [
            "Training job submitted successfully.",
            f"Model: {model_name}",
            f"Training table: {train_table}",
            f"Job ID: {job_id}",
        ]
        if validation_table:
            lines.append(f"Validation table: {validation_table}")
        if parsed_tags:
            lines.append(f"Tags: {', '.join(parsed_tags)}")

        return ToolResult.success_result("\n".join(lines))


class MLPredictTool(BaseTool):
    """Tool for running inference and saving predictions to a table."""

    def __init__(self, runtime: MLRuntime):
        self.runtime = runtime

    async def execute(
        self,
        *,
        model_name: str | None = None,
        table_name: str | None = None,
        output_table: str | None = None,
        batch_size: int | None = None,
        limit: int | None = None,
        device: str | None = None,
    ) -> ToolResult:
        if not model_name or not table_name or not output_table:
            return ToolResult.error_result(
                "Parameters 'model_name', 'table_name', and 'output_table' "
                "are required to run prediction."
            )

        try:
            parsed_batch_size = _as_optional_int(batch_size, "batch_size")
            parsed_limit = _as_optional_int(limit, "limit")
        except ValueError as exc:
            return ToolResult.error_result(str(exc))

        try:
            summary = await asyncio.to_thread(
                self.runtime.predict,
                model_name=str(model_name),
                table_name=str(table_name),
                batch_size=parsed_batch_size or 32,
                limit=parsed_limit,
                output_table=str(output_table),
                device=str(device) if device else None,
            )
        except MLRuntimeError as exc:
            return ToolResult.error_result(str(exc))
        except Exception as exc:  # noqa: BLE001
            return ToolResult.error_result(f"Unexpected error during prediction: {exc}")

        outputs = ", ".join(summary.outputs) if summary.outputs else "None"
        lines = [
            "Prediction completed successfully.",
            f"Model: {model_name}",
            f"Source table: {table_name}",
            f"Rows processed: {summary.total_predictions}",
            f"Outputs: {outputs}",
            f"Results saved to table: {summary.saved_table or output_table}",
        ]

        return ToolResult.success_result("\n".join(lines))


class MLModelGeneratorTool(BaseTool):
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
        context: str | None = None,
        data_table: str | None = None,
        target_column: str | None = None,
        output_path: str | None = None,
        auto_confirm: bool = False,
        category: str | None = None,
        ml_plan: dict | None = None,
    ) -> ToolResult:
        if not self.api_key:
            return ToolResult.error_result(
                "API key required for model generation. "
                "Set ARC_API_KEY or configure an API key before using this tool."
            )

        if not self.services:
            return ToolResult.error_result(
                "Model generation service unavailable. "
                "Database services not initialized."
            )

        # If ML plan is provided, use it for context
        if ml_plan:
            # Extract information from ML plan
            from arc.core.ml_plan import MLPlan

            plan = MLPlan.from_dict(ml_plan)

            # Use plan data if parameters not explicitly provided
            if not context:
                context = plan.summary
            if not data_table:
                data_table = ml_plan.get("data_table")
            if not target_column:
                target_column = ml_plan.get("target_column")
            if not category and plan.selected_components:
                category = plan.selected_components[0]  # Use primary component

            # Create aggregated context from the plan for the model generator
            aggregated_context = plan.to_generation_context()
        else:
            aggregated_context = None

        if not name or not context or not data_table:
            return ToolResult.error_result(
                "Parameters 'name', 'context', and 'data_table' are required "
                "to generate a model specification."
            )

        # Show UI feedback if UI is available
        if self.ui:
            if ml_plan:
                self.ui.show_info(
                    f"🤖 Generating model specification for '{name}' "
                    "using ML plan guidance..."
                )
            else:
                self.ui.show_info(f"🤖 Generating model specification for '{name}'...")

        agent = ModelGeneratorAgent(
            self.services,
            self.api_key,
            self.base_url,
            self.model,
        )

        try:
            # Prepare components list from category
            components = [category] if category else None

            model_spec, model_yaml = await agent.generate_model(
                name=str(name),
                table_name=str(data_table),
                target_column=target_column,
                components=components,
                aggregated_context=aggregated_context,
            )
        except Exception as exc:
            # Import here to avoid circular imports
            from arc.core.agents.model_generator import ModelGeneratorError

            if isinstance(exc, ModelGeneratorError):
                return ToolResult.error_result(str(exc))
            return ToolResult.error_result(
                f"Unexpected error during model generation: {exc}"
            )

        # Validate the generated model using Arc-Graph validator
        try:
            model_dict = yaml.safe_load(model_yaml)
            validate_model_dict(model_dict)
        except yaml.YAMLError as exc:
            return ToolResult.error_result(
                f"Generated model contains invalid YAML: {exc}"
            )
        except ModelValidationError as exc:
            return ToolResult.error_result(f"Generated model failed validation: {exc}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult.error_result(
                f"Unexpected error during model validation: {exc}"
            )

        # Interactive confirmation workflow (unless auto_confirm is True)
        if not auto_confirm:
            workflow = YamlConfirmationWorkflow(
                validator_func=self._create_validator(),
                editor_func=self._create_editor(aggregated_context),
                ui_interface=self.ui,
                yaml_type_name="model",
                yaml_suffix=".arc-model.yaml",
            )

            context_dict = {
                "model_name": str(name),
                "table_name": str(data_table),
                "target_column": target_column,
                "components": components,
            }

            try:
                proceed, final_yaml = await workflow.run_workflow(
                    model_yaml, context_dict, output_path
                )
                if not proceed:
                    return ToolResult.success_result(
                        "✗ Model generation cancelled by user."
                    )
                model_yaml = final_yaml
            finally:
                workflow.cleanup()

        # Save to file if output_path provided
        if output_path:
            try:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_text(model_yaml)
            except Exception as exc:
                return ToolResult.error_result(
                    f"Failed to save model to {output_path}: {exc}"
                )

        summary = (
            f"Inputs: {len(model_spec.inputs)} • Nodes: {len(model_spec.graph)} "
            f"• Outputs: {len(model_spec.outputs)}"
        )

        lines = [
            f"✓ Model specification generated for '{name}'.",
            summary,
        ]

        if output_path:
            lines.append(f" Saved to: {output_path}")

        if auto_confirm:
            lines.append("\n YAML:")
            lines.append(model_yaml.strip())
        else:
            lines.append("✓ Model approved and ready for use.")

        return ToolResult.success_result("\n".join(lines))

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

    def _create_editor(self, aggregated_context: dict[str, Any] | None = None):
        """Create editor function for AI-assisted editing in the workflow.

        Args:
            aggregated_context: Optional aggregated context from ML plan

        Returns:
            Async function that edits YAML based on user feedback
        """

        async def edit(
            yaml_content: str, feedback: str, context: dict[str, Any]
        ) -> str | None:
            agent = ModelGeneratorAgent(
                self.services,
                self.api_key,
                self.base_url,
                self.model,
            )

            try:
                _model_spec, edited_yaml = await agent.generate_model(
                    name=context["model_name"],
                    table_name=context["table_name"],
                    target_column=context.get("target_column"),
                    components=context.get("components"),
                    aggregated_context=aggregated_context,
                    existing_yaml=yaml_content,
                    editing_instructions=feedback,
                )
                return edited_yaml
            except Exception as e:
                if self.ui:
                    self.ui.show_system_error(f"❌ AI editing failed: {str(e)}")
                return None

        return edit


class MLTrainerGeneratorTool(BaseTool):
    """Tool for generating Arc-Graph trainer specifications via LLM."""

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
        model_spec_path: str | None = None,
        output_path: str | None = None,
    ) -> ToolResult:
        if not self.api_key:
            return ToolResult.error_result(
                "API key required for trainer generation. "
                "Set ARC_API_KEY or configure an API key before using this tool."
            )

        if not self.services:
            return ToolResult.error_result(
                "Trainer generation service unavailable. "
                "Database services not initialized."
            )

        if not name or not context or not model_spec_path:
            return ToolResult.error_result(
                "Parameters 'name', 'context', and 'model_spec_path' are required "
                "to generate a trainer specification."
            )

        # Check that model spec file exists
        if not Path(model_spec_path).exists():
            return ToolResult.error_result(
                f"Model specification file not found: {model_spec_path}"
            )

        # Show UI feedback if UI is available
        if self.ui:
            self.ui.show_info(f"🤖 Generating trainer specification for '{name}'...")

        agent = TrainerGeneratorAgent(
            self.services,
            self.api_key,
            self.base_url,
            self.model,
        )

        try:
            trainer_spec, trainer_yaml = await agent.generate_trainer(
                name=str(name),
                user_context=str(context),
                model_spec_path=str(model_spec_path),
            )
        except Exception as exc:
            # Import here to avoid circular imports
            from arc.core.agents.trainer_generator import TrainerGeneratorError

            if isinstance(exc, TrainerGeneratorError):
                return ToolResult.error_result(str(exc))
            return ToolResult.error_result(
                f"Unexpected error during trainer generation: {exc}"
            )

        lines = [
            f"Trainer specification generated for '{name}'.",
            f"Loss: {trainer_spec.loss.type} • "
            f"Optimizer: {trainer_spec.optimizer.type}",
        ]

        if output_path:
            lines.append(f"Saved to: {output_path}")

        lines.append("YAML:")
        lines.append(trainer_yaml.strip())

        return ToolResult.success_result("\n".join(lines))


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
        user_context: str | None = None,
        data_table: str | None = None,
        target_column: str | None = None,
        conversation_history: list[dict] | None = None,
        feedback: str | None = None,
        previous_plan: dict | None = None,
    ) -> ToolResult:
        if not self.api_key:
            return ToolResult.error_result(
                "API key required for ML planning. "
                "Set ARC_API_KEY or configure an API key before using this tool."
            )

        if not self.services:
            return ToolResult.error_result(
                "ML planning service unavailable. Database services not initialized."
            )

        if not user_context or not data_table or not target_column:
            return ToolResult.error_result(
                "Parameters 'user_context', 'data_table', and 'target_column' "
                "are required for ML planning."
            )

        if conversation_history is None:
            return ToolResult.error_result(
                "Parameter 'conversation_history' is required for comprehensive "
                "ML planning. The full conversation history enables context-aware "
                "planning."
            )

        agent = MLPlanAgent(
            self.services,
            self.api_key,
            self.base_url,
            self.model,
        )

        try:
            # Import MLPlan for plan management
            from arc.core.ml_plan import MLPlan

            # Check if auto-accept is enabled
            if self.agent and self.agent.ml_plan_auto_accept:
                # Auto-accept mode - skip workflow
                pass  # Continue to generate plan but skip confirmation

            # Filter conversation history by timestamp for revisions
            # Only include messages since last plan (to avoid context pollution)
            filtered_history = conversation_history
            if previous_plan and "created_at" in previous_plan:
                try:
                    # For now, just use recent messages (simple approach)
                    # TODO: Filter by timestamp when message timestamps are available
                    filtered_history = conversation_history[-10:]  # Last 10 messages
                except Exception:
                    # If filtering fails, use all history
                    filtered_history = conversation_history

            # Internal loop for handling feedback (option C)
            current_feedback = feedback
            version = previous_plan.get("version", 0) + 1 if previous_plan else 1

            while True:
                # Generate the plan
                analysis = await agent.analyze_problem(
                    user_context=str(user_context),
                    table_name=str(data_table),
                    target_column=str(target_column),
                    conversation_history=filtered_history,
                    feedback=current_feedback,
                    stream=False,
                )

                # Determine stage
                if previous_plan:
                    stage = previous_plan.get("stage", "initial")
                    if current_feedback and "training" in str(current_feedback).lower():
                        stage = "post_training"
                    elif (
                        current_feedback
                        and "evaluation" in str(current_feedback).lower()
                    ):
                        stage = "post_evaluation"
                    reason = (
                        f"Revised based on feedback: {current_feedback[:100]}..."
                        if current_feedback
                        else "Plan revision"
                    )
                else:
                    stage = "initial"
                    reason = None

                plan = MLPlan.from_analysis(
                    analysis, version=version, stage=stage, reason=reason
                )

                # If auto-accept is enabled, skip workflow
                if self.agent and self.agent.ml_plan_auto_accept:
                    output_message = "Plan automatically accepted (auto-accept enabled)"
                    break

                # Display plan and run confirmation workflow
                if self.ui:
                    from arc.utils.ml_plan_workflow import MLPlanConfirmationWorkflow

                    workflow = MLPlanConfirmationWorkflow(self.ui)
                    result = await workflow.run_workflow(
                        plan, previous_plan is not None
                    )

                    choice = result.get("choice")

                    if choice == "accept":
                        output_message = (
                            "Plan accepted. Ready to proceed with implementation."
                        )
                        break
                    elif choice == "accept_all":
                        # Enable auto-accept for this session
                        if self.agent:
                            self.agent.ml_plan_auto_accept = True
                        output_message = (
                            "Plan accepted. Auto-accept enabled for this session."
                        )
                        break
                    elif choice == "feedback":
                        # Get feedback and loop to revise
                        current_feedback = result.get("feedback", "")
                        version += 1
                        # Continue loop to generate revised plan
                        continue
                    elif choice == "cancel":
                        # Display cancellation message and return to main agent
                        if self.ui:
                            self.ui._printer.console.print()
                            self.ui._printer.console.print(
                                "[yellow]ML plan cancelled.[/yellow] "
                                "ML plan is the first step for the ML workflow, "
                                "including feature pipelines, model design, "
                                "training and evaluation."
                            )
                            self.ui._printer.console.print()

                        # Return to main agent with prompt
                        return ToolResult(
                            success=True,
                            output=(
                                "ML plan cancelled. What would you like to do instead?"
                            ),
                            metadata={"cancelled": True},
                        )
                else:
                    # Headless mode - auto-accept
                    formatted_result = plan.format_for_display()
                    output_message = (
                        "I've created a comprehensive ML workflow plan based on "
                        f"your requirements.\n\n{formatted_result}"
                    )
                    break

            # Return with plan in metadata for storage
            plan_dict = plan.to_dict()
            plan_dict["data_table"] = str(data_table)
            plan_dict["target_column"] = str(target_column)

            return ToolResult(
                success=True,
                output=output_message,
                metadata={
                    "ml_plan": plan_dict,
                    "is_revision": previous_plan is not None,
                },
            )

        except Exception as exc:
            from arc.core.agents.ml_plan import MLPlanError

            if isinstance(exc, MLPlanError):
                return ToolResult.error_result(str(exc))
            return ToolResult.error_result(
                f"Unexpected error during ML planning: {exc}"
            )


class MLPredictorGeneratorTool(BaseTool):
    """Tool for generating Arc-Graph predictor specifications via LLM."""

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
        context: str | None = None,
        model_spec_path: str | None = None,
        trainer_spec_path: str | None = None,
        output_path: str | None = None,
    ) -> ToolResult:
        if not self.api_key:
            return ToolResult.error_result(
                "API key required for predictor generation. "
                "Set ARC_API_KEY or configure an API key before using this tool."
            )

        if not self.services:
            return ToolResult.error_result(
                "Predictor generation service unavailable. "
                "Database services not initialized."
            )

        if not model_spec_path or not context:
            return ToolResult.error_result(
                "Parameters 'model_spec_path' and 'context' are required "
                "to generate a predictor specification."
            )

        # Check that model spec file exists
        if not Path(model_spec_path).exists():
            return ToolResult.error_result(
                f"Model specification file not found: {model_spec_path}"
            )

        # Check trainer spec file if provided
        if trainer_spec_path and not Path(trainer_spec_path).exists():
            return ToolResult.error_result(
                f"Trainer specification file not found: {trainer_spec_path}"
            )

        # Show UI feedback if UI is available
        if self.ui:
            self.ui.show_info("🤖 Generating predictor specification...")

        agent = PredictorGeneratorAgent(
            self.services,
            self.api_key,
            self.base_url,
            self.model,
        )

        try:
            predictor_yaml = await agent.generate_predictor(
                user_context=str(context),
                model_spec_path=str(model_spec_path),
                trainer_spec_path=str(trainer_spec_path) if trainer_spec_path else None,
            )
        except Exception as exc:
            # Import here to avoid circular imports
            from arc.core.agents.predictor_generator import PredictorGeneratorError

            if isinstance(exc, PredictorGeneratorError):
                return ToolResult.error_result(str(exc))
            return ToolResult.error_result(
                f"Unexpected error during predictor generation: {exc}"
            )

        if output_path:
            try:
                Path(output_path).write_text(predictor_yaml, encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                return ToolResult.error_result(
                    f"Predictor generated but failed to save file: {exc}"
                )

        lines = [
            "Predictor specification generated successfully.",
        ]

        if output_path:
            lines.append(f"Saved to: {output_path}")

        lines.append("YAML:")
        lines.append(predictor_yaml.strip())

        return ToolResult.success_result("\n".join(lines))
