"""Machine learning tool implementations."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from arc.database.models.model import Model

from arc.core.agents.ml_plan import MLPlanAgent
from arc.core.agents.model_generator import ModelGeneratorAgent
from arc.core.agents.trainer_generator import (
    TrainerGeneratorAgent,
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
        context: str | None = None,
        data_table: str | None = None,
        target_column: str | None = None,
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

        # Validate: either ml_plan or context must be provided
        if not ml_plan and not context:
            return ToolResult.error_result(
                "Either 'ml_plan' or 'context' must be provided. "
                "ML plan is recommended for full ML workflows."
            )

        # Extract from ML plan if provided (ml_plan is PRIMARY source)
        ml_plan_architecture = None
        if ml_plan:
            from arc.core.ml_plan import MLPlan

            plan = MLPlan.from_dict(ml_plan)

            # Use plan data if parameters not explicitly provided
            if not context:
                context = plan.summary
            if not data_table:
                data_table = ml_plan.get("data_table")
            if not target_column:
                target_column = ml_plan.get("target_column")

            # CRITICAL: Extract architecture guidance from ML plan
            ml_plan_architecture = plan.model_architecture_and_loss

        # Validate required parameters
        if not name or not data_table:
            return ToolResult.error_result(
                "Parameters 'name' and 'data_table' are required "
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
            model_spec, model_yaml = await agent.generate_model(
                name=str(name),
                user_context=context,  # Use context as user_context
                table_name=str(data_table),
                target_column=target_column,
                category=category,
                ml_plan_architecture=ml_plan_architecture,
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
                editor_func=self._create_editor(context, category),
                ui_interface=self.ui,
                yaml_type_name="model",
                yaml_suffix=".arc-model.yaml",
            )

            context_dict = {
                "model_name": str(name),
                "table_name": str(data_table),
                "target_column": target_column,
                "category": category,
            }

            try:
                proceed, final_yaml = await workflow.run_workflow(
                    model_yaml,
                    context_dict,
                    None,  # No file path
                )
                if not proceed:
                    return ToolResult.success_result(
                        "✗ Model generation cancelled by user."
                    )
                model_yaml = final_yaml
            finally:
                workflow.cleanup()

        # Feedback loop: Update ML plan if architecture changed
        updated_ml_plan = None
        if (
            ml_plan
            and ml_plan_architecture
            and self._detect_architecture_changes(ml_plan_architecture, model_yaml)
        ):
            if self.ui:
                self.ui.show_info(
                    "🔄 Detected architecture changes from ML plan. "
                    "Updating plan to reflect actual implementation..."
                )

            # Update ML plan with actual implementation
            updated_ml_plan = await self._update_ml_plan_with_changes(
                ml_plan, model_yaml
            )

            if self.ui and updated_ml_plan != ml_plan:
                self.ui.show_success(
                    "✓ ML plan updated to reflect model architecture changes."
                )

        # Save model to DB with plan_id if using ML plan
        try:
            plan_id = ml_plan.get("plan_id") if ml_plan else None
            model = self._save_model_to_db(
                name=str(name),
                yaml_content=model_yaml,
                description=context[:200] if context else "Generated model",
                plan_id=plan_id,
            )
            model_id = model.id
        except Exception as exc:
            return ToolResult.error_result(f"Failed to save model to DB: {exc}")

        summary = (
            f"Inputs: {len(model_spec.inputs)} • Nodes: {len(model_spec.graph)} "
            f"• Outputs: {len(model_spec.outputs)}"
        )

        lines = [
            f"✓ Model '{name}' generated and saved to DB.",
            f"Model ID: {model_id}",
            summary,
        ]

        if auto_confirm:
            lines.append("\n✓ YAML:")
            lines.append(model_yaml.strip())
        else:
            lines.append("✓ Model approved and ready for use.")

        # Build metadata
        result_metadata = {
            "model_id": model_id,
            "model_name": name,
            "yaml_content": model_yaml,
            "from_ml_plan": ml_plan is not None,
        }

        # Include updated ML plan if changes were detected
        if updated_ml_plan is not None:
            result_metadata["ml_plan"] = updated_ml_plan
            result_metadata["ml_plan_updated"] = True
        elif ml_plan is not None:
            # No changes, but still include original plan
            result_metadata["ml_plan"] = ml_plan
            result_metadata["ml_plan_updated"] = False

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

    def _create_editor(
        self, user_context: str | None = None, category: str | None = None
    ):
        """Create editor function for AI-assisted editing in the workflow.

        Args:
            user_context: User context description
            category: Model category (optional)

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
                    user_context=user_context or "",
                    table_name=context["table_name"],
                    target_column=context.get("target_column"),
                    category=category,
                    existing_yaml=yaml_content,
                    editing_instructions=feedback,
                )
                return edited_yaml
            except Exception as e:
                if self.ui:
                    self.ui.show_system_error(f"❌ AI editing failed: {str(e)}")
                return None

        return edit

    def _detect_architecture_changes(
        self, ml_plan_architecture: str, final_yaml: str
    ) -> bool:
        """Detect if final YAML differs significantly from ML plan architecture.

        Args:
            ml_plan_architecture: Original architecture guidance from ML plan
            final_yaml: Final generated/edited YAML content

        Returns:
            True if significant changes detected, False otherwise
        """
        # Simple heuristic: check if ML plan mentions specific components
        # and whether those appear in final YAML
        # More sophisticated: use LLM to compare semantically

        # For now, simple check: if plan is short or YAML is long enough
        # to have diverged, consider it changed
        plan_lower = ml_plan_architecture.lower()
        yaml_lower = final_yaml.lower()

        # Extract key architectural terms from plan
        key_terms = []
        for term in [
            "linear",
            "relu",
            "dropout",
            "batchnorm",
            "attention",
            "transformer",
            "embedding",
            "cross",
            "binary_cross_entropy",
            "mse_loss",
            "cross_entropy",
        ]:
            if term in plan_lower:
                key_terms.append(term)

        # If plan mentioned specific terms but they're not in YAML, flag it
        missing_terms = [term for term in key_terms if term not in yaml_lower]

        # If >30% of key architectural terms are missing, consider it changed
        return key_terms and len(missing_terms) / len(key_terms) > 0.3

    async def _update_ml_plan_with_changes(
        self, ml_plan: dict, final_yaml: str
    ) -> dict:
        """Update ML plan based on actual implemented architecture.

        Uses LLM to analyze differences between plan and implementation,
        then updates relevant sections of the plan.

        Args:
            ml_plan: Original ML plan dictionary
            final_yaml: Final generated/edited YAML content

        Returns:
            Updated ML plan dictionary with revised architecture section
        """

        from jinja2 import Environment, FileSystemLoader

        from arc.core.ml_plan import MLPlan

        plan = MLPlan.from_dict(ml_plan)

        # Load Jinja2 template
        template_dir = (
            Path(__file__).parent.parent / "core" / "agents" / "ml_plan" / "templates"
        )
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        template = env.get_template("update_plan.j2")

        # Render prompt from template
        prompt = template.render(
            original_architecture=plan.model_architecture_and_loss,
            final_yaml=final_yaml,
        )

        try:
            response = await self._call_llm(prompt)
            updated_architecture = response.strip()

            # Update the plan
            updated_plan_dict = ml_plan.copy()
            updated_plan_dict["model_architecture_and_loss"] = updated_architecture

            return updated_plan_dict

        except Exception as e:
            # If LLM call fails, return original plan
            if self.ui:
                self.ui.show_warning(f"⚠ Could not update ML plan automatically: {e}")
            return ml_plan

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with a prompt and return response.

        Args:
            prompt: Prompt to send to LLM

        Returns:
            LLM response text
        """
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key, base_url=self.base_url)

        message = client.messages.create(
            model=self.model or "claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )

        return message.content[0].text

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
        context: str | None = None,
        model_id: str | None = None,
        train_table: str | None = None,
        auto_confirm: bool = False,
    ) -> ToolResult:
        """Generate trainer spec, register it, and launch training with confirmation.

        Args:
            name: Trainer name
            context: Training goals and constraints for LLM generation
            model_id: Model ID (with version, e.g., 'my_model-v1')
            train_table: Training data table
            auto_confirm: Skip confirmation workflows (for testing only)

        Note:
            All training configuration (epochs, batch_size, learning_rate, validation,
            etc.) must be specified in the trainer YAML generated by the LLM.
            No runtime overrides are supported.
        """
        # Validate API key
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

        # Validate required parameters
        if not name or not context or not model_id or not train_table:
            return ToolResult.error_result(
                "Parameters 'name', 'context', 'model_id', and 'train_table' "
                "are required."
            )

        # Get the registered model by ID
        try:
            model_record = self.services.models.get_model_by_id(str(model_id))
            if not model_record:
                return ToolResult.error_result(
                    f"Model '{model_id}' not found in registry. "
                    "Please check the model ID or register the model first."
                )
        except Exception as exc:
            return ToolResult.error_result(
                f"Failed to retrieve model '{model_id}': {exc}"
            )

        # Show UI feedback if UI is available
        if self.ui:
            self.ui.show_info(f"📋 Using registered model: {model_record.id}")
            self.ui.show_info(f"🤖 Generating trainer specification for '{name}'...")

        # Generate trainer spec via LLM
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
                model_id=model_record.id,
                model_spec_yaml=model_record.spec,
            )
        except Exception as exc:
            from arc.core.agents.trainer_generator import TrainerGeneratorError

            if isinstance(exc, TrainerGeneratorError):
                return ToolResult.error_result(str(exc))
            return ToolResult.error_result(
                f"Unexpected error during trainer generation: {exc}"
            )

        # Validate the generated trainer
        try:
            trainer_dict = yaml.safe_load(trainer_yaml)
            validate_trainer_dict(trainer_dict)
        except yaml.YAMLError as exc:
            return ToolResult.error_result(
                f"Generated trainer contains invalid YAML: {exc}"
            )
        except TrainerValidationError as exc:
            return ToolResult.error_result(
                f"Generated trainer failed validation: {exc}"
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult.error_result(
                f"Unexpected error during trainer validation: {exc}"
            )

        # Interactive confirmation workflow (unless auto_confirm is True)
        if not auto_confirm:
            workflow = YamlConfirmationWorkflow(
                validator_func=self._create_validator(),
                editor_func=self._create_editor(context, model_record),
                ui_interface=self.ui,
                yaml_type_name="trainer",
                yaml_suffix=".arc-trainer.yaml",
            )

            context_dict = {
                "trainer_name": str(name),
                "context": str(context),
                "model_id": model_record.id,
                "model_spec_yaml": model_record.spec,
            }

            try:
                proceed, final_yaml = await workflow.run_workflow(
                    trainer_yaml,
                    context_dict,
                    None,  # No output path - we register to DB
                )
                if not proceed:
                    return ToolResult.success_result("✗ Trainer generation cancelled.")
                trainer_yaml = final_yaml
            finally:
                workflow.cleanup()

        # Auto-register trainer to database
        try:
            trainer_record = self.runtime.create_trainer(
                name=str(name),
                model_id=str(model_id),
                schema_yaml=trainer_yaml,
                description=f"Generated trainer for model {model_id}",
            )

            if self.ui:
                self.ui.show_system_success(
                    f"✓ Trainer registered: {trainer_record.id}"
                )
        except Exception as exc:
            return ToolResult.error_result(f"Failed to register trainer: {exc}")

        # Prepare trainer summary
        summary = (
            f"Model: {trainer_spec.model_ref} • "
            f"Optimizer: {trainer_spec.optimizer.type}"
        )

        lines = [
            f"✓ Trainer '{trainer_record.id}' created and registered.",
            summary,
        ]

        # Auto-launch training after trainer is accepted
        job_id = None

        if True:  # Always train when trainer is accepted
            if self.ui:
                self.ui.show_info(f"🚀 Launching training with trainer '{name}'...")

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

                # Handle TensorBoard launch
                if not auto_confirm and self.ui:
                    if self.tensorboard_manager:
                        try:
                            await self._handle_tensorboard_launch(job_id)
                        except Exception as e:  # noqa: BLE001
                            # Show error but don't fail the whole training
                            self.ui._printer.console.print(
                                f"[yellow]⚠️  TensorBoard setup failed: {e}[/yellow]"
                            )
                            self._show_manual_tensorboard_instructions(job_id)
                    else:
                        # No TensorBoard manager available
                        self.ui._printer.console.print(
                            "[dim]ℹ️  TensorBoard auto-launch not available "
                            "(restart arc chat to enable)[/dim]"
                        )
                        self._show_manual_tensorboard_instructions(job_id)

            except MLRuntimeError as exc:
                return ToolResult.error_result(
                    f"Trainer registered but training failed: {exc}"
                )
            except Exception as exc:  # noqa: BLE001
                return ToolResult.error_result(
                    f"Trainer registered but unexpected training error: {exc}"
                )
            lines.append(
                f"To train later, use: /ml jobs submit --trainer {name} "
                f"--data {train_table}"
            )

        # Build result metadata
        result_metadata = {
            "trainer_id": trainer_record.id,
            "trainer_name": name,
            "model_id": model_record.id,
            "yaml_content": trainer_yaml,
            "training_launched": job_id is not None,
        }

        if job_id:
            result_metadata["job_id"] = job_id

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata=result_metadata,
        )

    async def _handle_tensorboard_launch(self, job_id: str):
        """Handle TensorBoard launch based on user preference.

        Args:
            job_id: Training job identifier
        """

        from arc.core.config import SettingsManager
        from arc.utils.tensorboard_workflow import prompt_tensorboard_preference

        settings = SettingsManager()
        mode = settings.get_tensorboard_mode()

        # First time - no preference set, show combined dialog
        if mode is None:
            mode, should_launch = await prompt_tensorboard_preference(self.ui)
            settings.set_tensorboard_mode(mode)
            self.ui._printer.console.print()
            self.ui._printer.console.print(
                f"[green]✓ TensorBoard preference saved: {mode}[/green]"
            )

            # Launch immediately if user chose to
            if should_launch:
                await self._launch_tensorboard(job_id)
            else:
                self._show_manual_tensorboard_instructions(job_id)

        # Subsequent times - respect saved preference
        elif mode == "always":
            await self._launch_tensorboard(job_id)
        elif mode == "ask":
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
                self.ui._printer.console.print()
                self.ui._printer.console.print(
                    "[green]✓ TensorBoard preference updated: always[/green]"
                )
                await self._launch_tensorboard(job_id)
            elif choice == "yes":
                await self._launch_tensorboard(job_id)
            else:  # "no"
                self._show_manual_tensorboard_instructions(job_id)
        else:  # "never"
            self._show_manual_tensorboard_instructions(job_id)

    async def _launch_tensorboard(self, job_id: str):
        """Launch TensorBoard and show info.

        Args:
            job_id: Training job identifier
        """
        from pathlib import Path

        from arc.core.config import SettingsManager

        logdir = Path(f"tensorboard/run_{job_id}")

        try:
            settings = SettingsManager()
            port = settings.get_tensorboard_port()

            url, pid = self.tensorboard_manager.launch(job_id, logdir, port)

            self.ui._printer.console.print()
            self.ui._printer.console.print("[green]🚀 Launching TensorBoard...[/green]")
            self.ui._printer.console.print(f"   • URL: [bold]{url}[/bold]")
            self.ui._printer.console.print(f"   • Process ID: {pid}")
            self.ui._printer.console.print(f"   • Logs: {logdir}")
            self.ui._printer.console.print()
            self.ui._printer.console.print(
                f"[dim]Stop with: /ml tensorboard stop {job_id}[/dim]"
            )
        except Exception as e:  # noqa: BLE001
            self.ui._printer.console.print(
                f"[yellow]⚠️  Failed to launch TensorBoard: {e}[/yellow]"
            )
            self._show_manual_tensorboard_instructions(job_id)

    def _show_manual_tensorboard_instructions(self, job_id: str):
        """Show manual TensorBoard instructions.

        Args:
            job_id: Training job identifier
        """
        logdir = f"tensorboard/run_{job_id}"
        self.ui._printer.console.print()
        self.ui._printer.console.print("[cyan]📊 Track training:[/cyan]")
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

    def _create_editor(self, user_context: str, model_record):
        """Create editor function for AI-assisted editing."""

        async def edit(
            yaml_content: str, feedback: str, context: dict[str, Any]
        ) -> str | None:
            agent = TrainerGeneratorAgent(
                self.services,
                self.api_key,
                self.base_url,
                self.model,
            )

            try:
                _trainer_spec, edited_yaml = await agent.generate_trainer(
                    name=context["trainer_name"],
                    user_context=user_context,
                    model_id=model_record.id,
                    model_spec_yaml=model_record.spec,
                    existing_yaml=yaml_content,
                    editing_instructions=feedback,
                )
                return edited_yaml
            except Exception as e:
                if self.ui:
                    self.ui.show_system_error(f"❌ Edit failed: {e}")
                return None

        return edit


class MLEvaluateTool(BaseTool):
    """Unified tool for generating evaluator specs and running evaluation.

    This tool combines evaluator generation with evaluation execution in a single
    workflow, similar to the MLTrainTool pattern. It provides:
    1. Evaluator spec generation via LLM
    2. Interactive confirmation workflow for evaluator spec
    3. Auto-registration to database
    4. Interactive confirmation workflow for evaluation launch
    5. Evaluation execution with metrics display

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
    ) -> None:
        self.services = services
        self.runtime = runtime
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.ui = ui_interface

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
        context: str | None = None,
        trainer_id: str | None = None,
        data_table: str | None = None,
        auto_confirm: bool = False,
    ) -> ToolResult:
        """Generate evaluator spec, register it, and run evaluation.

        Args:
            name: Evaluator name
            context: Evaluation goals and context for LLM generation
            trainer_id: Trainer ID to evaluate (references the trained model)
            data_table: Test dataset table name
            auto_confirm: Skip confirmation workflows (for testing only)

        Returns:
            ToolResult with evaluation metrics
        """
        # Validate API key
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

        # Validate required parameters
        if not name or not context or not trainer_id or not data_table:
            return ToolResult.error_result(
                "Parameters 'name', 'context', 'trainer_id', and 'data_table' "
                "are required."
            )

        # Get the registered trainer
        try:
            trainer_record = self.services.trainers.get_trainer_by_id(
                str(trainer_id)
            )
            if not trainer_record:
                return ToolResult.error_result(
                    f"Trainer '{trainer_id}' not found in registry. "
                    "Please train a model first using /ml train"
                )
        except Exception as exc:
            return ToolResult.error_result(
                f"Failed to retrieve trainer '{trainer_id}': {exc}"
            )

        # Get model spec and infer target column
        try:
            model_record = self.services.models.get_model_by_id(trainer_record.model_id)
            if not model_record:
                return ToolResult.error_result(
                    f"Model '{trainer_record.model_id}' not found in registry"
                )

            # Infer target column from model spec
            target_column = self._infer_target_column_from_model(model_record.spec)
            if not target_column:
                return ToolResult.error_result(
                    "Cannot infer target column from model spec. "
                    "Ensure model's loss spec includes a 'target' input."
                )

            if self.ui:
                self.ui.show_info(
                    f"📌 Inferred target column from model: '{target_column}'"
                )

        except Exception as exc:
            return ToolResult.error_result(
                f"Failed to retrieve model for trainer: {exc}"
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
                        f"📊 Target column '{target_column}' found in data - "
                        "will compute metrics"
                    )
                else:
                    self.ui.show_info(
                        f"📋 Target column '{target_column}' not in data - "
                        "will generate predictions only"
                    )
        except Exception as exc:
            # If schema check fails, default to assuming target exists
            if self.ui:
                self.ui.show_info(
                    f"⚠️ Could not check if target column exists: {exc}"
                )
            target_column_exists = True

        # Show UI feedback if UI is available
        if self.ui:
            self.ui.show_info(f"📋 Using registered trainer: {trainer_record.id}")
            self.ui.show_info(f"🤖 Generating evaluator specification for '{name}'...")

        # Generate evaluator spec via LLM
        from arc.core.agents.evaluator_generator import EvaluatorGeneratorAgent

        agent = EvaluatorGeneratorAgent(
            self.services,
            self.api_key,
            self.base_url,
            self.model,
        )

        try:
            evaluator_spec, evaluator_yaml = await agent.generate_evaluator(
                name=str(name),
                user_context=str(context),
                trainer_ref=str(trainer_id),
                trainer_spec_yaml=trainer_record.spec,
                dataset=str(data_table),
                target_column=str(target_column),
                target_column_exists=target_column_exists,
            )
        except Exception as exc:
            from arc.core.agents.evaluator_generator import EvaluatorGeneratorError

            if isinstance(exc, EvaluatorGeneratorError):
                return ToolResult.error_result(str(exc))
            return ToolResult.error_result(
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
            return ToolResult.error_result(
                f"Generated evaluator contains invalid YAML: {exc}"
            )
        except EvaluatorValidationError as exc:
            return ToolResult.error_result(
                f"Generated evaluator failed validation: {exc}"
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult.error_result(
                f"Unexpected error during evaluator validation: {exc}"
            )

        # Interactive confirmation workflow (unless auto_confirm is True)
        if not auto_confirm:
            workflow = YamlConfirmationWorkflow(
                validator_func=self._create_validator(),
                editor_func=self._create_editor(
                    context, trainer_id, trainer_record, target_column_exists
                ),
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
                "target_column_exists": target_column_exists,
            }

            try:
                proceed, final_yaml = await workflow.run_workflow(
                    evaluator_yaml,
                    context_dict,
                    None,  # No output path - we register to DB
                )
                if not proceed:
                    return ToolResult.success_result("✗ Evaluator cancelled.")
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

            spec_matches = (
                existing_evaluator
                and existing_evaluator.spec.strip() == evaluator_yaml.strip()
            )

            if spec_matches:
                # Reuse existing evaluator (same spec)
                evaluator_record = existing_evaluator
                if self.ui:
                    self.ui.show_system_success(
                        f"✓ Using existing evaluator: {evaluator_record.id}"
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

                if self.ui:
                    self.ui.show_system_success(
                        f"✓ Evaluator registered: {evaluator_record.id}"
                    )
        except Exception as exc:
            return ToolResult.error_result(f"Failed to register evaluator: {exc}")

        # Prepare evaluator summary
        summary = (
            f"Trainer: {evaluator_spec.trainer_ref} • "
            f"Dataset: {evaluator_spec.dataset} • "
            f"Target: {evaluator_spec.target_column}"
        )

        lines = [
            f"✓ Evaluator '{evaluator_record.id}' created and registered.",
            summary,
        ]

        # Run evaluation automatically after registration
        metrics_dict = None
        if self.ui:
            self.ui.show_info(f"🔍 Running evaluation with '{name}'...")

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

        tracking_service = EvaluationTrackingService(self.services.trainers.db_manager)

        try:
            eval_run = tracking_service.create_run(
                evaluator_id=evaluator_record.id,
                trainer_id=trainer_record.id,
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

            # Run evaluation
            result = await asyncio.to_thread(
                evaluator.evaluate, self.services.ml_data, output_table
            )

            metrics_dict = result.metrics

            # Update run with results
            tracking_service.update_run_result(
                eval_run.run_id,
                metrics_result=metrics_dict,
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
                f"{k}={v:.4f}" for k, v in list(metrics_dict.items())[:3]
            )
            job.complete(f"Evaluation completed: {metrics_summary}")
            self.services.jobs.update_job(job)

            # Display metrics
            lines.append("")
            lines.append("✓ Evaluation completed successfully.")
            lines.append(f"  Job ID: {job.job_id}")
            lines.append(f"  Run ID: {eval_run.run_id}")
            lines.append("")
            lines.append("Metrics:")
            for metric_name, metric_value in metrics_dict.items():
                lines.append(f"  {metric_name}: {metric_value:.4f}")
            lines.append("")
            lines.append(f"Samples evaluated: {result.num_samples}")
            lines.append(f"Evaluation time: {result.evaluation_time:.2f}s")
            lines.append("")
            lines.append(f"Predictions saved to table: {output_table}")

        except Exception as exc:
            from arc.ml.evaluator import EvaluationError

            # Update run with error
            try:
                tracking_service.update_run_error(eval_run.run_id, str(exc))
                tracking_service.update_run_status(
                    eval_run.run_id,
                    EvaluationStatus.FAILED,
                )
            except Exception:  # noqa: S110
                pass  # Ignore tracking errors

            # Update job with error
            try:
                job.fail(f"Evaluation failed: {str(exc)[:200]}")
                self.services.jobs.update_job(job)
            except Exception:  # noqa: S110
                pass  # Ignore job update errors

            if isinstance(exc, EvaluationError):
                return ToolResult.error_result(
                    f"Evaluator registered but evaluation failed: {exc}"
                )
            return ToolResult.error_result(
                f"Evaluator registered but unexpected error: {exc}"
            )

        # Build result metadata
        result_metadata = {
            "evaluator_id": evaluator_record.id,
            "evaluator_name": name,
            "trainer_id": trainer_record.id,
            "yaml_content": evaluator_yaml,
            "evaluation_ran": True,
        }

        if metrics_dict:
            result_metadata["metrics"] = metrics_dict

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
        user_context: str,
        trainer_id: str,
        trainer_record,
        target_column_exists: bool,
    ):
        """Create editor function for AI-assisted editing."""

        async def edit(
            yaml_content: str, feedback: str, context: dict[str, Any]
        ) -> str | None:
            from arc.core.agents.evaluator_generator import EvaluatorGeneratorAgent

            agent = EvaluatorGeneratorAgent(
                self.services,
                self.api_key,
                self.base_url,
                self.model,
            )

            try:
                _evaluator_spec, edited_yaml = await agent.generate_evaluator(
                    name=context["evaluator_name"],
                    user_context=user_context,
                    trainer_ref=trainer_id,
                    trainer_spec_yaml=trainer_record.spec,
                    dataset=context["dataset"],
                    target_column=context["target_column"],
                    target_column_exists=target_column_exists,
                    existing_yaml=yaml_content,
                    editing_instructions=feedback,
                )
                return edited_yaml
            except Exception as e:
                if self.ui:
                    self.ui.show_system_error(f"❌ Edit failed: {e}")
                return None

        return edit


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
            trainer_record = self.services.trainers.get_trainer_by_id(
                str(trainer_id)
            )
            if not trainer_record:
                return ToolResult.error_result(
                    f"Trainer '{trainer_id}' not found in registry. "
                    "Please register the trainer first using /ml generate-trainer"
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
                        f"📊 Target column '{target_column}' found in data - "
                        "will include metrics"
                    )
                else:
                    self.ui.show_info(
                        f"📋 Target column '{target_column}' not in data - "
                        "prediction mode"
                    )
        except Exception as exc:
            # If schema check fails, default to assuming target exists
            if self.ui:
                self.ui.show_info(
                    f"⚠️ Could not check if target column exists: {exc}"
                )
            target_column_exists = True

        # Show UI feedback if UI is available
        if self.ui:
            self.ui.show_info(f"📋 Using registered trainer: {trainer_record.id}")
            self.ui.show_info(f"🤖 Generating evaluator specification for '{name}'...")

        from arc.core.agents.evaluator_generator import EvaluatorGeneratorAgent

        agent = EvaluatorGeneratorAgent(
            self.services,
            self.api_key,
            self.base_url,
            self.model,
        )

        try:
            evaluator_spec, evaluator_yaml = await agent.generate_evaluator(
                name=str(name),
                user_context=str(context),
                trainer_ref=str(trainer_id),
                trainer_spec_yaml=trainer_record.spec,
                dataset=str(data_table),
                target_column=str(target_column),
                target_column_exists=target_column_exists,
            )
        except Exception as exc:
            # Import here to avoid circular imports
            from arc.core.agents.evaluator_generator import EvaluatorGeneratorError

            if isinstance(exc, EvaluatorGeneratorError):
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
        except Exception as exc:  # noqa: BLE001
            return ToolResult.error_result(
                f"Unexpected error during evaluator validation: {exc}"
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
                )
                if not proceed:
                    return ToolResult.success_result(
                        "✗ Evaluator generation cancelled."
                    )
                evaluator_yaml = final_yaml
            finally:
                workflow.cleanup()

        # Auto-register evaluator to database
        try:
            from datetime import UTC, datetime

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
                    f"✓ Evaluator registered: {evaluator_record.id}"
                )
        except Exception as exc:
            return ToolResult.error_result(f"Failed to register evaluator: {exc}")

        summary = (
            f"Trainer: {evaluator_spec.trainer_ref} • "
            f"Dataset: {evaluator_spec.dataset} • "
            f"Target: {evaluator_spec.target_column}"
        )

        lines = [
            f"✓ Evaluator '{evaluator_record.id}' created and registered.",
            summary,
        ]

        if auto_confirm:
            lines.append("\n  YAML:")
            lines.append(evaluator_yaml.strip())
        else:
            lines.append("✓ Evaluator approved and ready for evaluation.")

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
        """Create editor function for AI-assisted editing in the workflow.

        Returns:
            Async function that edits YAML based on user feedback
        """

        async def edit(
            yaml_content: str, feedback: str, context: dict[str, Any]
        ) -> str | None:
            from arc.core.agents.evaluator_generator import EvaluatorGeneratorAgent

            agent = EvaluatorGeneratorAgent(
                self.services,
                self.api_key,
                self.base_url,
                self.model,
            )

            try:
                _evaluator_spec, edited_yaml = await agent.generate_evaluator(
                    name=context["evaluator_name"],
                    user_context=context["context"],
                    trainer_ref=context["trainer_ref"],
                    trainer_spec_yaml=context["trainer_spec_yaml"],
                    dataset=context["dataset"],
                    target_column=context["target_column"],
                    existing_yaml=yaml_content,
                    editing_instructions=feedback,
                )
                return edited_yaml
            except Exception as e:
                if self.ui:
                    self.ui.show_system_error(f"❌ Edit failed: {e}")
                return None

        return edit


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
        source_tables: str | None = None,
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

        if not user_context or not source_tables:
            return ToolResult.error_result(
                "Parameters 'user_context' and 'source_tables' "
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

            # Note: conversation_history is already filtered by the agent using
            # timestamps. For revisions, only messages after the last plan are included.
            # For initial plans, all conversation history is included.

            # Internal loop for handling feedback (option C)
            current_feedback = feedback

            # Get version from database to avoid conflicts
            latest_plan = self.services.ml_plans.get_latest_plan_for_tables(
                str(source_tables)
            )
            version = latest_plan.version + 1 if latest_plan else 1

            while True:
                try:
                    # Generate the plan (pass source_tables as comma-separated string)
                    analysis = await agent.analyze_problem(
                        user_context=str(user_context),
                        source_tables=str(source_tables),
                        conversation_history=conversation_history,
                        feedback=current_feedback,
                        stream=False,
                    )

                    # Determine stage
                    if previous_plan:
                        stage = previous_plan.get("stage", "initial")
                        feedback_lower = str(current_feedback).lower()
                        if current_feedback and "training" in feedback_lower:
                            stage = "post_training"
                        elif current_feedback and "evaluation" in feedback_lower:
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

                except Exception as e:
                    # Handle errors during plan generation
                    error_msg = f"Failed to generate ML plan: {str(e)}"
                    if self.ui:
                        self.ui._printer.console.print(f"[red]❌ {error_msg}[/red]")
                    return ToolResult.error_result(error_msg)

                # If auto-accept is enabled, skip workflow
                if self.agent and self.agent.ml_plan_auto_accept:
                    output_message = "Plan automatically accepted (auto-accept enabled)"
                    break

                # Display plan and run confirmation workflow
                if self.ui:
                    from arc.utils.ml_plan_workflow import MLPlanConfirmationWorkflow

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

            # Save plan to database after acceptance
            try:
                from datetime import UTC, datetime

                from arc.database.models.ml_plan import MLPlan as MLPlanModel
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
                    user_context=str(user_context),
                    source_tables=str(source_tables),
                    plan_yaml=plan_yaml,  # Store as YAML string
                    status="approved",  # Plan was accepted by user
                    created_at=now,
                    updated_at=now,
                )

                # Save to database
                self.services.ml_plans.create_plan(db_plan)

                # Add plan_id to metadata for linking
                plan_dict["plan_id"] = plan_id

            except Exception as e:
                # Log error but don't fail - plan still in memory
                if self.ui:
                    self.ui.show_warning(
                        f"⚠ Plan saved to session but not database: {e}"
                    )

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
