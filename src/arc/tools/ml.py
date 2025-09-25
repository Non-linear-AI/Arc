"""Machine learning tool implementations."""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

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
        ui_interface=None,
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
        exclude_columns: list[str] | None = None,
        output_path: str | None = None,
        auto_confirm: bool = False,
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

        if not name or not context or not data_table:
            return ToolResult.error_result(
                "Parameters 'name', 'context', and 'data_table' are required "
                "to generate a model specification."
            )


        agent = ModelGeneratorAgent(
            self.services,
            self.api_key,
            self.base_url,
            self.model,
        )

        try:
            model_spec, model_yaml = await agent.generate_model(
                name=str(name),
                user_context=str(context),
                table_name=str(data_table),
                exclude_columns=exclude_columns,
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
            proceed, final_yaml = await self._interactive_confirmation_workflow(
                model_spec, model_yaml, str(name)
            )
            if not proceed:
                return ToolResult.success_result(
                    "❌ Model generation cancelled by user."
                )
            model_yaml = final_yaml

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
            f"✅ Model specification generated for '{name}'.",
            summary,
        ]

        if output_path:
            lines.append(f"💾 Saved to: {output_path}")

        if auto_confirm:
            lines.append("\n📝 YAML:")
            lines.append(model_yaml.strip())
        else:
            lines.append("✅ Model approved and ready for use.")

        return ToolResult.success_result("\n".join(lines))

    async def _interactive_confirmation_workflow(
        self, model_spec, model_yaml: str, name: str
    ) -> tuple[bool, str]:
        """Interactive confirmation workflow with editing support.

        Returns:
            Tuple of (should_proceed, final_yaml)
        """
        while True:
            # Display preview
            await self._display_model_preview(model_spec, model_yaml)

            # Get user choice using UI choice selection
            options = [
                ("save", "Yes - Save this model"),
                ("edit", "Edit - Modify the YAML before saving"),
                ("cancel", "No - Cancel generation"),
            ]

            if self.ui:
                # Use UI choice selection (like permission requests)
                choice = await self.ui._printer.get_choice_async(
                    options, default="save"
                )
            else:
                # Fallback for non-UI usage
                print(f"\n🤖 Confirm model specification for '{name}'")
                for i, (_, label) in enumerate(options, 1):
                    print(f"  {i}. {label}")
                choice_input = input("Enter choice (1-3): ").strip()
                choice_map = {"1": "save", "2": "edit", "3": "cancel"}
                choice = choice_map.get(choice_input, "cancel")

            if choice == "save":
                return True, model_yaml
            elif choice == "edit":
                # User wants to edit
                edited_yaml = await self._edit_yaml_interactive(model_yaml)
                if edited_yaml is None:
                    continue  # Edit cancelled, show confirmation again

                # Validate edited YAML
                try:
                    edited_dict = yaml.safe_load(edited_yaml)
                    validate_model_dict(edited_dict)
                    model_yaml = edited_yaml  # Use edited version
                    # Update model_spec for preview
                    from ..graph.model import ModelSpec

                    model_spec = ModelSpec.from_dict(edited_dict)
                    continue  # Show updated preview and confirmation
                except (yaml.YAMLError, ModelValidationError) as e:
                    error_msg = f"❌ Validation error in edited model: {e}"
                    instruction_msg = "Please edit again or cancel."
                    if self.ui:
                        self.ui.show_system_error(error_msg)
                        self.ui.show_info(instruction_msg)
                    else:
                        print(error_msg)
                        print(instruction_msg)
                    continue
            elif choice == "cancel" or choice == "__esc__":
                # User cancelled
                return False, model_yaml
            else:
                # This shouldn't happen with UI choice selection, but handle gracefully
                if self.ui:
                    self.ui.show_system_error("❌ Invalid choice. Please try again.")
                else:
                    print("❌ Invalid choice. Please try again.")
                continue

    async def _display_model_preview(self, model_spec, model_yaml: str) -> None:
        """Display formatted model preview."""

        def output(text):
            if self.ui:
                self.ui.show_info(text)
            else:
                print(text)

        output("\n" + "=" * 50)
        output("🤖 Arc-Graph Model Preview")
        output("=" * 50)
        input_count = len(model_spec.inputs)
        node_count = len(model_spec.graph)
        output_count = len(model_spec.outputs)
        output(
            f"📊 Inputs: {input_count} • 🔗 Nodes: {node_count} • "
            f"📤 Outputs: {output_count}"
        )
        output("\n📝 YAML Specification:")

        # Use Rich syntax highlighting for YAML display
        await self._display_yaml_with_rich_syntax(model_yaml)

    async def _display_yaml_with_rich_syntax(self, yaml_content: str) -> None:
        """Display YAML using Rich syntax highlighting with line numbers."""
        try:
            from rich.console import Console
            from rich.syntax import Syntax

            # Create Rich console
            console = Console()

            # Create syntax-highlighted YAML with line numbers
            syntax = Syntax(
                yaml_content,
                "yaml",
                theme="github-dark",
                line_numbers=True,
                word_wrap=False,
            )

            # Display using Rich console
            console.print(syntax)

        except ImportError:
            # Fallback to simple display if Rich is not available
            if self.ui:
                self.ui.show_system_error(
                    "Rich library not available, using simple display"
                )
            else:
                print("Rich library not available, using simple display")
            self._display_yaml_simple(yaml_content)
        except Exception as e:
            # Fallback to simple display on any error
            if self.ui:
                self.ui.show_system_error(f"Error with Rich display: {e}")
            else:
                print(f"Error with Rich display: {e}")
            self._display_yaml_simple(yaml_content)

    def _display_yaml_simple(self, yaml_content: str) -> None:
        """Simple fallback YAML display without Rich."""
        lines = yaml_content.strip().split("\n")
        for i, line in enumerate(lines, 1):
            line_with_number = f"{i:4d}: {line}"
            if self.ui:
                self.ui.show_info(line_with_number)
            else:
                print(line_with_number)

    async def _edit_yaml_interactive(self, yaml_content: str) -> str | None:
        """Interactive YAML editing with automatic editor detection."""
        # Try to detect and use system editor automatically

        editor = os.environ.get("EDITOR")
        if not editor:
            # Try common editors in order of preference
            import shutil

            common_editors = ["code", "nano", "vim", "vi", "emacs"]
            for ed in common_editors:
                if shutil.which(ed):
                    editor = ed
                    break

        if not editor:
            # No editor found, provide inline editing option
            if self.ui:
                self.ui.show_system_error(
                    "❌ No text editor found. Please set $EDITOR or install nano/vim."
                )
                self.ui.show_info("💡 You can:")
                self.ui.show_info(
                    "   1. Set EDITOR environment variable: export EDITOR=nano"
                )
                self.ui.show_info("   2. Install a text editor: brew install nano")
                self.ui.show_info("   3. Cancel this edit and try again")
            else:
                print(
                    "❌ No text editor found. Please set $EDITOR or install nano/vim."
                )
            return None

        # Use the detected/configured editor
        if self.ui:
            self.ui.show_info(f"🔧 Opening YAML in {editor}...")

        return await self._edit_yaml_system_editor(yaml_content, editor)

    async def _edit_yaml_system_editor(
        self, yaml_content: str, editor: str = None
    ) -> str | None:
        """Launch system editor for YAML editing."""

        try:
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".arc-model.yaml", prefix="arc_model_", delete=False
            ) as f:
                # Add helpful header
                header = (
                    "# Arc-Graph Model Specification\n"
                    "# Edit and save to confirm changes\n\n"
                )
                f.write(header + yaml_content)
                temp_path = f.name

            # Use provided editor or fall back to environment
            if not editor:
                editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "nano"))

            # Create subprocess without redirecting stdout/stderr
            # so editor can interact with terminal
            process = await asyncio.create_subprocess_exec(
                editor,
                temp_path,
            )

            await process.wait()

            if process.returncode == 0:
                with open(temp_path) as f:
                    content = f.read()
                    # Strip header if present
                    if content.startswith(header):
                        content = content[len(header) :]
                    return content.strip()
            else:
                error_msg = "❌ Editor failed or was cancelled"
                if self.ui:
                    self.ui.show_system_error(error_msg)
                else:
                    print(error_msg)
                return None

        except Exception as e:
            error_msg = f"❌ Failed to launch editor: {e}"
            if self.ui:
                self.ui.show_system_error(error_msg)
            else:
                print(error_msg)
            return None
        finally:
            # Cleanup temp file
            if "temp_path" in locals():
                with contextlib.suppress(OSError):
                    os.unlink(temp_path)


class MLTrainerGeneratorTool(BaseTool):
    """Tool for generating Arc-Graph trainer specifications via LLM."""

    def __init__(
        self,
        services,
        api_key: str | None,
        base_url: str | None,
        model: str | None,
    ) -> None:
        self.services = services
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

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


class MLPredictorGeneratorTool(BaseTool):
    """Tool for generating Arc-Graph predictor specifications via LLM."""

    def __init__(
        self,
        services,
        api_key: str | None,
        base_url: str | None,
        model: str | None,
    ) -> None:
        self.services = services
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

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
