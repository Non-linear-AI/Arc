"""Machine learning tool implementations."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Sequence

from ..ml.runtime import MLRuntime, MLRuntimeError
from .base import BaseTool, ToolResult


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
                "Parameters 'model_name' and 'train_table' are required to train a model."
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
            return ToolResult.error_result(f"Unexpected error launching training: {exc}")

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
                "Parameters 'model_name', 'table_name', and 'output_table' are required "
                "to run prediction."
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
