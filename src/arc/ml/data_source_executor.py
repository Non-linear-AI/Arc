"""Data source pipeline executor for SQL transformation workflows."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from arc.utils.validation import (
    ValidationError,
    quote_sql_identifier,
    validate_sql_syntax,
    validate_table_name,
)

if TYPE_CHECKING:
    from arc.database.manager import DatabaseManager
    from arc.graph.features.data_source import DataSourceSpec


@dataclass
class DataSourceExecutionResult:
    """Result of data source pipeline execution."""

    created_tables: list[str]
    execution_time: float
    step_count: int
    intermediate_views_cleaned: int
    steps_executed: list[tuple[str, str]]  # [(step_name, step_type), ...]
    progress_log: list[tuple[str, str]]  # [(message, level), ...]


class DataSourceExecutionError(Exception):
    """Exception raised when data source pipeline execution fails."""


async def execute_data_source_pipeline(
    spec: DataSourceSpec,
    target_db: str,
    db_manager: DatabaseManager,
    progress_callback: Callable[[str, str], None] | None = None,
) -> DataSourceExecutionResult:
    """Execute a DataSourceSpec pipeline with proper cleanup.

    Args:
        spec: DataSourceSpec containing steps and outputs
        target_db: Target database ("system" or "user")
        db_manager: DatabaseManager instance for execution
        progress_callback: Optional callback for progress updates (message, level)

    Returns:
        DataSourceExecutionResult with execution details

    Raises:
        DataSourceExecutionError: If pipeline execution fails
    """
    start_time = time.time()

    if target_db not in ["system", "user"]:
        raise DataSourceExecutionError(
            f"Invalid target database: {target_db}. Must be 'system' or 'user'."
        )

    # Get execution order (topologically sorted)
    try:
        ordered_steps = spec.get_execution_order()
    except ValueError as e:
        raise DataSourceExecutionError(
            f"Failed to determine execution order: {e}"
        ) from e

    step_count = len(ordered_steps)
    intermediate_views = []  # Track views that need cleanup
    steps_executed = []  # Track executed steps
    progress_log = []  # Accumulate progress messages

    # Validate all SQL syntax before execution
    validation_errors = []
    for step in ordered_steps:
        # Validate step name
        try:
            validate_table_name(step.name)
        except ValidationError as e:
            validation_errors.append(f"Step '{step.name}': {e}")
            continue

        # Validate SQL syntax
        sql = spec.substitute_vars(step.sql)
        sql_errors = validate_sql_syntax(sql)
        if sql_errors:
            validation_errors.append(f"Step '{step.name}': " + "; ".join(sql_errors))

    # Fail early if validation errors found
    if validation_errors:
        error_msg = "Pipeline validation failed:\n  " + "\n  ".join(validation_errors)
        raise DataSourceExecutionError(error_msg)

    # Report progress: starting
    if progress_callback:
        progress_callback("🤖 Executing pipeline...", "info")
        progress_log.append(("🤖 Executing pipeline...", "info"))

    try:
        for i, step in enumerate(ordered_steps, 1):
            # Get step type from the step itself (defaults to 'table' if not set)
            step_type = getattr(step, "type", "table")
            steps_executed.append((step.name, step_type))

            # Report progress: step
            step_msg = f"Step {i}/{step_count}: {step.name} ({step_type})"
            if progress_callback:
                progress_callback(step_msg, "step")
                progress_log.append((step_msg, "step"))

            # Substitute variables in SQL
            sql = spec.substitute_vars(step.sql)

            # Strip trailing semicolons (they cause syntax errors in
            # CREATE TABLE/VIEW AS)
            sql = sql.rstrip().rstrip(";").rstrip()

            # Quote step name for safe SQL execution (not used for execute type)
            quoted_name = quote_sql_identifier(step.name)

            try:
                if step_type == "execute":
                    # Execute directly without wrapping (DDL/DML statements)
                    # These don't create artifacts - just run the SQL as-is
                    if target_db == "system":
                        db_manager.system_execute(sql)
                    else:
                        db_manager.user_execute(sql)
                elif step_type == "view":
                    # Create temporary view for intermediate steps
                    # Use CREATE OR REPLACE to handle re-runs
                    create_sql = f"CREATE OR REPLACE VIEW {quoted_name} AS ({sql})"
                    intermediate_views.append(step.name)  # Track for cleanup
                    if target_db == "system":
                        db_manager.system_execute(create_sql)
                    else:
                        db_manager.user_execute(create_sql)
                elif step_type == "table":
                    # Create persistent table for output steps
                    # Use CREATE OR REPLACE to handle re-runs
                    create_sql = f"CREATE OR REPLACE TABLE {quoted_name} AS ({sql})"
                    if target_db == "system":
                        db_manager.system_execute(create_sql)
                    else:
                        db_manager.user_execute(create_sql)
                else:
                    raise DataSourceExecutionError(
                        f"Unknown step type '{step_type}' for step '{step.name}'"
                    )

            except Exception as step_error:
                raise DataSourceExecutionError(
                    f"Failed to execute step '{step.name}': {str(step_error)}"
                ) from step_error

    finally:
        # Clean up intermediate views
        cleaned_count = 0
        if intermediate_views:
            for view_name in intermediate_views:
                try:
                    # Quote view name for safe cleanup
                    quoted_view = quote_sql_identifier(view_name)
                    cleanup_sql = f"DROP VIEW IF EXISTS {quoted_view}"
                    if target_db == "system":
                        db_manager.system_execute(cleanup_sql)
                    else:
                        db_manager.user_execute(cleanup_sql)
                    cleaned_count += 1
                except Exception:
                    # Don't fail the entire pipeline if cleanup fails
                    warning_msg = f"Could not clean up view '{view_name}'"
                    if progress_callback:
                        progress_callback(warning_msg, "warning")
                        progress_log.append((warning_msg, "warning"))

    # Calculate execution time
    execution_time = time.time() - start_time

    # Report progress: completion
    success_msg = "Pipeline completed successfully"
    if progress_callback:
        progress_callback(success_msg, "success")
        progress_log.append((success_msg, "success"))

        output_list = ", ".join(spec.outputs)
        output_msg = f"Output tables: {output_list}"
        progress_callback(output_msg, "info")
        progress_log.append((output_msg, "info"))

        # Add blank line after output tables
        progress_callback("", "info")
        progress_log.append(("", "info"))

    return DataSourceExecutionResult(
        created_tables=spec.outputs.copy(),
        execution_time=execution_time,
        step_count=step_count,
        intermediate_views_cleaned=cleaned_count,
        steps_executed=steps_executed,
        progress_log=progress_log,
    )
