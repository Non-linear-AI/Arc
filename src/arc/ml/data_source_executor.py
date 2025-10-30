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
    steps_executed: list[tuple[str, str]]  # [(step_name, step_type), ...]
    progress_log: list[tuple[str, str]]  # [(message, level), ...]
    sql: str | None = None  # Full SQL (all steps combined) for context
    outputs: list[dict] | None = None  # Output table schemas and row counts


class DataSourceExecutionError(Exception):
    """Exception raised when data source pipeline execution fails."""


async def dry_run_data_source_pipeline(
    spec: DataSourceSpec,
    target_db: str,
    db_manager: DatabaseManager,
) -> tuple[bool, str | None]:
    """Dry-run a DataSourceSpec pipeline using DuckDB transactions.

    This executes all steps in a transaction and then rolls back,
    catching runtime errors without modifying the database.

    Args:
        spec: DataSourceSpec containing steps and outputs
        target_db: Target database ("system" or "user")
        db_manager: DatabaseManager instance for execution

    Returns:
        Tuple of (success: bool, error_message: str | None)
    """
    if target_db not in ["system", "user"]:
        return (
            False,
            f"Invalid target database: {target_db}. Must be 'system' or 'user'.",
        )

    # Get execution order (topologically sorted)
    try:
        ordered_steps = spec.get_execution_order()
    except ValueError as e:
        return False, f"Failed to determine execution order: {e}"

    # Choose execute method based on target database
    execute_fn = (
        db_manager.system_execute if target_db == "system" else db_manager.user_execute
    )

    try:
        # Start transaction
        execute_fn("BEGIN TRANSACTION")

        # Execute all steps in the transaction
        for step in ordered_steps:
            sql = spec.substitute_vars(step.sql)
            sql = sql.rstrip().rstrip(";").rstrip()

            quoted_name = quote_sql_identifier(step.name)

            if step.type == "execute":
                # Execute directly without wrapping
                execute_fn(sql)
            elif step.type == "view":
                # Drop both table and view (in case object exists with wrong type)
                execute_fn(f"DROP TABLE IF EXISTS {quoted_name}")
                execute_fn(f"DROP VIEW IF EXISTS {quoted_name}")
                # Create view
                create_sql = f"CREATE VIEW {quoted_name} AS ({sql})"
                execute_fn(create_sql)
            elif step.type == "table":
                # Drop both table and view (in case object exists with wrong type)
                execute_fn(f"DROP VIEW IF EXISTS {quoted_name}")
                execute_fn(f"DROP TABLE IF EXISTS {quoted_name}")
                # Create persistent table
                create_sql = f"CREATE TABLE {quoted_name} AS ({sql})"
                execute_fn(create_sql)
            else:
                # Unknown type - rollback and fail
                execute_fn("ROLLBACK")
                return False, f"Unknown step type '{step.type}' for step '{step.name}'"

        # If we got here, all steps executed successfully
        # Rollback to undo all changes
        execute_fn("ROLLBACK")
        return True, None

    except Exception as e:
        # Error during execution - try to rollback
        import contextlib

        with contextlib.suppress(Exception):
            # Rollback failed, but we're already handling an error
            execute_fn("ROLLBACK")

        # Return the error message
        error_msg = str(e)
        # Try to extract which step failed (if error contains step name)
        for step in ordered_steps:
            if step.name in error_msg:
                return False, f"Step '{step.name}' failed during dry-run: {error_msg}"

        return False, f"Dry-run failed: {error_msg}"


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

    # Choose execute method based on target database
    execute_fn = (
        db_manager.system_execute if target_db == "system" else db_manager.user_execute
    )

    # Build complete SQL (all steps combined) for context
    sql_parts = ["BEGIN TRANSACTION;\n\n"]

    try:
        # Start transaction for atomic execution
        execute_fn("BEGIN TRANSACTION")

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
                    sql_parts.append(f"-- Step {i}/{step_count}: {step.name} (execute)\n")
                    sql_parts.append(f"{sql};\n\n")

                    execute_fn(sql)
                elif step_type == "view":
                    # Create persistent view for intermediate steps
                    # Drop both table and view first (in case object exists with wrong type)
                    sql_parts.append(f"-- Step {i}/{step_count}: {step.name} (view)\n")
                    sql_parts.append(f"DROP TABLE IF EXISTS {quoted_name};\n")
                    sql_parts.append(f"DROP VIEW IF EXISTS {quoted_name};\n")
                    execute_fn(f"DROP TABLE IF EXISTS {quoted_name}")
                    execute_fn(f"DROP VIEW IF EXISTS {quoted_name}")

                    # Create view
                    create_sql = f"CREATE VIEW {quoted_name} AS ({sql})"
                    sql_parts.append(f"{create_sql};\n\n")
                    execute_fn(create_sql)
                elif step_type == "table":
                    # Create persistent table for output steps
                    # Drop both table and view first (in case object exists with wrong type)
                    sql_parts.append(f"-- Step {i}/{step_count}: {step.name} (table)\n")
                    sql_parts.append(f"DROP VIEW IF EXISTS {quoted_name};\n")
                    sql_parts.append(f"DROP TABLE IF EXISTS {quoted_name};\n")
                    execute_fn(f"DROP VIEW IF EXISTS {quoted_name}")
                    execute_fn(f"DROP TABLE IF EXISTS {quoted_name}")

                    # Create table
                    create_sql = f"CREATE TABLE {quoted_name} AS ({sql})"
                    sql_parts.append(f"{create_sql};\n\n")
                    execute_fn(create_sql)
                else:
                    raise DataSourceExecutionError(
                        f"Unknown step type '{step_type}' for step '{step.name}'"
                    )

            except Exception as step_error:
                raise DataSourceExecutionError(
                    f"Failed to execute step '{step.name}': {str(step_error)}"
                ) from step_error

        # All steps succeeded - commit the transaction
        execute_fn("COMMIT")
        sql_parts.append("COMMIT;")

    except Exception as e:
        # Error during execution - rollback transaction
        import contextlib

        with contextlib.suppress(Exception):
            execute_fn("ROLLBACK")

        # Re-raise the error (will be a DataSourceExecutionError from step execution)
        raise

    # Calculate execution time
    execution_time = time.time() - start_time

    # Complete the SQL string
    complete_sql = "".join(sql_parts)

    # Collect output schemas and row counts
    outputs = []
    for table_name in spec.outputs:
        quoted_table = quote_sql_identifier(table_name)

        try:
            # Get schema using DESCRIBE
            if target_db == "system":
                schema_result = db_manager.system_query(f"DESCRIBE {quoted_table}")
            else:
                schema_result = db_manager.user_query(f"DESCRIBE {quoted_table}")

            columns = [
                {"name": row["column_name"], "type": row["column_type"]}
                for row in schema_result.rows
            ]

            # Get row count
            if target_db == "system":
                count_result = db_manager.system_query(
                    f"SELECT COUNT(*) as count FROM {quoted_table}"
                )
            else:
                count_result = db_manager.user_query(
                    f"SELECT COUNT(*) as count FROM {quoted_table}"
                )

            row_count = count_result.rows[0]["count"]

            outputs.append({
                "name": table_name,
                "type": "table",
                "row_count": row_count,
                "columns": columns,
            })

        except Exception as e:
            # If we can't get schema/count, still include basic info
            outputs.append({
                "name": table_name,
                "type": "table",
                "error": f"Failed to collect metadata: {str(e)}",
            })

    # Report progress: completion
    success_msg = "Pipeline completed successfully"
    if progress_callback:
        progress_callback(success_msg, "success")
        progress_log.append((success_msg, "success"))

        # Only show output tables if there are any
        if spec.outputs:
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
        steps_executed=steps_executed,
        progress_log=progress_log,
        sql=complete_sql,
        outputs=outputs,
    )
