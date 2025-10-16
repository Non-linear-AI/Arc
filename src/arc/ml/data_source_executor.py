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


def _quote_ddl_identifiers(sql: str) -> str:
    """Quote table/view names in DDL statements if they contain special characters.

    Args:
        sql: DDL SQL statement (DROP, ALTER, TRUNCATE, etc.)

    Returns:
        SQL with properly quoted identifiers

    Examples:
        DROP TABLE my-table => DROP TABLE "my-table"
        DROP TABLE IF EXISTS my-table => DROP TABLE IF EXISTS "my-table"
    """
    import re

    # Pattern to match table/view names in DDL statements
    # Matches: DROP TABLE [IF EXISTS] table_name
    #          DROP VIEW [IF EXISTS] view_name
    #          ALTER TABLE table_name
    #          TRUNCATE TABLE table_name

    # First, check if identifiers are already quoted
    if '"' in sql:
        # Already has quotes, return as-is
        return sql

    # For DROP/TRUNCATE TABLE/VIEW [IF EXISTS] <name>
    # Match the table/view name after TABLE/VIEW [IF EXISTS]
    patterns = [
        # DROP TABLE [IF EXISTS] name
        (r"(DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?)([\w\-]+)", r'\1"\2"'),
        # DROP VIEW [IF EXISTS] name
        (r"(DROP\s+VIEW\s+(?:IF\s+EXISTS\s+)?)([\w\-]+)", r'\1"\2"'),
        # ALTER TABLE name
        (r"(ALTER\s+TABLE\s+)([\w\-]+)", r'\1"\2"'),
        # TRUNCATE TABLE name
        (r"(TRUNCATE\s+TABLE\s+)([\w\-]+)", r'\1"\2"'),
    ]

    result = sql
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result


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
            # Substitute variables in SQL first
            sql = spec.substitute_vars(step.sql)

            # Strip trailing semicolons
            sql = sql.rstrip().rstrip(";").rstrip()

            # Check if this is a DDL statement (DROP, ALTER, TRUNCATE, etc.)
            sql_upper = sql.strip().upper()
            is_ddl = any(
                sql_upper.startswith(stmt)
                for stmt in ["DROP ", "ALTER ", "TRUNCATE ", "GRANT ", "REVOKE "]
            )

            # DDL statements are executed directly, not wrapped in CREATE TABLE/VIEW
            if is_ddl:
                step_type = "ddl"
                steps_executed.append((step.name, step_type))

                # Report progress: step
                step_msg = f"Step {i}/{step_count}: {step.name} ({step_type})"
                if progress_callback:
                    progress_callback(step_msg, "step")
                    progress_log.append((step_msg, "step"))

                try:
                    # For DDL statements, we need to quote table names with special chars
                    # Extract table name from DROP/ALTER/TRUNCATE statements and quote if needed
                    ddl_sql = _quote_ddl_identifiers(sql)

                    # Execute DDL directly without wrapping
                    if target_db == "system":
                        db_manager.system_execute(ddl_sql)
                    else:
                        db_manager.user_execute(ddl_sql)

                except Exception as step_error:
                    raise DataSourceExecutionError(
                        f"Failed to execute step '{step.name}': {str(step_error)}"
                    ) from step_error

            else:
                # For DML/DQL statements (SELECT, INSERT, etc.), wrap in CREATE TABLE/VIEW
                step_type = "table" if step.name in spec.outputs else "view"
                steps_executed.append((step.name, step_type))

                # Report progress: step
                step_msg = f"Step {i}/{step_count}: {step.name} ({step_type})"
                if progress_callback:
                    progress_callback(step_msg, "step")
                    progress_log.append((step_msg, "step"))

                # Quote step name for safe SQL execution
                quoted_name = quote_sql_identifier(step.name)

                try:
                    if step.name in spec.outputs:
                        # Create persistent table for output steps
                        create_sql = f"CREATE TABLE {quoted_name} AS ({sql})"
                    else:
                        # Create regular view for intermediate steps (allows debugging)
                        create_sql = f"CREATE VIEW {quoted_name} AS ({sql})"
                        intermediate_views.append(step.name)  # Track for cleanup

                    # Execute using database manager directly to ensure same session
                    if target_db == "system":
                        db_manager.system_execute(create_sql)
                    else:
                        db_manager.user_execute(create_sql)

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
