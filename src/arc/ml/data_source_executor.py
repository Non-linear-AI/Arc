"""Data source pipeline executor for SQL transformation workflows."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arc.database.manager import DatabaseManager
    from arc.graph.features.data_source import DataSourceSpec
    from arc.ui.console import InteractiveInterface


@dataclass
class DataSourceExecutionResult:
    """Result of data source pipeline execution."""

    created_tables: list[str]
    execution_time: float
    step_count: int
    intermediate_views_cleaned: int


class DataSourceExecutionError(Exception):
    """Exception raised when data source pipeline execution fails."""


async def execute_data_source_pipeline(
    spec: DataSourceSpec,
    target_db: str,
    db_manager: DatabaseManager,
    ui: InteractiveInterface | None = None,
) -> DataSourceExecutionResult:
    """Execute a DataSourceSpec pipeline with proper cleanup.

    Args:
        spec: DataSourceSpec containing steps and outputs
        target_db: Target database ("system" or "user")
        db_manager: DatabaseManager instance for execution
        ui: Optional UI interface for progress reporting

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

    # Show progress header
    if ui:
        ui.show_info("")
        ui.show_info("Executing pipeline...")

    step_count = len(ordered_steps)
    intermediate_views = []  # Track views that need cleanup

    try:
        for i, step in enumerate(ordered_steps, 1):
            # Determine if this is an output step (should create table) or
            # intermediate (view)
            step_type = "table" if step.name in spec.outputs else "view"

            if ui:
                ui.show_info(f"  Step {i}/{step_count}: {step.name} ({step_type})")

            # Substitute variables in SQL
            sql = spec.substitute_vars(step.sql)

            # Strip trailing semicolons (they cause syntax errors in
            # CREATE TABLE/VIEW AS)
            sql = sql.rstrip().rstrip(";").rstrip()

            try:
                if step.name in spec.outputs:
                    # Create persistent table for output steps
                    create_sql = f"CREATE TABLE {step.name} AS ({sql})"
                else:
                    # Create regular view for intermediate steps (allows debugging)
                    create_sql = f"CREATE VIEW {step.name} AS ({sql})"
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
                    cleanup_sql = f"DROP VIEW IF EXISTS {view_name}"
                    if target_db == "system":
                        db_manager.system_execute(cleanup_sql)
                    else:
                        db_manager.user_execute(cleanup_sql)
                    cleaned_count += 1
                except Exception:
                    # Don't fail the entire pipeline if cleanup fails
                    if ui:
                        ui.show_info(f"Warning: Could not clean up view '{view_name}'")

    # Calculate execution time
    execution_time = time.time() - start_time

    # Show completion summary
    if ui:
        ui.show_info("")
        ui.show_info("Pipeline completed successfully")
        output_list = ", ".join(spec.outputs)
        ui.show_info(f"Output tables: {output_list}")

        if target_db == "user":
            ui.show_info(
                "Note: Intermediate views have been cleaned up. Query results "
                "with: /sql SELECT * FROM <table_name>"
            )

    return DataSourceExecutionResult(
        created_tables=spec.outputs.copy(),
        execution_time=execution_time,
        step_count=step_count,
        intermediate_views_cleaned=cleaned_count,
    )
