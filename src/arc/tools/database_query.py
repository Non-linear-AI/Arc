"""Database query execution tool."""

import time
from typing import TYPE_CHECKING

from arc.database.base import DatabaseError, QueryValidationError
from arc.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from arc.database.services.container import ServiceContainer


class DatabaseQueryTool(BaseTool):
    """Tool for executing SQL queries against system and user databases."""

    def __init__(self, services: "ServiceContainer"):
        """Initialize DatabaseQueryTool.

        Args:
            services: ServiceContainer instance providing database access
        """
        self.services = services

    async def execute(
        self, query: str, target_db: str = "user", validate_schema: bool = True
    ) -> ToolResult:
        """Execute a SQL query against the specified database.

        Args:
            query: SQL query to execute
            target_db: Target database - "user" (default, full access) or
                "system" (read-only)
            validate_schema: Whether to validate query against database schema

        Returns:
            ToolResult with query results or error information
        """
        try:
            # Validate target database
            if target_db not in ["system", "user"]:
                return ToolResult.error_result(
                    f"Invalid target database: {target_db}. "
                    "Must be 'system' or 'user'.",
                    recovery_actions=(
                        "Use 'system' for read-only queries or 'user' for full access."
                    ),
                )

            # Optional schema validation before execution
            schema_warnings = []
            if validate_schema:
                try:
                    validation_result = self.services.schema.validate_query_schema(
                        query, target_db
                    )
                    if not validation_result.get("valid", True):
                        schema_warnings.append("Schema validation warnings found.")

                    # Add suggestions to output if available
                    suggestions = validation_result.get("suggestions", [])
                    if suggestions:
                        schema_warnings.extend(suggestions)

                except Exception:
                    # Don't fail the query if schema validation fails
                    schema_warnings.append("Schema validation unavailable.")

            # Execute query using the interactive query service (read-only mode)
            start_time = time.time()
            result = self.services.query.execute_query(query, target_db, read_only=True)
            execution_time = time.time() - start_time

            # Use full query (no truncation)
            display_query = query

            # Prepare metadata for section title
            metadata = {"execution_time": execution_time, "target_db": target_db}

            # Format results using Rich Table (like /sql command)
            if result.empty():
                output = "No results returned."
            else:
                import json

                from rich import box
                from rich.table import Table

                # Get basic result information
                row_count = result.count()
                first_row = result.first()

                # Add row count to metadata
                metadata["row_count"] = row_count

                if not first_row:
                    output = "No results returned."
                else:
                    # Build Rich table for clean display (will be dimmed via
                    # Padding wrapper)
                    table = Table(
                        show_header=True,
                        header_style="bold",
                        border_style="color(240)",
                        box=box.HORIZONTALS,
                    )

                    # Add columns from first row
                    for column_name in first_row:
                        table.add_column(str(column_name), no_wrap=False)

                    # Add data rows (limit to allow aggregation results)
                    max_rows = 100
                    for row_count_idx, row in enumerate(result):
                        if row_count_idx >= max_rows:
                            table.add_row(*["..." for _ in first_row], style="dim")
                            break

                        # Convert all values to strings and handle None
                        row_values = []
                        for value in row.values():
                            if value is None:
                                row_values.append("[dim]NULL[/dim]")
                            elif isinstance(value, (dict, list)):
                                # Format JSON-like objects
                                row_values.append(
                                    json.dumps(
                                        value, indent=None, separators=(",", ":")
                                    )
                                )
                            else:
                                row_values.append(str(value))

                        table.add_row(*row_values)

                    # Store Rich table in metadata for rendering
                    total_rows = result.count()
                    row_text = "row" if total_rows == 1 else "rows"
                    if total_rows > max_rows:
                        summary = f"Showing {max_rows} of {total_rows} rows"
                    else:
                        summary = f"{total_rows} {row_text} returned"

                    # Build JSON output for agent
                    # Convert rows to list of dictionaries
                    json_rows = []
                    for row_idx, row in enumerate(result):
                        if row_idx >= max_rows:
                            break
                        row_dict = {}
                        for key, value in row.items():
                            # Keep None as null in JSON
                            row_dict[key] = value
                        json_rows.append(row_dict)

                    # Format as JSON (compact)
                    json_output = json.dumps(json_rows, default=str)

                    # Add truncation note if needed
                    truncation_note = ""
                    if total_rows > max_rows:
                        truncation_note = f"\n(Showing {max_rows} of {total_rows} rows)"

                    agent_output = json_output + truncation_note

                    # Return metadata for Rich rendering (minimal style)
                    metadata["rich_table"] = table
                    metadata["summary"] = summary
                    metadata["query"] = display_query
                    metadata["agent_output"] = agent_output
                    output = agent_output

            # Add schema warnings if any
            if schema_warnings:
                output += "\nSchema Notes:\n"
                for warning in schema_warnings:
                    output += f"• {warning}\n"

            return ToolResult.success_result(output, metadata=metadata)

        except QueryValidationError as e:
            # Query validation errors (e.g., invalid SQL, permission issues)
            recovery_actions = None
            if "read-only" in str(e).lower():
                recovery_actions = (
                    "System database is read-only. Use SELECT queries only, "
                    "or switch to user database for write operations."
                )
            elif "not configured" in str(e).lower():
                recovery_actions = (
                    "User database is not configured. Set user database path in "
                    "settings or use system database for read-only queries."
                )

            return ToolResult.error_result(
                f"Query validation error: {str(e)}", recovery_actions=recovery_actions
            )

        except DatabaseError as e:
            # Database execution errors (e.g., SQL syntax errors, constraint violations)
            recovery_actions = (
                "Check SQL syntax and ensure all referenced tables/columns exist. "
                "Use schema_discovery tool to explore database schema."
            )

            # Try to provide schema-aware suggestions for common errors
            error_msg = str(e).lower()
            if "table" in error_msg and "does not exist" in error_msg:
                try:
                    schema_info = self.services.schema.get_schema_info(target_db)
                    available_tables = schema_info.get_table_names()
                    if available_tables:
                        recovery_actions += (
                            f" Available tables: {', '.join(available_tables[:5])}"
                        )
                        if len(available_tables) > 5:
                            recovery_actions += (
                                f" and {len(available_tables) - 5} more."
                            )
                except Exception:
                    pass

            return ToolResult.error_result(
                f"Database error: {str(e)}", recovery_actions=recovery_actions
            )

        except Exception as e:
            # Unexpected errors
            return ToolResult.error_result(
                f"Unexpected error executing query: {str(e)}",
                recovery_actions="Check query syntax and database connectivity.",
            )
