"""Database query execution tool."""

import time
from typing import TYPE_CHECKING

from ..database.base import DatabaseError, QueryValidationError
from .base import BaseTool, ToolResult

if TYPE_CHECKING:
    from ..database.services.container import ServiceContainer


class DatabaseQueryTool(BaseTool):
    """Tool for executing SQL queries against system and user databases."""

    def __init__(self, services: "ServiceContainer"):
        """Initialize DatabaseQueryTool.

        Args:
            services: ServiceContainer instance providing database access
        """
        self.services = services

    async def execute(self, query: str, target_db: str = "system") -> ToolResult:
        """Execute a SQL query against the specified database.

        Args:
            query: SQL query to execute
            target_db: Target database - "system" (read-only) or "user" (full access)

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

            # Execute query using the interactive query service
            start_time = time.time()
            result = self.services.query.execute_query(query, target_db)
            execution_time = time.time() - start_time

            # Format results for output
            if result.empty():
                output = (
                    f"Query executed successfully ({execution_time:.3f}s)\n"
                    "No results returned."
                )
            else:
                # Get basic result information
                row_count = result.count()
                first_row = result.first()

                if first_row:
                    column_names = list(first_row.keys())
                    columns_info = f"Columns: {', '.join(column_names)}"
                else:
                    columns_info = "No columns"

                # Format output with summary
                output = (
                    f"Query executed successfully ({execution_time:.3f}s)\n"
                    f"Results: {row_count} row{'s' if row_count != 1 else ''}\n"
                    f"{columns_info}\n\n"
                )

                # Add sample data for small result sets
                if row_count <= 10:
                    output += "Data:\n"
                    for i, row in enumerate(result):
                        row_data = []
                        for value in row.values():
                            if value is None:
                                row_data.append("NULL")
                            elif isinstance(value, str) and len(value) > 50:
                                row_data.append(f"{value[:47]}...")
                            else:
                                row_data.append(str(value))
                        output += f"  Row {i + 1}: {', '.join(row_data)}\n"
                else:
                    output += (
                        f"Use `/sql {query}` in interactive mode to view full results."
                    )

            return ToolResult.success_result(output)

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
                "Use DESCRIBE or SHOW TABLES to explore database schema."
            )

            return ToolResult.error_result(
                f"Database error: {str(e)}", recovery_actions=recovery_actions
            )

        except Exception as e:
            # Unexpected errors
            return ToolResult.error_result(
                f"Unexpected error executing query: {str(e)}",
                recovery_actions="Check query syntax and database connectivity.",
            )
