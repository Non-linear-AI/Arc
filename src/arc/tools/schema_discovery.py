"""Schema discovery tool for exploring database structures."""

from typing import TYPE_CHECKING

from arc.database.base import DatabaseError
from arc.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from arc.database.services.container import ServiceContainer


class SchemaDiscoveryTool(BaseTool):
    """Tool for discovering and exploring database schema information."""

    def __init__(self, services: "ServiceContainer"):
        """Initialize SchemaDiscoveryTool.

        Args:
            services: ServiceContainer instance providing database access
        """
        self.services = services

    async def execute(
        self,
        action: str = "list_tables",
        target_db: str = "system",
        table_name: str | None = None,
        force_refresh: bool = False,
    ) -> ToolResult:
        """Execute schema discovery operation.

        Args:
            action: Type of schema discovery action
                   - "list_tables": List all tables in database
                   - "describe_table": Show detailed table structure
                   - "show_schema": Show complete database schema summary
                   - "refresh_cache": Force refresh of schema cache
            target_db: Target database - "system" (default) or "user"
            table_name: Specific table name (required for "describe_table")
            force_refresh: Force refresh of schema cache before operation

        Returns:
            ToolResult with schema information
        """
        try:
            # Validate target database
            if target_db not in ["system", "user"]:
                return ToolResult.error_result(
                    f"Invalid target database: {target_db}. "
                    "Must be 'system' or 'user'.",
                    recovery_actions=(
                        "Use 'system' for Arc metadata or 'user' for training data."
                    ),
                )

            # Validate action
            valid_actions = [
                "list_tables",
                "describe_table",
                "show_schema",
                "refresh_cache",
            ]
            if action not in valid_actions:
                return ToolResult.error_result(
                    f"Invalid action: {action}. "
                    f"Must be one of: {', '.join(valid_actions)}",
                    recovery_actions=(
                        f"Use one of the supported actions: {', '.join(valid_actions)}"
                    ),
                )

            # Force refresh if requested (useful after DDL operations)
            if force_refresh:
                self.services.schema.invalidate_cache(target_db)

            # Execute the requested action
            if action == "list_tables":
                return await self._list_tables(target_db)
            elif action == "describe_table":
                if not table_name:
                    return ToolResult.error_result(
                        "Table name is required for 'describe_table' action.",
                        recovery_actions="Provide a table_name parameter.",
                    )
                return await self._describe_table(table_name, target_db)
            elif action == "show_schema":
                return await self._show_schema(target_db)
            elif action == "refresh_cache":
                self.services.schema.invalidate_cache(target_db)
                # Re-fetch to repopulate cache
                schema_info = self.services.schema.get_schema_info(
                    target_db, force_refresh=True
                )
                table_count = len(schema_info.tables)
                return ToolResult.success_result(
                    f"✓ Schema cache refreshed for {target_db} database\n"
                    f"  Found {table_count} table{'s' if table_count != 1 else ''}"
                )

        except DatabaseError as e:
            # Database-specific errors
            recovery_actions = None
            if "not configured" in str(e).lower():
                recovery_actions = (
                    "User database is not configured. Set user database path in "
                    "settings or use system database for Arc metadata."
                )

            return ToolResult.error_result(
                f"Database error: {str(e)}", recovery_actions=recovery_actions
            )

        except Exception as e:
            # Unexpected errors
            return ToolResult.error_result(
                f"Unexpected error during schema discovery: {str(e)}",
                recovery_actions="Check database connectivity and try again.",
            )

    async def _list_tables(self, target_db: str) -> ToolResult:
        """List all tables in the specified database.

        Args:
            target_db: Target database ("system" or "user")

        Returns:
            ToolResult with table list
        """
        try:
            schema_info = self.services.schema.get_schema_info(target_db)
            tables = schema_info.tables

            if not tables:
                output = f"No tables found in {target_db} database."
                metadata = {"table_count": 0}
            else:
                total = len(tables)
                # Show first 5 tables
                show_count = min(5, total)
                output = ""

                for i, table in enumerate(tables[:show_count]):
                    column_count = len(schema_info.get_columns_for_table(table.name))
                    output += f"• {table.name} [dim]({column_count} cols)[/dim]"
                    # Add 3 tables per line
                    if (i + 1) % 3 == 0 or i == show_count - 1:
                        output += "\n"
                    else:
                        output += " "

                # Add "… and N more" if there are more tables
                if total > show_count:
                    output += f"… and {total - show_count} more"

                metadata = {"table_count": total}

            return ToolResult.success_result(output, metadata=metadata)

        except Exception as e:
            return ToolResult.error_result(f"Failed to list tables: {str(e)}")

    async def _describe_table(self, table_name: str, target_db: str) -> ToolResult:
        """Describe the structure of a specific table.

        Args:
            table_name: Name of the table to describe
            target_db: Target database ("system" or "user")

        Returns:
            ToolResult with table structure details
        """
        try:
            schema_info = self.services.schema.get_schema_info(target_db)

            # Check if table exists
            if not schema_info.table_exists(table_name):
                available_tables = ", ".join(schema_info.get_table_names())
                return ToolResult.error_result(
                    f"Table '{table_name}' not found in {target_db} database.",
                    recovery_actions=f"Available tables: {available_tables}",
                )

            # Get table details
            columns = schema_info.get_columns_for_table(table_name)
            total_cols = len(columns)
            show_count = min(5, total_cols)

            # Check if all columns are nullable
            all_nullable = all(col.is_nullable for col in columns)

            # Add nullable info as first line if all columns are nullable
            output = ""

            if columns:
                # Group columns by data type
                type_groups = {}
                for col in columns[:show_count]:
                    dtype = col.data_type
                    if dtype not in type_groups:
                        type_groups[dtype] = []
                    type_groups[dtype].append(col.column_name)

                # Output grouped by type
                for dtype, col_names in type_groups.items():
                    output += f"• {', '.join(col_names)} [dim]({dtype})[/dim]\n"

                # Add "… and N more" if there are more columns
                if total_cols > show_count:
                    output += f"… and {total_cols - show_count} more columns"

            metadata = {"column_count": total_cols}
            return ToolResult.success_result(output, metadata=metadata)

        except Exception as e:
            return ToolResult.error_result(f"Failed to describe table: {str(e)}")

    async def _show_schema(self, target_db: str) -> ToolResult:
        """Show complete schema summary for the database.

        Args:
            target_db: Target database ("system" or "user")

        Returns:
            ToolResult with complete schema information
        """
        try:
            summary = self.services.schema.get_table_summary(target_db)
            return ToolResult.success_result(summary)

        except Exception as e:
            return ToolResult.error_result(f"Failed to get schema summary: {str(e)}")

    def _get_table_insights(self, columns: list) -> list[str]:
        """Generate insights about table structure based on column patterns.

        Args:
            columns: List of ColumnInfo objects

        Returns:
            List of insight strings
        """
        insights = []

        # Check for common patterns
        column_names = [col.column_name.lower() for col in columns]

        # Primary key detection
        if "id" in column_names:
            insights.append("Contains 'id' column (likely primary key)")

        # Timestamp detection
        timestamp_cols = [
            name
            for name in column_names
            if any(
                time_word in name
                for time_word in ["created", "updated", "timestamp", "date"]
            )
        ]
        if timestamp_cols:
            insights.append(f"Contains timestamp columns: {', '.join(timestamp_cols)}")

        # JSON/TEXT columns for complex data
        json_cols = [
            col.column_name for col in columns if "json" in col.data_type.lower()
        ]
        if json_cols:
            insights.append(f"Contains JSON columns: {', '.join(json_cols)}")

        # Foreign key patterns
        fk_cols = [
            name for name in column_names if name.endswith("_id") and name != "id"
        ]
        if fk_cols:
            insights.append(f"Possible foreign key columns: {', '.join(fk_cols)}")

        return insights
