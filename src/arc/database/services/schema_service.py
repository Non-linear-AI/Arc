"""Schema discovery service for Arc databases."""

import re
from dataclasses import dataclass
from typing import Any

from ..base import DatabaseError
from .base import BaseService


@dataclass
class TableInfo:
    """Information about a database table."""

    name: str
    schema: str = "main"
    table_type: str = "BASE TABLE"


@dataclass
class ColumnInfo:
    """Information about a table column."""

    table_name: str
    column_name: str
    data_type: str
    is_nullable: bool = True
    default_value: str | None = None
    column_position: int = 0


@dataclass
class SchemaInfo:
    """Complete schema information for a database."""

    tables: list[TableInfo]
    columns: list[ColumnInfo]

    def get_table_names(self) -> list[str]:
        """Get list of table names."""
        return [table.name for table in self.tables]

    def get_columns_for_table(self, table_name: str) -> list[ColumnInfo]:
        """Get columns for a specific table."""
        return [col for col in self.columns if col.table_name == table_name]

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        return table_name in self.get_table_names()

    def get_column_names(self, table_name: str) -> list[str]:
        """Get column names for a table."""
        return [col.column_name for col in self.get_columns_for_table(table_name)]


class SchemaService(BaseService):
    """Service for discovering and caching database schema information."""

    def __init__(self, db_manager):
        """Initialize SchemaService.

        Args:
            db_manager: DatabaseManager instance
        """
        super().__init__(db_manager)
        self._system_schema_cache: SchemaInfo | None = None
        self._user_schema_cache: SchemaInfo | None = None

    def get_schema_info(
        self, target_db: str = "system", force_refresh: bool = False
    ) -> SchemaInfo:
        """Get schema information for the specified database.

        Args:
            target_db: Target database ("system" or "user")
            force_refresh: Force refresh of cached schema

        Returns:
            SchemaInfo containing tables and columns

        Raises:
            DatabaseError: If schema discovery fails
        """
        if target_db not in ["system", "user"]:
            raise DatabaseError(f"Invalid target database: {target_db}")

        # Check cache first
        cache = (
            self._system_schema_cache
            if target_db == "system"
            else self._user_schema_cache
        )

        if not force_refresh and cache is not None:
            return cache

        # Refresh schema information
        schema_info = self._discover_schema(target_db)

        # Update cache
        if target_db == "system":
            self._system_schema_cache = schema_info
        else:
            self._user_schema_cache = schema_info

        return schema_info

    def _discover_schema(self, target_db: str) -> SchemaInfo:
        """Discover schema information for the specified database.

        Args:
            target_db: Target database ("system" or "user")

        Returns:
            SchemaInfo with discovered tables and columns

        Raises:
            DatabaseError: If schema discovery fails
        """
        try:
            # Discover tables
            tables = self._discover_tables(target_db)

            # Discover columns for all tables
            columns = []
            for table in tables:
                table_columns = self._discover_columns(table.name, target_db)
                columns.extend(table_columns)

            return SchemaInfo(tables=tables, columns=columns)

        except Exception as e:
            raise DatabaseError(
                f"Schema discovery failed for {target_db} database: {e}"
            ) from e

    def _discover_tables(self, target_db: str) -> list[TableInfo]:
        """Discover tables in the specified database.

        Args:
            target_db: Target database ("system" or "user")

        Returns:
            List of TableInfo objects
        """
        # Use DuckDB's INFORMATION_SCHEMA for standard SQL compliance
        query = """
        SELECT
            table_name,
            table_schema,
            table_type
        FROM INFORMATION_SCHEMA.TABLES
        WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
        ORDER BY table_name
        """

        result = (
            self.db_manager.system_query(query)
            if target_db == "system"
            else self.db_manager.user_query(query)
        )

        tables = []
        for row in result:
            tables.append(
                TableInfo(
                    name=row["table_name"],
                    schema=row.get("table_schema", "main"),
                    table_type=row.get("table_type", "BASE TABLE"),
                )
            )

        return tables

    def _discover_columns(self, table_name: str, target_db: str) -> list[ColumnInfo]:
        """Discover columns for a specific table.

        Args:
            table_name: Name of the table
            target_db: Target database ("system" or "user")

        Returns:
            List of ColumnInfo objects
        """
        # Use DuckDB's INFORMATION_SCHEMA for column information
        query = """
        SELECT
            table_name,
            column_name,
            data_type,
            is_nullable,
            column_default,
            ordinal_position
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE table_name = ?
        ORDER BY ordinal_position
        """

        if target_db == "system":
            result = self.db_manager.system_query(query, [table_name])
        else:
            # For user database, we need to use parameterized query differently
            # Replace ? with actual table name (safe since table_name is validated)
            safe_query = query.replace("?", f"'{table_name}'")
            result = self.db_manager.user_query(safe_query)

        columns = []
        for row in result:
            columns.append(
                ColumnInfo(
                    table_name=row["table_name"],
                    column_name=row["column_name"],
                    data_type=row["data_type"],
                    is_nullable=row.get("is_nullable", "YES") == "YES",
                    default_value=row.get("column_default"),
                    column_position=row.get("ordinal_position", 0),
                )
            )

        return columns

    def get_table_summary(
        self, target_db: str = "system", table_name: str | None = None
    ) -> str:
        """Get a formatted summary of database schema.

        Args:
            target_db: Target database ("system" or "user")
            table_name: Specific table name (optional)

        Returns:
            Formatted string with schema information
        """
        try:
            schema_info = self.get_schema_info(target_db)

            if table_name:
                # Show specific table information
                if not schema_info.table_exists(table_name):
                    return f"Table '{table_name}' not found in {target_db} database."

                columns = schema_info.get_columns_for_table(table_name)
                summary = f"Table: {table_name} ({target_db} DB)\n"
                summary += f"Columns ({len(columns)}):\n"

                for col in columns:
                    nullable = "NULL" if col.is_nullable else "NOT NULL"
                    default = (
                        f" DEFAULT {col.default_value}" if col.default_value else ""
                    )
                    summary += (
                        f"  - {col.column_name}: {col.data_type} {nullable}{default}\n"
                    )

            else:
                # Show all tables summary
                tables = schema_info.tables
                summary = f"{target_db.title()} Database Schema:\n"
                summary += f"Tables ({len(tables)}):\n"

                for table in tables:
                    column_count = len(schema_info.get_columns_for_table(table.name))
                    summary += f"  - {table.name} ({column_count} columns)\n"

                if tables:
                    summary += (
                        "\nUse schema discovery with specific table name for "
                        "detailed column information."
                    )

            return summary

        except Exception as e:
            return f"Error getting schema summary: {str(e)}"

    def validate_query_schema(
        self, query: str, target_db: str = "system"
    ) -> dict[str, Any]:
        """Validate that referenced tables and columns exist in the schema.

        Args:
            query: SQL query to validate
            target_db: Target database ("system" or "user")

        Returns:
            Dictionary with validation results
        """
        try:
            schema_info = self.get_schema_info(target_db)

            # This is a basic implementation - could be enhanced with SQL parsing
            # For now, we'll just check if query contains known table names
            query_upper = query.upper()

            referenced_tables = []
            missing_tables = []

            for table in schema_info.tables:
                if table.name.upper() in query_upper:
                    referenced_tables.append(table.name)

            # Extract potential table names from FROM/JOIN clauses
            # (basic regex could help)
            # This is a simplified check - real implementation would parse SQL

            return {
                "valid": len(missing_tables) == 0,
                "referenced_tables": referenced_tables,
                "missing_tables": missing_tables,
                "available_tables": schema_info.get_table_names(),
                "suggestions": self._get_schema_suggestions(query, schema_info),
            }

        except Exception as e:
            return {"valid": False, "error": str(e), "suggestions": []}

    def _get_schema_suggestions(self, query: str, schema_info: SchemaInfo) -> list[str]:
        """Get schema-based suggestions for improving a query.

        Args:
            query: SQL query
            schema_info: Schema information

        Returns:
            List of suggestion strings
        """
        suggestions = []

        # If no tables are referenced, suggest available tables
        if not any(table.name.upper() in query.upper() for table in schema_info.tables):
            table_list = ", ".join(schema_info.get_table_names())
            suggestions.append(f"Available tables: {table_list}")

        return suggestions

    def is_ddl_statement(self, sql: str) -> bool:
        """Check if a SQL statement is a DDL (Data Definition Language) operation.

        DDL operations modify database schema and require cache invalidation.

        Args:
            sql: SQL statement to check

        Returns:
            True if the statement is a DDL operation
        """
        if not sql or not sql.strip():
            return False

        # Normalize SQL: remove extra whitespace and convert to uppercase
        normalized_sql = re.sub(r"\s+", " ", sql.strip().upper())

        # DDL statement patterns that modify schema
        ddl_patterns = [
            r"^CREATE\s+(TABLE|INDEX|VIEW|SEQUENCE|TRIGGER|FUNCTION|PROCEDURE)",
            r"^DROP\s+(TABLE|INDEX|VIEW|SEQUENCE|TRIGGER|FUNCTION|PROCEDURE)",
            r"^ALTER\s+(TABLE|INDEX|VIEW|SEQUENCE)",
            r"^TRUNCATE\s+TABLE",
            r"^RENAME\s+(TABLE|COLUMN)",
            # DuckDB specific
            r"^PRAGMA\s+(table_info|index_info)",
        ]

        return any(re.match(pattern, normalized_sql) for pattern in ddl_patterns)

    def invalidate_cache(self, target_db: str = "user") -> None:
        """Invalidate schema cache for specific database after DDL operations.

        Args:
            target_db: Target database to invalidate ("system" or "user")
        """
        if target_db == "system":
            self._system_schema_cache = None
        elif target_db == "user":
            self._user_schema_cache = None

    def invalidate_all_caches(self) -> None:
        """Invalidate all schema caches."""
        self._system_schema_cache = None
        self._user_schema_cache = None

    def clear_cache(self, target_db: str | None = None) -> None:
        """Clear schema cache for specified database or all databases.

        Args:
            target_db: Target database ("system", "user", or None for both)
        """
        if target_db is None or target_db == "system":
            self._system_schema_cache = None
        if target_db is None or target_db == "user":
            self._user_schema_cache = None

    def generate_system_schema_prompt(self) -> str:
        """Generate formatted system database schema for prompt injection.

        Returns:
            Formatted string describing system database schema for AI context
        """
        try:
            schema_info = self.get_schema_info("system")

            # System database description with ML workflow context
            prompt_sections = [
                "# System Database Schema",
                "",
                "Arc's system database contains the following tables for ML workflow "
                "management:",
                "",
            ]

            # Table descriptions with ML context
            table_descriptions = {
                "models": "Model registry - Stores ML model definitions, versions, and "
                "Arc-Graph specifications",
                "jobs": "Job tracking - Manages training, evaluation, and processing "
                "jobs with status monitoring",
                "trained_models": "Model artifacts - Catalogs successful training "
                "outputs with metrics and paths",
                "deployments": "Model serving - Tracks deployed models for real-time "
                "inference",
                "plugins": "Plugin registry - Available ML algorithms and custom "
                "components",
                "plugin_components": "Plugin specs - Detailed component "
                "specifications for plugins",
                "plugin_schemas": "Plugin metadata - Schema validation and "
                "documentation for algorithms",
            }

            for table in schema_info.tables:
                table_name = table.name
                description = table_descriptions.get(
                    table_name, f"System table: {table_name}"
                )

                prompt_sections.append(f"## {table_name}")
                prompt_sections.append(f"**Purpose**: {description}")

                columns = schema_info.get_columns_for_table(table_name)
                if columns:
                    prompt_sections.append("**Columns**:")
                    for col in columns:
                        nullable = "NULL" if col.is_nullable else "NOT NULL"
                        col_desc = f"  - {col.column_name}: {col.data_type} {nullable}"
                        if col.default_value:
                            col_desc += f" DEFAULT {col.default_value}"
                        prompt_sections.append(col_desc)

                prompt_sections.append("")

            prompt_sections.extend(
                [
                    "**Usage Guidelines**:",
                    "- Use system database queries for ML workflow operations "
                    "(models, jobs, deployments)",
                    "- System database is read-only - use SELECT queries only",
                    "- For user data analysis, use schema_discovery tool to explore "
                    "user database first",
                    "",
                ]
            )

            return "\n".join(prompt_sections)

        except Exception as e:
            # Fallback to basic description if schema discovery fails
            return (
                "# System Database Schema\n\n"
                "Arc system database contains tables for ML workflow management:\n"
                "- models: ML model registry\n"
                "- jobs: Training/processing job tracking\n"
                "- trained_models: Model artifact catalog\n"
                "- deployments: Model serving registry\n"
                "- plugins: Available ML algorithms\n"
                "- plugin_components: Plugin specifications\n"
                "- plugin_schemas: Plugin metadata\n\n"
                f"Note: Schema discovery failed ({str(e)}), using basic description.\n"
            )
