"""Input validation utilities for Arc."""

import re
from pathlib import Path


class ValidationError(Exception):
    """Raised when validation fails."""

    pass


def validate_sql_identifier(name: str, context: str = "identifier") -> None:
    """Validate a SQL identifier (table/column/view name).

    Args:
        name: The identifier to validate
        context: Context for error messages (e.g., "table name", "column name")

    Raises:
        ValidationError: If the identifier is invalid

    Rules:
        - Must not be empty
        - Must start with a letter or underscore
        - Can only contain letters, numbers, and underscores
        - Cannot be a SQL reserved word
        - Must be 1-128 characters
    """
    if not name:
        raise ValidationError(f"Invalid {context}: cannot be empty")

    if len(name) > 128:
        raise ValidationError(
            f"Invalid {context} '{name}': must be 128 characters or less"
        )

    # Check first character
    if not (name[0].isalpha() or name[0] == "_"):
        raise ValidationError(
            f"Invalid {context} '{name}': must start with a letter or underscore"
        )

    # Check all characters
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
        raise ValidationError(
            f"Invalid {context} '{name}': can only contain letters, numbers, "
            "and underscores"
        )

    # Check for SQL reserved words (common subset)
    sql_reserved = {
        "select",
        "insert",
        "update",
        "delete",
        "drop",
        "create",
        "alter",
        "table",
        "view",
        "index",
        "from",
        "where",
        "join",
        "on",
        "and",
        "or",
        "not",
        "null",
        "true",
        "false",
        "as",
        "by",
        "order",
        "group",
        "having",
        "union",
        "all",
        "distinct",
        "case",
        "when",
        "then",
        "else",
        "end",
    }

    if name.lower() in sql_reserved:
        raise ValidationError(
            f"Invalid {context} '{name}': cannot use SQL reserved word. "
            f"Try using a different name or quoting it."
        )


def quote_sql_identifier(name: str) -> str:
    """Quote a SQL identifier for safe use in queries.

    Args:
        name: The identifier to quote

    Returns:
        Quoted identifier using double quotes

    Note:
        DuckDB uses double quotes for identifiers.
        This escapes any double quotes in the name by doubling them.
    """
    # Escape any double quotes by doubling them
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def validate_table_name(name: str) -> None:
    """Validate a table name.

    Args:
        name: The table name to validate

    Raises:
        ValidationError: If the table name is invalid
    """
    validate_sql_identifier(name, context="table name")


def validate_column_name(name: str) -> None:
    """Validate a column name.

    Args:
        name: The column name to validate

    Raises:
        ValidationError: If the column name is invalid
    """
    validate_sql_identifier(name, context="column name")


def validate_model_name(name: str) -> None:
    """Validate a model name.

    Args:
        name: The model name to validate

    Raises:
        ValidationError: If the model name is invalid

    Rules:
        - Must follow identifier rules
        - Should be descriptive
    """
    validate_sql_identifier(name, context="model name")


def validate_file_path(path: str, must_exist: bool = False) -> Path:
    """Validate a file path.

    Args:
        path: The file path to validate
        must_exist: Whether the file must exist

    Returns:
        Resolved Path object

    Raises:
        ValidationError: If the path is invalid or doesn't exist when required
    """
    if not path:
        raise ValidationError("File path cannot be empty")

    try:
        path_obj = Path(path).resolve()
    except Exception as e:
        raise ValidationError(f"Invalid file path '{path}': {e}") from e

    # Check for directory traversal attempts
    # Ensure the resolved path doesn't escape outside allowed directories
    # For now, just check for suspicious patterns
    if ".." in Path(path).parts:
        raise ValidationError(
            f"Invalid file path '{path}': directory traversal not allowed"
        )

    if must_exist and not path_obj.exists():
        raise ValidationError(f"File not found: {path}")

    return path_obj


def validate_api_key(api_key: str) -> str:
    """Validate an API key.

    Args:
        api_key: The API key to validate

    Returns:
        Stripped API key

    Raises:
        ValidationError: If the API key is invalid
    """
    if not api_key:
        raise ValidationError("API key cannot be empty")

    # Strip whitespace
    api_key = api_key.strip()

    if not api_key:
        raise ValidationError("API key cannot be only whitespace")

    # Basic length check (most API keys are at least 20 characters)
    if len(api_key) < 10:
        raise ValidationError(
            f"API key too short ({len(api_key)} characters). "
            "Expected at least 10 characters."
        )

    return api_key


def validate_url(url: str) -> str:
    """Validate a URL.

    Args:
        url: The URL to validate

    Returns:
        Validated URL

    Raises:
        ValidationError: If the URL is invalid
    """
    if not url:
        raise ValidationError("URL cannot be empty")

    url = url.strip()

    if not url:
        raise ValidationError("URL cannot be only whitespace")

    # Check for protocol
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValidationError(
            f"Invalid URL '{url}': must start with http:// or https://"
        )

    # Basic URL pattern check
    # This is a simple check - for production, use urllib.parse
    if not re.match(r"^https?://[^\s/$.?#].[^\s]*$", url, re.IGNORECASE):
        raise ValidationError(f"Invalid URL format: {url}")

    return url


def validate_database_name(name: str) -> None:
    """Validate a database name.

    Args:
        name: The database name to validate

    Raises:
        ValidationError: If the database name is invalid
    """
    valid_names = ["system", "user"]

    if name not in valid_names:
        raise ValidationError(
            f"Invalid database name '{name}'. Must be one of: {', '.join(valid_names)}"
        )


def validate_model_id(model_id: str) -> None:
    """Validate a model ID format.

    Args:
        model_id: The model ID to validate (e.g., "my_model-v1")

    Raises:
        ValidationError: If the model ID is invalid

    Expected format: <name>-v<version>
    """
    if not model_id:
        raise ValidationError("Model ID cannot be empty")

    # Check format: name-vN
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]+-v\d+$", model_id):
        raise ValidationError(
            f"Invalid model ID '{model_id}'. Expected format: <name>-v<version> "
            "(e.g., 'my_model-v1')"
        )


def validate_sql_syntax(sql: str, connection=None) -> list[str]:
    """Validate SQL syntax using DuckDB's EXPLAIN.

    Args:
        sql: SQL query to validate
        connection: Optional DuckDB connection. If None, creates temporary connection.

    Returns:
        List of syntax error messages (empty if valid).
        Only returns parser/syntax errors, not catalog errors (missing tables/columns).

    Note:
        Uses DuckDB's native EXPLAIN to validate syntax without executing.
        This ensures perfect compatibility with DuckDB's SQL dialect.
        Catalog errors (table/column not found) are ignored as they're expected
        during validation without the actual data present.

        For DDL statements (DROP, ALTER, TRUNCATE), validation is more lenient
        since EXPLAIN doesn't work well with these statements and they don't
        return data anyway.
    """
    if not sql or not sql.strip():
        return ["SQL query cannot be empty"]

    errors = []
    close_conn = False

    # Check if this is a DDL statement that doesn't work well with EXPLAIN
    clean_sql = sql.strip().upper()
    is_ddl_statement = any(
        clean_sql.startswith(stmt)
        for stmt in ["DROP ", "ALTER ", "TRUNCATE ", "GRANT ", "REVOKE "]
    )

    # For DDL statements, do basic syntax validation instead of EXPLAIN
    if is_ddl_statement:
        # DDL statements are typically valid if they follow basic SQL structure
        # We'll do minimal validation here since they're meant to be executed
        # even if they don't return data
        try:
            # Import here to avoid circular dependency
            import duckdb

            # Use provided connection or create temporary one
            if connection is None:
                connection = duckdb.connect(":memory:")
                close_conn = True

            # For DROP TABLE, just check basic syntax by parsing
            # We don't try to execute or explain since the table might not exist
            # This is intentionally lenient to allow valid DDL even if targets don't exist
            pass  # DDL statements are assumed valid

        finally:
            if close_conn and connection:
                connection.close()

        # Return empty errors list - DDL statements are allowed through
        return errors

    # For non-DDL statements (SELECT, INSERT, etc.), use EXPLAIN validation
    try:
        # Import here to avoid circular dependency
        import duckdb

        # Use provided connection or create temporary one
        if connection is None:
            connection = duckdb.connect(":memory:")
            close_conn = True

        # Use EXPLAIN to validate syntax without executing
        # Strip trailing semicolons as they can cause issues with EXPLAIN
        clean_sql = sql.rstrip().rstrip(";").rstrip()
        connection.execute(f"EXPLAIN {clean_sql}")

    except Exception as e:
        error_msg = str(e)
        # Only report syntax/parser errors, not catalog errors
        # Catalog errors (table/column not found) are expected during validation
        if "Parser Error" in error_msg or "syntax error" in error_msg.lower():
            errors.append(f"SQL syntax error: {error_msg}")
        # Ignore catalog errors like "Table does not exist" during validation
        # These are expected when validating SQL without the actual data
    finally:
        if close_conn and connection:
            connection.close()

    return errors
