"""DuckDB database implementation."""

import time
from pathlib import Path
from typing import Any

import duckdb

from arc.database.base import Database, DatabaseError, QueryResult


class DuckDBDatabase(Database):
    """DuckDB implementation of the Database interface."""

    def __init__(self, db_path: str | Path):
        """Initialize DuckDB database connection.

        Args:
            db_path: Path to the DuckDB database file.
                Use ":memory:" for in-memory database.
        """
        self.db_path = str(db_path)
        self._connection: duckdb.DuckDBPyConnection | None = None
        self._connect()

    def _connect(self) -> None:
        """Establish connection to the database."""
        try:
            self._connection = duckdb.connect(self.db_path)
        except Exception as e:
            raise DatabaseError(
                f"Failed to connect to DuckDB at {self.db_path}: {e}"
            ) from e

    def _ensure_connected(self) -> duckdb.DuckDBPyConnection:
        """Ensure we have a valid database connection."""
        if self._connection is None:
            self._connect()

        if self._connection is None:
            raise DatabaseError("Database connection is not available")

        return self._connection

    def query(self, sql: str, params: list | None = None) -> QueryResult:
        """Execute a SELECT query and return results.

        Args:
            sql: SQL SELECT statement to execute
            params: Optional list of parameters for the query

        Returns:
            QueryResult containing the query results

        Raises:
            DatabaseError: If query execution fails
        """
        start_time = time.time()

        try:
            conn = self._ensure_connected()

            # Execute query and fetch results
            cursor = conn.execute(sql, params) if params else conn.execute(sql)

            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description or []]

            # Convert to list of dictionaries
            result_rows = [dict(zip(columns, row, strict=False)) for row in rows]

            execution_time = time.time() - start_time

            return QueryResult(rows=result_rows, execution_time=execution_time)

        except Exception as e:
            raise DatabaseError(f"Query execution failed: {e}") from e

    def execute(self, sql: str, params: list | None = None) -> None:
        """Execute a DDL or DML statement (CREATE, INSERT, UPDATE, DELETE).

        Args:
            sql: SQL statement to execute
            params: Optional list of parameters for the statement

        Raises:
            DatabaseError: If statement execution fails
        """
        try:
            conn = self._ensure_connected()
            if params:
                conn.execute(sql, params)
            else:
                conn.execute(sql)

        except Exception as e:
            raise DatabaseError(f"Statement execution failed: {e}") from e

    def close(self) -> None:
        """Close the database connection and clean up resources."""
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                # Ignore errors when closing
                pass
            finally:
                self._connection = None

    def init_schema(self) -> None:
        """Initialize the database schema with required tables and indexes.

        Creates all necessary tables for Arc's data model. This method is
        idempotent and can be called multiple times safely.

        Raises:
            DatabaseError: If schema creation fails
        """
        try:
            # Main registry for versioned model definitions with inheritance support
            self.execute("""
                CREATE TABLE IF NOT EXISTS models(
                    id TEXT PRIMARY KEY,
                    type TEXT,
                    -- Rich schema fields kept for compatibility with other parts
                    -- of the codebase
                    name VARCHAR(255),
                    version INTEGER,
                    description TEXT,
                    spec TEXT,
                    plan_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (name, version)
                );
            """)

            # Create indexes for common lookups
            self.execute("""
                CREATE INDEX IF NOT EXISTS idx_models_name ON models(name);
            """)

            self.execute("""
                CREATE INDEX IF NOT EXISTS idx_models_name_version
                ON models(name, version);
            """)

            # Registry for trainer specifications linked to models
            self.execute("""
                CREATE TABLE IF NOT EXISTS trainers(
                    id TEXT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    version INTEGER NOT NULL,
                    model_id TEXT NOT NULL,
                    model_version INTEGER NOT NULL,
                    spec TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (name, version)
                );
            """)

            # Create indexes for trainer lookups
            self.execute("""
                CREATE INDEX IF NOT EXISTS idx_trainers_name ON trainers(name);
            """)

            self.execute("""
                CREATE INDEX IF NOT EXISTS idx_trainers_model_id ON trainers(model_id);
            """)

            self.execute("""
                CREATE INDEX IF NOT EXISTS idx_trainers_name_version
                ON trainers(name, version);
            """)

            # Tracks long-running processes like training
            self.execute("""
                CREATE TABLE IF NOT EXISTS jobs(
                    job_id TEXT PRIMARY KEY,
                    model_id INTEGER,
                    trainer_id TEXT,
                    trainer_version INTEGER,
                    type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT,
                    sql_query TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Stores plugin schema metadata for validation and documentation
            self.execute("""
                CREATE TABLE IF NOT EXISTS plugin_schemas(
                    algorithm_type TEXT NOT NULL,
                    version TEXT NOT NULL,
                    schema_json TEXT,
                    description TEXT,
                    author TEXT,
                    UNIQUE(algorithm_type, version)
                );
            """)

            # New plugin system with graph components
            # Stores metadata for each plugin version
            self.execute("""
                CREATE SEQUENCE IF NOT EXISTS plugins_seq;
            """)

            self.execute("""
                CREATE TABLE IF NOT EXISTS plugins(
                    id BIGINT PRIMARY KEY DEFAULT nextval('plugins_seq'),
                    name VARCHAR(255) NOT NULL,
                    version VARCHAR(50) NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (name, version)
                );
            """)

            # Stores detailed specification for each component
            self.execute("""
                CREATE SEQUENCE IF NOT EXISTS plugin_components_seq;
            """)

            self.execute("""
                CREATE TABLE IF NOT EXISTS plugin_components(
                    id BIGINT PRIMARY KEY DEFAULT nextval('plugin_components_seq'),
                    plugin_id BIGINT NOT NULL,
                    component_name VARCHAR(255) NOT NULL,
                    component_spec TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (plugin_id, component_name)
                );
            """)

            # Create indexes for component lookup performance
            self.execute("""
                CREATE INDEX IF NOT EXISTS idx_plugins_name ON plugins(name);
            """)

            self.execute("""
                CREATE INDEX IF NOT EXISTS idx_plugin_components_name
                ON plugin_components(component_name);
            """)

            # Plans table - stores comprehensive ML workflow plans
            self.execute("""
                CREATE TABLE IF NOT EXISTS plans(
                    plan_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    user_context TEXT NOT NULL,
                    source_tables TEXT NOT NULL,
                    plan_yaml TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Migrate old schema: data_table/target_column -> source_tables
            # Check if old columns exist and migrate
            from contextlib import suppress

            with suppress(Exception):
                # Check if data_table column exists (old schema)
                check_result = self.query("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'plans' AND column_name = 'data_table'
                """)

                if check_result.rows:
                    # Old schema exists, migrate it
                    # Add new column if it doesn't exist
                    with suppress(Exception):
                        self.execute("ALTER TABLE plans ADD COLUMN source_tables TEXT")

                    # Copy data_table to source_tables for existing rows
                    self.execute("""
                        UPDATE plans
                        SET source_tables = data_table
                        WHERE source_tables IS NULL
                    """)

                    # Drop old columns
                    self.execute("ALTER TABLE plans DROP COLUMN data_table")
                    self.execute("ALTER TABLE plans DROP COLUMN target_column")

            # Training tracking tables
            self.execute("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    run_id VARCHAR PRIMARY KEY,
                    job_id VARCHAR,
                    model_id VARCHAR,
                    trainer_id VARCHAR,
                    run_name VARCHAR,
                    description TEXT,
                    tensorboard_enabled BOOLEAN DEFAULT TRUE,
                    tensorboard_log_dir VARCHAR,
                    metric_log_frequency INTEGER DEFAULT 100,
                    checkpoint_frequency INTEGER DEFAULT 5,
                    status VARCHAR DEFAULT 'pending',
                    started_at TIMESTAMP,
                    paused_at TIMESTAMP,
                    resumed_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    artifact_path VARCHAR,
                    final_metrics JSON,
                    original_config JSON,
                    current_config JSON,
                    config_history JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            self.execute("""
                CREATE TABLE IF NOT EXISTS training_metrics (
                    metric_id VARCHAR PRIMARY KEY,
                    run_id VARCHAR,
                    metric_name VARCHAR,
                    metric_type VARCHAR,
                    step INTEGER,
                    epoch INTEGER,
                    value DOUBLE,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            self.execute("""
                CREATE INDEX IF NOT EXISTS idx_training_metrics_run_metric
                ON training_metrics(run_id, metric_name, step);
            """)

            self.execute("""
                CREATE TABLE IF NOT EXISTS training_checkpoints (
                    checkpoint_id VARCHAR PRIMARY KEY,
                    run_id VARCHAR,
                    epoch INTEGER,
                    step INTEGER,
                    checkpoint_path VARCHAR,
                    metrics JSON,
                    is_best BOOLEAN DEFAULT FALSE,
                    file_size_bytes BIGINT,
                    status VARCHAR DEFAULT 'saved',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            self.execute("""
                CREATE INDEX IF NOT EXISTS idx_training_checkpoints_run
                ON training_checkpoints(run_id, epoch);
            """)

        except Exception as e:
            raise DatabaseError(f"Schema initialization failed: {e}") from e

    def __enter__(self) -> "DuckDBDatabase":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensure connection is closed."""
        self.close()

    def __del__(self) -> None:
        """Destructor - ensure connection is closed."""
        self.close()
