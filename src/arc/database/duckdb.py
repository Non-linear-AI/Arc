"""DuckDB database implementation."""

import time
from contextlib import suppress
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
            self._setup_s3_extensions()
            self._setup_snowflake_extensions()
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

    def _setup_s3_extensions(self) -> None:
        """Setup S3 extensions and configure credentials if available.

        Loads httpfs, aws, and iceberg extensions for S3 support.
        If S3 credentials are configured, creates a DuckDB secret for authentication.
        """
        if self._connection is None:
            return

        try:
            # Install and load httpfs extension for S3 access
            self._connection.execute("INSTALL httpfs")
            self._connection.execute("LOAD httpfs")
        except Exception:
            # Silent failure - httpfs is optional
            pass

        try:
            # Install and load aws extension for authentication
            self._connection.execute("INSTALL aws")
            self._connection.execute("LOAD aws")
        except Exception:
            # Silent failure - aws is optional
            pass

        try:
            # Install and load iceberg extension for Iceberg support
            self._connection.execute("INSTALL iceberg")
            self._connection.execute("LOAD iceberg")
        except Exception:
            # Silent failure - iceberg is optional
            pass

        # Configure S3 credentials if available
        self._configure_s3_credentials()

    def _configure_s3_credentials(self) -> None:
        """Configure S3 credentials using credential chain (IAM roles) or explicit
        config.

        Uses credential_chain provider by default to enable:
        - IAM roles (EC2 instance profiles, ECS task roles, Lambda execution roles)
        - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        - AWS credential files (~/.aws/credentials)

        Falls back to config provider only for custom S3-compatible endpoints
        (MinIO, etc.)
        """
        if self._connection is None:
            return

        # Import here to avoid circular dependency
        from arc.core.config import SettingsManager

        try:
            settings_manager = SettingsManager()
            s3_config = settings_manager.get_s3_config()

            # Check if custom endpoint is configured (MinIO, Wasabi, etc.)
            if s3_config and s3_config.get("endpoint"):
                # Custom endpoint requires config provider with explicit credentials
                secret_parts = [
                    "CREATE OR REPLACE SECRET (",
                    "TYPE s3,",
                    "PROVIDER config",
                ]

                if s3_config.get("access_key_id"):
                    secret_parts.append(f", KEY_ID '{s3_config['access_key_id']}'")

                if s3_config.get("secret_access_key"):
                    secret_parts.append(f", SECRET '{s3_config['secret_access_key']}'")

                if s3_config.get("region"):
                    secret_parts.append(f", REGION '{s3_config['region']}'")

                secret_parts.append(f", ENDPOINT '{s3_config['endpoint']}'")
                secret_parts.append(")")

                secret_sql = " ".join(secret_parts)
                self._connection.execute(secret_sql)
            else:
                # Use credential_chain for automatic credential discovery
                # This enables IAM roles, env vars, ~/.aws/credentials, etc.
                # If credential_chain creation fails (no credentials available),
                # we don't create any secret - this allows anonymous access to
                # public S3 buckets
                try:
                    self._connection.execute(
                        "CREATE OR REPLACE SECRET (TYPE s3, PROVIDER credential_chain)"
                    )

                    # Optionally override region if specified in config
                    if s3_config and s3_config.get("region"):
                        region_override = (
                            "CREATE OR REPLACE SECRET ("
                            "TYPE s3, "
                            "PROVIDER credential_chain, "
                            f"REGION '{s3_config['region']}'"
                            ")"
                        )
                        self._connection.execute(region_override)
                except Exception:
                    # credential_chain creation failed - no credentials available
                    # This is OK: public S3 buckets will use anonymous access
                    pass

        except Exception:
            # Silent failure - S3 credentials are optional
            pass

    def _setup_snowflake_extensions(self) -> None:
        """Setup Snowflake extensions and auto-attach if configured.

        Loads DuckDB Snowflake extension from community repository.
        If Snowflake credentials are configured, creates a secret and attaches
        the Snowflake database to make it accessible as a schema in DuckDB.
        """
        if self._connection is None:
            return

        try:
            # Install and load Snowflake extension from community
            self._connection.execute("INSTALL snowflake FROM community")
            self._connection.execute("LOAD snowflake")
        except Exception:
            # Silent failure - Snowflake extension is optional
            pass

        # Configure Snowflake credentials and auto-attach if available
        self._configure_snowflake_credentials()

    def _configure_snowflake_credentials(self) -> None:
        """Configure Snowflake credentials and auto-attach database if configured.

        Creates a DuckDB secret for Snowflake authentication and automatically
        attaches the Snowflake database as a read-only schema in DuckDB.

        The attached schema allows querying Snowflake tables directly:
        - SELECT * FROM snowflake.public.customers
        - CREATE TABLE local AS SELECT * FROM snowflake.public.orders

        Snowflake tables appear in INFORMATION_SCHEMA, making them discoverable
        via schema discovery tools without any code changes.
        """
        if self._connection is None:
            return

        # Import here to avoid circular dependency
        from arc.core.config import SettingsManager

        try:
            settings_manager = SettingsManager()
            snowflake_config = settings_manager.get_snowflake_config()

            # Skip if Snowflake not configured
            if not snowflake_config:
                return

            # Create Snowflake secret for authentication
            secret_sql = f"""
                CREATE OR REPLACE SECRET arc_snowflake_secret (
                    TYPE snowflake,
                    ACCOUNT '{snowflake_config["account"]}',
                    USER '{snowflake_config["user"]}',
                    PASSWORD '{snowflake_config["password"]}',
                    DATABASE '{snowflake_config["database"]}',
                    WAREHOUSE '{snowflake_config["warehouse"]}'
                )
            """
            self._connection.execute(secret_sql)

            # Auto-attach Snowflake database as read-only schema
            # This makes Snowflake tables appear in DuckDB's catalog
            attach_sql = """
                ATTACH '' AS snowflake (
                    TYPE snowflake,
                    SECRET arc_snowflake_secret,
                    READ_ONLY
                )
            """
            self._connection.execute(attach_sql)

        except Exception:
            # Silent failure - Snowflake is optional
            # If attach fails, users can manually attach using /sql
            # See docs/snowflake-setup.md for troubleshooting
            pass

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

            # Registry for evaluator specifications linked to models
            self.execute("""
                CREATE TABLE IF NOT EXISTS evaluators(
                    id TEXT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    version INTEGER NOT NULL,
                    model_id TEXT NOT NULL,
                    model_version INTEGER NOT NULL,
                    spec TEXT NOT NULL,
                    description TEXT,
                    plan_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (name, version)
                );
            """)

            # Create indexes for evaluator lookups
            self.execute("""
                CREATE INDEX IF NOT EXISTS idx_evaluators_name ON evaluators(name);
            """)

            self.execute("""
                CREATE INDEX IF NOT EXISTS idx_evaluators_model_id
                ON evaluators(model_id);
            """)

            self.execute("""
                CREATE INDEX IF NOT EXISTS idx_evaluators_name_version
                ON evaluators(name, version);
            """)

            # Tracks long-running processes like training
            self.execute("""
                CREATE TABLE IF NOT EXISTS jobs(
                    job_id TEXT PRIMARY KEY,
                    model_id TEXT,
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

            # Data processors table - stores versioned data processing pipelines
            self.execute("""
                CREATE TABLE IF NOT EXISTS data_processors(
                    id TEXT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    version INTEGER NOT NULL,
                    spec TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (name, version)
                );
            """)

            # Create indexes for data processor lookups
            self.execute("""
                CREATE INDEX IF NOT EXISTS idx_data_processors_name
                ON data_processors(name);
            """)

            self.execute("""
                CREATE INDEX IF NOT EXISTS idx_data_processors_name_version
                ON data_processors(name, version);
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

            # Migrate plans table to add name column (for name-based versioning)
            with suppress(Exception):
                # Check if name column exists
                check_result = self.query("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'plans' AND column_name = 'name'
                """)

                if not check_result.rows:
                    # name column doesn't exist, add it
                    self.execute("ALTER TABLE plans ADD COLUMN name TEXT")

                    # For existing plans, derive name from plan_id (remove -vN suffix)
                    self.execute("""
                        UPDATE plans
                        SET name = REGEXP_REPLACE(plan_id, '-v[0-9]+$', '')
                        WHERE name IS NULL
                    """)

            # Training tracking tables
            self.execute("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    run_id VARCHAR PRIMARY KEY,
                    job_id VARCHAR,
                    model_id VARCHAR,
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
                    training_config JSON,
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

            # Evaluation runs table - tracks evaluation executions
            self.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id VARCHAR PRIMARY KEY,
                    evaluator_id VARCHAR NOT NULL,
                    job_id VARCHAR,
                    model_id VARCHAR,
                    dataset VARCHAR,
                    target_column VARCHAR,
                    status VARCHAR DEFAULT 'pending',
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    metrics_result JSON,
                    prediction_table VARCHAR,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            self.execute("""
                CREATE INDEX IF NOT EXISTS idx_evaluation_runs_evaluator
                ON evaluation_runs(evaluator_id, created_at);
            """)

            self.execute("""
                CREATE INDEX IF NOT EXISTS idx_evaluation_runs_model
                ON evaluation_runs(model_id, created_at);
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
