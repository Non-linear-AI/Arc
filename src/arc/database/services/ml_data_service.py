"""ML Data Service for high-level data access during ML operations.

This service provides semantic data access for training/prediction workflows
without exposing SQL. It offers dataset management, feature extraction,
and data validation functionality.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import torch

from ..base import QueryResult
from ..manager import DatabaseManager
from .base import BaseService


class DatasetInfo:
    """Information about a dataset."""

    def __init__(self, name: str, row_count: int, columns: list[dict[str, Any]]):
        self.name = name
        self.row_count = row_count
        self.columns = columns

    @property
    def column_names(self) -> list[str]:
        """Get list of column names."""
        return [col["name"] for col in self.columns]

    @property
    def numeric_columns(self) -> list[str]:
        """Get list of numeric column names."""
        numeric_types = ["INTEGER", "DOUBLE", "FLOAT", "NUMERIC", "DECIMAL", "REAL"]
        return [
            col["name"]
            for col in self.columns
            if any(nt in col["type"].upper() for nt in numeric_types)
        ]

    @property
    def categorical_columns(self) -> list[str]:
        """Get list of categorical/text column names."""
        text_types = ["VARCHAR", "TEXT", "STRING", "CHAR"]
        return [
            col["name"]
            for col in self.columns
            if any(tt in col["type"].upper() for tt in text_types)
        ]


class MLDataService(BaseService):
    """Service for high-level ML data access and processing."""

    def __init__(self, db_manager: DatabaseManager):
        """Initialize ML data service.

        Args:
            db_manager: Database manager instance
        """
        super().__init__(db_manager)

    def list_datasets(self) -> list[str]:
        """Get list of available datasets (tables).

        Returns:
            List of dataset names
        """
        try:
            tables_sql = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_name
            """
            result = self.db_manager.user_query(tables_sql)
            return [row["table_name"] for row in result.rows]
        except Exception:
            return []

    def get_dataset_info(self, dataset_name: str) -> DatasetInfo | None:
        """Get information about a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            DatasetInfo object or None if dataset doesn't exist
        """
        if not self._is_valid_table_name(dataset_name):
            return None

        try:
            # Get schema
            schema = self._get_table_schema(dataset_name)
            if not schema:
                return None

            # Get row count
            row_count = self._get_table_row_count(dataset_name)

            return DatasetInfo(dataset_name, row_count, schema)

        except Exception:
            return None

    def dataset_exists(self, dataset_name: str) -> bool:
        """Check if dataset exists.

        Args:
            dataset_name: Name of the dataset

        Returns:
            True if dataset exists, False otherwise
        """
        return self.get_dataset_info(dataset_name) is not None

    def get_features_and_targets(
        self,
        dataset_name: str,
        feature_columns: list[str],
        target_columns: list[str] | None = None,
        limit: int | None = None,
        sample_fraction: float | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Extract features and targets from a dataset.

        Args:
            dataset_name: Name of the dataset
            feature_columns: List of feature column names
            target_columns: Optional list of target column names
            limit: Maximum number of rows to return
            sample_fraction: Fraction of data to sample (0.0-1.0)

        Returns:
            Tuple of (features_df, targets_df). targets_df is None if no target columns

        Raises:
            ValueError: If dataset or columns don't exist
        """
        if not self.dataset_exists(dataset_name):
            raise ValueError(f"Dataset '{dataset_name}' does not exist")

        # Validate columns exist
        dataset_info = self.get_dataset_info(dataset_name)
        available_columns = dataset_info.column_names

        for col in feature_columns:
            if col not in available_columns:
                raise ValueError(f"Feature column '{col}' not found in dataset")

        if target_columns:
            for col in target_columns:
                if col not in available_columns:
                    raise ValueError(f"Target column '{col}' not found in dataset")

        # Build query
        all_columns = feature_columns[:]
        if target_columns:
            all_columns.extend(target_columns)

        columns_str = ", ".join(f'"{col}"' for col in all_columns)
        query = f'SELECT {columns_str} FROM "{dataset_name}"'

        # Add sampling if requested
        if sample_fraction is not None and 0 < sample_fraction < 1:
            query += f" USING SAMPLE {sample_fraction * 100}%"

        # Add limit if requested
        if limit is not None and limit > 0:
            query += f" LIMIT {limit}"

        try:
            result = self.db_manager.user_query(query)
            df = pd.DataFrame(result.rows)

            # Split into features and targets
            features_df = df[feature_columns]

            targets_df = None
            if target_columns:
                targets_df = df[target_columns]

            return features_df, targets_df

        except Exception as e:
            raise ValueError(f"Failed to extract data from dataset: {e}") from e

    def get_features_as_tensors(
        self,
        dataset_name: str,
        feature_columns: list[str],
        target_columns: list[str] | None = None,
        limit: int | None = None,
        sample_fraction: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Get features and targets as PyTorch tensors.

        Args:
            dataset_name: Name of the dataset
            feature_columns: List of feature column names
            target_columns: Optional list of target column names
            limit: Maximum number of rows to return
            sample_fraction: Fraction of data to sample (0.0-1.0)

        Returns:
            Tuple of (features_tensor, targets_tensor). targets_tensor is None if
            no targets

        Raises:
            ValueError: If dataset/columns don't exist or data isn't numeric
        """
        features_df, targets_df = self.get_features_and_targets(
            dataset_name, feature_columns, target_columns, limit, sample_fraction
        )

        try:
            # Convert features to tensor
            features_tensor = torch.tensor(
                features_df.values.astype(float), dtype=torch.float32
            )

            targets_tensor = None
            if targets_df is not None:
                targets_tensor = torch.tensor(
                    targets_df.values.astype(float), dtype=torch.float32
                )
                # Squeeze single target column
                if targets_tensor.shape[1] == 1:
                    targets_tensor = targets_tensor.squeeze(1)

            return features_tensor, targets_tensor

        except Exception as e:
            raise ValueError(f"Failed to convert data to tensors: {e}") from e

    def get_column_statistics(
        self, dataset_name: str, column_name: str
    ) -> dict[str, Any] | None:
        """Get statistics for a numeric column.

        Args:
            dataset_name: Name of the dataset
            column_name: Name of the column

        Returns:
            Dictionary with min, max, mean, std, count stats or None if error
        """
        if not self.dataset_exists(dataset_name):
            return None

        if not self._is_valid_column_name(column_name):
            return None

        try:
            stats_sql = f'''
            SELECT
                MIN("{column_name}") as min_val,
                MAX("{column_name}") as max_val,
                AVG("{column_name}") as avg_val,
                STDDEV("{column_name}") as std_val,
                COUNT("{column_name}") as count_val,
                COUNT(*) as total_rows
            FROM "{dataset_name}"
            WHERE "{column_name}" IS NOT NULL
            '''

            result = self.db_manager.user_query(stats_sql)
            if not result.rows:
                return None

            row = result.rows[0]
            return {
                "min": row["min_val"],
                "max": row["max_val"],
                "mean": row["avg_val"],
                "std": row["std_val"],
                "count": row["count_val"],
                "total_rows": row["total_rows"],
            }

        except Exception:
            return None

    def get_unique_values(
        self, dataset_name: str, column_name: str, limit: int = 100
    ) -> list[Any]:
        """Get unique values from a categorical column.

        Args:
            dataset_name: Name of the dataset
            column_name: Name of the column
            limit: Maximum number of unique values to return

        Returns:
            List of unique values
        """
        if not self.dataset_exists(dataset_name):
            return []

        if not self._is_valid_column_name(column_name):
            return []

        try:
            unique_sql = f'''
            SELECT DISTINCT "{column_name}"
            FROM "{dataset_name}"
            WHERE "{column_name}" IS NOT NULL
            ORDER BY "{column_name}"
            LIMIT {limit}
            '''

            result = self.db_manager.user_query(unique_sql)
            return [row[column_name] for row in result.rows]

        except Exception:
            return []

    def get_data(
        self,
        dataset_name: str,
        columns: list[str] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Get data from a dataset.

        Args:
            dataset_name: Name of the dataset
            columns: Optional list of columns to include (all if None)
            limit: Optional maximum number of rows to return (all if None)

        Returns:
            DataFrame with data
        """
        if not self.dataset_exists(dataset_name):
            return pd.DataFrame()

        try:
            if columns:
                # Validate columns
                for col in columns:
                    if not self._is_valid_column_name(col):
                        return pd.DataFrame()
                columns_str = ", ".join(f'"{col}"' for col in columns)
            else:
                columns_str = "*"

            query = f'SELECT {columns_str} FROM "{dataset_name}"'
            if limit is not None:
                query += f" LIMIT {limit}"

            result = self.db_manager.user_query(query)
            return pd.DataFrame(result.rows)

        except Exception:
            return pd.DataFrame()

    def sample_data(
        self, dataset_name: str, columns: list[str] | None = None, limit: int = 100
    ) -> pd.DataFrame:
        """Get a sample of data from a dataset.

        Args:
            dataset_name: Name of the dataset
            columns: Optional list of columns to include (all if None)
            limit: Maximum number of rows to return

        Returns:
            DataFrame with sample data
        """
        return self.get_data(dataset_name, columns, limit)

    def validate_columns(
        self, dataset_name: str, columns: list[str]
    ) -> dict[str, bool]:
        """Validate that columns exist in a dataset.

        Args:
            dataset_name: Name of the dataset
            columns: List of column names to validate

        Returns:
            Dictionary mapping column names to existence (True/False)
        """
        dataset_info = self.get_dataset_info(dataset_name)
        if not dataset_info:
            return dict.fromkeys(columns, False)

        available_columns = set(dataset_info.column_names)
        return {col: col in available_columns for col in columns}

    # Internal helper methods

    def _get_table_schema(self, table_name: str) -> list[dict[str, Any]]:
        """Get table schema information."""
        try:
            schema_sql = f"""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
            """

            result = self.db_manager.user_query(schema_sql)

            schema = []
            for row in result.rows:
                schema.append(
                    {
                        "name": row["column_name"],
                        "type": row["data_type"],
                        "nullable": row["is_nullable"] == "YES",
                    }
                )

            return schema

        except Exception:
            return []

    def _get_table_row_count(self, table_name: str) -> int:
        """Get row count for a table."""
        try:
            count_sql = f'SELECT COUNT(*) FROM "{table_name}"'
            result = self.db_manager.user_query(count_sql)
            if result.rows:
                count_key = list(result.rows[0].keys())[0]
                return result.rows[0][count_key]
            return 0
        except Exception:
            return 0

    def create_temp_table(self, table_name: str, sql: str) -> bool:
        """Create temporary table for plugin processing.

        Args:
            table_name: Name for the temporary table
            sql: SQL query to populate the table

        Returns:
            True if successful, False otherwise
        """
        if not self.is_valid_sql(sql):
            return False

        if not self._is_valid_table_name(table_name):
            return False

        try:
            # Drop existing temp table if it exists
            self.drop_temp_table(table_name)

            # Create new temp table
            create_sql = f"CREATE TEMP TABLE {table_name} AS {sql}"
            self.db_manager.user_execute(create_sql)
            return True

        except Exception:
            return False

    def drop_temp_table(self, table_name: str) -> bool:
        """Drop temporary table.

        Args:
            table_name: Name of table to drop

        Returns:
            True if successful, False otherwise
        """
        if not self._is_valid_table_name(table_name):
            return False

        try:
            # Check if table exists first
            if self.table_exists(table_name):
                drop_sql = f"DROP TABLE IF EXISTS {table_name}"
                self.db_manager.user_execute(drop_sql)
            return True

        except Exception:
            return False

    def get_table_schema(self, table_name: str) -> list[dict[str, Any]]:
        """Get column schema for a table.

        Args:
            table_name: Name of the table

        Returns:
            List of column specifications with name, type, nullable info
        """
        if not self._is_valid_table_name(table_name):
            return []

        try:
            # Use DuckDB's information schema
            schema_sql = f"""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
            """

            result = self.db_manager.user_query(schema_sql)

            schema = []
            for row in result.rows:
                schema.append(
                    {
                        "name": row["column_name"],
                        "type": row["data_type"],
                        "nullable": row["is_nullable"] == "YES",
                    }
                )

            return schema

        except Exception:
            return []

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists in user database.

        Args:
            table_name: Name of table to check

        Returns:
            True if table exists, False otherwise
        """
        if not self._is_valid_table_name(table_name):
            return False

        try:
            # Check in information schema
            check_sql = f"""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = '{table_name}'
            """

            result = self.db_manager.user_query(check_sql)
            if result.rows:
                count_key = list(result.rows[0].keys())[0]
                return result.rows[0][count_key] > 0
            return False

        except Exception:
            return False

    def fetch_columns(self, table_name: str, columns: list[str]) -> QueryResult:
        """Fetch specific columns from a table.

        Args:
            table_name: Name of the table
            columns: List of column names to fetch

        Returns:
            Query result with requested columns

        Raises:
            ValueError: If table or columns are invalid
        """
        if not self._is_valid_table_name(table_name):
            raise ValueError(f"Invalid table name: {table_name}")

        if not columns:
            raise ValueError("Column list cannot be empty")

        # Validate column names
        for col in columns:
            if not self._is_valid_column_name(col):
                raise ValueError(f"Invalid column name: {col}")

        if not self.table_exists(table_name):
            raise ValueError(f"Table does not exist: {table_name}")

        # Build SELECT query
        columns_str = ", ".join(f'"{col}"' for col in columns)
        sql = f'SELECT {columns_str} FROM "{table_name}"'

        return self.execute_sql(sql)

    def get_table_row_count(self, table_name: str) -> int:
        """Get row count for a table.

        Args:
            table_name: Name of the table

        Returns:
            Number of rows in the table
        """
        if not self.table_exists(table_name):
            return 0

        try:
            count_sql = f'SELECT COUNT(*) FROM "{table_name}"'
            result = self.execute_sql(count_sql)
            if result.rows:
                count_key = list(result.rows[0].keys())[0]
                return result.rows[0][count_key]
            return 0

        except Exception:
            return 0

    def get_column_stats(
        self, table_name: str, column_name: str
    ) -> dict[str, Any] | None:
        """Get basic statistics for a numeric column.

        Args:
            table_name: Name of the table
            column_name: Name of the column

        Returns:
            Dictionary with min, max, avg, count stats or None if error
        """
        if not self.table_exists(table_name):
            return None

        if not self._is_valid_column_name(column_name):
            return None

        try:
            stats_sql = f'''
            SELECT
                MIN("{column_name}") as min_val,
                MAX("{column_name}") as max_val,
                AVG("{column_name}") as avg_val,
                COUNT("{column_name}") as count_val,
                COUNT(*) as total_rows
            FROM "{table_name}"
            WHERE "{column_name}" IS NOT NULL
            '''

            result = self.execute_sql(stats_sql)
            if not result.rows:
                return None

            row = result.rows[0]
            return {
                "min": row[0],
                "max": row[1],
                "avg": row[2],
                "count": row[3],
                "total_rows": row[4],
            }

        except Exception:
            return None

    def get_tables_list(self) -> list[str]:
        """Get list of all tables in user database.

        Returns:
            List of table names
        """
        try:
            tables_sql = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_name
            """

            result = self.db_manager.user_query(tables_sql)
            return [row[0] for row in result.rows]

        except Exception:
            return []

    def sample_table_data(self, table_name: str, limit: int = 100) -> pd.DataFrame:
        """Get sample data from a table as DataFrame.

        Args:
            table_name: Name of the table
            limit: Maximum number of rows to return

        Returns:
            DataFrame with sample data
        """
        if not self.table_exists(table_name):
            return pd.DataFrame()

        try:
            sample_sql = f'SELECT * FROM "{table_name}" LIMIT {limit}'
            result = self.execute_sql(sample_sql)

            if not result.rows:
                return pd.DataFrame()

            # Get column names from schema
            schema = self.get_table_schema(table_name)
            columns = [col["name"] for col in schema]

            return pd.DataFrame(result.rows, columns=columns)

        except Exception:
            return pd.DataFrame()

    def _is_user_data_query(self, sql: str) -> bool:
        """Check if SQL query only accesses user data tables.

        Args:
            sql: SQL query to check

        Returns:
            True if query only accesses user data, False otherwise
        """
        sql_lower = sql.lower()

        # Block access to system tables/schemas
        system_patterns = [
            r"\bsystem\.",
            r"\binformation_schema\.",
            r"\bpg_catalog\.",
            r"\bsqlite_master\b",
            r"\bsqlite_temp_master\b",
        ]

        return all(not re.search(pattern, sql_lower) for pattern in system_patterns)

    def _is_valid_table_name(self, table_name: str) -> bool:
        """Validate table name for security.

        Args:
            table_name: Table name to validate

        Returns:
            True if valid, False otherwise
        """
        if not table_name or not table_name.strip():
            return False

        # Allow alphanumeric, underscores, and dots
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_\.]*$", table_name):
            return False

        # Block suspicious patterns
        suspicious = ["..", "__", "system", "information_schema", "pg_"]
        return not any(sus in table_name.lower() for sus in suspicious)

    def _is_valid_column_name(self, column_name: str) -> bool:
        """Validate column name for security.

        Args:
            column_name: Column name to validate

        Returns:
            True if valid, False otherwise
        """
        if not column_name or not column_name.strip():
            return False

        # Allow alphanumeric, underscores
        return re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", column_name) is not None
