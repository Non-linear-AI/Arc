"""ML Data Service for high-level data access during ML operations.

This service provides semantic data access for training/prediction workflows
without exposing SQL. It offers dataset management, feature extraction,
and data validation functionality.
"""

from __future__ import annotations

import contextlib
import re
import time
from typing import Any

import pandas as pd
import torch

from arc.database.base import QueryResult
from arc.database.manager import DatabaseManager
from arc.database.services.base import BaseService


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
        # Row count cache: {table_name: (row_count, timestamp)}
        # Cache TTL: 5 minutes (300 seconds)
        self._row_count_cache: dict[str, tuple[int, float]] = {}
        self._cache_ttl = 300  # 5 minutes

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

    def get_dataset_info(
        self, dataset_name: str, include_row_count: bool = True
    ) -> DatasetInfo | None:
        """Get information about a dataset.

        Args:
            dataset_name: Name of the dataset
            include_row_count: Whether to fetch row count (default: True).
                Set to False to skip potentially slow COUNT(*) queries.

        Returns:
            DatasetInfo object or None if dataset doesn't exist
        """
        if not self._is_valid_table_name(dataset_name):
            return None

        try:
            schema = self._fetch_table_schema(dataset_name)
            if not schema:
                return None

            row_count = 0
            if include_row_count:
                row_count = self._fetch_table_row_count(dataset_name)

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

    def invalidate_row_count_cache(self, table_name: str | None = None) -> None:
        """Invalidate row count cache for a specific table or all tables.

        Args:
            table_name: Name of the table to invalidate. If None, clears entire cache.
        """
        if table_name is None:
            self._row_count_cache.clear()
        elif table_name in self._row_count_cache:
            del self._row_count_cache[table_name]

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

    def get_features_and_targets_paginated(
        self,
        dataset_name: str,
        feature_columns: list[str],
        target_columns: list[str] | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Extract features and targets from a dataset with pagination support.

        Args:
            dataset_name: Name of the dataset
            feature_columns: List of feature column names
            target_columns: Optional list of target column names
            offset: Number of rows to skip from the beginning
            limit: Maximum number of rows to return

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

        # Build query with pagination
        all_columns = feature_columns[:]
        if target_columns:
            all_columns.extend(target_columns)

        columns_str = ", ".join(f'"{col}"' for col in all_columns)
        query = f'SELECT {columns_str} FROM "{dataset_name}"'

        # Add OFFSET and LIMIT for pagination
        if offset > 0:
            query += f" OFFSET {offset}"
        if limit is not None and limit > 0:
            query += f" LIMIT {limit}"

        try:
            result = self.db_manager.user_query(query)

            if not result.rows:
                # Return empty DataFrames with correct columns
                features_df = pd.DataFrame(columns=feature_columns)
                targets_df = None
                if target_columns:
                    targets_df = pd.DataFrame(columns=target_columns)
                return features_df, targets_df

            # Create DataFrame with proper column names
            df = pd.DataFrame(result.rows, columns=all_columns)

            # Split into features and targets
            features_df = df[feature_columns]

            targets_df = None
            if target_columns:
                targets_df = df[target_columns]

            return features_df, targets_df

        except Exception as e:
            raise ValueError(
                f"Failed to extract paginated data from dataset: {e}"
            ) from e

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
        # Validate that feature columns are numeric (not categorical)
        dataset_info = self.get_dataset_info(dataset_name, include_row_count=False)
        if dataset_info:
            categorical_cols = set(dataset_info.categorical_columns)
            categorical_features = [
                col for col in feature_columns if col in categorical_cols
            ]

            if categorical_features:
                # Get column type for better error message
                col_info = next(
                    (
                        c
                        for c in dataset_info.columns
                        if c["name"] == categorical_features[0]
                    ),
                    None,
                )
                col_type = col_info["type"] if col_info else "VARCHAR"

                raise ValueError(
                    f"Cannot convert categorical column '{categorical_features[0]}' "
                    f"(type: {col_type}) to tensor. Categorical features must be "
                    f"encoded as integers before training. Solutions:\n"
                    f"1. Use label encoding in ml_data tool with fit.label_encoder + "
                    f"transform.label_encode operators\n"
                    f"2. Use hash bucketing with transform.hash_bucket operator for "
                    f"high-cardinality features\n"
                    f"3. Use embeddings in model specification "
                    f"(requires encoded integers)"
                )

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
        return self._compute_column_statistics(dataset_name, column_name)

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

    def _fetch_table_schema(self, table_name: str) -> list[dict[str, Any]]:
        """Fetch column metadata for a table."""
        try:
            schema_sql = f"""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
            """

            result = self._user_query(schema_sql)
            return [
                {
                    "name": row["column_name"],
                    "type": row["data_type"],
                    "nullable": row["is_nullable"] == "YES",
                }
                for row in result.rows
            ]

        except Exception:
            return []

    def _fetch_table_row_count(self, table_name: str, use_cache: bool = True) -> int:
        """Fetch the total number of rows for a table.

        Args:
            table_name: Name of the table
            use_cache: Whether to use cached row counts (default: True)

        Returns:
            Number of rows in the table
        """
        # Check cache if enabled
        if use_cache and table_name in self._row_count_cache:
            row_count, timestamp = self._row_count_cache[table_name]
            # Check if cache is still valid
            if time.time() - timestamp < self._cache_ttl:
                return row_count
            # Cache expired, remove entry
            del self._row_count_cache[table_name]

        try:
            count_sql = f'SELECT COUNT(*) AS row_count FROM "{table_name}"'
            result = self._user_query(count_sql)
            if result.rows:
                row_count = result.rows[0].get("row_count", 0)
                # Cache the result
                if use_cache:
                    self._row_count_cache[table_name] = (row_count, time.time())
                return row_count
            return 0
        except Exception:
            return 0

    def _compute_column_statistics(
        self, table_name: str, column_name: str
    ) -> dict[str, Any] | None:
        """Compute statistics for a numeric column."""
        if not self.table_exists(table_name):
            return None

        if not self._is_valid_column_name(column_name):
            return None

        try:
            stats_sql = f'''
            SELECT
                MIN("{column_name}") AS min_val,
                MAX("{column_name}") AS max_val,
                AVG("{column_name}") AS avg_val,
                STDDEV("{column_name}") AS std_val,
                COUNT("{column_name}") AS count_val,
                COUNT(*) AS total_rows
            FROM "{table_name}"
            WHERE "{column_name}" IS NOT NULL
            '''

            result = self._user_query(stats_sql)
            if not result.rows:
                return None

            row = result.rows[0]
            return {
                "min": row.get("min_val"),
                "max": row.get("max_val"),
                "mean": row.get("avg_val"),
                "std": row.get("std_val"),
                "count": row.get("count_val", 0),
                "total_rows": row.get("total_rows", 0),
            }

        except Exception:
            return None

    def get_table_schema(self, table_name: str) -> list[dict[str, Any]]:
        """Get column schema for a table."""
        if not self._is_valid_table_name(table_name):
            return []
        return self._fetch_table_schema(table_name)

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
            check_sql = f"""
            SELECT COUNT(*) AS table_count
            FROM information_schema.tables
            WHERE table_name = '{table_name}'
            """

            result = self._user_query(check_sql)
            if result.rows:
                return result.rows[0].get("table_count", 0) > 0
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

        return self._user_query(sql)

    def get_table_row_count(self, table_name: str) -> int:
        """Get row count for a table.

        Args:
            table_name: Name of the table

        Returns:
            Number of rows in the table
        """
        if not self._is_valid_table_name(table_name):
            return 0
        return self._fetch_table_row_count(table_name)

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
        stats = self._compute_column_statistics(table_name, column_name)
        if stats is None:
            return None

        return {
            "min": stats["min"],
            "max": stats["max"],
            "avg": stats["mean"],
            "count": stats["count"],
            "total_rows": stats["total_rows"],
        }

    def get_tables_list(self) -> list[str]:
        """Get list of all tables in user database.

        Returns:
            List of table names
        """
        return self.list_datasets()

    def sample_table_data(self, table_name: str, limit: int = 100) -> pd.DataFrame:
        """Get sample data from a table as DataFrame.

        Args:
            table_name: Name of the table
            limit: Maximum number of rows to return

        Returns:
            DataFrame with sample data
        """
        return self.sample_data(table_name, limit=limit)

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

    def save_prediction_results(
        self,
        source_table: str,
        output_table: str,
        feature_columns: list[str],
        predictions: dict[str, Any],
        batch_size: int = 1000,
        limit: int | None = None,
    ) -> None:
        """Save prediction results with original features to a new table.

        Streams data in batches to handle large datasets efficiently.

        Args:
            source_table: Name of the source table containing features
            output_table: Name of the output table to create
            feature_columns: List of feature column names from Arc-Graph
            predictions: Dictionary mapping output names to prediction tensors
            batch_size: Number of rows to process per batch
            limit: Optional limit on number of rows to process

        Raises:
            ValueError: If table doesn't exist or columns are invalid
        """
        if not self.dataset_exists(source_table):
            raise ValueError(f"Source table '{source_table}' does not exist")

        # Validate feature columns exist
        validation_result = self.validate_columns(source_table, feature_columns)
        missing_cols = [col for col, exists in validation_result.items() if not exists]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        # Get prediction info
        if not predictions:
            raise ValueError("No predictions provided")

        num_predictions = list(predictions.values())[0].shape[0]
        prediction_columns = list(predictions.keys())

        # Create table schema
        schema_parts = []

        # Add feature columns (get types from source table)
        source_schema = self.get_table_schema(source_table)
        feature_types = {col["name"]: col["type"] for col in source_schema}

        for col in feature_columns:
            col_type = feature_types.get(col, "DOUBLE")
            schema_parts.append(f'"{col}" {col_type}')

        # Add prediction columns
        for pred_col in prediction_columns:
            schema_parts.append(f'"pred_{pred_col}" DOUBLE')

        schema_cols = ", ".join(schema_parts)
        schema = f'CREATE TABLE IF NOT EXISTS "{output_table}" ({schema_cols})'

        try:
            # Drop existing table and create new one
            self.db_manager.user_execute(f'DROP TABLE IF EXISTS "{output_table}"')
            self.db_manager.user_execute(schema)

            # Process data in batches
            total_rows = min(num_predictions, self.get_table_row_count(source_table))
            if limit is not None:
                total_rows = min(total_rows, limit)

            processed = 0

            while processed < total_rows:
                # Calculate batch size for this iteration
                current_batch = min(batch_size, total_rows - processed)

                # Get feature data for this batch
                features_query = f"""
                    SELECT {", ".join(f'"{col}"' for col in feature_columns)}
                    FROM "{source_table}"
                    LIMIT {current_batch} OFFSET {processed}
                """

                feature_result = self.db_manager.user_query(features_query)

                if not feature_result.rows:
                    break

                # Prepare batch data for insertion
                batch_data = []
                for i, feature_row in enumerate(feature_result.rows):
                    prediction_idx = processed + i
                    if prediction_idx >= num_predictions:
                        break

                    # Combine features and predictions
                    row_data = [feature_row[col] for col in feature_columns]

                    # Add prediction values
                    for pred_col in prediction_columns:
                        pred_tensor = predictions[pred_col][prediction_idx]
                        if isinstance(pred_tensor, torch.Tensor):
                            if pred_tensor.dim() == 0 or (
                                pred_tensor.dim() == 1 and pred_tensor.shape[0] == 1
                            ):
                                pred_value = pred_tensor.item()
                            else:
                                # For multi-dimensional outputs, take first element
                                pred_value = float(pred_tensor.flatten()[0].item())
                        else:
                            pred_value = float(pred_tensor)

                        row_data.append(pred_value)

                    batch_data.append(row_data)

                # Insert batch
                if batch_data:
                    for row in batch_data:
                        # Format values for SQL insertion
                        formatted_values = []
                        for value in row:
                            if isinstance(value, str):
                                # Escape single quotes and wrap in quotes
                                escaped_value = value.replace("'", "''")
                                formatted_values.append(f"'{escaped_value}'")
                            elif value is None:
                                formatted_values.append("NULL")
                            else:
                                formatted_values.append(str(value))

                        values_str = ", ".join(formatted_values)
                        insert_sql = (
                            f'INSERT INTO "{output_table}" VALUES ({values_str})'
                        )
                        self.db_manager.user_execute(insert_sql)

                processed += len(batch_data)

        except Exception as e:
            # Clean up on error
            with contextlib.suppress(Exception):
                self.db_manager.user_execute(f'DROP TABLE IF EXISTS "{output_table}"')
            raise ValueError(f"Failed to save prediction results: {e}") from e

    def analyze_target_column(
        self, table_name: str, column_name: str
    ) -> dict[str, Any]:
        """Analyze target column to provide factual statistics for model generation.

        Args:
            table_name: Name of the dataset table
            column_name: Name of the target column to analyze

        Returns:
            Dictionary with target column statistics and facts

        Raises:
            ValueError: If table or column doesn't exist
        """
        if not self.dataset_exists(table_name):
            raise ValueError(f"Table '{table_name}' does not exist")

        if not self._is_valid_column_name(column_name):
            raise ValueError(f"Invalid column name: {column_name}")

        # Validate column exists
        dataset_info = self.get_dataset_info(table_name)
        if column_name not in dataset_info.column_names:
            raise ValueError(
                f"Column '{column_name}' not found in table '{table_name}'"
            )

        try:
            # Get basic column information
            target_col_info = None
            for col in dataset_info.columns:
                if col["name"] == column_name:
                    target_col_info = col
                    break

            # Comprehensive analysis query
            analysis_sql = f'''
            SELECT
                COUNT(*) AS total_count,
                COUNT("{column_name}") AS non_null_count,
                COUNT(*) - COUNT("{column_name}") AS null_count,
                COUNT(DISTINCT "{column_name}") AS unique_count
            FROM "{table_name}"
            '''

            result = self.db_manager.user_query(analysis_sql)
            basic_stats = result.rows[0] if result.rows else {}

            analysis = {
                "total_count": basic_stats.get("total_count", 0),
                "non_null_count": basic_stats.get("non_null_count", 0),
                "null_count": basic_stats.get("null_count", 0),
                "unique_count": basic_stats.get("unique_count", 0),
                "data_type": target_col_info["type"] if target_col_info else "UNKNOWN",
            }

            # Check if column is numeric based on type and values
            is_numeric = self._is_numeric_column_type(target_col_info["type"])

            if is_numeric:
                # Get numeric statistics
                numeric_stats = self._compute_column_statistics(table_name, column_name)
                if numeric_stats:
                    analysis.update(
                        {
                            "is_numeric": True,
                            "min_value": numeric_stats.get("min"),
                            "max_value": numeric_stats.get("max"),
                            "mean_value": numeric_stats.get("mean"),
                            "std_value": numeric_stats.get("std"),
                        }
                    )
                else:
                    analysis["is_numeric"] = False
            else:
                # Get sample categorical values
                sample_values = self.get_unique_values(
                    table_name, column_name, limit=10
                )
                analysis.update(
                    {
                        "is_numeric": False,
                        "sample_values": sample_values,
                    }
                )

            return analysis

        except Exception as e:
            raise ValueError(
                f"Failed to analyze target column '{column_name}': {e}"
            ) from e

    def _is_numeric_column_type(self, data_type: str) -> bool:
        """Check if a data type represents numeric data.

        Args:
            data_type: SQL data type string

        Returns:
            True if numeric type, False otherwise
        """
        numeric_types = [
            "INTEGER",
            "INT",
            "BIGINT",
            "SMALLINT",
            "TINYINT",
            "DOUBLE",
            "FLOAT",
            "REAL",
            "DECIMAL",
            "NUMERIC",
        ]
        return any(nt in data_type.upper() for nt in numeric_types)
