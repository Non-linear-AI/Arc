"""Built-in data processors for common transformations."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch

from ...database.base import Database
from .base import NormalizationProcessor, ProcessorError, StatefulProcessor


class StandardNormalizationProcessor(NormalizationProcessor):
    """Z-score normalization using database-wide mean and standard deviation.

    Learns global statistics from the entire database to ensure consistent
    normalization across all data splits.
    """

    def fit(self, database: Database, config: dict[str, Any]) -> None:
        """Learn normalization parameters from database.

        Args:
            database: Database connection
            config: Must contain 'table_name' and 'columns' keys

        Raises:
            ProcessorError: If fitting fails
        """
        table_name = config.get("table_name")
        columns = config.get("columns")

        if not table_name or not columns:
            raise ProcessorError("Config must contain 'table_name' and 'columns' keys")

        try:
            stats = self._compute_database_stats(database, table_name, columns)

            # Extract mean and std for each column
            means = []
            stds = []
            for col in columns:
                mean_val = stats[f"{col}_mean"]
                std_val = stats[f"{col}_std"]

                # Handle zero standard deviation
                if std_val is None or std_val <= self.eps:
                    std_val = 1.0

                means.append(mean_val)
                stds.append(std_val)

            self.state = {
                "mean": torch.tensor(means, dtype=torch.float32).unsqueeze(0),
                "std": torch.tensor(stds, dtype=torch.float32).unsqueeze(0),
                "columns": columns,
            }
            self.is_fitted = True

        except Exception as e:
            raise ProcessorError(f"Failed to fit StandardNormalization: {e}") from e

    def transform(self, _data: torch.Tensor) -> torch.Tensor:
        """Apply z-score normalization to data.

        Args:
            data: Input data tensor [batch_size, num_features]

        Returns:
            Normalized data tensor

        Raises:
            ProcessorError: If processor not fitted or data invalid
        """
        self._check_fitted()
        self._validate_data(_data)

        expected_features = len(self.state["columns"])
        if _data.shape[1] != expected_features:
            raise ProcessorError(
                f"Expected {expected_features} features, got {_data.shape[1]}"
            )

        mean = self.state["mean"].to(_data.device)
        std = self.state["std"].to(_data.device)

        return (_data - mean) / std


class MinMaxNormalizationProcessor(NormalizationProcessor):
    """Min-max scaling using database-wide minimum and maximum values.

    Scales features to a specified range (default [0, 1]).
    """

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0), **kwargs):
        """Initialize min-max processor.

        Args:
            feature_range: Desired range for transformed features
            **kwargs: Additional configuration parameters
        """
        super().__init__(feature_range=feature_range, **kwargs)
        self.feature_range = feature_range

    def fit(self, database: Database, config: dict[str, Any]) -> None:
        """Learn min-max parameters from database.

        Args:
            database: Database connection
            config: Must contain 'table_name' and 'columns' keys
        """
        table_name = config.get("table_name")
        columns = config.get("columns")

        if not table_name or not columns:
            raise ProcessorError("Config must contain 'table_name' and 'columns' keys")

        try:
            stats = self._compute_database_stats(database, table_name, columns)

            mins = []
            maxs = []
            for col in columns:
                min_val = stats[f"{col}_min"]
                max_val = stats[f"{col}_max"]

                # Handle case where min == max
                if max_val - min_val <= self.eps:
                    max_val = min_val + 1.0

                mins.append(min_val)
                maxs.append(max_val)

            self.state = {
                "min": torch.tensor(mins, dtype=torch.float32).unsqueeze(0),
                "max": torch.tensor(maxs, dtype=torch.float32).unsqueeze(0),
                "feature_range": self.feature_range,
                "columns": columns,
            }
            self.is_fitted = True

        except Exception as e:
            raise ProcessorError(f"Failed to fit MinMaxNormalization: {e}") from e

    def transform(self, _data: torch.Tensor) -> torch.Tensor:
        """Apply min-max scaling to data.

        Args:
            data: Input data tensor [batch_size, num_features]

        Returns:
            Scaled data tensor
        """
        self._check_fitted()
        self._validate_data(_data)

        expected_features = len(self.state["columns"])
        if _data.shape[1] != expected_features:
            raise ProcessorError(
                f"Expected {expected_features} features, got {_data.shape[1]}"
            )

        min_vals = self.state["min"].to(_data.device)
        max_vals = self.state["max"].to(_data.device)
        feature_min, feature_max = self.state["feature_range"]

        # Scale to [0, 1] then to desired range
        normalized = (_data - min_vals) / (max_vals - min_vals)
        scaled = normalized * (feature_max - feature_min) + feature_min

        return scaled


class RobustNormalizationProcessor(StatefulProcessor):
    """Robust scaling using median and interquartile range.

    Less sensitive to outliers than standard normalization.
    """

    def fit(self, database: Database, config: dict[str, Any]) -> None:
        """Learn robust scaling parameters from database.

        Args:
            database: Database connection
            config: Must contain 'table_name' and 'columns' keys
        """
        table_name = config.get("table_name")
        columns = config.get("columns")

        if not table_name or not columns:
            raise ProcessorError("Config must contain 'table_name' and 'columns' keys")

        try:
            # Query to get all data for quantile computation
            columns_str = ", ".join(
                [f"CAST({col} AS DOUBLE) AS {col}" for col in columns]
            )
            query = f"SELECT {columns_str} FROM {table_name}"
            result = database.query(query)

            if not result.rows:
                raise ProcessorError(f"No data found in table {table_name}")

            # Convert to DataFrame for quantile computation
            df = pd.DataFrame(result.rows, columns=columns)

            medians = []
            scales = []

            for col in columns:
                values = df[col].dropna()
                median_val = float(values.median())
                q25 = float(values.quantile(0.25))
                q75 = float(values.quantile(0.75))
                iqr = q75 - q25

                # Avoid division by zero
                if iqr <= 1e-8:
                    iqr = 1.0

                medians.append(median_val)
                scales.append(iqr)

            self.state = {
                "median": torch.tensor(medians, dtype=torch.float32).unsqueeze(0),
                "scale": torch.tensor(scales, dtype=torch.float32).unsqueeze(0),
                "columns": columns,
            }
            self.is_fitted = True

        except Exception as e:
            raise ProcessorError(f"Failed to fit RobustNormalization: {e}") from e

    def transform(self, _data: torch.Tensor) -> torch.Tensor:
        """Apply robust scaling to data.

        Args:
            data: Input data tensor [batch_size, num_features]

        Returns:
            Scaled data tensor
        """
        self._check_fitted()
        self._validate_data(_data)

        expected_features = len(self.state["columns"])
        if _data.shape[1] != expected_features:
            raise ProcessorError(
                f"Expected {expected_features} features, got {_data.shape[1]}"
            )

        median = self.state["median"].to(_data.device)
        scale = self.state["scale"].to(_data.device)

        return (_data - median) / scale


class CategoricalEncodingProcessor(StatefulProcessor):
    """One-hot encoding for categorical variables.

    Learns unique categories from the database and applies consistent
    encoding across all data.
    """

    def __init__(self, handle_unknown: str = "ignore", **kwargs):
        """Initialize categorical encoder.

        Args:
            handle_unknown: How to handle unknown categories ('ignore' or 'error')
            **kwargs: Additional configuration parameters
        """
        super().__init__(handle_unknown=handle_unknown, **kwargs)
        self.handle_unknown = handle_unknown

    def fit(self, database: Database, config: dict[str, Any]) -> None:
        """Learn categorical mappings from database.

        Args:
            database: Database connection
            config: Must contain 'table_name' and 'columns' keys
        """
        table_name = config.get("table_name")
        columns = config.get("columns")

        if not table_name or not columns:
            raise ProcessorError("Config must contain 'table_name' and 'columns' keys")

        try:
            category_mappings = {}

            for col in columns:
                # Get unique values for this column
                query = (
                    f"SELECT DISTINCT {col} FROM {table_name} WHERE {col} IS NOT NULL"
                )
                result = database.query(query)

                if not result.rows:
                    raise ProcessorError(f"No data found for column {col}")

                # Sort categories for consistent ordering
                categories = sorted([str(row[0]) for row in result.rows])
                category_mappings[col] = {
                    cat: idx for idx, cat in enumerate(categories)
                }

            self.state = {
                "category_mappings": category_mappings,
                "columns": columns,
                "num_categories": {
                    col: len(mapping) for col, mapping in category_mappings.items()
                },
            }
            self.is_fitted = True

        except Exception as e:
            raise ProcessorError(f"Failed to fit CategoricalEncoding: {e}") from e

    def transform(self, _data: torch.Tensor) -> torch.Tensor:
        """Apply one-hot encoding to categorical data.

        Note: This method expects string data, so it should be used
        with a special categorical dataset that preserves string values.

        Args:
            data: Input categorical data (as string representations)

        Returns:
            One-hot encoded tensor

        Raises:
            ProcessorError: This method needs special handling for categorical data
        """
        raise ProcessorError(
            "CategoricalEncodingProcessor requires special handling for string data. "
            "Use transform_categorical() method instead."
        )

    def transform_categorical(self, categorical_data: list[list[str]]) -> torch.Tensor:
        """Transform categorical data to one-hot encoding.

        Args:
            categorical_data: List of rows, each row is a list of categorical values

        Returns:
            One-hot encoded tensor
        """
        self._check_fitted()

        if not categorical_data:
            raise ProcessorError("Empty categorical data provided")

        columns = self.state["columns"]
        category_mappings = self.state["category_mappings"]

        encoded_rows = []
        for row in categorical_data:
            if len(row) != len(columns):
                raise ProcessorError(
                    f"Expected {len(columns)} categorical values, got {len(row)}"
                )

            encoded_row = []
            for _col_idx, (col, value) in enumerate(zip(columns, row, strict=False)):
                mapping = category_mappings[col]
                num_cats = len(mapping)

                # Create one-hot vector for this categorical value
                one_hot = torch.zeros(num_cats)

                if str(value) in mapping:
                    one_hot[mapping[str(value)]] = 1.0
                elif self.handle_unknown == "error":
                    raise ProcessorError(
                        f"Unknown category '{value}' for column '{col}'"
                    )
                # If handle_unknown == "ignore", leave as all zeros

                encoded_row.append(one_hot)

            # Concatenate all one-hot vectors for this row
            encoded_rows.append(torch.cat(encoded_row))

        return torch.stack(encoded_rows)

    def get_feature_names(self) -> list[str]:
        """Get feature names for the encoded output.

        Returns:
            List of feature names in the encoded output
        """
        self._check_fitted()

        feature_names = []
        for col in self.state["columns"]:
            mapping = self.state["category_mappings"][col]
            categories = sorted(mapping.keys(), key=lambda x: mapping[x])
            for cat in categories:
                feature_names.append(f"{col}_{cat}")

        return feature_names
