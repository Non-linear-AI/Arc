"""Base classes for data processors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from arc.database.base import Database


class ProcessorError(Exception):
    """Base exception for processor-related errors."""


class StatefulProcessor(ABC):
    """Base class for processors that learn parameters from data.

    Processors can learn statistics and transformations from the entire
    database, then apply these learned transformations consistently to
    new data.
    """

    def __init__(self, **kwargs):
        """Initialize processor with configuration parameters.

        Args:
            **kwargs: Processor-specific configuration parameters
        """
        self.config = kwargs
        self.is_fitted = False
        self.state: dict[str, Any] = {}

    @abstractmethod
    def fit(self, database: Database, config: dict[str, Any]) -> None:
        """Learn parameters from database.

        Args:
            database: Database connection to learn from
            config: Configuration including table_name, columns, etc.

        Raises:
            ProcessorError: If fitting fails
        """

    @abstractmethod
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply learned transformation to data.

        Args:
            data: Input data tensor

        Returns:
            Transformed data tensor

        Raises:
            ProcessorError: If processor not fitted or transformation fails
        """

    def fit_transform(
        self, database: Database, config: dict[str, Any], data: torch.Tensor
    ) -> torch.Tensor:
        """Fit processor and transform data in one step.

        Args:
            database: Database connection
            config: Fit configuration
            data: Data to transform

        Returns:
            Transformed data
        """
        self.fit(database, config)
        return self.transform(data)

    def save_state(self) -> dict[str, Any]:
        """Save learned state for persistence.

        Returns:
            Dictionary containing processor state
        """
        if not self.is_fitted:
            raise ProcessorError("Cannot save state of unfitted processor")

        return {
            "config": self.config,
            "state": self.state,
            "processor_type": self.__class__.__name__,
        }

    def load_state(self, saved_state: dict[str, Any]) -> None:
        """Load previously saved processor state.

        Args:
            saved_state: Previously saved state dictionary

        Raises:
            ProcessorError: If state is invalid or incompatible
        """
        if saved_state.get("processor_type") != self.__class__.__name__:
            raise ProcessorError(
                f"State type mismatch: expected {self.__class__.__name__}, "
                f"got {saved_state.get('processor_type')}"
            )

        self.config = saved_state["config"]
        self.state = saved_state["state"]
        self.is_fitted = True

    def _check_fitted(self) -> None:
        """Check if processor is fitted, raise error if not."""
        if not self.is_fitted:
            raise ProcessorError(
                f"Processor {self.__class__.__name__} must be fitted before use"
            )

    def _validate_data(self, data: torch.Tensor) -> None:
        """Validate input data tensor.

        Args:
            data: Input data to validate

        Raises:
            ProcessorError: If data is invalid
        """
        if not isinstance(data, torch.Tensor):
            raise ProcessorError("Input data must be a torch.Tensor")

        if data.dim() != 2:
            raise ProcessorError(
                f"Input data must be 2D [batch_size, features], got {data.dim()}D"
            )


class NormalizationProcessor(StatefulProcessor):
    """Base class for normalization processors."""

    def __init__(self, eps: float = 1e-8, **kwargs):
        """Initialize normalization processor.

        Args:
            eps: Small constant to avoid division by zero
            **kwargs: Additional configuration parameters
        """
        super().__init__(eps=eps, **kwargs)
        self.eps = eps

    def _compute_database_stats(
        self, database: Database, table_name: str, columns: list[str]
    ) -> dict[str, torch.Tensor]:
        """Compute statistics from database.

        Args:
            database: Database connection
            table_name: Name of table containing data
            columns: List of column names to compute stats for

        Returns:
            Dictionary containing computed statistics

        Raises:
            ProcessorError: If database query fails
        """
        try:
            # Build query to compute statistics
            stat_queries = []
            for col in columns:
                stat_queries.extend(
                    [
                        f"AVG(CAST({col} AS DOUBLE)) AS {col}_mean",
                        f"STDDEV(CAST({col} AS DOUBLE)) AS {col}_std",
                        f"MIN(CAST({col} AS DOUBLE)) AS {col}_min",
                        f"MAX(CAST({col} AS DOUBLE)) AS {col}_max",
                    ]
                )

            query = f"SELECT {', '.join(stat_queries)} FROM {table_name}"
            result = database.query(query)

            if not result.rows:
                raise ProcessorError(f"No data found in table {table_name}")

            row = result.rows[0]
            stats = {}

            for i, col in enumerate(columns):
                stats[f"{col}_mean"] = float(row[i * 4])
                stats[f"{col}_std"] = float(row[i * 4 + 1])
                stats[f"{col}_min"] = float(row[i * 4 + 2])
                stats[f"{col}_max"] = float(row[i * 4 + 3])

            return stats

        except Exception as e:
            raise ProcessorError(f"Failed to compute database statistics: {e}") from e
