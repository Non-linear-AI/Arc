"""Data loading and preprocessing utilities for Arc ML."""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..database.base import Database


class ArcDataset(Dataset):
    """PyTorch Dataset for Arc-Graph data."""

    def __init__(self, features: torch.Tensor, targets: torch.Tensor | None = None):
        """Initialize dataset.

        Args:
            features: Feature tensor of shape [num_samples, num_features]
            targets: Optional target tensor of shape [num_samples, ...] for training
        """
        self.features = features
        self.targets = targets

        if targets is not None and len(features) != len(targets):
            raise ValueError(
                f"Features and targets must have same length: "
                f"{len(features)} vs {len(targets)}"
            )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]


class DataProcessor:
    """Processes data for Arc-Graph models."""

    def __init__(self, database: Database | None = None):
        self.database = database

    def load_from_table(
        self,
        table_name: str,
        feature_columns: list[str],
        target_columns: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Load data from database table.

        Args:
            table_name: Name of database table
            feature_columns: List of feature column names
            target_columns: Optional list of target column names

        Returns:
            Tuple of (features, targets) tensors

        Raises:
            RuntimeError: If database not provided
            ValueError: If columns not found or data invalid
        """
        if not self.database:
            raise RuntimeError("Database connection required to load from table")

        # Build query
        all_columns = feature_columns[:]
        if target_columns:
            all_columns.extend(target_columns)

        columns_str = ", ".join(all_columns)
        query = f"SELECT {columns_str} FROM {table_name}"

        # Execute query
        result = self.database.query(query)
        if not result.rows:
            raise ValueError(f"No data found in table {table_name}")

        # Convert to DataFrame for easier processing
        df = pd.DataFrame(result.rows)

        # Extract features
        try:
            feature_data = df[feature_columns].values.astype(float)
            features = torch.tensor(feature_data, dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"Failed to process feature columns: {e}") from e

        # Extract targets if provided
        targets = None
        if target_columns:
            try:
                target_data = df[target_columns].values.astype(float)
                targets = torch.tensor(target_data, dtype=torch.float32)
                if len(target_columns) == 1:
                    targets = targets.squeeze(
                        1
                    )  # Remove extra dimension for single target
            except Exception as e:
                raise ValueError(f"Failed to process target columns: {e}") from e

        return features, targets

    def create_dataloader(
        self,
        features: torch.Tensor,
        targets: torch.Tensor | None = None,
        batch_size: int = 32,
        shuffle: bool = True,
        **dataloader_kwargs,
    ) -> DataLoader:
        """Create PyTorch DataLoader from tensors.

        Args:
            features: Feature tensor
            targets: Optional target tensor
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle data
            **dataloader_kwargs: Additional arguments for DataLoader

        Returns:
            PyTorch DataLoader
        """
        dataset = ArcDataset(features, targets)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, **dataloader_kwargs
        )

    def normalize_features(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Normalize features using standard scaling.

        Args:
            features: Input features to normalize

        Returns:
            Tuple of (normalized_features, normalization_stats)
            normalization_stats contains 'mean' and 'std' for denormalization
        """
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)

        # Avoid division by zero
        std = torch.where(std == 0, torch.ones_like(std), std)

        normalized = (features - mean) / std

        stats = {"mean": mean, "std": std}
        return normalized, stats

    def apply_normalization(
        self, features: torch.Tensor, normalization_stats: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Apply existing normalization to new features.

        Args:
            features: Features to normalize
            normalization_stats: Stats from normalize_features()

        Returns:
            Normalized features
        """
        mean = normalization_stats["mean"]
        std = normalization_stats["std"]
        return (features - mean) / std

    def create_train_val_split(
        self,
        features: torch.Tensor,
        targets: torch.Tensor | None = None,
        val_ratio: float = 0.2,
        random_seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Split data into training and validation sets.

        Args:
            features: Feature tensor
            targets: Optional target tensor
            val_ratio: Fraction of data to use for validation
            random_seed: Random seed for reproducible splits

        Returns:
            Tuple of (train_features, val_features, train_targets, val_targets)
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)

        num_samples = len(features)
        indices = torch.randperm(num_samples)

        val_size = int(num_samples * val_ratio)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        train_features = features[train_indices]
        val_features = features[val_indices]

        train_targets = None
        val_targets = None
        if targets is not None:
            train_targets = targets[train_indices]
            val_targets = targets[val_indices]

        return train_features, val_features, train_targets, val_targets
