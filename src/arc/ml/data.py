"""Data loading and preprocessing utilities for Arc ML."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from arc.database.services import MLDataService
from arc.ml.processors.base import ProcessorError, StatefulProcessor
from arc.plugins import get_plugin_manager


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


class MultiInputDataset(Dataset):
    """PyTorch Dataset for models with multiple named inputs.

    This dataset splits a single feature tensor into multiple input tensors
    according to the model's input specification, enabling models with
    separate inputs (e.g., user_id, item_id, features) to work correctly.
    """

    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor | None,
        input_spec: dict[str, dict],
        feature_columns: list[str],
    ):
        """Initialize multi-input dataset.

        Args:
            features: Feature tensor of shape [num_samples, num_features]
            targets: Optional target tensor of shape [num_samples, ...]
            input_spec: Model input specification (from model_spec.inputs)
                Each input should have 'columns', 'dtype', and 'shape' fields
            feature_columns: Ordered list of column names in features tensor

        Example:
            >>> input_spec = {
            ...     'user_id': {'columns': ['UserID'], 'dtype': 'long', 'shape': [None, 1]},
            ...     'features': {'columns': ['age', 'income'], 'dtype': 'float32', 'shape': [None, 2]}
            ... }
            >>> dataset = MultiInputDataset(features, targets, input_spec, ['UserID', 'age', 'income'])
        """
        self.features = features
        self.targets = targets
        self.feature_columns = feature_columns

        if targets is not None and len(features) != len(targets):
            raise ValueError(
                f"Features and targets must have same length: "
                f"{len(features)} vs {len(targets)}"
            )

        # Compute column indices for each input
        self.input_slices = {}
        self.input_dtypes = {}

        for input_name, spec in input_spec.items():
            input_columns = spec.get("columns", [])
            if not input_columns:
                raise ValueError(
                    f"Input '{input_name}' has no columns specified in model spec"
                )

            # Find indices of these columns in the feature tensor
            try:
                indices = [feature_columns.index(col) for col in input_columns]
            except ValueError as e:
                missing_col = str(e).split("'")[1]
                raise ValueError(
                    f"Input '{input_name}' expects column '{missing_col}' which is not in feature_columns. "
                    f"Available columns: {feature_columns}"
                ) from e

            self.input_slices[input_name] = indices

            # Store target dtype for this input
            dtype_str = spec.get("dtype", "float32")
            self.input_dtypes[input_name] = self._parse_dtype(dtype_str)

    def _parse_dtype(self, dtype_str: str) -> torch.dtype:
        """Parse dtype string to torch.dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float": torch.float32,
            "double": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "long": torch.long,
            "int": torch.int32,
            "bool": torch.bool,
        }
        return dtype_map.get(dtype_str, torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(
        self, idx: int
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Get a sample with features split into named inputs.

        Returns:
            If targets is None: dict mapping input names to tensors
            If targets exists: (input_dict, target_tensor)
        """
        # Split features according to input specs
        input_dict = {}
        for input_name, indices in self.input_slices.items():
            # Extract columns for this input
            input_tensor = self.features[idx, indices]

            # Convert to appropriate dtype
            target_dtype = self.input_dtypes[input_name]
            if input_tensor.dtype != target_dtype:
                input_tensor = input_tensor.to(target_dtype)

            input_dict[input_name] = input_tensor

        if self.targets is not None:
            return input_dict, self.targets[idx]
        return input_dict


class DataProcessor:
    """Processes data for Arc-Graph models with plugin support."""

    def __init__(
        self,
        ml_data_service: MLDataService | None = None,
    ):
        """Initialize DataProcessor with MLDataService.

        Args:
            ml_data_service: MLDataService instance for data access
        """
        self.ml_data_service = ml_data_service
        self.plugin_manager = get_plugin_manager()
        self.fitted_processors: dict[str, StatefulProcessor] = {}
        self._processor_configs: dict[str, dict[str, Any]] = {}

    def load_from_table(
        self,
        table_name: str,
        feature_columns: list[str],
        target_columns: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Load data from database table using MLDataService.

        Args:
            table_name: Name of database table
            feature_columns: List of feature column names
            target_columns: Optional list of target column names

        Returns:
            Tuple of (features, targets) tensors

        Raises:
            RuntimeError: If no MLDataService available
            ValueError: If columns not found or data invalid
        """
        # Use MLDataService for data access
        if self.ml_data_service:
            try:
                return self.ml_data_service.get_features_as_tensors(
                    dataset_name=table_name,
                    feature_columns=feature_columns,
                    target_columns=target_columns,
                )
            except Exception as e:
                raise ValueError(f"Failed to load data via MLDataService: {e}") from e

        # No MLDataService available
        raise RuntimeError("MLDataService is required for data access")

    def create_dataloader(
        self,
        features: torch.Tensor,
        targets: torch.Tensor | None = None,
        batch_size: int = 32,
        shuffle: bool = True,
        input_spec: dict[str, dict] | None = None,
        feature_columns: list[str] | None = None,
        **dataloader_kwargs,
    ) -> DataLoader:
        """Create PyTorch DataLoader from tensors.

        Args:
            features: Feature tensor
            targets: Optional target tensor
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle data
            input_spec: Optional model input specification for multi-input models
            feature_columns: Optional ordered list of column names (required for multi-input)
            **dataloader_kwargs: Additional arguments for DataLoader

        Returns:
            PyTorch DataLoader

        Note:
            If input_spec and feature_columns are provided, creates a MultiInputDataset
            that splits features according to the model's input specification.
        """
        # Use MultiInputDataset if we have input spec with multiple inputs
        if input_spec and feature_columns and len(input_spec) > 1:
            dataset = MultiInputDataset(features, targets, input_spec, feature_columns)
        else:
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

    # Enhanced processor methods using plugin system

    def fit_processor(
        self,
        processor_name: str,
        processor_type: str,
        config: dict[str, Any],
        alias: str | None = None,
    ) -> None:
        """Fit a processor on database data.

        Args:
            processor_name: Unique name for this processor instance
            processor_type: Type of processor (e.g., 'core.StandardNormalization')
            config: Configuration including table_name, columns, etc.
            alias: Optional alias for easier reference

        Raises:
            ProcessorError: If processor not found or fitting fails
            RuntimeError: If database not available
        """
        if not self.database:
            raise RuntimeError("Database connection required to fit processors")

        # Get processor class from plugin manager
        processor_class = self.plugin_manager.get_processor(processor_type)
        if not processor_class:
            raise ProcessorError(f"Processor type '{processor_type}' not found")

        # Validate configuration
        if not self.plugin_manager.validate_component_config(
            "processor", processor_type, config
        ):
            raise ProcessorError(
                f"Invalid configuration for processor '{processor_type}'"
            )

        try:
            # Create and fit processor
            processor = processor_class(**config.get("init_params", {}))
            processor.fit(self.database, config)

            # Store fitted processor
            self.fitted_processors[processor_name] = processor
            self._processor_configs[processor_name] = {
                "type": processor_type,
                "config": config,
                "alias": alias,
            }

            if alias:
                self.fitted_processors[alias] = processor

        except Exception as e:
            raise ProcessorError(
                f"Failed to fit processor '{processor_name}': {e}"
            ) from e

    def apply_processor(self, processor_name: str, data: torch.Tensor) -> torch.Tensor:
        """Apply a fitted processor to data.

        Args:
            processor_name: Name or alias of the fitted processor
            data: Input data tensor

        Returns:
            Transformed data tensor

        Raises:
            ProcessorError: If processor not fitted or transformation fails
        """
        if processor_name not in self.fitted_processors:
            raise ProcessorError(f"Processor '{processor_name}' not fitted")

        try:
            processor = self.fitted_processors[processor_name]
            return processor.transform(data)
        except Exception as e:
            raise ProcessorError(
                f"Failed to apply processor '{processor_name}': {e}"
            ) from e

    def apply_processor_pipeline(
        self, processor_names: list[str], data: torch.Tensor
    ) -> torch.Tensor:
        """Apply multiple processors in sequence.

        Args:
            processor_names: List of processor names to apply in order
            data: Input data tensor

        Returns:
            Transformed data tensor
        """
        result = data
        for processor_name in processor_names:
            result = self.apply_processor(processor_name, result)
        return result

    def get_available_processors(self) -> dict[str, type]:
        """Get all available processor types from plugins.

        Returns:
            Dictionary mapping processor type names to classes
        """
        return self.plugin_manager.get_processors()

    def get_fitted_processors(self) -> list[str]:
        """Get names of all fitted processors.

        Returns:
            List of fitted processor names
        """
        return list(self._processor_configs.keys())

    def remove_processor(self, processor_name: str) -> None:
        """Remove a fitted processor.

        Args:
            processor_name: Name of processor to remove
        """
        if processor_name in self.fitted_processors:
            # Also remove alias if it exists
            config = self._processor_configs.get(processor_name, {})
            alias = config.get("alias")
            if alias and alias in self.fitted_processors:
                del self.fitted_processors[alias]

            del self.fitted_processors[processor_name]
            del self._processor_configs[processor_name]

    def save_processors(self, filepath: str | Path) -> None:
        """Save all fitted processors to file.

        Args:
            filepath: Path to save processors to

        Raises:
            ProcessorError: If saving fails
        """
        filepath = Path(filepath)

        try:
            saved_data = {}
            for name, processor in self.fitted_processors.items():
                # Skip aliases to avoid duplication
                if name not in self._processor_configs:
                    continue

                config = self._processor_configs[name]
                saved_data[name] = {
                    "processor_state": processor.save_state(),
                    "config": config,
                }

            with open(filepath, "w") as f:
                json.dump(saved_data, f, indent=2, default=str)

        except Exception as e:
            raise ProcessorError(f"Failed to save processors: {e}") from e

    def load_processors(self, filepath: str | Path) -> None:
        """Load fitted processors from file.

        Args:
            filepath: Path to load processors from

        Raises:
            ProcessorError: If loading fails
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise ProcessorError(f"Processor file not found: {filepath}")

        try:
            with open(filepath) as f:
                saved_data = json.load(f)

            for name, data in saved_data.items():
                processor_state = data["processor_state"]
                config = data["config"]
                processor_type = config["type"]

                # Get processor class
                processor_class = self.plugin_manager.get_processor(processor_type)
                if not processor_class:
                    raise ProcessorError(f"Processor type '{processor_type}' not found")

                # Create processor and load state
                init_params = config["config"].get("init_params", {})
                processor = processor_class(**init_params)
                processor.load_state(processor_state)

                # Store processor
                self.fitted_processors[name] = processor
                self._processor_configs[name] = config

                # Handle alias
                alias = config.get("alias")
                if alias:
                    self.fitted_processors[alias] = processor

        except Exception as e:
            raise ProcessorError(f"Failed to load processors: {e}") from e

    # Enhanced convenience methods

    def normalize_features_with_plugin(
        self,
        features: torch.Tensor,
        table_name: str,
        columns: list[str],
        method: str = "standard",
        processor_name: str | None = None,
    ) -> tuple[torch.Tensor, str]:
        """Normalize features using database-wide statistics via plugins.

        Args:
            features: Features to normalize
            table_name: Database table containing the full dataset
            columns: Column names corresponding to features
            method: Normalization method ('standard', 'minmax', 'robust')
            processor_name: Optional name for the processor (auto-generated if None)

        Returns:
            Tuple of (normalized_features, processor_name)
        """
        # Map method to processor type
        method_mapping = {
            "standard": "core.StandardNormalization",
            "minmax": "core.MinMaxNormalization",
            "robust": "core.RobustNormalization",
            "zscore": "core.StandardNormalization",
        }

        if method not in method_mapping:
            raise ValueError(f"Unknown normalization method: {method}")

        processor_type = method_mapping[method]
        if processor_name is None:
            processor_name = f"normalize_{method}_{len(self.fitted_processors)}"

        # Fit processor if not already fitted
        if processor_name not in self.fitted_processors:
            config = {
                "table_name": table_name,
                "columns": columns,
            }
            self.fit_processor(processor_name, processor_type, config)

        # Apply normalization
        normalized_features = self.apply_processor(processor_name, features)
        return normalized_features, processor_name

    # === Spec-driven features pipeline ===

    def run_feature_pipeline(
        self,
        table_name: str,
        features_spec: Mapping[str, Any],
        *,
        training: bool = True,
        initial_context: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[dict[str, dict[str, Any]], torch.Tensor | None, torch.Tensor | None]:
        """Execute the spec-defined features processors pipeline.

        Supports namespaces: tensors, vars, states; and operator kinds:
        - inspect.* (train-only)
        - fit.* (train-only)
        - transform.* (train and predict)

        Returns a context with populated namespaces, the default features tensor
        (taken from tensors.features if present), and an optional targets tensor.
        """
        if not self.ml_data_service:
            raise RuntimeError("MLDataService required to run feature pipeline")

        feature_columns: list[str] = list(features_spec.get("feature_columns", []))
        target_columns: list[str] = list(features_spec.get("target_columns", []))
        processors: list[dict[str, Any]] = list(features_spec.get("processors", []))

        # Determine needed raw columns
        raw_cols = set(feature_columns) | set(target_columns)
        raw_cols |= self._collect_raw_columns_from_processors(processors)

        # Load dataframe with needed columns
        df = self._load_table_df(table_name, sorted(raw_cols))

        # Initialize context
        context: dict[str, dict[str, Any]] = {
            "tensors": {},
            "vars": {},
            "states": {},
        }
        if initial_context:
            for ns in ("tensors", "vars", "states"):
                if ns in initial_context:
                    context[ns].update(dict(initial_context[ns]))

        top_level = {
            "feature_columns": feature_columns,
            "target_columns": target_columns,
        }

        # Execute steps
        for step in processors:
            name = step.get("name") or step.get("op", "<unnamed>")
            op: str = step.get("op", "")
            if not op:
                raise ProcessorError(f"Processor '{name}' missing 'op'")

            # Skip train-only or inspect/fit during prediction
            if not training:
                if step.get("train_only", False):
                    continue
                if op.startswith("inspect.") or op.startswith("fit."):
                    continue

            # Prepare inputs
            inputs_spec: Mapping[str, Any] = step.get("inputs", {})
            resolved_inputs = {
                k: self._resolve_input_value(v, context, top_level, df)
                for k, v in inputs_spec.items()
            }

            # Execute operation
            outputs = self._execute_operator(op, resolved_inputs)

            # Map outputs into namespaces
            outputs_map: Mapping[str, str] = step.get("outputs", {})
            for global_name, src_name in outputs_map.items():
                ns, key = self._split_ns_key(global_name)
                if src_name not in outputs:
                    raise ProcessorError(
                        f"Processor '{name}' expected output '{src_name}' from '{op}'"
                    )
                context[ns][key] = outputs[src_name]

        # Default features tensor
        default_features = context["tensors"].get("features")

        # Targets tensor (optional)
        targets_tensor: torch.Tensor | None = None
        if target_columns:
            try:
                target_data = df[target_columns].values
                targets_tensor = torch.tensor(target_data, dtype=torch.float32)
                if len(target_columns) == 1:
                    targets_tensor = targets_tensor.squeeze(1)
            except Exception as e:
                raise ProcessorError(f"Failed to build targets: {e}") from e

        return context, default_features, targets_tensor

    # ---- helpers for spec pipeline ----

    def _load_table_df(self, table_name: str, columns: Iterable[str]) -> pd.DataFrame:
        cols = list(columns)
        if not cols:
            raise ProcessorError("No columns specified to load from table")

        # Use MLDataService if available (preferred)
        if self.ml_data_service:
            try:
                return self.ml_data_service.get_data(
                    dataset_name=table_name,
                    columns=cols,
                    limit=None,  # Get all data for processing pipeline
                )
            except Exception as e:
                raise ProcessorError(
                    f"Failed to load data via MLDataService: {e}"
                ) from e

        # Fallback to direct database access
        if not self.database:
            raise ProcessorError("Either MLDataService or Database connection required")

        cols_clause = ", ".join(cols)
        result = self.database.query(f"SELECT {cols_clause} FROM {table_name}")
        if not result.rows:
            raise ProcessorError(f"No data found in table {table_name}")
        try:
            return pd.DataFrame(result.rows, columns=cols)
        except Exception:
            return pd.DataFrame(result.rows)

    def _collect_raw_columns_from_processors(
        self, processors: Iterable[Mapping[str, Any]]
    ) -> set[str]:
        raw: set[str] = set()
        for step in processors:
            inputs = step.get("inputs", {})
            for v in inputs.values():
                raw |= self._extract_raw_columns(v)
        return raw

    def _extract_raw_columns(self, val: Any) -> set[str]:
        cols: set[str] = set()
        if isinstance(val, str):
            # Namespaced references or known top-level lists are not raw columns
            if "." in val or val in ("feature_columns", "target_columns"):
                return cols
            cols.add(val)
        elif isinstance(val, (list, tuple)):
            for item in val:
                cols |= self._extract_raw_columns(item)
        elif isinstance(val, dict):
            for item in val.values():
                cols |= self._extract_raw_columns(item)
        return cols

    def _resolve_input_value(
        self,
        token: Any,
        context: Mapping[str, Mapping[str, Any]],
        top_level: Mapping[str, Any],
        df: pd.DataFrame,
    ) -> Any:
        # Primitive/structured literals
        if not isinstance(token, str):
            if isinstance(token, list):
                return [
                    self._resolve_input_value(t, context, top_level, df) for t in token
                ]
            if isinstance(token, tuple):
                return tuple(
                    self._resolve_input_value(t, context, top_level, df) for t in token
                )
            if isinstance(token, dict):
                return {
                    k: self._resolve_input_value(v, context, top_level, df)
                    for k, v in token.items()
                }
            return token

        # String token
        if token.startswith("tensors."):
            k = token.split(".", 1)[1]
            if k not in context["tensors"]:
                raise ProcessorError(f"Missing tensor '{k}' in context")
            return context["tensors"][k]
        if token.startswith("vars."):
            k = token.split(".", 1)[1]
            if k not in context["vars"]:
                raise ProcessorError(f"Missing var '{k}' in context")
            return context["vars"][k]
        if token.startswith("states."):
            k = token.split(".", 1)[1]
            if k not in context["states"]:
                raise ProcessorError(f"Missing state '{k}' in context")
            return context["states"][k]
        if token == "feature_columns" or token == "target_columns":
            cols = top_level[token]
            return [self._resolve_input_value(c, context, top_level, df) for c in cols]

        # Treat as a raw column name -> return appropriate column representation
        if token not in df.columns:
            raise ProcessorError(f"Unknown input token or column: {token}")
        series = df[token]
        # If non-numeric/textual column, return list of strings for downstream ops
        if series.dtype.kind in ("O",) or str(series.dtype).startswith("string"):
            return [None if pd.isna(v) else str(v) for v in series.tolist()]
        # Otherwise numeric tensor [N]
        arr = pd.to_numeric(series, errors="coerce").fillna(0).values
        return torch.tensor(arr, dtype=torch.float32)

    def _split_ns_key(self, global_name: str) -> tuple[str, str]:
        if "." not in global_name:
            raise ProcessorError(
                f"Output name '{global_name}' must be namespaced as '<ns>.<key>'"
            )
        ns, key = global_name.split(".", 1)
        if ns not in ("tensors", "vars", "states"):
            raise ProcessorError(
                f"Unknown namespace '{ns}' in output name '{global_name}'"
            )
        return ns, key

    def _execute_operator(self, op: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Execute a supported operator and return its outputs dict."""
        # Transform operators
        if op == "transform.assemble_vector":
            return self._op_transform_assemble_vector(inputs)
        if op == "transform.standard_scaler":
            return self._op_transform_standard_scaler(inputs)
        if op == "transform.sequence_shift":
            return self._op_transform_sequence_shift(inputs)
        if op == "transform.hash_bucket":
            return self._op_transform_hash_bucket(inputs)
        if op == "transform.hash_and_pad":
            return self._op_transform_hash_and_pad(inputs)

        # Inspect operators (train-only)
        if op == "inspect.feature_stats":
            return self._op_inspect_feature_stats(inputs)

        # Fit operators (train-only)
        if op == "fit.standard_scaler":
            return self._op_fit_standard_scaler(inputs)
        if op == "fit.label_encoder":
            return self._op_fit_label_encoder(inputs)

        # Label encoding transform
        if op == "transform.label_encode":
            return self._op_transform_label_encode(inputs)

        raise ProcessorError(f"Unsupported operator: {op}")

    # ---- concrete operator implementations ----

    def _op_transform_assemble_vector(
        self, inputs: Mapping[str, Any]
    ) -> dict[str, Any]:
        # inputs: { columns: list[str] or list[Tensor] }
        cols = inputs.get("columns")
        if cols is None:
            raise ProcessorError("transform.assemble_vector requires 'columns'")
        tensors: list[torch.Tensor] = []
        # Allow list of column names (strings) or list of 1D tensors
        for item in cols:
            if isinstance(item, torch.Tensor):
                t = item
            else:
                # If item is a scalar column vector disguised as
                # Python list/ndarray, coerce to a tensor
                t = item if isinstance(item, torch.Tensor) else torch.as_tensor(item)
            t = t.reshape(-1, 1)
            tensors.append(t.to(torch.float32))
        output = torch.cat(tensors, dim=1) if tensors else torch.empty(0, 0)
        return {"output": output}

    def _op_inspect_feature_stats(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        # inputs: { tensor: Tensor }
        x = inputs.get("tensor")
        if not isinstance(x, torch.Tensor):
            raise ProcessorError("inspect.feature_stats requires 'tensor' tensor input")
        if x.dim() != 2:
            raise ProcessorError("inspect.feature_stats expects 2D tensor [N, F]")
        n_features = int(x.shape[1])
        return {"n_features": n_features}

    def _op_fit_standard_scaler(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        # Accept either { x: Tensor } or { mean: ..., std: ... } in future
        x = inputs.get("x")
        if not isinstance(x, torch.Tensor):
            raise ProcessorError("fit.standard_scaler requires 'x' tensor input")
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
        return {"state": {"mean": mean, "std": std}}

    def _op_transform_standard_scaler(
        self, inputs: Mapping[str, Any]
    ) -> dict[str, Any]:
        # inputs: { x: Tensor, state: { mean, std } }
        x = inputs.get("x")
        state = inputs.get("state")
        if not isinstance(x, torch.Tensor):
            raise ProcessorError("transform.standard_scaler requires 'x' tensor input")
        if not isinstance(state, dict) or "mean" not in state or "std" not in state:
            raise ProcessorError(
                "transform.standard_scaler requires 'state' with 'mean' and 'std'"
            )
        mean = state["mean"].to(x.device)
        std = state["std"].to(x.device)
        return {"output": (x - mean) / std}

    def _op_transform_sequence_shift(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        x = inputs.get("x")
        direction = inputs.get("direction", "left")
        fill_value = inputs.get("fill_value", 0)

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        squeeze_back = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_back = True
        if x.dim() != 2:
            raise ProcessorError("transform.sequence_shift expects 1D or 2D tensor")

        y = torch.empty_like(x)
        if direction == "left":
            y[:, :-1] = x[:, 1:]
            y[:, -1] = fill_value
        elif direction == "right":
            y[:, 1:] = x[:, :-1]
            y[:, 0] = fill_value
        else:
            raise ProcessorError("sequence_shift 'direction' must be 'left' or 'right'")

        if squeeze_back:
            y = y.squeeze(0)
        return {"output": y}

    def _stable_hash_to_bucket(self, value: Any, num_buckets: int) -> int:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                return int(value) % num_buckets
            except Exception:
                value = str(value)
        b = str(value).encode("utf-8", errors="ignore")
        h = hashlib.blake2b(b, digest_size=8).digest()
        return int.from_bytes(h, "little") % num_buckets

    def _op_transform_hash_bucket(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        x = inputs.get("x")
        num_buckets = int(inputs.get("num_buckets", 1000))
        if isinstance(x, torch.Tensor):
            if x.dim() != 1:
                raise ProcessorError("hash_bucket expects 1D input tensor or list")
            vals = x.tolist()
        elif isinstance(x, list):
            vals = x
        else:
            raise ProcessorError("hash_bucket requires 'x' as list or 1D tensor")
        buckets = [self._stable_hash_to_bucket(v, num_buckets) for v in vals]
        return {"output": torch.tensor(buckets, dtype=torch.long)}

    def _op_transform_hash_and_pad(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        x = inputs.get("x")
        num_buckets = int(inputs.get("num_buckets", 1000))
        max_length = int(inputs.get("max_length", 0))
        fill_value = int(inputs.get("fill_value", 0))
        if max_length <= 0:
            raise ProcessorError("hash_and_pad requires positive 'max_length'")

        sequences: list[list[Any]]
        if isinstance(x, torch.Tensor):
            if x.dim() == 1:
                sequences = [x.tolist()]
            elif x.dim() == 2:
                sequences = [row.tolist() for row in x]
            else:
                raise ProcessorError(
                    "hash_and_pad expects 1D/2D tensor or list of lists"
                )
        elif isinstance(x, list):
            sequences = x if x and isinstance(x[0], list) else [x]
        else:
            raise ProcessorError("hash_and_pad requires 'x' as list of lists or tensor")

        rows: list[torch.Tensor] = []
        for seq in sequences:
            hashed = [self._stable_hash_to_bucket(v, num_buckets) for v in seq]
            if len(hashed) >= max_length:
                hashed = hashed[:max_length]
            else:
                hashed = hashed + [fill_value] * (max_length - len(hashed))
            rows.append(torch.tensor(hashed, dtype=torch.long))
        output = torch.stack(rows)
        return {"output": output}

    def _op_fit_label_encoder(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Fit a label encoder to build vocabulary from categorical data.

        Args:
            inputs: { x: list of categorical values (strings or None) }

        Returns:
            { state: { vocabulary: dict, vocab_size: int } }
        """
        x = inputs.get("x")
        if x is None:
            raise ProcessorError("fit.label_encoder requires 'x' input")

        if not isinstance(x, list):
            raise ProcessorError(
                f"fit.label_encoder expects list input, got {type(x).__name__}"
            )

        # Build vocabulary: map unique values to sequential indices
        # Use sorted order for deterministic vocabulary
        unique_values = sorted(set(x), key=lambda v: (v is None, v))

        vocabulary = {val: idx for idx, val in enumerate(unique_values)}
        vocab_size = len(vocabulary)

        return {"state": {"vocabulary": vocabulary, "vocab_size": vocab_size}}

    def _op_transform_label_encode(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Transform categorical data to integer indices using fitted vocabulary.

        Args:
            inputs: {
                x: list of categorical values,
                state: { vocabulary: dict, vocab_size: int },
                handle_unknown: "use_unknown_value" (default) | "error",
                unknown_value: int (defaults to vocab_size)
            }

        Returns:
            { output: torch.Tensor (long) }

        Raises:
            ProcessorError: If unknown category found and handle_unknown="error"
        """
        x = inputs.get("x")
        state = inputs.get("state")
        handle_unknown = inputs.get("handle_unknown", "use_unknown_value")
        unknown_value = inputs.get("unknown_value")

        if x is None:
            raise ProcessorError("transform.label_encode requires 'x' input")
        if state is None or not isinstance(state, dict):
            raise ProcessorError(
                "transform.label_encode requires 'state' with vocabulary"
            )

        vocabulary = state.get("vocabulary")
        if vocabulary is None or not isinstance(vocabulary, dict):
            raise ProcessorError(
                "transform.label_encode state must contain 'vocabulary' dict"
            )

        vocab_size = state.get("vocab_size")
        if vocab_size is None:
            raise ProcessorError(
                "transform.label_encode state must contain 'vocab_size'"
            )

        if not isinstance(x, list):
            raise ProcessorError(
                f"transform.label_encode expects list input, got {type(x).__name__}"
            )

        # Default unknown_value to vocab_size if not specified
        if unknown_value is None:
            unknown_value = vocab_size

        # Encode values to indices
        indices = []
        for val in x:
            if val in vocabulary:
                indices.append(vocabulary[val])
            else:
                # Handle unknown category
                if handle_unknown == "error":
                    # Provide helpful error with available categories
                    available = list(vocabulary.keys())[:10]  # Show first 10
                    available_str = ", ".join(repr(k) for k in available)
                    if len(vocabulary) > 10:
                        available_str += f", ... ({len(vocabulary)} total)"
                    raise ProcessorError(
                        f"Unknown category {repr(val)} not in vocabulary. "
                        f"Available categories: {available_str}"
                    )
                else:
                    # Use unknown_value for unknown categories
                    indices.append(unknown_value)

        return {"output": torch.tensor(indices, dtype=torch.long)}

    # High-level convenience methods for training integration

    def create_dataloader_from_dataset(
        self,
        dataset_name: str,
        feature_columns: list[str],
        target_columns: list[str] | None = None,
        batch_size: int = 32,
        shuffle: bool = True,
        input_spec: dict[str, dict] | None = None,
        **dataloader_kwargs,
    ) -> DataLoader:
        """Create PyTorch DataLoader from dataset using integrated MLDataService.

        Args:
            dataset_name: Name of the dataset
            feature_columns: List of feature column names
            target_columns: Optional list of target column names
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle data
            input_spec: Optional model input specification for multi-input models
            **dataloader_kwargs: Additional arguments for DataLoader

        Returns:
            PyTorch DataLoader ready for training

        Raises:
            RuntimeError: If no data service available
            ValueError: If dataset or columns don't exist
        """
        # Load data using the integrated approach
        features, targets = self.load_from_table(
            table_name=dataset_name,
            feature_columns=feature_columns,
            target_columns=target_columns,
        )

        # Create dataset and dataloader
        return self.create_dataloader(
            features=features,
            targets=targets,
            batch_size=batch_size,
            shuffle=shuffle,
            input_spec=input_spec,
            feature_columns=feature_columns,
            **dataloader_kwargs,
        )

    def create_dataloader_from_table(
        self,
        ml_data_service,
        table_name: str,
        feature_columns: list[str],
        target_columns: list[str] | None = None,
        batch_size: int = 32,
        shuffle: bool = True,
        chunk_size: int = 10000,
        input_spec: dict[str, dict] | None = None,
        **dataloader_kwargs,
    ) -> DataLoader:
        """Create PyTorch DataLoader from database table using incremental loading.

        Args:
            ml_data_service: MLDataService instance for data access
            table_name: Name of the database table
            feature_columns: List of feature column names
            target_columns: Optional list of target column names
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle data
            chunk_size: Number of rows to load per chunk for streaming
            input_spec: Optional model input specification for multi-input models
            **dataloader_kwargs: Additional arguments for DataLoader

        Returns:
            PyTorch DataLoader ready for training

        Raises:
            ValueError: If table or columns don't exist

        Note:
            For multi-input models, input_spec should be provided to split features
            according to the model's input specification. The streaming dataset
            will load all data as a single tensor and wrap it in MultiInputDataset
            via a custom collate function.
        """
        # Check if table exists as a dataset
        if not ml_data_service.dataset_exists(table_name):
            raise ValueError(f"Dataset/table '{table_name}' does not exist")

        # Get dataset info to determine size and validate columns
        dataset_info = ml_data_service.get_dataset_info(table_name)
        if not dataset_info:
            raise ValueError(f"Could not get info for dataset '{table_name}'")

        # Validate columns exist
        available_columns = dataset_info.column_names
        for col in feature_columns:
            if col not in available_columns:
                raise ValueError(
                    f"Feature column '{col}' not found in dataset '{table_name}'"
                )

        if target_columns:
            for col in target_columns:
                if col not in available_columns:
                    raise ValueError(
                        f"Target column '{col}' not found in dataset '{table_name}'"
                    )

        # Create streaming dataset that loads data incrementally
        streaming_dataset = StreamingTableDataset(
            ml_data_service=ml_data_service,
            table_name=table_name,
            feature_columns=feature_columns,
            target_columns=target_columns,
            total_rows=dataset_info.row_count,
            chunk_size=chunk_size,
            input_spec=input_spec,
        )

        return DataLoader(
            streaming_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **dataloader_kwargs,
        )


class StreamingTableDataset(Dataset):
    """PyTorch Dataset that streams data from database in chunks using MLDataService."""

    def __init__(
        self,
        ml_data_service,
        table_name: str,
        feature_columns: list[str],
        target_columns: list[str] | None,
        total_rows: int,
        chunk_size: int = 10000,
        input_spec: dict[str, dict] | None = None,
    ):
        """Initialize streaming dataset.

        Args:
            ml_data_service: MLDataService instance for data access
            table_name: Name of the database table
            feature_columns: List of feature column names
            target_columns: Optional list of target column names
            total_rows: Total number of rows in the dataset
            chunk_size: Number of rows to load per chunk
            input_spec: Optional model input specification for multi-input models
        """
        self.ml_data_service = ml_data_service
        self.table_name = table_name
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.total_rows = total_rows
        self.chunk_size = chunk_size
        self.input_spec = input_spec

        # Cache for loaded chunks
        self._chunk_cache = {}
        self._current_chunk_id = -1
        self._current_chunk_data = None
        self._max_cached_chunks = 3  # Keep max 3 chunks in memory

        # For multi-input models, prepare column index mappings
        self.is_multi_input = input_spec is not None and len(input_spec) > 1
        if self.is_multi_input:
            # Borrow logic from MultiInputDataset
            self.input_slices = {}
            self.input_dtypes = {}
            for input_name, spec in input_spec.items():
                input_columns = spec.get("columns", [])
                if not input_columns:
                    raise ValueError(
                        f"Input '{input_name}' has no columns specified in model spec"
                    )
                try:
                    indices = [feature_columns.index(col) for col in input_columns]
                except ValueError as e:
                    missing_col = str(e).split("'")[1]
                    raise ValueError(
                        f"Input '{input_name}' expects column '{missing_col}' which is not in feature_columns. "
                        f"Available columns: {feature_columns}"
                    ) from e
                self.input_slices[input_name] = indices
                dtype_str = spec.get("dtype", "float32")
                self.input_dtypes[input_name] = self._parse_dtype(dtype_str)

    def _parse_dtype(self, dtype_str: str) -> torch.dtype:
        """Parse dtype string to torch.dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float": torch.float32,
            "double": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "long": torch.long,
            "int": torch.int32,
            "bool": torch.bool,
        }
        return dtype_map.get(dtype_str, torch.float32)

    def __len__(self) -> int:
        return self.total_rows

    def __getitem__(
        self, idx: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor | None]
        | tuple[dict[str, torch.Tensor], torch.Tensor | None]
    ):
        """Get a single sample by index.

        Returns:
            For single-input models: (features_tensor, targets_tensor)
            For multi-input models: (input_dict, targets_tensor)
        """
        if idx >= self.total_rows:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {self.total_rows}"
            )

        # Determine which chunk this index belongs to
        chunk_id = idx // self.chunk_size
        idx_in_chunk = idx % self.chunk_size

        # Load chunk if not already loaded
        if chunk_id != self._current_chunk_id:
            self._load_chunk(chunk_id)

        # Extract the sample from the current chunk
        if self._current_chunk_data is None:
            raise RuntimeError(f"Failed to load chunk {chunk_id}")

        features, targets = self._current_chunk_data

        # Handle case where chunk might be smaller than chunk_size (last chunk)
        if idx_in_chunk >= len(features):
            raise IndexError(
                f"Index {idx_in_chunk} out of range for chunk of size {len(features)}"
            )

        sample_features = features[idx_in_chunk]
        sample_targets = targets[idx_in_chunk] if targets is not None else None

        # For multi-input models, split features into named inputs
        if self.is_multi_input:
            input_dict = {}
            for input_name, indices in self.input_slices.items():
                input_tensor = sample_features[indices]
                target_dtype = self.input_dtypes[input_name]
                if input_tensor.dtype != target_dtype:
                    input_tensor = input_tensor.to(target_dtype)
                input_dict[input_name] = input_tensor

            if sample_targets is not None:
                return input_dict, sample_targets
            return input_dict, None

        # For single-input models, return tensor directly
        return sample_features, sample_targets

    def _load_chunk(self, chunk_id: int) -> None:
        """Load a specific chunk of data using MLDataService pagination."""
        # Check cache first
        if chunk_id in self._chunk_cache:
            self._current_chunk_data = self._chunk_cache[chunk_id]
            self._current_chunk_id = chunk_id
            return

        # Calculate offset and limit for this chunk
        offset = chunk_id * self.chunk_size
        limit = min(self.chunk_size, self.total_rows - offset)

        try:
            # Use MLDataService with pagination to get chunk data
            features_df, targets_df = (
                self.ml_data_service.get_features_and_targets_paginated(
                    dataset_name=self.table_name,
                    feature_columns=self.feature_columns,
                    target_columns=self.target_columns,
                    offset=offset,
                    limit=limit,
                )
            )

            # Convert to tensors
            import torch

            if features_df.empty:
                # Handle empty chunk (shouldn't happen with proper bounds checking)
                features = torch.empty(
                    0, len(self.feature_columns), dtype=torch.float32
                )
                targets = None
                if self.target_columns:
                    targets = torch.empty(
                        0, len(self.target_columns), dtype=torch.float32
                    )
            else:
                features = torch.tensor(features_df.values, dtype=torch.float32)
                targets = None
                if targets_df is not None:
                    targets = torch.tensor(targets_df.values, dtype=torch.float32)

            chunk_data = (features, targets)

            # Cache the chunk with LRU-like eviction
            if len(self._chunk_cache) >= self._max_cached_chunks:
                # Remove oldest chunk (simple FIFO for now)
                oldest_chunk = min(self._chunk_cache.keys())
                del self._chunk_cache[oldest_chunk]

            self._chunk_cache[chunk_id] = chunk_data
            self._current_chunk_data = chunk_data
            self._current_chunk_id = chunk_id

        except Exception as e:
            raise RuntimeError(
                f"Failed to load chunk {chunk_id} from table '{self.table_name}': {e}"
            ) from e
