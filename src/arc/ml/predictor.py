"""Model inference and prediction engine for Arc-Graph models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from arc.database.services.ml_data_service import MLDataService
from arc.graph import ModelSpec
from arc.ml.artifacts import ModelArtifact, ModelArtifactManager
from arc.ml.builder import ModelBuilder

logger = logging.getLogger(__name__)


class PredictionError(Exception):
    """Raised when prediction fails."""


class ArcPredictor:
    """Handles model inference for Arc-Graph models.

    Supports both single predictions and batch predictions from database tables.
    Returns all outputs defined in the model specification.
    """

    def __init__(
        self,
        model: nn.Module,
        model_spec: ModelSpec,
        artifact: ModelArtifact,
        device: str | torch.device = "cpu",
    ):
        """Initialize predictor.

        Args:
            model: Trained PyTorch model
            model_spec: Model specification
            artifact: Model artifact metadata
            device: Device for inference
        """
        self.model = model
        self.model_spec = model_spec
        self.artifact = artifact
        self.device = torch.device(device)

        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()

        # Determine which outputs to return from model spec
        self.output_keys = self._get_output_keys()

        logger.info(f"ArcPredictor initialized for model {artifact.model_id}")
        logger.info(f"Will return outputs: {self.output_keys}")

    def _get_output_keys(self) -> list[str]:
        """Determine which outputs to return from model spec."""
        # Use all model outputs from ModelSpec
        if self.model_spec.outputs:
            return list(self.model_spec.outputs.keys())

        # Fallback: assume single output named 'output'
        return ["output"]

    def _extract_outputs(
        self, model_output: dict[str, torch.Tensor] | torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Extract outputs from model."""
        if isinstance(model_output, dict):
            # Handle dictionary output from model
            extracted = {}

            for output_name in self.output_keys:
                if output_name not in model_output:
                    available = list(model_output.keys())
                    raise PredictionError(
                        f"Model output '{output_name}' not found in model "
                        f"outputs. Available: {available}"
                    )

                extracted[output_name] = model_output[output_name]

            return extracted
        else:
            # Single tensor output from model
            if len(self.output_keys) == 1:
                return {self.output_keys[0]: model_output}
            else:
                raise PredictionError(
                    f"Model returns single tensor but expects "
                    f"multiple outputs: {self.output_keys}"
                )

    def predict_batch(
        self,
        features: torch.Tensor,
        batch_size: int = 32,
    ) -> dict[str, torch.Tensor]:
        """Run batch prediction on features tensor.

        Args:
            features: Input features tensor [num_samples, num_features]
            batch_size: Batch size for processing

        Returns:
            Dictionary mapping output names to prediction tensors

        Raises:
            PredictionError: If prediction fails
        """
        try:
            features = features.to(self.device)

            # Create DataLoader for batched processing
            dataset = torch.utils.data.TensorDataset(features)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            predictions = {key: [] for key in self.output_keys}

            with torch.no_grad():
                for batch in dataloader:
                    batch_features = batch[0]

                    # Forward pass
                    output = self.model(batch_features)

                    # Extract and map outputs according to predictor spec
                    extracted_outputs = self._extract_outputs(output)

                    # Collect outputs for this batch
                    for key, tensor in extracted_outputs.items():
                        predictions[key].append(tensor.cpu())

            # Concatenate all batches
            final_predictions = {}
            for key in self.output_keys:
                final_predictions[key] = torch.cat(predictions[key], dim=0)

            return final_predictions

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise PredictionError(f"Batch prediction failed: {e}") from e

    def predict_single(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run prediction on a single sample.

        Args:
            features: Input features tensor [num_features] or [1, num_features]

        Returns:
            Dictionary mapping output names to prediction tensors

        Raises:
            PredictionError: If prediction fails
        """
        try:
            # Ensure batch dimension
            if features.dim() == 1:
                features = features.unsqueeze(0)
            elif features.dim() != 2 or features.size(0) != 1:
                raise PredictionError(
                    f"Expected features with shape [num_features] or "
                    f"[1, num_features], got {features.shape}"
                )

            # Use batch prediction for single sample
            predictions = self.predict_batch(features, batch_size=1)

            # Remove batch dimension for single prediction
            # Use squeeze() without argument to remove all dimensions of size 1
            return {key: tensor.squeeze() for key, tensor in predictions.items()}

        except Exception as e:
            logger.error(f"Single prediction failed: {e}")
            raise PredictionError(f"Single prediction failed: {e}") from e

    def predict_from_table(
        self,
        ml_data_service: MLDataService,
        table_name: str,
        feature_columns: list[str] | None = None,
        batch_size: int = 32,
        limit: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run prediction on data from a database table.

        Args:
            ml_data_service: Service for accessing ML data
            table_name: Name of the table containing features
            feature_columns: List of feature column names
                (uses model_spec default if None)
            batch_size: Batch size for processing
            limit: Maximum number of rows to process

        Returns:
            Dictionary mapping output names to prediction tensors

        Raises:
            PredictionError: If prediction fails
        """
        try:
            # Use feature columns from model_spec if not provided
            if feature_columns is None:
                # Extract feature columns from model inputs
                if not self.model_spec.inputs:
                    raise PredictionError(
                        "No feature columns specified and no inputs found in model_spec"
                    )
                # Get columns from the first input (assuming single input for
                # simplicity)
                first_input = next(iter(self.model_spec.inputs.values()))
                if not hasattr(first_input, "columns") or not first_input.columns:
                    raise PredictionError(
                        "No feature columns found in model input specification"
                    )
                feature_columns = first_input.columns

            logger.info(
                f"Loading features from table '{table_name}' with columns: "
                f"{feature_columns}"
            )

            # Load features as tensors
            features, _ = ml_data_service.get_features_as_tensors(
                dataset_name=table_name,
                feature_columns=feature_columns,
                target_columns=None,  # No targets needed for prediction
            )

            if limit is not None:
                features = features[:limit]

            logger.info(f"Loaded {features.shape[0]} samples for prediction")

            # Run batch prediction
            return self.predict_batch(features, batch_size=batch_size)

        except Exception as e:
            logger.error(f"Table prediction failed: {e}")
            raise PredictionError(f"Table prediction failed: {e}") from e

    def predict_from_table_streaming(
        self,
        ml_data_service: MLDataService,
        table_name: str,
        feature_columns: list[str] | None = None,
        batch_size: int = 32,
        chunk_size: int = 10000,
        limit: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run prediction on data from a database table using streaming for efficiency.

        This method loads data in chunks to handle large datasets without memory issues,
        similar to the approach used in training with StreamingTableDataset.

        Args:
            ml_data_service: Service for accessing ML data
            table_name: Name of the table containing features
            feature_columns: List of feature column names
                (uses model_spec default if None)
            batch_size: Batch size for model inference processing
            chunk_size: Number of rows to load per chunk from database
            limit: Maximum number of rows to process

        Returns:
            Dictionary mapping output names to prediction tensors

        Raises:
            PredictionError: If prediction fails
        """
        try:
            # Use feature columns from model_spec if not provided
            if feature_columns is None:
                if (
                    not self.model_spec.features
                    or not self.model_spec.features.feature_columns
                ):
                    raise PredictionError(
                        "No feature columns specified and none found in model_spec"
                    )
                feature_columns = self.model_spec.features.feature_columns

            # Get dataset info to validate and determine total size
            dataset_info = ml_data_service.get_dataset_info(table_name)
            if not dataset_info:
                raise PredictionError(f"Dataset '{table_name}' does not exist")

            total_rows = dataset_info.row_count
            if limit is not None:
                total_rows = min(total_rows, limit)

            logger.info(
                f"Starting streaming prediction on table '{table_name}' "
                f"with {total_rows} rows using chunk_size={chunk_size}"
            )

            # Initialize result accumulation
            predictions = {key: [] for key in self.output_keys}
            processed_rows = 0

            # Process data in chunks
            while processed_rows < total_rows:
                # Calculate chunk size for this iteration
                current_chunk_size = min(chunk_size, total_rows - processed_rows)

                logger.info(
                    f"Processing chunk: rows {processed_rows} to "
                    f"{processed_rows + current_chunk_size - 1}"
                )

                # Load chunk data using pagination
                features_df, _ = ml_data_service.get_features_and_targets_paginated(
                    dataset_name=table_name,
                    feature_columns=feature_columns,
                    target_columns=None,  # No targets needed for prediction
                    offset=processed_rows,
                    limit=current_chunk_size,
                )

                # Break if no more data
                if features_df.empty:
                    break

                # Convert to tensor
                features_tensor = torch.tensor(
                    features_df.values.astype(float), dtype=torch.float32
                )

                # Run prediction on this chunk
                chunk_predictions = self.predict_batch(features_tensor, batch_size)

                # Accumulate results
                for key in self.output_keys:
                    predictions[key].append(chunk_predictions[key])

                processed_rows += len(features_df)

                # Stop if we've hit the limit
                if limit is not None and processed_rows >= limit:
                    break

            # Concatenate all chunk results
            final_predictions = {}
            for key in self.output_keys:
                if predictions[key]:
                    final_predictions[key] = torch.cat(predictions[key], dim=0)
                else:
                    # Handle case where no data was processed
                    final_predictions[key] = torch.empty(0)

            logger.info(
                f"Streaming prediction completed: processed {processed_rows} rows, "
                f"generated {len(final_predictions)} output types"
            )

            return final_predictions

        except Exception as e:
            logger.error(f"Streaming table prediction failed: {e}")
            raise PredictionError(f"Streaming table prediction failed: {e}") from e

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        feature_columns: list[str] | None = None,
        batch_size: int = 32,
    ) -> dict[str, torch.Tensor]:
        """Run prediction on data from a pandas DataFrame.

        Args:
            df: DataFrame containing features
            feature_columns: List of feature column names
                (uses model_spec default if None)
            batch_size: Batch size for processing

        Returns:
            Dictionary mapping output names to prediction tensors

        Raises:
            PredictionError: If prediction fails
        """
        try:
            # Use feature columns from model_spec if not provided
            if feature_columns is None:
                # Extract feature columns from model inputs
                if not self.model_spec.inputs:
                    raise PredictionError(
                        "No feature columns specified and no inputs found in model_spec"
                    )
                # Get columns from the first input (assuming single input for
                # simplicity)
                first_input = next(iter(self.model_spec.inputs.values()))
                if not hasattr(first_input, "columns") or not first_input.columns:
                    raise PredictionError(
                        "No feature columns found in model input specification"
                    )
                feature_columns = first_input.columns

            # Validate columns exist
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                raise PredictionError(
                    f"Missing feature columns in DataFrame: {missing_cols}"
                )

            # Extract features and convert to tensor
            feature_data = df[feature_columns].values
            features = torch.tensor(feature_data, dtype=torch.float32)

            logger.info(f"Loaded {features.shape[0]} samples from DataFrame")

            # Run batch prediction
            return self.predict_batch(features, batch_size=batch_size)

        except Exception as e:
            logger.error(f"DataFrame prediction failed: {e}")
            raise PredictionError(f"DataFrame prediction failed: {e}") from e

    def to_dict(self) -> dict[str, Any]:
        """Get predictor information as dictionary."""
        return {
            "model_id": self.artifact.model_id,
            "model_name": self.artifact.model_name,
            "version": self.artifact.version,
            "device": str(self.device),
            "output_keys": self.output_keys,
            "feature_columns": (
                next(iter(self.model_spec.inputs.values())).columns
                if self.model_spec.inputs
                and hasattr(next(iter(self.model_spec.inputs.values())), "columns")
                else None
            ),
        }

    @classmethod
    def load_from_artifact(
        cls,
        artifact_manager: ModelArtifactManager,
        model_id: str,
        version: int | None = None,
        device: str | torch.device = "cpu",
    ) -> ArcPredictor:
        """Load predictor from saved artifact.

        Args:
            artifact_manager: Manager for loading artifacts
            model_id: Model identifier
            version: Specific version to load (latest if None)
            device: Device for inference

        Returns:
            Loaded predictor ready for inference

        Raises:
            PredictionError: If loading fails
        """
        try:
            logger.info(f"Loading predictor for model {model_id}, version {version}")

            # Load model state dict and metadata
            state_dict, artifact = artifact_manager.load_model_state_dict(
                model_id=model_id,
                version=version,
                device=device,
            )

            # Reconstruct ModelSpec from metadata
            if not artifact.model_spec:
                raise PredictionError(
                    f"No model specification found in artifact for model {model_id}"
                )

            # Convert old model_spec format to new ModelSpec if needed
            if "model" in artifact.model_spec:
                model_dict = artifact.model_spec["model"]
            else:
                model_dict = artifact.model_spec

            from arc.graph.model import (
                GraphNode,
                ModelInput,
                ModelSpec,
                validate_model_dict,
            )

            # Validate the model structure
            validate_model_dict(model_dict)

            # Create ModelSpec from dict (similar to ModelSpec.from_yaml)
            # Parse inputs
            inputs = {}
            for input_name, input_spec in model_dict["inputs"].items():
                inputs[input_name] = ModelInput(
                    dtype=input_spec["dtype"],
                    shape=input_spec["shape"],
                    columns=input_spec.get("columns"),
                )

            # Parse graph nodes
            graph = []
            for node_data in model_dict["graph"]:
                graph.append(
                    GraphNode(
                        name=node_data["name"],
                        type=node_data["type"],
                        params=node_data.get("params", {}),
                        inputs=node_data.get("inputs", {}),
                    )
                )

            # Parse outputs
            outputs = model_dict["outputs"]

            model_spec = ModelSpec(
                inputs=inputs,
                graph=graph,
                outputs=outputs,
            )

            # Build model from ModelSpec
            builder = ModelBuilder()
            model = builder.build_model(model_spec)

            # Load trained weights
            model.load_state_dict(state_dict)

            return cls(
                model=model,
                model_spec=model_spec,
                artifact=artifact,
                device=device,
            )

        except Exception as e:
            logger.error(f"Failed to load predictor: {e}")
            raise PredictionError(f"Failed to load predictor: {e}") from e

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_spec: ModelSpec,
        device: str | torch.device = "cpu",
    ) -> ArcPredictor:
        """Load predictor from training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model_spec: Model specification
            device: Device for inference

        Returns:
            Loaded predictor ready for inference

        Raises:
            PredictionError: If loading fails
        """
        try:
            logger.info(f"Loading predictor from checkpoint: {checkpoint_path}")

            # Load checkpoint
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )

            # Build model from ModelSpec
            builder = ModelBuilder()
            model = builder.build_model(model_spec)

            # Load trained weights
            model.load_state_dict(checkpoint["model_state_dict"])

            # Create minimal artifact metadata
            artifact = ModelArtifact(
                model_id="checkpoint_model",
                model_name="Checkpoint Model",
                version=checkpoint.get("epoch", 1),
                model_state_path="model.pt",
                model_spec=model_spec.__dict__,
            )

            return cls(
                model=model,
                model_spec=model_spec,
                artifact=artifact,
                device=device,
            )

        except Exception as e:
            logger.error(f"Failed to load predictor from checkpoint: {e}")
            raise PredictionError(
                f"Failed to load predictor from checkpoint: {e}"
            ) from e
