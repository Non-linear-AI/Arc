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
from arc.graph.spec import ArcGraph
from arc.ml.artifacts import ModelArtifact, ModelArtifactManager
from arc.ml.builder import ModelBuilder

logger = logging.getLogger(__name__)


class PredictionError(Exception):
    """Raised when prediction fails."""


class ArcPredictor:
    """Handles model inference for Arc-Graph models.

    Supports both single predictions and batch predictions from database tables.
    Uses the predictor specification to determine which outputs to return.
    """

    def __init__(
        self,
        model: nn.Module,
        arc_graph: ArcGraph,
        artifact: ModelArtifact,
        device: str | torch.device = "cpu",
    ):
        """Initialize predictor.

        Args:
            model: Trained PyTorch model
            arc_graph: Arc-Graph specification
            artifact: Model artifact metadata
            device: Device for inference
        """
        self.model = model
        self.arc_graph = arc_graph
        self.artifact = artifact
        self.device = torch.device(device)

        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()

        # Determine which outputs to return
        self.output_keys = self._get_output_keys()

        logger.info(f"ArcPredictor initialized for model {artifact.model_id}")
        logger.info(f"Will return outputs: {self.output_keys}")

    def _get_output_keys(self) -> list[str]:
        """Determine which outputs to return based on predictor spec."""
        if self.arc_graph.predictor and self.arc_graph.predictor.returns:
            return self.arc_graph.predictor.returns

        # If no predictor spec, return all model outputs
        if self.arc_graph.model and self.arc_graph.model.outputs:
            return list(self.arc_graph.model.outputs.keys())

        # Fallback: assume single output named 'output'
        return ["output"]

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

                    # Extract requested outputs
                    if isinstance(output, dict):
                        for key in self.output_keys:
                            if key not in output:
                                available = list(output.keys())
                                raise PredictionError(
                                    f"Output '{key}' not found in model outputs. "
                                    f"Available: {available}"
                                )
                            predictions[key].append(output[key].cpu())
                    else:
                        # Single tensor output
                        if len(self.output_keys) == 1:
                            predictions[self.output_keys[0]].append(output.cpu())
                        else:
                            raise PredictionError(
                                f"Model returns single tensor but predictor expects "
                                f"multiple outputs: {self.output_keys}"
                            )

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
                (uses arc_graph default if None)
            batch_size: Batch size for processing
            limit: Maximum number of rows to process

        Returns:
            Dictionary mapping output names to prediction tensors

        Raises:
            PredictionError: If prediction fails
        """
        try:
            # Use feature columns from arc_graph if not provided
            if feature_columns is None:
                if (
                    not self.arc_graph.features
                    or not self.arc_graph.features.feature_columns
                ):
                    raise PredictionError(
                        "No feature columns specified and none found in arc_graph"
                    )
                feature_columns = self.arc_graph.features.feature_columns

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
                (uses arc_graph default if None)
            batch_size: Batch size for processing

        Returns:
            Dictionary mapping output names to prediction tensors

        Raises:
            PredictionError: If prediction fails
        """
        try:
            # Use feature columns from arc_graph if not provided
            if feature_columns is None:
                if (
                    not self.arc_graph.features
                    or not self.arc_graph.features.feature_columns
                ):
                    raise PredictionError(
                        "No feature columns specified and none found in arc_graph"
                    )
                feature_columns = self.arc_graph.features.feature_columns

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
                self.arc_graph.features.feature_columns
                if self.arc_graph.features
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

            # Reconstruct Arc-Graph from metadata
            if not artifact.arc_graph:
                raise PredictionError(
                    f"No Arc-Graph specification found in artifact for model {model_id}"
                )

            arc_graph = ArcGraph.from_dict(artifact.arc_graph)

            # Build model from Arc-Graph
            builder = ModelBuilder()
            model = builder.build_model(arc_graph)

            # Load trained weights
            model.load_state_dict(state_dict)

            return cls(
                model=model,
                arc_graph=arc_graph,
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
        arc_graph: ArcGraph,
        device: str | torch.device = "cpu",
    ) -> ArcPredictor:
        """Load predictor from training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            arc_graph: Arc-Graph specification
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

            # Build model from Arc-Graph
            builder = ModelBuilder()
            model = builder.build_model(arc_graph)

            # Load trained weights
            model.load_state_dict(checkpoint["model_state_dict"])

            # Create minimal artifact metadata
            artifact = ModelArtifact(
                model_id="checkpoint_model",
                model_name="Checkpoint Model",
                version=checkpoint.get("epoch", 1),
                model_state_path="model.pt",
                arc_graph=arc_graph.__dict__,
            )

            return cls(
                model=model,
                arc_graph=arc_graph,
                artifact=artifact,
                device=device,
            )

        except Exception as e:
            logger.error(f"Failed to load predictor from checkpoint: {e}")
            raise PredictionError(
                f"Failed to load predictor from checkpoint: {e}"
            ) from e
