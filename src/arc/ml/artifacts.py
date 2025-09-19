"""Model artifact storage and management for Arc Graph."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..graph import TrainingConfig
from .trainer import TrainingResult


@dataclass
class ModelArtifact:
    """Metadata for a saved model artifact."""

    # Model identification
    model_id: str
    model_name: str
    version: int

    # Training information
    training_config: TrainingConfig | None = None
    training_result: TrainingResult | None = None
    arc_graph: dict[str, Any] | None = None

    # Model architecture info
    model_class: str | None = None
    model_params: dict[str, Any] | None = None
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None

    # Metrics and performance
    final_metrics: dict[str, float] | None = None
    best_metrics: dict[str, float] | None = None

    # File paths (relative to artifact directory)
    model_state_path: str = "model_state.pt"
    optimizer_state_path: str | None = None
    metadata_path: str = "metadata.json"
    training_history_path: str | None = None

    # Timestamps
    created_at: str = ""
    updated_at: str = ""

    # Additional metadata
    tags: list[str] | None = None
    description: str | None = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at


class ModelArtifactManager:
    """Manages model artifacts storage and retrieval."""

    def __init__(self, artifacts_dir: str | Path):
        """Initialize artifact manager.

        Args:
            artifacts_dir: Base directory for storing artifacts
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def save_model_artifact(
        self,
        model: nn.Module,
        artifact: ModelArtifact,
        optimizer: torch.optim.Optimizer | None = None,
        training_history: dict[str, Any] | None = None,
        arc_graph: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Save a complete model artifact.

        Args:
            model: PyTorch model to save
            artifact: Artifact metadata
            optimizer: Optional optimizer state to save
            training_history: Optional training history data
            arc_graph: Optional Arc Graph specification
            overwrite: Whether to overwrite existing artifacts

        Returns:
            Path to the saved artifact directory
        """
        # Create artifact directory
        artifact_dir = self.get_artifact_path(artifact.model_id, artifact.version)

        if artifact_dir.exists() and not overwrite:
            raise FileExistsError(f"Artifact already exists: {artifact_dir}")

        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_state_path = artifact_dir / artifact.model_state_path
        torch.save(model.state_dict(), model_state_path)

        # Save optimizer state if provided
        if optimizer is not None:
            optimizer_path = artifact_dir / "optimizer_state.pt"
            torch.save(optimizer.state_dict(), optimizer_path)
            artifact.optimizer_state_path = "optimizer_state.pt"

        # Save training history if provided
        if training_history is not None:
            history_path = artifact_dir / "training_history.json"
            with open(history_path, "w") as f:
                json.dump(training_history, f, indent=2)
            artifact.training_history_path = "training_history.json"

        # Save Arc Graph specification if provided
        if arc_graph is not None:
            graph_path = artifact_dir / "arc_graph.json"
            with open(graph_path, "w") as f:
                json.dump(asdict(arc_graph), f, indent=2, default=str)
            artifact.arc_graph = asdict(arc_graph)

        # Update timestamps
        artifact.updated_at = datetime.now().isoformat()

        # Save metadata
        metadata_path = artifact_dir / artifact.metadata_path
        self._save_metadata(artifact, metadata_path)

        return artifact_dir

    def load_model_artifact(
        self,
        model_id: str,
        version: int | None = None,
    ) -> tuple[nn.Module | None, ModelArtifact]:
        """Load a model artifact.

        Args:
            model_id: Model identifier
            version: Specific version to load (latest if None)

        Returns:
            Tuple of (loaded model, artifact metadata)
        """
        if version is None:
            version = self.get_latest_version(model_id)

        artifact_dir = self.get_artifact_path(model_id, version)
        if not artifact_dir.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_dir}")

        # Load metadata
        metadata_path = artifact_dir / "metadata.json"
        artifact = self._load_metadata(metadata_path)

        # Load model state if available
        model = None
        model_state_path = artifact_dir / artifact.model_state_path
        if model_state_path.exists():
            # Note: This loads only the state dict. The model architecture
            # needs to be reconstructed separately using the Arc Graph
            # or model class information stored in metadata
            pass

        return model, artifact

    def load_model_state_dict(
        self,
        model_id: str,
        version: int | None = None,
        device: str | torch.device = "cpu",
    ) -> tuple[dict[str, Any], ModelArtifact]:
        """Load model state dict and metadata.

        Args:
            model_id: Model identifier
            version: Specific version to load (latest if None)
            device: Device to load model on

        Returns:
            Tuple of (state dict, artifact metadata)
        """
        if version is None:
            version = self.get_latest_version(model_id)

        artifact_dir = self.get_artifact_path(model_id, version)
        if not artifact_dir.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_dir}")

        # Load metadata
        metadata_path = artifact_dir / "metadata.json"
        artifact = self._load_metadata(metadata_path)

        # Load model state dict
        model_state_path = artifact_dir / artifact.model_state_path
        state_dict = torch.load(
            model_state_path, map_location=device, weights_only=False
        )

        return state_dict, artifact

    def list_artifacts(self, model_id: str | None = None) -> list[ModelArtifact]:
        """List available artifacts.

        Args:
            model_id: Optional model ID to filter by

        Returns:
            List of artifact metadata
        """
        artifacts = []

        if model_id:
            # List versions for specific model
            model_dir = self.artifacts_dir / model_id
            if model_dir.exists():
                for version_dir in model_dir.iterdir():
                    if version_dir.is_dir():
                        metadata_path = version_dir / "metadata.json"
                        if metadata_path.exists():
                            artifact = self._load_metadata(metadata_path)
                            artifacts.append(artifact)
        else:
            # List all artifacts
            for model_dir in self.artifacts_dir.iterdir():
                if model_dir.is_dir():
                    for version_dir in model_dir.iterdir():
                        if version_dir.is_dir():
                            metadata_path = version_dir / "metadata.json"
                            if metadata_path.exists():
                                artifact = self._load_metadata(metadata_path)
                                artifacts.append(artifact)

        # Sort by created_at descending
        artifacts.sort(key=lambda x: x.created_at, reverse=True)
        return artifacts

    def get_latest_version(self, model_id: str) -> int:
        """Get the latest version number for a model.

        Args:
            model_id: Model identifier

        Returns:
            Latest version number

        Raises:
            FileNotFoundError: If no versions exist
        """
        model_dir = self.artifacts_dir / model_id
        if not model_dir.exists():
            raise FileNotFoundError(f"No artifacts found for model: {model_id}")

        versions = []
        for version_dir in model_dir.iterdir():
            if version_dir.is_dir() and version_dir.name.isdigit():
                versions.append(int(version_dir.name))

        if not versions:
            raise FileNotFoundError(f"No valid versions found for model: {model_id}")

        return max(versions)

    def delete_artifact(self, model_id: str, version: int) -> None:
        """Delete a specific artifact version.

        Args:
            model_id: Model identifier
            version: Version to delete
        """
        artifact_dir = self.get_artifact_path(model_id, version)
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)

        # Clean up empty model directory
        model_dir = self.artifacts_dir / model_id
        if model_dir.exists() and not any(model_dir.iterdir()):
            model_dir.rmdir()

    def delete_model_artifacts(self, model_id: str) -> None:
        """Delete all artifacts for a model.

        Args:
            model_id: Model identifier
        """
        model_dir = self.artifacts_dir / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir)

    def get_artifact_path(self, model_id: str, version: int) -> Path:
        """Get path to artifact directory.

        Args:
            model_id: Model identifier
            version: Model version

        Returns:
            Path to artifact directory
        """
        return self.artifacts_dir / model_id / str(version)

    def copy_artifact(
        self,
        source_model_id: str,
        source_version: int,
        target_model_id: str,
        target_version: int,
    ) -> Path:
        """Copy an artifact to a new location.

        Args:
            source_model_id: Source model ID
            source_version: Source version
            target_model_id: Target model ID
            target_version: Target version

        Returns:
            Path to copied artifact
        """
        source_dir = self.get_artifact_path(source_model_id, source_version)
        target_dir = self.get_artifact_path(target_model_id, target_version)

        if not source_dir.exists():
            raise FileNotFoundError(f"Source artifact not found: {source_dir}")

        if target_dir.exists():
            raise FileExistsError(f"Target artifact already exists: {target_dir}")

        # Copy directory
        shutil.copytree(source_dir, target_dir)

        # Update metadata
        metadata_path = target_dir / "metadata.json"
        if metadata_path.exists():
            artifact = self._load_metadata(metadata_path)
            artifact.model_id = target_model_id
            artifact.version = target_version
            artifact.updated_at = datetime.now().isoformat()
            self._save_metadata(artifact, metadata_path)

        return target_dir

    def _save_metadata(self, artifact: ModelArtifact, path: Path) -> None:
        """Save artifact metadata to JSON file."""
        # Convert dataclass to dict, handling special types
        metadata = asdict(artifact)

        # Convert training config and result to dicts if present
        if metadata.get("training_config") and is_dataclass(
            metadata["training_config"]
        ):
            metadata["training_config"] = asdict(metadata["training_config"])
        if metadata.get("training_result") and is_dataclass(
            metadata["training_result"]
        ):
            metadata["training_result"] = asdict(metadata["training_result"])

        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def _load_metadata(self, path: Path) -> ModelArtifact:
        """Load artifact metadata from JSON file."""
        with open(path) as f:
            data = json.load(f)

        # Convert nested dicts back to dataclasses if present
        if data.get("training_config"):
            data["training_config"] = TrainingConfig(**data["training_config"])
        if data.get("training_result"):
            data["training_result"] = TrainingResult(**data["training_result"])

        return ModelArtifact(**data)

    def export_artifact(
        self,
        model_id: str,
        version: int,
        export_path: str | Path,
        format: str = "torch",
    ) -> Path:
        """Export artifact to different format.

        Args:
            model_id: Model identifier
            version: Model version
            export_path: Path to export to
            format: Export format ('torch', 'onnx', 'torchscript')

        Returns:
            Path to exported file
        """
        export_path = Path(export_path)

        if format == "torch":
            # Export as PyTorch checkpoint
            state_dict, artifact = self.load_model_state_dict(model_id, version)

            export_data = {
                "model_state_dict": state_dict,
                "metadata": asdict(artifact),
            }

            torch.save(export_data, export_path)

        else:
            raise ValueError(f"Unsupported export format: {format}")

        return export_path


def create_artifact_from_training(
    model_id: str,
    model_name: str,
    version: int,
    training_config: TrainingConfig,
    training_result: TrainingResult,
    arc_graph: dict[str, Any] | None = None,
    model_info: dict[str, Any] | None = None,
    **kwargs,
) -> ModelArtifact:
    """Create a model artifact from training results.

    Args:
        model_id: Model identifier
        model_name: Human-readable model name
        version: Model version
        training_config: Training configuration used
        training_result: Results from training
        arc_graph: Optional Arc Graph specification
        model_info: Optional model architecture information
        **kwargs: Additional metadata

    Returns:
        Model artifact
    """
    # Extract final metrics from training result
    final_metrics = {}
    if training_result.train_losses:
        final_metrics["final_train_loss"] = training_result.final_train_loss
    if training_result.final_val_loss is not None:
        final_metrics["final_val_loss"] = training_result.final_val_loss
    if training_result.best_val_loss is not None:
        final_metrics["best_val_loss"] = training_result.best_val_loss

    # Extract best metrics
    best_metrics = {}
    if training_result.metrics_history:
        for metric_name, values in training_result.metrics_history.items():
            if values:
                best_metrics[f"best_{metric_name}"] = (
                    min(values) if "loss" in metric_name else max(values)
                )

    artifact = ModelArtifact(
        model_id=model_id,
        model_name=model_name,
        version=version,
        training_config=training_config,
        training_result=training_result,
        arc_graph=asdict(arc_graph) if arc_graph else None,
        final_metrics=final_metrics,
        best_metrics=best_metrics,
        **kwargs,
    )

    # Add model info if provided
    if model_info:
        artifact.model_class = model_info.get("model_class")
        artifact.model_params = model_info.get("model_params")
        artifact.input_shape = model_info.get("input_shape")
        artifact.output_shape = model_info.get("output_shape")

    return artifact
