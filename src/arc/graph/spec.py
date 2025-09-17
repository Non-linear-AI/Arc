"""Data models for Arc-Graph (v0.1 overview schema)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

try:
    import yaml
except ImportError as e:
    raise RuntimeError(
        "PyYAML is required for Arc-Graph. "
        "Install with 'uv add pyyaml' or 'pip install pyyaml'."
    ) from e

# === Features ===


@dataclass
class Processor:
    name: str
    op: str
    train_only: bool = False
    inputs: dict[str, str] | None = None
    outputs: dict[str, str] | None = None


@dataclass
class Features:
    feature_columns: list[str]
    target_columns: list[str] | None
    processors: list[Processor]


# === Model ===


@dataclass
class ModelInput:
    dtype: str
    shape: list[int | None | str]


@dataclass
class GraphNode:
    name: str
    type: str
    params: dict[str, Any] | None = None
    inputs: dict[str, str] | None = None


@dataclass
class ModelSpec:
    inputs: dict[str, ModelInput]
    graph: list[GraphNode]
    outputs: dict[str, str]


# === Trainer / Predictor ===


@dataclass
class OptimizerSpec:
    type: str
    config: dict[str, Any] | None = None


@dataclass
class LossSpec:
    type: str
    inputs: dict[str, str]


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Core training parameters
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001

    # Optimizer configuration
    optimizer: str = "adam"
    optimizer_params: dict[str, Any] = field(default_factory=dict)

    # Loss function configuration
    loss_function: str = "cross_entropy"
    loss_params: dict[str, Any] = field(default_factory=dict)

    # Training behavior
    validation_split: float = 0.2
    shuffle: bool = True
    drop_last: bool = False

    # Checkpointing
    checkpoint_every: int = 5  # epochs
    save_best_only: bool = True

    # Early stopping
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.001

    # Hardware
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    # Logging
    log_every: int = 10  # batches
    verbose: bool = True


@dataclass
class TrainerSpec:
    optimizer: OptimizerSpec
    loss: LossSpec
    config: TrainingConfig | None = None


@dataclass
class PredictorSpec:
    returns: list[str]


# === Root ===


@dataclass
class ArcGraph:
    version: str
    model_name: str
    description: str | None
    features: Features
    model: ModelSpec
    trainer: TrainerSpec
    predictor: PredictorSpec | None

    @classmethod
    def from_yaml(cls, yaml_str: str) -> ArcGraph:
        """Parse Arc-Graph from YAML string.

        Args:
            yaml_str: YAML string containing Arc-Graph specification

        Returns:
            ArcGraph: Parsed and validated Arc-Graph

        Raises:
            ValueError: If YAML is invalid or doesn't contain valid Arc-Graph
            GraphValidationError: If Arc-Graph structure is invalid
        """
        # Import here to avoid circular import
        from .validator import validate_graph_dict

        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML must be a mapping")

        validate_graph_dict(data)
        return cls.from_dict(data)

    @classmethod
    def from_yaml_file(cls, path: str) -> ArcGraph:
        """Parse Arc-Graph from YAML file.

        Args:
            path: Path to YAML file containing Arc-Graph specification

        Returns:
            ArcGraph: Parsed and validated Arc-Graph

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid or doesn't contain valid Arc-Graph
            GraphValidationError: If Arc-Graph structure is invalid
        """
        with open(path, encoding="utf-8") as f:
            return cls.from_yaml(f.read())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArcGraph:
        # Features
        f = data["features"]
        processors: list[Processor] = []
        for p in f.get("processors", []) or []:
            processors.append(
                Processor(
                    name=str(p.get("name")),
                    op=str(p.get("op")),
                    train_only=bool(p.get("train_only", False)),
                    inputs=p.get("inputs"),
                    outputs=p.get("outputs"),
                )
            )
        features = Features(
            feature_columns=[str(c) for c in f.get("feature_columns", [])],
            target_columns=[str(c) for c in f.get("target_columns", [])]
            if f.get("target_columns") is not None
            else None,
            processors=processors,
        )

        # Model
        m = data["model"]
        inputs: dict[str, ModelInput] = {}
        for key, val in (m.get("inputs") or {}).items():
            shape = []
            for s in val.get("shape", []):
                if s is None:
                    shape.append(None)
                elif isinstance(s, str):
                    shape.append(s)  # Keep string references like "vars.n_features"
                else:
                    shape.append(int(s))
            inputs[key] = ModelInput(dtype=str(val.get("dtype")), shape=shape)
        graph_nodes: list[GraphNode] = []
        for n in m.get("graph", []) or []:
            graph_nodes.append(
                GraphNode(
                    name=str(n.get("name")),
                    type=str(n.get("type")),
                    params=n.get("params"),
                    inputs=n.get("inputs"),
                )
            )
        model_spec = ModelSpec(
            inputs=inputs,
            graph=graph_nodes,
            outputs=m.get("outputs") or {},
        )

        # Trainer
        t = data["trainer"]
        opt = t.get("optimizer") or {}
        optimizer = OptimizerSpec(type=str(opt.get("type")), config=opt.get("config"))
        loss = t.get("loss") or {}
        loss_spec = LossSpec(
            type=str(loss.get("type")), inputs=loss.get("inputs") or {}
        )

        # Parse training config section
        trainer_config = None
        if "config" in t:
            config_data = t["config"]
            trainer_config = TrainingConfig(
                epochs=config_data.get("epochs", 10),
                batch_size=config_data.get("batch_size", 32),
                learning_rate=config_data.get("learning_rate", 0.001),
                optimizer=optimizer.type.lower(),
                optimizer_params=optimizer.config or {},
                loss_function=loss_spec.type.lower(),
                loss_params={},
                validation_split=config_data.get("validation_split", 0.2),
                shuffle=config_data.get("shuffle", True),
                drop_last=config_data.get("drop_last", False),
                checkpoint_every=config_data.get("checkpoint_every", 5),
                save_best_only=config_data.get("save_best_only", True),
                early_stopping_patience=config_data.get("early_stopping_patience"),
                early_stopping_min_delta=config_data.get(
                    "early_stopping_min_delta", 0.001
                ),
                device=config_data.get("device", "auto"),
                log_every=config_data.get("log_every", 10),
                verbose=config_data.get("verbose", True),
            )

        trainer = TrainerSpec(
            optimizer=optimizer, loss=loss_spec, config=trainer_config
        )

        # Predictor (optional)
        pred = data.get("predictor")
        predictor = (
            PredictorSpec(returns=list(pred.get("returns", []))) if pred else None
        )

        return cls(
            version=str(data.get("version")),
            model_name=str(data.get("model_name")),
            description=data.get("description"),
            features=features,
            model=model_spec,
            trainer=trainer,
            predictor=predictor,
        )

    def to_training_config(
        self, override_params: dict[str, Any] | None = None
    ) -> TrainingConfig:
        """Get training configuration from Arc-Graph specification.

        Args:
            override_params: Optional parameters to override defaults

        Returns:
            TrainingConfig: Training configuration for PyTorch trainer
        """
        if self.trainer.config is None:
            raise ValueError(
                "Arc-Graph trainer.config is required for training. "
                "Add a trainer.config section to the graph specification."
            )

        config_data = asdict(self.trainer.config)

        if override_params:
            unknown_keys = set(override_params) - set(config_data)
            if unknown_keys:
                raise ValueError(
                    "Unsupported training config override(s): "
                    + ", ".join(sorted(unknown_keys))
                )
            config_data.update(override_params)

        return TrainingConfig(**config_data)
