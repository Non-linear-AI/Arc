"""Data models for Arc-Graph (v0.1 overview schema)."""

from __future__ import annotations

from dataclasses import dataclass
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
class TrainerSpec:
    optimizer: OptimizerSpec
    loss: LossSpec


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
        trainer = TrainerSpec(optimizer=optimizer, loss=loss_spec)

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
