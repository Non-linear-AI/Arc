"""Utility functions for Arc ML package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    pass


def auto_detect_input_size(data: torch.Tensor) -> int:
    """Auto-detect input size from data tensor.

    Args:
        data: Input data tensor with shape [batch_size, features]

    Returns:
        Number of input features

    Raises:
        ValueError: If data shape is invalid
    """
    if data.dim() != 2:
        raise ValueError(
            f"Expected 2D tensor [batch_size, features], got shape {data.shape}"
        )

    return data.shape[1]


def validate_tensor_shape(
    tensor: torch.Tensor, expected_shape: list[int | None]
) -> None:
    """Validate tensor shape against expected shape.

    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape, with None for variable dimensions

    Raises:
        ValueError: If shape doesn't match
    """
    actual_shape = list(tensor.shape)

    if len(actual_shape) != len(expected_shape):
        raise ValueError(
            f"Shape dimension mismatch: expected {len(expected_shape)} dims, "
            f"got {len(actual_shape)} dims"
        )

    for i, (actual, expected) in enumerate(
        zip(actual_shape, expected_shape, strict=False)
    ):
        if expected is not None and actual != expected:
            raise ValueError(
                f"Shape mismatch at dimension {i}: expected {expected}, got {actual}"
            )


def resolve_variable_references(
    params: dict[str, Any], var_registry: dict[str, Any]
) -> dict[str, Any]:
    """Resolve variable references in layer parameters.

    Args:
        params: Layer parameters that may contain variable references
        var_registry: Registry of resolved variables

    Returns:
        Parameters with variables resolved to actual values

    Raises:
        ValueError: If variable reference cannot be resolved
    """
    resolved_params = {}

    for key, value in params.items():
        if isinstance(value, str) and value.startswith("vars."):
            var_name = value
            if var_name not in var_registry:
                raise ValueError(f"Cannot resolve variable reference: {var_name}")
            resolved_params[key] = var_registry[var_name]
        else:
            resolved_params[key] = value

    return resolved_params


def create_sample_data(
    num_samples: int,
    num_features: int,
    binary_classification: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create sample data for testing.

    Args:
        num_samples: Number of samples to generate
        num_features: Number of features per sample
        binary_classification: Whether to create binary classification targets

    Returns:
        Tuple of (features, targets)
    """
    # Generate random features
    features = torch.randn(num_samples, num_features)

    # Generate targets
    if binary_classification:
        targets = torch.randint(0, 2, (num_samples,)).float()
    else:
        targets = torch.randn(num_samples)

    return features, targets


def count_parameters(model: torch.nn.Module) -> int:
    """Count total number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ShapeInferenceError(Exception):
    """Raised when shape inference fails."""

    pass


class ShapeValidator:
    """Validates and infers tensor shapes in Arc-Graph models."""

    def __init__(self, var_registry: dict[str, Any] | None = None):
        self.var_registry = var_registry or {}
        self.shape_cache: dict[str, list[int | None]] = {}
        self.layer_output_shapes: dict[str, list[int | None]] = {}

    def parse_shape_spec(self, shape_spec: list[int | str | None]) -> list[int | None]:
        """Parse shape specification, resolving variables.

        Args:
            shape_spec: Shape specification from model definition

        Returns:
            Resolved shape with None for dynamic dimensions

        Raises:
            ShapeInferenceError: If variable cannot be resolved
        """
        resolved_shape = []
        for dim in shape_spec:
            if dim is None or (isinstance(dim, str) and dim == "null"):
                resolved_shape.append(None)  # Dynamic dimension
            elif isinstance(dim, int):
                resolved_shape.append(dim)
            elif isinstance(dim, str) and dim.startswith("vars."):
                if dim not in self.var_registry:
                    raise ShapeInferenceError(f"Cannot resolve shape variable: {dim}")
                resolved_shape.append(self.var_registry[dim])
            else:
                raise ShapeInferenceError(f"Invalid shape dimension: {dim}")
        return resolved_shape

    def validate_input_shapes(
        self,
        inputs: dict[str, torch.Tensor],
        input_specs: dict[str, Any],  # Using Any to avoid import issues
    ) -> None:
        """Validate model inputs against their specifications.

        Args:
            inputs: Actual input tensors
            input_specs: Expected input specifications

        Raises:
            ShapeInferenceError: If validation fails
        """
        for input_name, input_spec in input_specs.items():
            if input_name not in inputs:
                raise ShapeInferenceError(f"Missing required input: {input_name}")

            tensor = inputs[input_name]
            expected_shape = self.parse_shape_spec(input_spec.shape)

            try:
                validate_tensor_shape(tensor, expected_shape)
            except ValueError as e:
                raise ShapeInferenceError(
                    f"Input '{input_name}' shape validation failed: {e}"
                ) from e

            # Cache the actual shape for inference
            self.shape_cache[input_name] = list(tensor.shape)

    def infer_layer_output_shape(
        self,
        layer_name: str,
        layer_type: str,
        layer_params: dict[str, Any],
        input_shapes: dict[str, list[int | None]],
    ) -> list[int | None]:
        """Infer output shape for a layer given its inputs.

        Args:
            layer_name: Name of the layer
            layer_type: Type of the layer (e.g., 'core.Linear')
            layer_params: Layer parameters
            input_shapes: Shapes of input tensors

        Returns:
            Inferred output shape

        Raises:
            ShapeInferenceError: If inference fails
        """
        # Resolve parameters first
        resolved_params = resolve_variable_references(layer_params, self.var_registry)

        if layer_type == "core.Linear":
            return self._infer_linear_shape(resolved_params, input_shapes)
        elif layer_type in ["core.ReLU", "core.Sigmoid", "core.Dropout"]:
            return self._infer_activation_shape(input_shapes)
        elif layer_type == "core.Embedding":
            return self._infer_embedding_shape(resolved_params, input_shapes)
        elif layer_type == "core.MultiHeadAttention":
            return self._infer_attention_shape(resolved_params, input_shapes)
        elif layer_type == "core.TransformerEncoderLayer":
            return self._infer_transformer_shape(resolved_params, input_shapes)
        elif layer_type == "core.PositionalEncoding":
            return self._infer_positional_encoding_shape(input_shapes)
        elif layer_type in ["core.LayerNorm", "core.BatchNorm1d"]:
            return self._infer_normalization_shape(input_shapes)
        elif layer_type == "core.Concatenate":
            return self._infer_concatenate_shape(resolved_params, input_shapes)
        elif layer_type == "core.Add":
            return self._infer_add_shape(input_shapes)
        elif layer_type in ["core.LSTM", "core.GRU"]:
            return self._infer_rnn_shape(resolved_params, input_shapes)
        else:
            # Unknown layer type - return first input shape as fallback
            if input_shapes:
                return next(iter(input_shapes.values()))
            raise ShapeInferenceError(
                f"Cannot infer shape for unknown layer type: {layer_type}"
            )

    def _infer_linear_shape(
        self, params: dict[str, Any], input_shapes: dict[str, list[int | None]]
    ) -> list[int | None]:
        """Infer Linear layer output shape."""
        if len(input_shapes) != 1:
            raise ShapeInferenceError("Linear layer expects exactly one input")

        input_shape = next(iter(input_shapes.values()))
        in_features = params.get("in_features")
        out_features = params.get("out_features")

        if in_features is None:
            raise ShapeInferenceError("Linear layer missing in_features parameter")
        if out_features is None:
            raise ShapeInferenceError("Linear layer missing out_features parameter")

        # Validate input shape matches in_features
        if len(input_shape) >= 1:
            actual_in_features = input_shape[-1]
            if actual_in_features is not None and actual_in_features != in_features:
                raise ShapeInferenceError(
                    f"Linear layer input size mismatch: layer expects {in_features} "
                    f"features but input has {actual_in_features} features"
                )

        # Output shape: [..., out_features]
        output_shape = input_shape[:-1] + [out_features]
        return output_shape

    def _infer_activation_shape(
        self, input_shapes: dict[str, list[int | None]]
    ) -> list[int | None]:
        """Infer activation layer output shape (same as input)."""
        if len(input_shapes) != 1:
            raise ShapeInferenceError("Activation layer expects exactly one input")
        return next(iter(input_shapes.values()))

    def _infer_embedding_shape(
        self, params: dict[str, Any], input_shapes: dict[str, list[int | None]]
    ) -> list[int | None]:
        """Infer Embedding layer output shape."""
        if len(input_shapes) != 1:
            raise ShapeInferenceError("Embedding layer expects exactly one input")

        input_shape = next(iter(input_shapes.values()))
        embedding_dim = params.get("embedding_dim")

        if embedding_dim is None:
            raise ShapeInferenceError("Embedding layer missing embedding_dim parameter")

        # Output shape: [..., embedding_dim]
        output_shape = input_shape + [embedding_dim]
        return output_shape

    def _infer_attention_shape(
        self, params: dict[str, Any], input_shapes: dict[str, list[int | None]]
    ) -> list[int | None]:
        """Infer MultiHeadAttention output shape."""
        if len(input_shapes) == 1:
            # Self-attention: output shape same as input
            return next(iter(input_shapes.values()))
        elif "query" in input_shapes:
            # Cross-attention: output shape matches query
            return input_shapes["query"]
        else:
            # Use first input shape
            return next(iter(input_shapes.values()))

    def _infer_transformer_shape(
        self, params: dict[str, Any], input_shapes: dict[str, list[int | None]]
    ) -> list[int | None]:
        """Infer TransformerEncoderLayer output shape (same as input)."""
        return self._infer_activation_shape(input_shapes)

    def _infer_positional_encoding_shape(
        self, input_shapes: dict[str, list[int | None]]
    ) -> list[int | None]:
        """Infer PositionalEncoding output shape (same as input)."""
        return self._infer_activation_shape(input_shapes)

    def _infer_normalization_shape(
        self, input_shapes: dict[str, list[int | None]]
    ) -> list[int | None]:
        """Infer normalization layer output shape (same as input)."""
        return self._infer_activation_shape(input_shapes)

    def _infer_concatenate_shape(
        self, params: dict[str, Any], input_shapes: dict[str, list[int | None]]
    ) -> list[int | None]:
        """Infer Concatenate layer output shape."""
        if len(input_shapes) < 2:
            raise ShapeInferenceError("Concatenate layer expects at least 2 inputs")

        concat_dim = params.get("dim", -1)
        shapes = list(input_shapes.values())

        # All shapes must have same number of dimensions
        first_shape = shapes[0]
        for i, shape in enumerate(shapes[1:], 1):
            if len(shape) != len(first_shape):
                raise ShapeInferenceError(
                    f"Cannot concatenate tensors with different number of dimensions: "
                    f"{len(first_shape)} vs {len(shape)}"
                )

        # Normalize concatenation dimension
        if concat_dim < 0:
            concat_dim = len(first_shape) + concat_dim

        # All dimensions except concat_dim must match
        output_shape = first_shape[:]
        concat_size = 0

        for shape in shapes:
            for i, (dim1, dim2) in enumerate(zip(first_shape, shape, strict=True)):
                if i == concat_dim:
                    if dim2 is not None:
                        concat_size += dim2
                elif dim1 != dim2 and dim1 is not None and dim2 is not None:
                    raise ShapeInferenceError(
                        f"Cannot concatenate tensors: dimension {i} mismatch ({dim1} vs {dim2})"
                    )

        # Set concatenated dimension size
        if concat_size > 0:
            output_shape[concat_dim] = concat_size
        else:
            output_shape[concat_dim] = None  # Dynamic

        return output_shape

    def _infer_add_shape(
        self, input_shapes: dict[str, list[int | None]]
    ) -> list[int | None]:
        """Infer Add layer output shape (all inputs must have same shape)."""
        if len(input_shapes) < 2:
            raise ShapeInferenceError("Add layer expects at least 2 inputs")

        shapes = list(input_shapes.values())
        first_shape = shapes[0]

        for i, shape in enumerate(shapes[1:], 1):
            if len(shape) != len(first_shape):
                raise ShapeInferenceError(
                    f"Cannot add tensors with different shapes: {first_shape} vs {shape}"
                )
            for j, (dim1, dim2) in enumerate(zip(first_shape, shape, strict=True)):
                if dim1 != dim2 and dim1 is not None and dim2 is not None:
                    raise ShapeInferenceError(
                        f"Cannot add tensors: dimension {j} mismatch ({dim1} vs {dim2})"
                    )

        return first_shape

    def _infer_rnn_shape(
        self, params: dict[str, Any], input_shapes: dict[str, list[int | None]]
    ) -> list[int | None]:
        """Infer LSTM/GRU output shape."""
        if len(input_shapes) != 1:
            raise ShapeInferenceError("RNN layer expects exactly one input")

        input_shape = next(iter(input_shapes.values()))
        hidden_size = params.get("hidden_size")
        bidirectional = params.get("bidirectional", False)

        if hidden_size is None:
            raise ShapeInferenceError("RNN layer missing hidden_size parameter")

        # Output shape: [batch, seq_len, hidden_size * (2 if bidirectional else 1)]
        output_hidden_size = hidden_size * (2 if bidirectional else 1)
        output_shape = input_shape[:-1] + [output_hidden_size]
        return output_shape

    def validate_model_shapes(
        self,
        input_specs: dict[str, Any],
        graph_nodes: list[Any],  # Using Any to avoid import issues
        input_mappings: dict[str, str | dict[str, str]],
        sample_inputs: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, list[int | None]]:
        """Validate shapes throughout the entire model.

        Args:
            input_specs: Model input specifications
            graph_nodes: Graph nodes in execution order
            input_mappings: Layer input mappings
            sample_inputs: Optional sample inputs for validation

        Returns:
            Dictionary mapping layer names to their output shapes

        Raises:
            ShapeInferenceError: If validation fails
        """
        # Validate sample inputs if provided
        if sample_inputs:
            self.validate_input_shapes(sample_inputs, input_specs)
        else:
            # Initialize shape cache from input specs
            for input_name, input_spec in input_specs.items():
                self.shape_cache[input_name] = self.parse_shape_spec(input_spec.shape)

        # Process each layer in execution order
        for node in graph_nodes:
            # Get input shapes for this layer
            layer_input_shapes = self._get_layer_input_shapes(node.name, input_mappings)

            # Infer output shape
            output_shape = self.infer_layer_output_shape(
                node.name, node.type, node.params or {}, layer_input_shapes
            )

            # Cache the output shape
            self.shape_cache[node.name] = output_shape
            self.shape_cache[f"{node.name}.output"] = output_shape
            self.layer_output_shapes[node.name] = output_shape

        return self.layer_output_shapes

    def _get_layer_input_shapes(
        self, layer_name: str, input_mappings: dict[str, str | dict[str, str]]
    ) -> dict[str, list[int | None]]:
        """Get input shapes for a specific layer."""
        input_spec = input_mappings.get(layer_name)
        layer_input_shapes = {}

        if input_spec is None:
            # Sequential fallback - use most recent available shape
            available_shapes = [
                (k, v)
                for k, v in self.shape_cache.items()
                if not k.endswith(".output") and k != layer_name
            ]
            if available_shapes:
                last_key, last_shape = available_shapes[-1]
                layer_input_shapes["input"] = last_shape
        elif isinstance(input_spec, str):
            # Single input reference
            if input_spec in self.shape_cache:
                layer_input_shapes["input"] = self.shape_cache[input_spec]
            elif f"{input_spec}.output" in self.shape_cache:
                layer_input_shapes["input"] = self.shape_cache[f"{input_spec}.output"]
            else:
                raise ShapeInferenceError(
                    f"Cannot find shape for input reference: {input_spec}"
                )
        elif isinstance(input_spec, dict):
            # Multi-input mapping
            for input_port, tensor_ref in input_spec.items():
                if tensor_ref in self.shape_cache:
                    layer_input_shapes[input_port] = self.shape_cache[tensor_ref]
                elif f"{tensor_ref}.output" in self.shape_cache:
                    layer_input_shapes[input_port] = self.shape_cache[
                        f"{tensor_ref}.output"
                    ]
                else:
                    raise ShapeInferenceError(
                        f"Cannot find shape for tensor reference: {tensor_ref}"
                    )

        return layer_input_shapes
