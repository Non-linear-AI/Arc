"""PyTorch model builder from Arc-Graph specifications."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from arc.graph import GraphNode, ModelSpec
from arc.ml.utils import (
    ShapeInferenceError,
    ShapeValidator,
    resolve_variable_references,
)


class CustomModuleWrapper(nn.Module):
    """Wrapper for custom module definitions.

    This class wraps a custom module definition (from the modules: section)
    and executes its internal graph.
    """

    def __init__(
        self,
        module_def: Any,
        input_names: list[str],
        builder: ModelBuilder,
        model_spec: ModelSpec | None = None,
    ):
        super().__init__()
        self.module_def = module_def
        self.input_names = input_names

        # Build internal layers
        self.layers = nn.ModuleDict()
        for node in module_def.graph:
            layer = builder._build_layer(node, model_spec)
            self.layers[node.name] = layer

        # Compute execution order for module's internal graph
        self.execution_order, self.input_mappings = self._compute_execution_order()

        # Store output mapping
        self.output_mapping = (
            module_def.outputs if hasattr(module_def, "outputs") else {}
        )

    def get_output_names(self) -> list[str]:
        """Get list of custom output names defined by this module."""
        return list(self.output_mapping.keys()) if self.output_mapping else []

    def _compute_execution_order(self) -> tuple[list[str], dict[str, Any]]:
        """Compute execution order for module's internal graph."""
        dependencies = {}
        input_mappings = {}

        for node in self.module_def.graph:
            node_deps = set()

            if hasattr(node, "inputs") and node.inputs:
                input_mappings[node.name] = node.inputs

                # Extract dependencies
                if isinstance(node.inputs, dict):
                    for tensor_ref in node.inputs.values():
                        if isinstance(tensor_ref, str):
                            dep = self._extract_dependency(tensor_ref)
                            if dep:
                                node_deps.add(dep)
                elif isinstance(node.inputs, list):
                    for tensor_ref in node.inputs:
                        if isinstance(tensor_ref, str):
                            dep = self._extract_dependency(tensor_ref)
                            if dep:
                                node_deps.add(dep)
                elif isinstance(node.inputs, str):
                    dep = self._extract_dependency(node.inputs)
                    if dep:
                        node_deps.add(dep)

            dependencies[node.name] = node_deps

        # Topological sort
        execution_order = []
        remaining = {node.name for node in self.module_def.graph}
        declaration_index = {
            node.name: idx for idx, node in enumerate(self.module_def.graph)
        }

        while remaining:
            ready = [node for node in remaining if not (dependencies[node] & remaining)]
            if not ready:
                raise ValueError(f"Circular dependency in module {self.module_def}")
            next_node = min(ready, key=lambda n: declaration_index.get(n, float("inf")))
            execution_order.append(next_node)
            remaining.remove(next_node)

        return execution_order, input_mappings

    def _extract_dependency(self, tensor_ref: str) -> str | None:
        """Extract node dependency from tensor reference."""
        if "." in tensor_ref:
            node_name = tensor_ref.split(".")[0]
            if any(node.name == node_name for node in self.module_def.graph):
                return node_name

        if any(node.name == tensor_ref for node in self.module_def.graph):
            return tensor_ref

        return None

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """Forward pass through the custom module."""
        # Initialize tensor cache with module inputs
        tensor_cache = {}
        for input_name, tensor in inputs.items():
            tensor_cache[input_name] = tensor

        # Execute internal graph
        for layer_name in self.execution_order:
            layer = self.layers[layer_name]
            layer_inputs = self._resolve_layer_inputs(layer_name, tensor_cache)

            if isinstance(layer_inputs, dict):
                # Unpack dict as keyword arguments
                layer_output = layer(**layer_inputs)
            else:
                # Single input
                layer_output = layer(layer_inputs)

            tensor_cache[layer_name] = layer_output
            tensor_cache[f"{layer_name}.output"] = layer_output

        # Return outputs based on output mapping
        if self.output_mapping:
            if len(self.output_mapping) == 1:
                # Single output - return tensor directly
                output_ref = next(iter(self.output_mapping.values()))
                return self._resolve_tensor_reference(output_ref, tensor_cache)
            else:
                # Multiple outputs - return dict
                outputs = {}
                for output_name, output_ref in self.output_mapping.items():
                    outputs[output_name] = self._resolve_tensor_reference(
                        output_ref, tensor_cache
                    )
                return outputs
        else:
            # No output mapping - return last layer output
            last_layer = self.execution_order[-1]
            return tensor_cache[last_layer]

    def _resolve_layer_inputs(
        self, layer_name: str, tensor_cache: dict[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Resolve input tensors for a layer."""
        input_spec = self.input_mappings.get(layer_name)

        if input_spec is None:
            return self._sequential_fallback(tensor_cache)

        if isinstance(input_spec, str):
            return self._resolve_tensor_reference(input_spec, tensor_cache)

        if isinstance(input_spec, list):
            # For list inputs, convert to dict with numeric keys (input_0, input_1, etc.)
            # FunctionalWrapper will handle both binary ops (add, mul) and tuple ops (cat, stack)
            resolved_inputs = {}
            for i, tensor_ref in enumerate(input_spec):
                resolved_inputs[f"input_{i}"] = self._resolve_tensor_reference(
                    tensor_ref, tensor_cache
                )
            return resolved_inputs

        if isinstance(input_spec, dict):
            resolved_inputs = {}
            for input_port, tensor_ref in input_spec.items():
                resolved_inputs[input_port] = self._resolve_tensor_reference(
                    tensor_ref, tensor_cache
                )
            return resolved_inputs

        raise ValueError(f"Invalid input spec for {layer_name}: {input_spec}")

    def _resolve_tensor_reference(
        self, reference: str, tensor_cache: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Resolve a tensor reference to an actual tensor."""
        if reference in tensor_cache:
            return tensor_cache[reference]

        if not reference.endswith(".output"):
            output_ref = f"{reference}.output"
            if output_ref in tensor_cache:
                return tensor_cache[output_ref]

        raise ValueError(f"Cannot resolve tensor reference: {reference}")

    def _sequential_fallback(
        self, tensor_cache: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Fallback for sequential execution."""
        non_input_tensors = [
            (k, v)
            for k, v in tensor_cache.items()
            if k not in self.input_names and not k.endswith(".output")
        ]

        if not non_input_tensors:
            # Use first module input
            first_input = self.input_names[0]
            return tensor_cache[first_input]

        return non_input_tensors[-1][1]


class ArcModel(nn.Module):
    """PyTorch model built from Arc-Graph specification."""

    def __init__(
        self,
        layers: nn.ModuleDict,
        input_names: list[str],
        output_mapping: dict[str, str],
        execution_order: list[str],
        input_mappings: dict[str, str | dict[str, str]],
    ):
        super().__init__()
        self.layers = layers
        self.input_names = input_names
        self.output_mapping = output_mapping
        self.execution_order = execution_order
        self.input_mappings = input_mappings

    def forward(
        self, inputs: dict[str, torch.Tensor] | torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the model with DAG execution.

        Args:
            inputs: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output names to tensors
        """
        if isinstance(inputs, torch.Tensor):
            if len(self.input_names) != 1:
                raise ValueError(
                    "Tensor input provided but model requires multiple named inputs"
                )
            inputs = {self.input_names[0]: inputs}

        # Validate inputs
        for input_name in self.input_names:
            if input_name not in inputs:
                raise ValueError(f"Missing required input: {input_name}")

        # Store intermediate outputs with input tensors
        tensor_cache = {}
        tensor_cache.update(inputs)  # Add model inputs

        # Execute layers in topological order
        for layer_name in self.execution_order:
            layer = self.layers[layer_name]

            # Resolve layer inputs from cache
            layer_inputs = self._resolve_layer_inputs(layer_name, tensor_cache)

            # Execute layer with proper input handling
            if isinstance(layer_inputs, dict):
                # Multi-input layer
                layer_output = layer(**layer_inputs)
            else:
                # Single-input layer
                layer_output = layer(layer_inputs)

            # Store output with both layer name and explicit output port
            tensor_cache[layer_name] = layer_output
            tensor_cache[f"{layer_name}.output"] = layer_output

            # If layer returns a dict (custom module with multiple outputs),
            # cache each named output separately
            if isinstance(layer_output, dict):
                for output_name, output_tensor in layer_output.items():
                    tensor_cache[f"{layer_name}.{output_name}"] = output_tensor
            # If layer is a CustomModuleWrapper with single custom output,
            # cache it with the custom output name too
            elif isinstance(layer, CustomModuleWrapper):
                output_names = layer.get_output_names()
                if len(output_names) == 1:
                    # Single custom output - also cache with custom name
                    custom_name = output_names[0]
                    tensor_cache[f"{layer_name}.{custom_name}"] = layer_output

        # Map to final model outputs
        final_outputs = {}
        for output_name, source in self.output_mapping.items():
            resolved_tensor = self._resolve_tensor_reference(source, tensor_cache)
            final_outputs[output_name] = resolved_tensor

        return final_outputs

    def _resolve_layer_inputs(
        self, layer_name: str, tensor_cache: dict[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Resolve input tensors for a layer from the tensor cache.

        Args:
            layer_name: Name of the layer to get inputs for
            tensor_cache: Cache of available tensors

        Returns:
            Single tensor or dict of tensors for multi-input layers
        """
        input_spec = self.input_mappings.get(layer_name)

        if input_spec is None:
            # No explicit input mapping - use sequential fallback
            return self._sequential_input_fallback(layer_name, tensor_cache)

        if isinstance(input_spec, str):
            # Single input reference
            return self._resolve_tensor_reference(input_spec, tensor_cache)

        if isinstance(input_spec, list):
            # List of inputs - convert to dict keyed by module's expected input names
            layer = self.layers[layer_name]
            if isinstance(layer, CustomModuleWrapper):
                # Custom module - use its defined input names
                if len(input_spec) != len(layer.input_names):
                    raise ValueError(
                        f"Layer {layer_name} expects {len(layer.input_names)} inputs "
                        f"but got {len(input_spec)}"
                    )
                resolved_inputs = {}
                for input_name, tensor_ref in zip(
                    layer.input_names, input_spec, strict=False
                ):
                    resolved_inputs[input_name] = self._resolve_tensor_reference(
                        tensor_ref, tensor_cache
                    )
                return resolved_inputs
            elif len(input_spec) == 1:
                # Single input in list form - unwrap to single tensor
                return self._resolve_tensor_reference(input_spec[0], tensor_cache)
            else:
                # Multiple inputs without defined names - use numeric keys
                resolved_inputs = {}
                for i, tensor_ref in enumerate(input_spec):
                    resolved_inputs[f"input_{i}"] = self._resolve_tensor_reference(
                        tensor_ref, tensor_cache
                    )
                return resolved_inputs

        if isinstance(input_spec, dict):
            # Multi-input mapping
            resolved_inputs = {}
            for input_port, tensor_ref in input_spec.items():
                resolved_inputs[input_port] = self._resolve_tensor_reference(
                    tensor_ref, tensor_cache
                )
            return resolved_inputs

        raise ValueError(
            f"Invalid input specification for layer {layer_name}: {input_spec}"
        )

    def _resolve_tensor_reference(
        self, reference: str, tensor_cache: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Resolve a tensor reference to an actual tensor.

        Args:
            reference: Tensor reference (e.g., 'layer1', 'layer1.output', 'input_name')
            tensor_cache: Cache of available tensors

        Returns:
            Resolved tensor
        """
        if reference in tensor_cache:
            return tensor_cache[reference]

        # Try with .output suffix for node references
        if not reference.endswith(".output"):
            output_ref = f"{reference}.output"
            if output_ref in tensor_cache:
                return tensor_cache[output_ref]

        raise ValueError(f"Cannot resolve tensor reference: {reference}")

    def _sequential_input_fallback(
        self, _layer_name: str, tensor_cache: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Fallback for sequential execution when no explicit input mapping exists."""
        # Get all non-input tensors (layer outputs)
        layer_outputs = [
            (k, v)
            for k, v in tensor_cache.items()
            if k not in self.input_names and not k.endswith(".output")
        ]

        if not layer_outputs:
            # First layer - use first declared model input for determinism
            first_input = self.input_names[0]
            if first_input in tensor_cache:
                return tensor_cache[first_input]
            # Fallback to first available tensor
            return list(tensor_cache.values())[0]

        # Use the most recent layer output
        return layer_outputs[-1][1]


class ModelBuilder:
    """Builds PyTorch models from Arc-Graph specifications."""

    def __init__(self, enable_shape_validation: bool = True):
        self.var_registry: dict[str, Any] = {}
        self.enable_shape_validation = enable_shape_validation
        self.shape_validator: ShapeValidator | None = None

    def build_model(
        self, graph: ModelSpec, sample_data: torch.Tensor | None = None
    ) -> ArcModel:
        """Build a PyTorch model from ModelSpec specification.

        Args:
            graph: ModelSpec specification
            sample_data: Optional sample data for auto-detecting input sizes

        Returns:
            Built PyTorch model

        Raises:
            ValueError: If model building fails
            ShapeInferenceError: If shape validation fails
        """
        # Don't reset variable registry - preserve manually set variables
        # Only clear if we have sample data for auto-detection
        if sample_data is not None:
            self.var_registry.clear()

        # Auto-detect input sizes if sample data provided
        if sample_data is not None:
            self._auto_detect_sizes(graph, sample_data)

        # Initialize shape validator
        if self.enable_shape_validation:
            self.shape_validator = ShapeValidator(self.var_registry, graph.modules)

        # Compute execution order and input mappings
        execution_order, input_mappings = self._compute_execution_order(graph)

        # Validate shapes if enabled
        if self.enable_shape_validation and self.shape_validator:
            try:
                # Convert sample_data to input dict for validation
                sample_inputs = None
                if sample_data is not None:
                    first_input_name = next(iter(graph.inputs.keys()))
                    sample_inputs = {first_input_name: sample_data}

                # Sort nodes in execution order
                node_dict = {node.name: node for node in graph.graph}
                ordered_nodes = [
                    node_dict[name] for name in execution_order if name in node_dict
                ]

                self.shape_validator.validate_model_shapes(
                    graph.inputs, ordered_nodes, input_mappings, sample_inputs
                )
            except ShapeInferenceError as e:
                raise ValueError(f"Shape validation failed: {e}") from e

        # Build layers
        layers = self._build_layers(graph)

        # Get input names and output mapping
        input_names = list(graph.inputs.keys())
        output_mapping = graph.outputs

        return ArcModel(
            layers, input_names, output_mapping, execution_order, input_mappings
        )

    def _auto_detect_sizes(
        self, _model_spec: ModelSpec, sample_data: torch.Tensor
    ) -> None:
        """Auto-detect input sizes from sample data."""
        if sample_data.dim() != 2:
            raise ValueError("Sample data must be 2D [batch_size, features]")

        num_features = sample_data.shape[1]

        # Register common size variables
        self.var_registry["vars.n_features"] = num_features
        self.var_registry["vars.input_size"] = num_features
        self.var_registry["vars.batch_size"] = sample_data.shape[0]

    def _build_layers(self, model_spec: ModelSpec) -> nn.ModuleDict:
        """Build PyTorch layers from model specification."""
        layers = nn.ModuleDict()

        for node in model_spec.graph:
            layer = self._build_layer(node, model_spec)
            layers[node.name] = layer

        return layers

    def _build_layer(
        self, node: GraphNode, model_spec: ModelSpec | None = None
    ) -> nn.Module:
        """Build a single PyTorch layer from graph node."""
        # Get component (layer class or function)
        from arc.graph.model.components import get_component_class_or_function

        component, component_kind = get_component_class_or_function(node.type)

        # Resolve parameters
        params = node.params or {}
        resolved_params = resolve_variable_references(params, self.var_registry)

        # Create layer based on component kind
        try:
            if component_kind == "module":
                # Standard module instantiation
                return component(**resolved_params)
            elif component_kind == "function":
                # Wrap function in a module
                from arc.ml.layers import FunctionalWrapper

                return FunctionalWrapper(component, **resolved_params)
            elif component_kind == "custom_module":
                # Build custom module from module definition
                # Extract module name from type (e.g., "module.cross_layer" -> "cross_layer")
                module_name = node.type.split(".", 1)[1]

                # Get module definition from model spec
                if model_spec is None:
                    raise ValueError(
                        f"Cannot build custom module '{node.name}': model_spec not provided"
                    )

                if (
                    not hasattr(model_spec, "modules")
                    or module_name not in model_spec.modules
                ):
                    # Show available modules in error message
                    available = []
                    if hasattr(model_spec, "modules") and model_spec.modules:
                        available = sorted(model_spec.modules.keys())

                    if available:
                        available_str = ", ".join(available)
                        raise ValueError(
                            f"Custom module '{module_name}' not found in model specification. "
                            f"Available modules: {available_str}"
                        )
                    else:
                        raise ValueError(
                            f"Custom module '{module_name}' not found. "
                            f"No modules defined in model specification."
                        )

                module_def = model_spec.modules[module_name]

                # Get input names from module definition
                input_names = module_def.inputs if hasattr(module_def, "inputs") else []

                # Create custom module wrapper
                return CustomModuleWrapper(module_def, input_names, self, model_spec)
            else:
                raise ValueError(
                    f"Unsupported component kind '{component_kind}' "
                    f"for layer '{node.name}'"
                )
        except Exception as e:
            raise ValueError(
                f"Failed to create layer '{node.name}' of type '{node.type}': {e}"
            ) from e

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable value manually.

        Args:
            name: Variable name (e.g., "vars.n_features")
            value: Variable value
        """
        self.var_registry[name] = value

    def get_variable(self, name: str) -> Any:
        """Get a variable value.

        Args:
            name: Variable name

        Returns:
            Variable value

        Raises:
            KeyError: If variable not found
        """
        return self.var_registry[name]

    def get_shape_info(self) -> dict[str, list[int | None]]:
        """Get inferred shape information for all layers.

        Returns:
            Dictionary mapping layer names to their output shapes
        """
        if self.shape_validator:
            return self.shape_validator.layer_output_shapes
        return {}

    def print_shape_summary(self) -> None:
        """Print a summary of inferred shapes for debugging."""
        shape_info = self.get_shape_info()
        if not shape_info:
            print("No shape information available (shape validation may be disabled)")
            return

        print("\n=== Model Shape Summary ===")
        for layer_name, shape in shape_info.items():
            shape_str = (
                "["
                + ", ".join(str(dim) if dim is not None else "?" for dim in shape)
                + "]"
            )
            print(f"{layer_name}: {shape_str}")
        print("==========================\n")

    def _compute_execution_order(
        self, model_spec: ModelSpec
    ) -> tuple[list[str], dict[str, str | dict[str, str]]]:
        """Compute topological execution order and input mappings for the graph.

        Args:
            model_spec: Model specification with graph nodes

        Returns:
            Tuple of (execution_order, input_mappings)
        """
        # Build dependency graph
        dependencies = {}  # node_name -> set of dependencies
        input_mappings = {}  # node_name -> input specification

        for node in model_spec.graph:
            node_deps = set()

            # Extract dependencies from inputs field
            if hasattr(node, "inputs") and node.inputs:
                input_mappings[node.name] = node.inputs

                # Parse input references to find dependencies
                if isinstance(node.inputs, dict):
                    for tensor_ref in node.inputs.values():
                        if isinstance(tensor_ref, str):
                            dep = self._extract_dependency(tensor_ref, model_spec)
                            if dep:
                                node_deps.add(dep)
                elif isinstance(node.inputs, str):
                    dep = self._extract_dependency(node.inputs, model_spec)
                    if dep:
                        node_deps.add(dep)

            dependencies[node.name] = node_deps

        # Topological sort
        execution_order = []
        remaining = {node.name for node in model_spec.graph}
        # Preserve original YAML declaration order when multiple nodes are ready
        declaration_index = {
            node.name: idx for idx, node in enumerate(model_spec.graph)
        }

        while remaining:
            # Find nodes with no remaining dependencies
            ready = [node for node in remaining if not (dependencies[node] & remaining)]

            if not ready:
                msg = (
                    "Circular dependency detected in graph. Remaining nodes: "
                    f"{remaining}"
                )
                raise ValueError(msg)

            # Pick the next node in original declaration order
            next_node = min(ready, key=lambda n: declaration_index.get(n, float("inf")))
            execution_order.append(next_node)
            remaining.remove(next_node)

        return execution_order, input_mappings

    def _extract_dependency(self, tensor_ref: str, model_spec: ModelSpec) -> str | None:
        """Extract node dependency from tensor reference.

        Args:
            tensor_ref: Tensor reference string
            model_spec: Model specification

        Returns:
            Node name that this reference depends on, or None if it's a model input
        """
        # Handle node.output references
        if "." in tensor_ref:
            node_name = tensor_ref.split(".")[0]
            # Check if it's a graph node (not a model input)
            if any(node.name == node_name for node in model_spec.graph):
                return node_name

        # Handle direct node references
        if any(node.name == tensor_ref for node in model_spec.graph):
            return tensor_ref

        # Must be a model input or other reference
        return None

    def validate_model_at_runtime(
        self, graph: ModelSpec, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Validate model inputs and outputs at runtime.

        Args:
            graph: ModelSpec specification
            inputs: Input tensors to validate

        Returns:
            Validated inputs (potentially with corrections)

        Raises:
            ShapeInferenceError: If validation fails
        """
        if not self.enable_shape_validation:
            return inputs

        validator = ShapeValidator(self.var_registry, graph.modules)
        validator.validate_input_shapes(inputs, graph.model.inputs)
        return inputs
