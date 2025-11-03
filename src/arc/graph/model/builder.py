"""Model builder for constructing PyTorch models from Arc-Graph specifications."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from arc.graph.model.components import get_component_class_or_function
from arc.graph.model.spec import GraphNode, ModelSpec, ModuleDefinition
from arc.graph.model.validator import resolve_node_reference


def _apply_tuple_indices(value, indices):
    """Apply a list of indices to a value for nested tuple access."""
    if indices is None:
        return value

    current = value
    for idx in indices:
        current = current[idx]
    return current


class ArcGraphModel(nn.Module):
    """PyTorch model built from Arc-Graph specification."""

    def __init__(self, spec: ModelSpec):
        """Initialize the model from specification.

        Args:
            spec: The Arc-Graph model specification
        """
        super().__init__()
        self.spec = spec
        self.input_names = list(spec.inputs.keys())
        self.output_names = list(spec.outputs.keys())

        # Build embedding layers for categorical inputs
        self.embeddings = self._build_embedding_layers(spec.inputs)

        # Build the model from the specification
        self.custom_modules = self._build_custom_modules(spec.modules or {})
        component_dict, self.execution_order = self._build_main_graph(spec.graph)
        self.graph_modules = component_dict["modules"]
        self.graph_functions = component_dict["functions"]

    def _build_embedding_layers(self, inputs: dict) -> nn.ModuleDict:
        """Build embedding layers for categorical inputs.

        Args:
            inputs: Dictionary of model inputs from spec

        Returns:
            ModuleDict containing embedding layers for categorical inputs

        Raises:
            ValueError: If categorical input missing embedding_dim or vocab_size
        """
        embeddings = nn.ModuleDict()

        for input_name, input_spec in inputs.items():
            if input_spec.categorical:
                # Validate that required embedding parameters are present
                if input_spec.embedding_dim is None or input_spec.vocab_size is None:
                    raise ValueError(
                        f"Categorical input '{input_name}' must specify both "
                        f"'embedding_dim' and 'vocab_size'"
                    )

                # Create embedding layer
                embeddings[input_name] = nn.Embedding(
                    num_embeddings=input_spec.vocab_size,
                    embedding_dim=input_spec.embedding_dim,
                )

        return embeddings

    def _build_custom_modules(
        self, modules: dict[str, ModuleDefinition]
    ) -> dict[str, nn.Module]:
        """Build custom modules as PyTorch nn.Module instances.

        Args:
            modules: Dictionary of module definitions

        Returns:
            Dictionary of built PyTorch modules
        """
        built_modules = {}
        remaining_modules = modules.copy()

        # Build modules in multiple passes to handle dependencies
        max_passes = len(modules) + 1  # Safety limit
        pass_count = 0

        while remaining_modules and pass_count < max_passes:
            built_in_this_pass = []

            for module_name, module_def in list(remaining_modules.items()):
                # Check if all dependencies are satisfied
                dependencies_satisfied = True
                for node in module_def.graph:
                    if node.type.startswith("module."):
                        dep_name = node.type[7:]
                        if dep_name not in built_modules:
                            dependencies_satisfied = False
                            break

                if dependencies_satisfied:
                    # Temporarily set self.custom_modules for nested module building
                    old_custom_modules = getattr(self, "custom_modules", {})
                    self.custom_modules = built_modules

                    built_modules[module_name] = self._build_module_from_definition(
                        module_def, module_name
                    )
                    built_in_this_pass.append(module_name)

                    # Restore old custom_modules
                    self.custom_modules = old_custom_modules

            # Remove built modules from remaining
            for module_name in built_in_this_pass:
                del remaining_modules[module_name]

            # If no modules were built in this pass and we still have remaining modules,
            # there might be circular dependencies or missing modules
            if not built_in_this_pass and remaining_modules:
                remaining_names = list(remaining_modules.keys())
                raise ValueError(
                    f"Cannot resolve module dependencies: {remaining_names}"
                )

            pass_count += 1

        return built_modules

    def _build_module_from_definition(
        self, module_def: ModuleDefinition, module_name: str
    ) -> nn.Module:
        """Build a single custom module from its definition.

        Args:
            module_def: Module definition
            module_name: Name of the module (for error reporting)

        Returns:
            Built PyTorch module
        """

        # Create a mini-model for the module
        class CustomModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_names = module_def.inputs
                self.output_names = list(module_def.outputs.keys())

                # Build internal graph
                component_dict, self.execution_order = builder._build_graph_modules(
                    module_def.graph, set(module_def.inputs), builder.custom_modules
                )
                self.internal_modules = component_dict["modules"]
                self.internal_functions = component_dict["functions"]

            def forward(self, *args, **kwargs):
                # Handle both positional and keyword arguments
                if args and not kwargs:
                    # Positional arguments (for nn.Sequential compatibility)
                    if len(args) != len(self.input_names):
                        raise ValueError(
                            f"Module {module_name} expected {len(self.input_names)} "
                            f"positional arguments, got {len(args)}"
                        )
                    inputs = dict(zip(self.input_names, args, strict=True))
                elif kwargs and not args:
                    # Keyword arguments (original behavior)
                    inputs = kwargs
                    for input_name in self.input_names:
                        if input_name not in inputs:
                            raise ValueError(
                                f"Module {module_name} missing required input: "
                                f"{input_name}"
                            )
                else:
                    raise ValueError(
                        f"Module {module_name} cannot mix positional and keyword "
                        f"arguments"
                    )

                # Execute internal graph
                node_outputs = inputs.copy()  # Start with module inputs
                for node_name in self.execution_order:
                    node_def = next(n for n in module_def.graph if n.name == node_name)

                    # Get the component (module or function)
                    if node_name in self.internal_modules:
                        component = self.internal_modules[node_name]
                        is_module = True
                    else:
                        component = self.internal_functions[node_name]
                        is_module = False

                    # Prepare inputs for this node
                    node_inputs = builder._prepare_node_inputs(node_def, node_outputs)

                    # Execute node
                    if is_module:
                        if isinstance(node_inputs, dict) and len(node_inputs) == 1:
                            # Single input - pass as positional
                            result = component(list(node_inputs.values())[0])
                        elif isinstance(node_inputs, dict):
                            # Multiple named inputs
                            result = component(**node_inputs)
                        else:
                            # List inputs
                            result = component(*node_inputs)
                    else:
                        # Function call
                        params = node_def.params or {}

                        # Special handling for functions that expect tensor sequences
                        if node_def.type in ["torch.cat", "torch.stack"]:
                            if isinstance(node_inputs, list):
                                # For torch.cat/stack: convert list to tuple
                                result = component(tuple(node_inputs), **params)
                            else:
                                result = component(**node_inputs, **params)
                        elif isinstance(node_inputs, dict):
                            # Merge node inputs with params
                            all_args = {**node_inputs, **params}
                            result = component(**all_args)
                        else:
                            # Positional inputs with keyword params
                            result = component(*node_inputs, **params)

                    node_outputs[node_name] = result

                # Return specified outputs
                if len(self.output_names) == 1:
                    output_ref = module_def.outputs[self.output_names[0]]
                    node_name, attr, indices = resolve_node_reference(output_ref)
                    output = node_outputs[node_name]
                    output = _apply_tuple_indices(output, indices)
                    return output
                else:
                    results = {}
                    for output_name, output_ref in module_def.outputs.items():
                        node_name, attr, indices = resolve_node_reference(output_ref)
                        output = node_outputs[node_name]
                        output = _apply_tuple_indices(output, indices)
                        results[output_name] = output
                    return results

        # Keep reference to builder for nested calls
        builder = self
        return CustomModule()

    def _build_main_graph(
        self, graph: list[GraphNode]
    ) -> tuple[nn.ModuleDict, list[str]]:
        """Build the main computation graph.

        Args:
            graph: List of graph nodes

        Returns:
            Tuple of (modules dict, execution order)
        """
        return self._build_graph_modules(
            graph, set(self.input_names), self.custom_modules
        )

    def _build_graph_modules(
        self,
        graph: list[GraphNode],
        _: set[str],  # initial_nodes unused in this context
        custom_modules: dict[str, nn.Module],
    ) -> tuple[dict[str, Any], list[str]]:
        """Build PyTorch modules and functions from graph nodes.

        Args:
            graph: List of graph node definitions
            initial_nodes: Set of initially available nodes (inputs)
            custom_modules: Dictionary of custom modules

        Returns:
            Tuple of (dict of modules/functions, execution order)
        """
        modules = nn.ModuleDict()  # For actual PyTorch modules
        functions = {}  # For PyTorch functions
        execution_order = []

        for node in graph:
            component, component_kind = get_component_class_or_function(node.type)

            if component_kind == "module":
                # Standard PyTorch module
                params = node.params or {}
                modules[node.name] = component(**params)

            elif component_kind == "function":
                # Store function for later execution (not a module)
                functions[node.name] = component

            elif component_kind == "custom_module":
                # Reference to custom module
                module_name = node.type[7:]  # Remove "module." prefix
                if module_name not in custom_modules:
                    raise ValueError(
                        f"Referenced module '{module_name}' not found in available "
                        f"modules"
                    )
                modules[node.name] = copy.deepcopy(custom_modules[module_name])

            elif component_kind == "stack":
                # Handle arc.stack
                module_name = node.params["module"]
                count = node.params["count"]
                base_module = custom_modules[module_name]

                # Create sequential stack
                stack_layers = []
                for _ in range(count):
                    stack_layers.append(copy.deepcopy(base_module))

                modules[node.name] = nn.Sequential(*stack_layers)

            execution_order.append(node.name)

        # Combine modules and functions into a single dictionary
        # We'll use a wrapper class to hold both
        return {"modules": modules, "functions": functions}, execution_order

    def _prepare_node_inputs(
        self, node: GraphNode, available_outputs: dict[str, Any]
    ) -> dict[str, Any] | list[Any]:
        """Prepare inputs for a node based on its input specification.

        Args:
            node: Graph node definition
            available_outputs: Dictionary of available node outputs

        Returns:
            Prepared inputs (dict for named, list for positional)
        """
        if node.inputs is None:
            return {}

        if isinstance(node.inputs, dict):
            # Named inputs
            prepared = {}
            for arg_name, source_ref in node.inputs.items():
                node_name, attr, indices = resolve_node_reference(source_ref)
                value = available_outputs[node_name]
                value = _apply_tuple_indices(value, indices)
                prepared[arg_name] = value
            return prepared
        else:
            # Positional inputs (list)
            prepared = []
            for source_ref in node.inputs:
                node_name, attr, indices = resolve_node_reference(source_ref)
                value = available_outputs[node_name]
                value = _apply_tuple_indices(value, indices)
                prepared.append(value)
            return prepared

    def forward(self, **inputs) -> dict[str, torch.Tensor] | torch.Tensor:
        """Forward pass through the model.

        Args:
            **inputs: Named input tensors matching the model's input specification

        Returns:
            Dictionary of output tensors (or single tensor if only one output)
        """
        # Validate inputs
        for input_name in self.input_names:
            if input_name not in inputs:
                raise ValueError(f"Missing required input: {input_name}")

        # Apply embeddings to categorical inputs
        embedded_inputs = inputs.copy()
        for input_name, embedding_layer in self.embeddings.items():
            categorical_input = embedded_inputs[input_name]
            # Apply embedding (handles both 1D and 2D+ inputs)
            embedded_inputs[input_name] = embedding_layer(categorical_input)

        # Execute the computation graph
        node_outputs = embedded_inputs  # Start with embedded inputs

        for node_name in self.execution_order:
            node_def = next(n for n in self.spec.graph if n.name == node_name)

            # Get the component (module or function)
            if node_name in self.graph_modules:
                component = self.graph_modules[node_name]
                is_module = True
            else:
                component = self.graph_functions[node_name]
                is_module = False

            # Prepare inputs for this node
            node_inputs = self._prepare_node_inputs(node_def, node_outputs)

            # Execute the node
            if is_module:
                # PyTorch module
                if isinstance(node_inputs, dict) and len(node_inputs) == 1:
                    # Single input - pass as positional argument
                    result = component(list(node_inputs.values())[0])
                elif isinstance(node_inputs, dict):
                    # Multiple named inputs - unpack as keyword arguments
                    result = component(**node_inputs)
                else:
                    # List inputs - unpack as positional arguments
                    result = component(*node_inputs)
            else:
                # PyTorch function
                params = node_def.params or {}

                # Special handling for functions that expect tensor sequences
                if node_def.type in ["torch.cat", "torch.stack"]:
                    if isinstance(node_inputs, list):
                        # For torch.cat/stack: convert list to tuple as first arg
                        result = component(tuple(node_inputs), **params)
                    else:
                        result = component(**node_inputs, **params)
                elif isinstance(node_inputs, dict):
                    # Merge node inputs with params
                    all_args = {**node_inputs, **params}
                    result = component(**all_args)
                else:
                    # Positional inputs with keyword params
                    result = component(*node_inputs, **params)

            node_outputs[node_name] = result

        # Prepare final outputs
        if len(self.output_names) == 1:
            # Single output - return tensor directly
            output_name = self.output_names[0]
            output_ref = self.spec.outputs[output_name]
            node_name, attr, indices = resolve_node_reference(output_ref)
            output = node_outputs[node_name]
            output = _apply_tuple_indices(output, indices)
            return output
        else:
            # Multiple outputs - return dictionary
            results = {}
            for output_name, output_ref in self.spec.outputs.items():
                node_name, attr, indices = resolve_node_reference(output_ref)
                output = node_outputs[node_name]
                output = _apply_tuple_indices(output, indices)
                results[output_name] = output
            return results


def build_model_from_spec(spec: ModelSpec) -> ArcGraphModel:
    """Build a PyTorch model from Arc-Graph specification.

    Args:
        spec: The Arc-Graph model specification

    Returns:
        Built PyTorch model

    Raises:
        ValueError: If specification is invalid or model cannot be built
    """
    return ArcGraphModel(spec)


def build_model_from_yaml(yaml_content: str) -> ArcGraphModel:
    """Build a PyTorch model from Arc-Graph YAML string.

    Args:
        yaml_content: YAML string containing model specification

    Returns:
        Built PyTorch model

    Raises:
        ValueError: If YAML is invalid or model cannot be built
    """
    spec = ModelSpec.from_yaml(yaml_content)
    return build_model_from_spec(spec)


def build_model_from_file(file_path: str) -> ArcGraphModel:
    """Build a PyTorch model from Arc-Graph YAML file.

    Args:
        file_path: Path to YAML file containing model specification

    Returns:
        Built PyTorch model

    Raises:
        ValueError: If file is invalid or model cannot be built
    """
    spec = ModelSpec.from_yaml_file(file_path)
    return build_model_from_spec(spec)
