import pytest
import torch

from arc.ml.utils import (
    ShapeValidator,
    auto_detect_input_size,
    count_parameters,
    create_sample_data,
    resolve_variable_references,
    validate_tensor_shape,
)


class TestUtilityFunctions:
    """Test ML utility functions."""

    def test_auto_detect_input_size(self):
        """Test automatic input size detection."""
        # Test 2D tensor
        data = torch.randn(100, 10)
        size = auto_detect_input_size(data)
        assert size == 10

        # Test different sizes
        data = torch.randn(50, 25)
        size = auto_detect_input_size(data)
        assert size == 25

    def test_auto_detect_input_size_invalid_shape(self):
        """Test auto detection with invalid tensor shape."""
        # Test 1D tensor
        data = torch.randn(100)
        with pytest.raises(ValueError, match="Expected 2D tensor"):
            auto_detect_input_size(data)

        # Test 3D tensor
        data = torch.randn(10, 5, 3)
        with pytest.raises(ValueError, match="Expected 2D tensor"):
            auto_detect_input_size(data)

    def test_validate_tensor_shape_valid(self):
        """Test tensor shape validation with valid shapes."""
        # Test exact match
        tensor = torch.randn(32, 10)
        validate_tensor_shape(tensor, [32, 10])  # Should not raise

        # Test with None for variable dimensions
        tensor = torch.randn(64, 5, 3)
        validate_tensor_shape(tensor, [None, 5, 3])  # Should not raise

        # Test all None
        tensor = torch.randn(16, 8)
        validate_tensor_shape(tensor, [None, None])  # Should not raise

    def test_validate_tensor_shape_invalid(self):
        """Test tensor shape validation with invalid shapes."""
        tensor = torch.randn(32, 10)

        # Wrong number of dimensions
        with pytest.raises(ValueError, match="dimension mismatch"):
            validate_tensor_shape(tensor, [32])

        # Wrong dimension size
        with pytest.raises(ValueError, match="Shape mismatch at dimension"):
            validate_tensor_shape(tensor, [32, 5])

    def test_resolve_variable_references(self):
        """Test resolving variable references in parameters."""
        var_registry = {
            "vars.n_features": 10,
            "vars.hidden_size": 64,
            "vars.learning_rate": 0.001,
        }

        # Test with variable references
        params = {
            "in_features": "vars.n_features",
            "out_features": "vars.hidden_size",
            "bias": True,
            "dropout_rate": 0.1,  # Non-variable parameter
        }

        resolved = resolve_variable_references(params, var_registry)

        assert resolved["in_features"] == 10
        assert resolved["out_features"] == 64
        assert resolved["bias"] is True
        assert resolved["dropout_rate"] == 0.1

    def test_resolve_variable_references_undefined(self):
        """Test resolving undefined variable references."""
        var_registry = {"vars.n_features": 10}

        params = {"in_features": "vars.undefined_var"}

        with pytest.raises(ValueError, match="Cannot resolve variable reference"):
            resolve_variable_references(params, var_registry)

    def test_resolve_variable_references_no_vars(self):
        """Test resolving parameters with no variable references."""
        var_registry = {"vars.n_features": 10}

        params = {
            "in_features": 5,
            "out_features": 3,
            "bias": False,
        }

        resolved = resolve_variable_references(params, var_registry)

        # Should return unchanged
        assert resolved == params

    def test_create_sample_data(self):
        """Test sample data creation."""
        features, targets = create_sample_data(100, 5, binary_classification=True)

        assert features.shape == (100, 5)
        assert targets.shape == (100,)
        assert features.dtype == torch.float32
        assert targets.dtype == torch.float32

    def test_count_parameters(self):
        """Test parameter counting for models."""
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),  # 10*5 + 5 = 55 parameters
            torch.nn.Linear(5, 1),  # 5*1 + 1 = 6 parameters
        )

        param_count = count_parameters(model)
        assert param_count == 61  # 55 + 6

    def test_count_parameters_no_grad(self):
        """Test parameter counting with non-trainable parameters."""
        model = torch.nn.Linear(10, 5)

        # Disable gradients for all parameters
        for param in model.parameters():
            param.requires_grad = False

        param_count = count_parameters(model)
        assert param_count == 0  # No trainable parameters

    def test_count_parameters_mixed(self):
        """Test parameter counting with mixed trainable/non-trainable."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.Linear(5, 1),
        )

        # Disable gradients for first layer only
        for param in model[0].parameters():
            param.requires_grad = False

        param_count = count_parameters(model)
        assert param_count == 6  # Only second layer parameters


class TestShapeValidator:
    """Test ShapeValidator for model shape inference and validation."""

    def test_module_instance_with_custom_outputs(self):
        """Test that module instances with custom named outputs are properly cached.

        This tests the fix for DCN cross layers where:
        - cross_layer module defines custom output 'cross_output'
        - cross_layer1, cross_layer2 are instances of the module
        - cross_layer2 references cross_layer1.cross_output
        """
        from types import SimpleNamespace

        # Define the cross_layer module with custom output name
        cross_layer_module = SimpleNamespace(
            inputs=["x_current", "x_original"],
            outputs={"cross_output": "residual_add.output"},  # Custom output name
            graph=[
                SimpleNamespace(
                    name="linear_transform",
                    type="torch.nn.Linear",
                    params={"in_features": 100, "out_features": 100},
                    inputs={"input": "x_current"},
                ),
                SimpleNamespace(
                    name="element_wise_product",
                    type="torch.mul",
                    params={},
                    inputs=["x_original", "linear_transform.output"],
                ),
                SimpleNamespace(
                    name="residual_add",
                    type="torch.add",
                    params={},
                    inputs=["element_wise_product.output", "x_current"],
                ),
            ],
        )

        # Define model graph with module instances
        graph_nodes = [
            # First cross layer instance
            SimpleNamespace(
                name="cross_layer1",
                type="module.cross_layer",
                params={},
                inputs={"x_current": "features", "x_original": "features"},
            ),
            # Second cross layer instance referencing first's custom output
            SimpleNamespace(
                name="cross_layer2",
                type="module.cross_layer",
                params={},
                inputs={
                    "x_current": "cross_layer1.cross_output",  # Reference custom output
                    "x_original": "features",
                },
            ),
            # Final layer
            SimpleNamespace(
                name="output_layer",
                type="torch.nn.Linear",
                params={"in_features": 100, "out_features": 1},
                inputs={"input": "cross_layer2.cross_output"},
            ),
        ]

        # Define input specs
        input_specs = {
            "features": SimpleNamespace(shape=[None, 100]),
        }

        # Define input mappings (what each layer uses as inputs)
        input_mappings = {
            "cross_layer1": {"x_current": "features", "x_original": "features"},
            "cross_layer2": {
                "x_current": "cross_layer1.cross_output",
                "x_original": "features",
            },
            "output_layer": {"input": "cross_layer2.cross_output"},
        }

        # Create validator with module definition
        validator = ShapeValidator(modules={"cross_layer": cross_layer_module})

        # Validate shapes - this should succeed with the fix
        shapes = validator.validate_model_shapes(
            input_specs=input_specs,
            graph_nodes=graph_nodes,
            input_mappings=input_mappings,
        )

        # Verify layer output shapes are present
        assert "cross_layer1" in shapes
        assert "cross_layer2" in shapes
        assert "output_layer" in shapes

        # Verify custom outputs are cached in shape_cache
        assert "cross_layer1.cross_output" in validator.shape_cache  # Custom output
        assert "cross_layer2.cross_output" in validator.shape_cache  # Custom output

        # Verify shape values are correct
        assert shapes["cross_layer1"] == [None, 100]
        assert shapes["cross_layer2"] == [None, 100]
        assert shapes["output_layer"] == [None, 1]
        assert validator.shape_cache["cross_layer1.cross_output"] == [None, 100]
        assert validator.shape_cache["cross_layer2.cross_output"] == [None, 100]

    def test_module_instance_without_custom_outputs(self):
        """Test module instances that use default 'output' name."""
        from types import SimpleNamespace

        # Module with default output name
        mlp_module = SimpleNamespace(
            inputs=["input_features"],
            outputs={"output": "dense2.output"},  # Default output name
            graph=[
                SimpleNamespace(
                    name="dense1",
                    type="torch.nn.Linear",
                    params={"in_features": 50, "out_features": 128},
                    inputs={"input": "input_features"},
                ),
                SimpleNamespace(
                    name="relu",
                    type="torch.nn.functional.relu",
                    params={},
                    inputs={"input": "dense1.output"},
                ),
                SimpleNamespace(
                    name="dense2",
                    type="torch.nn.Linear",
                    params={"in_features": 128, "out_features": 64},
                    inputs={"input": "relu.output"},
                ),
            ],
        )

        graph_nodes = [
            SimpleNamespace(
                name="mlp_branch",
                type="module.mlp",
                params={},
                inputs=["features"],
            ),
            SimpleNamespace(
                name="final_layer",
                type="torch.nn.Linear",
                params={"in_features": 64, "out_features": 1},
                inputs={"input": "mlp_branch.output"},  # Using default output
            ),
        ]

        input_specs = {"features": SimpleNamespace(shape=[None, 50])}
        input_mappings = {
            "mlp_branch": ["features"],
            "final_layer": {"input": "mlp_branch.output"},
        }

        validator = ShapeValidator(modules={"mlp": mlp_module})

        shapes = validator.validate_model_shapes(
            input_specs=input_specs,
            graph_nodes=graph_nodes,
            input_mappings=input_mappings,
        )

        assert "mlp_branch" in shapes
        assert "final_layer" in shapes
        assert shapes["mlp_branch"] == [None, 64]
        assert shapes["final_layer"] == [None, 1]
        # Also check that default output is cached
        assert "mlp_branch.output" in validator.shape_cache

    def test_module_instance_with_multiple_custom_outputs(self):
        """Test module with multiple custom named outputs."""
        from types import SimpleNamespace

        # Module that outputs multiple tensors
        split_module = SimpleNamespace(
            inputs=["input_data"],
            outputs={
                "left_output": "left_branch.output",
                "right_output": "right_branch.output",
            },
            graph=[
                SimpleNamespace(
                    name="left_branch",
                    type="torch.nn.Linear",
                    params={"in_features": 100, "out_features": 50},
                    inputs={"input": "input_data"},
                ),
                SimpleNamespace(
                    name="right_branch",
                    type="torch.nn.Linear",
                    params={"in_features": 100, "out_features": 30},
                    inputs={"input": "input_data"},
                ),
            ],
        )

        graph_nodes = [
            SimpleNamespace(
                name="splitter",
                type="module.split",
                params={},
                inputs=["features"],
            ),
            SimpleNamespace(
                name="combine",
                type="torch.cat",
                params={"dim": 1},
                inputs=["splitter.left_output", "splitter.right_output"],
            ),
        ]

        input_specs = {"features": SimpleNamespace(shape=[None, 100])}
        input_mappings = {
            "splitter": ["features"],
            "combine": ["splitter.left_output", "splitter.right_output"],
        }

        validator = ShapeValidator(modules={"split": split_module})

        shapes = validator.validate_model_shapes(
            input_specs=input_specs,
            graph_nodes=graph_nodes,
            input_mappings=input_mappings,
        )

        # Both custom outputs should be cached in shape_cache
        assert "splitter.left_output" in validator.shape_cache
        assert "splitter.right_output" in validator.shape_cache
        assert validator.shape_cache["splitter.left_output"] == [None, 50]
        assert validator.shape_cache["splitter.right_output"] == [None, 30]
        assert shapes["combine"] == [None, 80]  # 50 + 30

    def test_module_instance_unknown_module(self):
        """Test error handling when module instance references unknown module."""
        from types import SimpleNamespace

        from arc.ml.utils import ShapeInferenceError

        graph_nodes = [
            SimpleNamespace(
                name="unknown_instance",
                type="module.nonexistent",  # Module not defined
                params={},
                inputs=["features"],
            ),
        ]

        input_specs = {"features": SimpleNamespace(shape=[None, 100])}
        input_mappings = {"unknown_instance": ["features"]}

        validator = ShapeValidator(modules={})

        with pytest.raises(
            ShapeInferenceError, match="references unknown module: nonexistent"
        ):
            validator.validate_model_shapes(
                input_specs=input_specs,
                graph_nodes=graph_nodes,
                input_mappings=input_mappings,
            )
