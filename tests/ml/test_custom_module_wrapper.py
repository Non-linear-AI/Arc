"""Tests for CustomModuleWrapper to ensure correct multi-input handling."""

from types import SimpleNamespace

import pytest
import torch

from arc.ml.builder import CustomModuleWrapper, ModelBuilder


class TestCustomModuleWrapperBinaryOperations:
    """Test CustomModuleWrapper handling of various multi-input operations."""

    def test_binary_operations_torch_add(self):
        """Test torch.add with 2 inputs (should use positional args)."""
        # Define module with torch.add
        module_def = SimpleNamespace(
            inputs=["x", "y"],
            outputs={"output": "add_op.output"},
            graph=[
                SimpleNamespace(
                    name="add_op",
                    type="torch.add",
                    params={},
                    inputs=["x", "y"],
                ),
            ],
        )

        builder = ModelBuilder(enable_shape_validation=False)
        wrapper = CustomModuleWrapper(module_def, ["x", "y"], builder, None)

        # Test forward pass
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

        result = wrapper(x=x, y=y)
        expected = torch.tensor([[11.0, 22.0], [33.0, 44.0]])

        assert torch.allclose(result, expected)

    def test_binary_operations_torch_mul(self):
        """Test torch.mul with 2 inputs (should use positional args)."""
        module_def = SimpleNamespace(
            inputs=["a", "b"],
            outputs={"output": "mul_op.output"},
            graph=[
                SimpleNamespace(
                    name="mul_op",
                    type="torch.mul",
                    params={},
                    inputs=["a", "b"],
                ),
            ],
        )

        builder = ModelBuilder(enable_shape_validation=False)
        wrapper = CustomModuleWrapper(module_def, ["a", "b"], builder, None)

        a = torch.tensor([[2.0, 3.0]])
        b = torch.tensor([[4.0, 5.0]])

        result = wrapper(a=a, b=b)
        expected = torch.tensor([[8.0, 15.0]])

        assert torch.allclose(result, expected)

    def test_torch_cat_with_two_inputs(self):
        """Test torch.cat with 2 inputs (needs tuple + dim kwarg)."""
        module_def = SimpleNamespace(
            inputs=["tensor1", "tensor2"],
            outputs={"output": "cat_op.output"},
            graph=[
                SimpleNamespace(
                    name="cat_op",
                    type="torch.cat",
                    params={"dim": 1},
                    inputs=["tensor1", "tensor2"],
                ),
            ],
        )

        builder = ModelBuilder(enable_shape_validation=False)
        wrapper = CustomModuleWrapper(module_def, ["tensor1", "tensor2"], builder, None)

        t1 = torch.tensor([[1.0, 2.0]])
        t2 = torch.tensor([[3.0, 4.0]])

        result = wrapper(tensor1=t1, tensor2=t2)
        expected = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        assert torch.allclose(result, expected)

    def test_torch_cat_with_three_inputs(self):
        """Test torch.cat with 3+ inputs (critical test for tuple handling)."""
        module_def = SimpleNamespace(
            inputs=["t1", "t2", "t3"],
            outputs={"output": "cat_op.output"},
            graph=[
                SimpleNamespace(
                    name="cat_op",
                    type="torch.cat",
                    params={"dim": 1},
                    inputs=["t1", "t2", "t3"],
                ),
            ],
        )

        builder = ModelBuilder(enable_shape_validation=False)
        wrapper = CustomModuleWrapper(module_def, ["t1", "t2", "t3"], builder, None)

        t1 = torch.tensor([[1.0]])
        t2 = torch.tensor([[2.0]])
        t3 = torch.tensor([[3.0]])

        result = wrapper(t1=t1, t2=t2, t3=t3)
        expected = torch.tensor([[1.0, 2.0, 3.0]])

        assert torch.allclose(result, expected)

    def test_torch_stack_with_multiple_inputs(self):
        """Test torch.stack with multiple inputs (also needs tuple)."""
        module_def = SimpleNamespace(
            inputs=["a", "b", "c"],
            outputs={"output": "stack_op.output"},
            graph=[
                SimpleNamespace(
                    name="stack_op",
                    type="torch.stack",
                    params={"dim": 0},
                    inputs=["a", "b", "c"],
                ),
            ],
        )

        builder = ModelBuilder(enable_shape_validation=False)
        wrapper = CustomModuleWrapper(module_def, ["a", "b", "c"], builder, None)

        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        c = torch.tensor([5.0, 6.0])

        result = wrapper(a=a, b=b, c=c)
        expected = torch.stack([a, b, c], dim=0)

        assert torch.allclose(result, expected)

    def test_nested_operations_in_module(self):
        """Test module with mixed binary ops (add/mul) and torch.cat."""
        module_def = SimpleNamespace(
            inputs=["x", "y", "z"],
            outputs={"output": "final_cat.output"},
            graph=[
                SimpleNamespace(
                    name="add_xy",
                    type="torch.add",
                    params={},
                    inputs=["x", "y"],
                ),
                SimpleNamespace(
                    name="mul_with_z",
                    type="torch.mul",
                    params={},
                    inputs=["add_xy.output", "z"],
                ),
                SimpleNamespace(
                    name="final_cat",
                    type="torch.cat",
                    params={"dim": 1},
                    inputs=["x", "mul_with_z.output"],
                ),
            ],
        )

        builder = ModelBuilder(enable_shape_validation=False)
        wrapper = CustomModuleWrapper(module_def, ["x", "y", "z"], builder, None)

        x = torch.tensor([[1.0, 2.0]])
        y = torch.tensor([[3.0, 4.0]])
        z = torch.tensor([[2.0, 3.0]])

        result = wrapper(x=x, y=y, z=z)

        # x + y = [4.0, 6.0]
        # (x+y) * z = [8.0, 18.0]
        # cat([x, result]) = [1.0, 2.0, 8.0, 18.0]
        expected = torch.tensor([[1.0, 2.0, 8.0, 18.0]])

        assert torch.allclose(result, expected)


class TestCustomModuleWrapperErrorMessages:
    """Test error messages in CustomModuleWrapper and ModelBuilder."""

    def test_unknown_module_shows_available_modules(self):
        """Test that error for unknown module shows available modules."""
        # Create a model spec with some modules
        from arc.graph.model.spec import (
            GraphNode,
            ModelInput,
            ModelSpec,
            ModuleDefinition,
        )

        # Define some modules
        module1 = ModuleDefinition(
            inputs=["x"],
            outputs={"output": "layer1.output"},
            graph=[
                GraphNode(
                    name="layer1",
                    type="torch.nn.Linear",
                    params={"in_features": 10, "out_features": 5},
                    inputs={"input": "x"},
                )
            ],
        )

        module2 = ModuleDefinition(
            inputs=["x"],
            outputs={"output": "layer1.output"},
            graph=[
                GraphNode(
                    name="layer1",
                    type="torch.nn.ReLU",
                    params={},
                    inputs={"input": "x"},
                )
            ],
        )

        model_spec = ModelSpec(
            inputs={"features": ModelInput(dtype="float32", shape=[None, 10])},
            graph=[
                GraphNode(
                    name="unknown_layer",
                    type="module.nonexistent",  # Module doesn't exist
                    params={},
                    inputs=["features"],
                )
            ],
            outputs={"output": "unknown_layer.output"},
            modules={"mlp_block": module1, "activation": module2},
        )

        builder = ModelBuilder(enable_shape_validation=False)

        with pytest.raises(ValueError) as exc_info:
            builder._build_layer(model_spec.graph[0], model_spec)

        error_message = str(exc_info.value)
        # Should mention the unknown module
        assert "nonexistent" in error_message
        # Should show available modules
        assert "mlp_block" in error_message or "activation" in error_message


class TestCustomModuleWrapperComplexScenarios:
    """Test complex real-world scenarios."""

    def test_dcn_cross_layer_pattern(self):
        """Test the DCN cross layer pattern with element-wise ops."""
        # This is the actual DCN cross layer pattern from the guide
        module_def = SimpleNamespace(
            inputs=["x_current", "x_original"],
            outputs={"cross_output": "residual_add.output"},
            graph=[
                SimpleNamespace(
                    name="linear_transform",
                    type="torch.nn.Linear",
                    params={"in_features": 4, "out_features": 4},
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

        builder = ModelBuilder(enable_shape_validation=False)
        wrapper = CustomModuleWrapper(
            module_def, ["x_current", "x_original"], builder, None
        )

        # Test forward pass
        x_current = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        x_original = torch.tensor([[0.5, 1.0, 1.5, 2.0]])

        result = wrapper(x_current=x_current, x_original=x_original)

        # Result should have shape [1, 4] (dimension preserved)
        assert result.shape == (1, 4)
        # Should not raise any errors
        assert result is not None
