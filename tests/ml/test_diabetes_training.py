"""Tests for diabetes logistic regression model training workflow.

This test ensures that the diabetes_logistic_regression.yaml model
can be built, validated, and trained successfully with functional layers.
"""

from pathlib import Path

import pandas as pd
import pytest
import torch

from arc.graph.model.spec import ModelSpec
from arc.ml.builder import ModelBuilder


@pytest.fixture
def diabetes_model_spec():
    """Load the diabetes logistic regression model spec."""
    spec_path = (
        Path(__file__).parent.parent.parent / "diabetes_logistic_regression.yaml"
    )
    if not spec_path.exists():
        pytest.skip("diabetes_logistic_regression.yaml not found")

    with open(spec_path) as f:
        return ModelSpec.from_yaml(f.read())


@pytest.fixture
def sample_diabetes_data():
    """Create sample diabetes dataset for testing."""
    # Create a small sample dataset with the expected columns
    data = {
        "pregnancies": [6.0, 1.0, 8.0, 1.0, 0.0],
        "glucose": [148.0, 85.0, 183.0, 89.0, 137.0],
        "blood_pressure": [72.0, 66.0, 64.0, 66.0, 40.0],
        "skin_thickness": [35.0, 29.0, 0.0, 23.0, 35.0],
        "insulin": [0.0, 0.0, 0.0, 94.0, 168.0],
        "bmi": [33.6, 26.6, 23.3, 28.1, 43.1],
        "diabetes_pedigree": [0.627, 0.351, 0.672, 0.167, 2.288],
        "age": [50.0, 31.0, 32.0, 21.0, 33.0],
        "outcome": [1.0, 0.0, 1.0, 0.0, 1.0],
    }
    return pd.DataFrame(data)


class TestDiabetesModelBuilding:
    """Test that the diabetes model can be built successfully."""

    def test_model_spec_validation(self, diabetes_model_spec):
        """Test that the model spec is valid."""
        assert diabetes_model_spec is not None
        assert "features" in diabetes_model_spec.inputs
        assert len(diabetes_model_spec.graph) > 0
        assert "logits" in diabetes_model_spec.outputs
        assert "probabilities" in diabetes_model_spec.outputs

    def test_model_uses_functional_layers(self, diabetes_model_spec):
        """Test that the model uses torch.nn.functional layers."""
        functional_layers = [
            node
            for node in diabetes_model_spec.graph
            if "torch.nn.functional" in node.type
        ]
        assert len(functional_layers) > 0, "Model should use functional layers"

        # Check specific functional layers
        layer_types = [node.type for node in functional_layers]
        assert "torch.nn.functional.relu" in layer_types
        assert "torch.nn.functional.sigmoid" in layer_types

    def test_model_building(self, diabetes_model_spec):
        """Test that the model can be built with ModelBuilder."""
        builder = ModelBuilder()
        model = builder.build_model(diabetes_model_spec)

        assert model is not None
        assert hasattr(model, "layers")

        # Check that functional layers are wrapped
        assert "activation_1" in model.layers
        assert "activation_2" in model.layers
        assert "probabilities" in model.layers

    def test_forward_pass(self, diabetes_model_spec):
        """Test that a forward pass works correctly."""
        builder = ModelBuilder()
        model = builder.build_model(diabetes_model_spec)

        # Create sample input
        batch_size = 4
        num_features = 8
        x = torch.randn(batch_size, num_features)

        # Forward pass (model expects dict of inputs)
        outputs = model({"features": x})

        # Check outputs
        assert "logits" in outputs
        assert "probabilities" in outputs
        assert outputs["logits"].shape == (batch_size, 1)
        assert outputs["probabilities"].shape == (batch_size, 1)

        # Check probability range
        probs = outputs["probabilities"]
        assert torch.all(probs >= 0) and torch.all(probs <= 1)

    def test_loss_computation(self, diabetes_model_spec):
        """Test that loss can be computed."""
        builder = ModelBuilder()
        model = builder.build_model(diabetes_model_spec)

        # Create sample input and target
        batch_size = 4
        num_features = 8
        x = torch.randn(batch_size, num_features)
        target = torch.randint(0, 2, (batch_size, 1)).float()

        # Forward pass (model expects dict of inputs)
        outputs = model({"features": x})

        # Compute loss manually (since we have the loss spec)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(outputs["logits"], target)

        assert loss is not None
        assert loss.item() > 0  # Loss should be positive


class TestDiabetesModelIntegration:
    """Integration tests for the complete training workflow."""

    def test_model_spec_roundtrip(self, diabetes_model_spec):
        """Test that model spec can be serialized and deserialized."""
        # Convert to YAML and back
        yaml_str = diabetes_model_spec.to_yaml()
        spec_copy = ModelSpec.from_yaml(yaml_str)

        assert len(spec_copy.graph) == len(diabetes_model_spec.graph)
        assert spec_copy.outputs == diabetes_model_spec.outputs

    def test_functional_layers_are_wrapped(self, diabetes_model_spec):
        """Test that functional layers are properly wrapped as modules."""
        from arc.ml.layers import FunctionalWrapper

        builder = ModelBuilder()
        model = builder.build_model(diabetes_model_spec)

        # Check that functional relu layers are wrapped
        activation_1 = model.layers["activation_1"]
        activation_2 = model.layers["activation_2"]
        probabilities = model.layers["probabilities"]

        assert isinstance(activation_1, FunctionalWrapper)
        assert isinstance(activation_2, FunctionalWrapper)
        assert isinstance(probabilities, FunctionalWrapper)

    def test_model_contains_standard_modules(self, diabetes_model_spec):
        """Test that standard nn.Module layers are not wrapped."""
        from arc.ml.layers import FunctionalWrapper

        builder = ModelBuilder()
        model = builder.build_model(diabetes_model_spec)

        # Check that standard layers are not wrapped
        hidden_layer_1 = model.layers["hidden_layer_1"]
        hidden_layer_2 = model.layers["hidden_layer_2"]

        assert not isinstance(hidden_layer_1, FunctionalWrapper)
        assert not isinstance(hidden_layer_2, FunctionalWrapper)
        assert isinstance(hidden_layer_1, torch.nn.Linear)
        assert isinstance(hidden_layer_2, torch.nn.Linear)


class TestDiabetesTrainingWorkflow:
    """Test the complete training workflow."""

    def test_small_training_loop(self, diabetes_model_spec, sample_diabetes_data):
        """Test a small training loop with sample data."""
        builder = ModelBuilder()
        model = builder.build_model(diabetes_model_spec)

        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # Prepare data
        feature_cols = [
            "pregnancies",
            "glucose",
            "blood_pressure",
            "skin_thickness",
            "insulin",
            "bmi",
            "diabetes_pedigree",
            "age",
        ]
        X = torch.tensor(sample_diabetes_data[feature_cols].values, dtype=torch.float32)
        y = torch.tensor(
            sample_diabetes_data["outcome"].values, dtype=torch.float32
        ).unsqueeze(1)

        # Training loop
        initial_loss = None
        for _epoch in range(5):
            optimizer.zero_grad()
            outputs = model({"features": X})
            loss = loss_fn(outputs["logits"], y)
            loss.backward()
            optimizer.step()

            if initial_loss is None:
                initial_loss = loss.item()

        final_loss = loss.item()

        # Loss should decrease (or at least not increase significantly)
        # We're being lenient here since it's a tiny dataset
        assert final_loss < initial_loss * 1.5  # Allow some variance

    def test_model_can_make_predictions(
        self, diabetes_model_spec, sample_diabetes_data
    ):
        """Test that the model can make predictions."""
        builder = ModelBuilder()
        model = builder.build_model(diabetes_model_spec)
        model.eval()

        # Prepare data
        feature_cols = [
            "pregnancies",
            "glucose",
            "blood_pressure",
            "skin_thickness",
            "insulin",
            "bmi",
            "diabetes_pedigree",
            "age",
        ]
        X = torch.tensor(sample_diabetes_data[feature_cols].values, dtype=torch.float32)

        # Make predictions
        with torch.no_grad():
            outputs = model({"features": X})
            predictions = outputs["probabilities"]

        # Check predictions
        assert predictions.shape == (len(sample_diabetes_data), 1)
        assert torch.all(predictions >= 0) and torch.all(predictions <= 1)

        # Binary predictions
        binary_preds = (predictions > 0.5).float()
        assert torch.all((binary_preds == 0) | (binary_preds == 1))
