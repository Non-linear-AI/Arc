"""Tests for the example repository system."""

from arc.core.agents.shared.example_repository import (
    ExampleRepository,
    ModelExample,
)


class TestModelExample:
    """Test model example data structure."""

    def test_model_example_creation(self):
        """Test creating a model example."""
        data_profile = {
            "table_name": "test_data",
            "columns": [{"name": "feature1", "type": "INTEGER"}],
            "task_type": "binary_classification",
        }

        example_schema = (
            "inputs:\n  health_data:\n    dtype: float32\n    shape: [null, 3]\n"
        )
        example = ModelExample(
            name="Test Example",
            user_intent="Test binary classification",
            schema=example_schema,
            data_profile=data_profile,
            explanation="This is a test",
        )

        assert example.name == "Test Example"
        assert example.user_intent == "Test binary classification"
        assert example.schema == example_schema
        assert example.data_profile == data_profile
        assert example.explanation == "This is a test"


class TestExampleRepository:
    """Test the example repository."""

    def test_repository_initialization(self):
        """Test that repository initializes with builtin examples."""
        repo = ExampleRepository()

        assert len(repo.model_examples) >= 1  # At least diabetes example
        example_names = [ex.name for ex in repo.model_examples]

        # Check for diabetes example
        assert any("Diabetes" in name for name in example_names)

    def test_get_examples_by_type(self):
        """Test filtering examples by use case type."""
        repo = ExampleRepository()

        examples = repo.retrieve_relevant_model_examples(
            "binary classification", {}, max_examples=5
        )

        assert len(examples) >= 1  # At least diabetes example
        assert any("Diabetes" in ex.name for ex in examples)

    def test_retrieve_relevant_examples_binary(self):
        """Test retrieving relevant examples for binary classification."""
        repo = ExampleRepository()

        context = "Create a binary classifier for diabetes prediction"
        data_profile = {
            "task_type": "binary_classification",
            "num_features": 3,
        }

        examples = repo.retrieve_relevant_model_examples(
            context, data_profile, max_examples=2
        )

        assert len(examples) >= 1  # At least diabetes example

        # Should return diabetes example
        top_example = examples[0]
        assert "diabetes" in top_example.name.lower()

    def test_retrieve_relevant_examples_regression(self):
        """Test retrieving relevant examples for regression."""
        repo = ExampleRepository()

        context = "Build a regression model to predict house prices"
        data_profile = {
            "task_type": "regression",
            "num_features": 4,
        }

        examples = repo.retrieve_relevant_model_examples(
            context, data_profile, max_examples=2
        )

        assert len(examples) >= 1  # At least one example available
        # Note: May return diabetes example as closest match

    def test_max_examples_limit(self):
        """Test that max_examples parameter is respected."""
        repo = ExampleRepository()

        context = "Create any kind of ML model"
        data_profile = {}

        # Test different limits
        for max_examples in [1, 2, 3]:
            examples = repo.retrieve_relevant_model_examples(
                context, data_profile, max_examples
            )
            assert len(examples) >= 1  # At least one example available
            assert len(examples) <= max_examples
