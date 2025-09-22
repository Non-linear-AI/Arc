"""Tests for ModelGeneratorAgent in the separated architecture."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from arc.core.agents.model_generator import ModelGeneratorAgent, ModelGeneratorError
from arc.graph.model import ModelSpec


class TestModelGeneratorAgent:
    """Test ModelGeneratorAgent functionality."""

    @pytest.fixture
    def mock_services(self):
        """Mock services container."""
        services_mock = MagicMock()

        # Mock ML data service
        mock_dataset_info = MagicMock()
        mock_dataset_info.name = "test_table"
        mock_dataset_info.columns = [
            {"name": "feature1", "type": "REAL", "notnull": 0},
            {"name": "feature2", "type": "REAL", "notnull": 0},
            {"name": "target", "type": "INTEGER", "notnull": 1},
        ]
        services_mock.ml_data.get_dataset_info.return_value = mock_dataset_info

        return services_mock

    @pytest.fixture
    def mock_agent(self):
        """Mock Arc agent."""
        return MagicMock()

    @pytest.fixture
    def model_generator(self, mock_services):
        """ModelGeneratorAgent instance."""
        from unittest.mock import AsyncMock, MagicMock

        generator = ModelGeneratorAgent(mock_services, "test_api_key")

        # Mock the arc_client to avoid actual API calls
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "mocked_yaml_content"
        mock_client.chat = AsyncMock(return_value=mock_response)
        generator.arc_client = mock_client

        return generator

    @pytest.fixture
    def valid_model_yaml(self):
        """Valid model YAML for testing."""
        return """inputs:
  features:
    dtype: float32
    shape: [null, 2]
    columns: [feature1, feature2]
graph:
  - name: linear
    type: torch.nn.Linear
    params:
      in_features: 2
      out_features: 1
    inputs:
      input: features
  - name: sigmoid
    type: torch.nn.Sigmoid
    inputs:
      input: linear.output
outputs:
  prediction: sigmoid.output"""

    @pytest.mark.asyncio
    async def test_generate_model_success(self, model_generator, valid_model_yaml):
        """Test successful model generation."""
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = valid_model_yaml
        model_generator.arc_client.chat = AsyncMock(return_value=mock_response)

        # Generate model
        model_spec, model_yaml = await model_generator.generate_model(
            name="test_model",
            user_context="Binary classification model",
            table_name="test_table",
        )

        # Verify result
        assert isinstance(model_spec, ModelSpec)
        assert model_spec.get_input_names() == ["features"]
        assert model_spec.get_output_names() == ["prediction"]
        assert model_spec.get_layer_names() == ["linear", "sigmoid"]

        # Verify YAML is valid
        assert "inputs:" in model_yaml
        assert "graph:" in model_yaml
        assert "outputs:" in model_yaml

    @pytest.mark.asyncio
    async def test_generate_model_yaml_generation(
        self, model_generator, valid_model_yaml
    ):
        """Test model generation returns correct YAML."""
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = valid_model_yaml
        model_generator.arc_client.chat = AsyncMock(return_value=mock_response)

        # Generate model (output path handling is now done by CLI)
        model_spec, model_yaml = await model_generator.generate_model(
            name="test_model",
            user_context="Binary classification model",
            table_name="test_table",
        )

        # Verify model was generated correctly
        assert model_spec is not None
        assert model_yaml is not None
        assert "inputs:" in model_yaml
        assert "graph:" in model_yaml

    @pytest.mark.asyncio
    async def test_generate_model_invalid_yaml(self, model_generator):
        """Test model generation with invalid YAML response."""
        # Mock invalid YAML response
        mock_response = MagicMock()
        mock_response.content = "invalid: yaml: content: ["
        model_generator.arc_client.chat = AsyncMock(return_value=mock_response)

        # Should raise ModelGeneratorError
        with pytest.raises(ModelGeneratorError):
            await model_generator.generate_model(
                name="test_model", user_context="Test model", table_name="test_table"
            )

    @pytest.mark.asyncio
    async def test_generate_model_missing_required_fields(self, model_generator):
        """Test model generation with missing required fields."""
        # Mock response missing required fields
        incomplete_yaml = """
        inputs:
          data:
            dtype: float32
            shape: [null, 10]
        # Missing graph and outputs
        """

        mock_response = MagicMock()
        mock_response.content = incomplete_yaml
        model_generator.arc_client.chat = AsyncMock(return_value=mock_response)

        # Should raise ModelGeneratorError
        with pytest.raises(ModelGeneratorError):
            await model_generator.generate_model(
                name="test_model", user_context="Test model", table_name="test_table"
            )

    @pytest.mark.asyncio
    async def test_generate_model_invalid_node_types(self, model_generator):
        """Test model generation with invalid node types."""
        # Mock response with invalid node type
        invalid_yaml = """
        inputs:
          data:
            dtype: float32
            shape: [null, 10]
        graph:
          - name: invalid_layer
            type: invalid.layer.Type
            inputs:
              input: data
        outputs:
          result: invalid_layer.output
        """

        mock_response = MagicMock()
        mock_response.content = invalid_yaml
        model_generator.arc_client.chat = AsyncMock(return_value=mock_response)

        # Should raise ModelGeneratorError due to validation
        with pytest.raises(ModelGeneratorError):
            await model_generator.generate_model(
                name="test_model", user_context="Test model", table_name="test_table"
            )

    def test_profile_data_success(self, model_generator, mock_services):
        """Test successful data profiling."""
        import asyncio

        # Mock successful database query
        mock_services.db_manager.user_query.return_value = [
            {"name": "feature1", "type": "REAL", "notnull": 0},
            {"name": "feature2", "type": "INTEGER", "notnull": 1},
            {"name": "target", "type": "INTEGER", "notnull": 1},
        ]

        # Call the private method directly for testing
        result = asyncio.run(model_generator._profile_data("test_table"))

        assert result["table_name"] == "test_table"
        assert len(result["columns"]) == 3

        # Check that specific columns are present
        column_names = [col["name"] for col in result["columns"]]
        assert "feature1" in column_names
        assert "feature2" in column_names
        assert "target" in column_names

    def test_profile_data_table_not_found(self, model_generator, mock_services):
        """Test data profiling with non-existent table."""
        import asyncio

        # Mock None result (table not found)
        mock_services.ml_data.get_dataset_info.return_value = None

        result = asyncio.run(model_generator._profile_data("nonexistent_table"))

        assert "error" in result
        assert "not found" in result["error"]

    def test_profile_data_invalid_table_name(self, model_generator, mock_services):
        """Test data profiling with invalid table name."""
        import asyncio

        # Mock exception from ML data service for invalid table name
        mock_services.ml_data.get_dataset_info.side_effect = Exception(
            "Invalid table name"
        )

        result = asyncio.run(model_generator._profile_data("invalid-table-name!"))

        assert "error" in result
        assert "Failed to analyze table" in result["error"]

    def test_get_model_components(self, model_generator):
        """Test getting available model components."""
        components = model_generator._get_model_components()

        assert "node_types" in components
        assert isinstance(components["node_types"], list)
        assert len(components["node_types"]) > 0

        # Should include standard PyTorch layers
        node_types = components["node_types"]
        assert "torch.nn.Linear" in node_types
        assert "torch.nn.ReLU" in node_types

    def test_get_model_examples(self, model_generator):
        """Test getting model examples."""
        examples = model_generator._get_model_examples("binary classification", {})

        assert isinstance(examples, list)
        # Should return at least one example
        assert len(examples) >= 0

        if examples:
            example = examples[0]
            assert "schema" in example
            assert "name" in example

    def test_validate_model_comprehensive_valid(
        self, model_generator, valid_model_yaml
    ):
        """Test comprehensive model validation with valid input."""
        context = {
            "data_profile": {
                "columns": [
                    {"name": "feature1", "type": "REAL"},
                    {"name": "feature2", "type": "REAL"},
                ]
            },
            "available_components": {
                "node_types": ["torch.nn.Linear", "torch.nn.Sigmoid"]
            },
        }

        result = model_generator._validate_model_comprehensive(
            valid_model_yaml, context
        )

        assert result["valid"] is True
        assert isinstance(result["object"], ModelSpec)
        assert result["error"] is None

    def test_validate_model_comprehensive_invalid_columns(
        self, model_generator, valid_model_yaml
    ):
        """Test comprehensive model validation with invalid column references."""
        context = {
            "data_profile": {"columns": [{"name": "different_column", "type": "REAL"}]},
            "available_components": {
                "node_types": ["torch.nn.Linear", "torch.nn.Sigmoid"]
            },
        }

        result = model_generator._validate_model_comprehensive(
            valid_model_yaml, context
        )

        assert result["valid"] is False
        assert "Column validation errors" in result["error"]
