"""Tests for MLModelAgent in the separated architecture."""

import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from arc.core.agents.ml_model import MLModelAgent, MLModelError
from arc.graph.model import ModelSpec


class TestMLModelAgent:
    """Test MLModelAgent functionality."""

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
        """MLModelAgent instance."""
        from unittest.mock import AsyncMock, MagicMock

        generator = MLModelAgent(mock_services, "test_api_key")

        # Mock the arc_client to avoid actual API calls
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "mocked_yaml_content"
        mock_response.tool_calls = None  # No tool calls by default
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
  prediction: sigmoid.output
training:
  loss:
    type: torch.nn.functional.binary_cross_entropy
    inputs:
      input: prediction
      target: target
  optimizer:
    type: adam
    params:
      lr: 0.001"""

    @pytest.mark.asyncio
    async def test_generate_model_success(self, model_generator, valid_model_yaml):
        """Test successful model generation."""
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = valid_model_yaml
        mock_response.tool_calls = None  # No tool calls
        model_generator.arc_client.chat = AsyncMock(return_value=mock_response)

        # Generate model
        model_spec, model_yaml, _ = await model_generator.generate_model(
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
        mock_response.tool_calls = None  # No tool calls
        model_generator.arc_client.chat = AsyncMock(return_value=mock_response)

        # Generate model (output path handling is now done by CLI)
        model_spec, model_yaml, _ = await model_generator.generate_model(
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
        mock_response.tool_calls = None  # No tool calls
        model_generator.arc_client.chat = AsyncMock(return_value=mock_response)

        # Should raise MLModelError
        with pytest.raises(MLModelError):
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
        mock_response.tool_calls = None  # No tool calls
        model_generator.arc_client.chat = AsyncMock(return_value=mock_response)

        # Should raise MLModelError
        with pytest.raises(MLModelError):
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
        mock_response.tool_calls = None  # No tool calls
        model_generator.arc_client.chat = AsyncMock(return_value=mock_response)

        # Should raise MLModelError due to validation
        with pytest.raises(MLModelError):
            await model_generator.generate_model(
                name="test_model", user_context="Test model", table_name="test_table"
            )

    def test_profile_data_success(self, model_generator):
        """Test successful data profiling."""
        import asyncio

        # Call the new unified data profiling method
        result = asyncio.run(model_generator._get_unified_data_profile("test_table"))

        assert result["table_name"] == "test_table"
        assert len(result["feature_columns"]) == 3

        # Check that specific columns are present
        column_names = [col["name"] for col in result["feature_columns"]]
        assert "feature1" in column_names
        assert "feature2" in column_names
        assert "target" in column_names

    def test_profile_data_table_not_found(self, model_generator, mock_services):
        """Test data profiling with non-existent table."""
        import asyncio

        # Mock None result (table not found)
        mock_services.ml_data.get_dataset_info.return_value = None

        result = asyncio.run(
            model_generator._get_unified_data_profile("nonexistent_table")
        )

        assert "error" in result
        assert "not found" in result["error"]

    def test_profile_data_invalid_table_name(self, model_generator, mock_services):
        """Test data profiling with invalid table name."""
        import asyncio

        # Mock exception from ML data service for invalid table name
        mock_services.ml_data.get_dataset_info.side_effect = Exception(
            "Invalid table name"
        )

        result = asyncio.run(
            model_generator._get_unified_data_profile("invalid-table-name!")
        )

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

    @pytest.mark.asyncio
    async def test_validate_model_comprehensive_valid(
        self, model_generator, valid_model_yaml
    ):
        """Test comprehensive model validation with valid input."""
        context = {
            "data_profile": {
                "feature_columns": [
                    {"name": "feature1", "type": "REAL"},
                    {"name": "feature2", "type": "REAL"},
                ]
            },
            "available_components": {
                "node_types": ["torch.nn.Linear", "torch.nn.Sigmoid"]
            },
        }

        result = await model_generator._validate_model_comprehensive(
            valid_model_yaml, context
        )

        assert result["valid"] is True
        assert isinstance(result["object"], ModelSpec)
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_validate_model_comprehensive_invalid_columns(
        self, model_generator, valid_model_yaml
    ):
        """Test comprehensive model validation with invalid column references."""
        context = {
            "data_profile": {
                "feature_columns": [{"name": "different_column", "type": "REAL"}]
            },
            "available_components": {
                "node_types": ["torch.nn.Linear", "torch.nn.Sigmoid"]
            },
        }

        result = await model_generator._validate_model_comprehensive(
            valid_model_yaml, context
        )

        assert result["valid"] is False
        assert "Column validation errors" in result["error"]

    @pytest.mark.asyncio
    async def test_validate_model_without_dry_run(
        self, model_generator, valid_model_yaml
    ):
        """Test validation without dry-run (no train_table provided)."""
        context = {
            "data_profile": {
                "feature_columns": [
                    {"name": "feature1", "type": "REAL"},
                    {"name": "feature2", "type": "REAL"},
                ]
            },
            "available_components": {
                "node_types": ["torch.nn.Linear", "torch.nn.Sigmoid"]
            },
            # No train_table - dry-run should be skipped
        }

        result = await model_generator._validate_model_comprehensive(
            valid_model_yaml, context
        )

        # Validation should pass (no dry-run attempted)
        assert result["valid"] is True
        assert isinstance(result["object"], ModelSpec)

    @pytest.mark.asyncio
    async def test_retry_mechanism_with_validation_failures(
        self, model_generator, mock_services
    ):
        """Test that the agent retries when validation fails (max 3 attempts)."""
        # Mock responses: first 2 fail validation, third succeeds
        # Don't include train_table/target_column to avoid dry-run validation
        invalid_yaml_1 = """
inputs:
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
outputs:
  prediction: linear.output
"""  # Missing training section to fail validation

        invalid_yaml_2 = """
inputs:
  features:
    dtype: float32
    shape: [null, 2]
    columns: [nonexistent_column]
graph:
  - name: linear
    type: torch.nn.Linear
    params:
      in_features: 2
      out_features: 1
    inputs:
      input: features
outputs:
  prediction: linear.output
training:
  loss:
    type: torch.nn.functional.mse_loss
  optimizer:
    type: adam
    params:
      lr: 0.001
"""  # Invalid column reference

        valid_yaml = """
inputs:
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
  prediction: sigmoid.output
training:
  loss:
    type: torch.nn.functional.binary_cross_entropy
    inputs:
      input: prediction
      target: target
  optimizer:
    type: adam
    params:
      lr: 0.001
"""

        # Track call count
        call_count = [0]

        async def mock_chat_with_retries(messages, tools=None):
            call_count[0] += 1
            response = MagicMock()
            response.tool_calls = None

            # First attempt: missing optimizer
            if call_count[0] == 1:
                response.content = invalid_yaml_1
            # Second attempt: invalid column
            elif call_count[0] == 2:
                response.content = invalid_yaml_2
            # Third attempt: valid
            else:
                response.content = valid_yaml

            return response

        model_generator.arc_client.chat = mock_chat_with_retries

        # Generate model - should succeed after retries
        # Don't pass target_column to avoid dry-run validation
        model_spec, model_yaml, _ = await model_generator.generate_model(
            name="test_model",
            user_context="Binary classification model",
            table_name="test_table",
        )

        # Should have made 3 attempts
        assert call_count[0] == 3
        assert isinstance(model_spec, ModelSpec)
        assert "sigmoid" in model_yaml

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, model_generator, mock_services):
        """Test that agent fails after max retries (3) are exhausted."""
        # Always return invalid YAML
        invalid_yaml = """
inputs:
  features:
    dtype: float32
    shape: [null, 2]
    columns: [feature1, feature2]
# Missing graph, outputs, and training
"""

        mock_response = MagicMock()
        mock_response.content = invalid_yaml
        mock_response.tool_calls = None
        model_generator.arc_client.chat = AsyncMock(return_value=mock_response)

        # Should fail after 3 attempts
        with pytest.raises(MLModelError) as exc_info:
            await model_generator.generate_model(
                name="test_model",
                user_context="Test model",
                table_name="test_table",
            )

        # Error message should mention max attempts
        assert "3 attempts" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_no_nested_retry_loops(self, model_generator, mock_services):
        """Test that there are no nested retry loops - only one retry mechanism."""
        # Track all calls to chat
        chat_calls = []

        async def track_chat_calls(messages, tools=None):
            chat_calls.append(len(messages))
            response = MagicMock()
            response.tool_calls = None
            # Always return invalid YAML to force retries
            response.content = "inputs:\n  features:\n    dtype: float32"
            return response

        model_generator.arc_client.chat = track_chat_calls

        # Try to generate (will fail after 3 attempts)
        with contextlib.suppress(MLModelError):
            await model_generator.generate_model(
                name="test_model",
                user_context="Test model",
                table_name="test_table",
            )

        # Should have exactly 3 chat calls (3 attempts, no nesting)
        # If there were nested loops, we'd see 3*3=9 calls or more
        assert len(chat_calls) == 3, (
            f"Expected 3 chat calls (one per retry), but got {len(chat_calls)}. "
            f"This suggests nested retry loops."
        )

    @pytest.mark.asyncio
    async def test_validate_loss_function_invalid_output_reference(
        self, model_generator
    ):
        """Test validation catches when loss function references non-existent output."""
        # YAML where loss references 'logits' but outputs only has 'prediction'
        invalid_yaml = """
inputs:
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
outputs:
  prediction: linear.output
training:
  loss:
    type: torch.nn.BCEWithLogitsLoss
    inputs:
      input: logits
      target: target
  optimizer:
    type: torch.optim.Adam
    lr: 0.001
  epochs: 10
  batch_size: 32
"""

        context = {
            "data_profile": {
                "feature_columns": [
                    {"name": "feature1", "type": "REAL"},
                    {"name": "feature2", "type": "REAL"},
                ]
            },
            "available_components": {
                "node_types": ["torch.nn.Linear", "torch.nn.Sigmoid"]
            },
        }

        result = await model_generator._validate_model_comprehensive(
            invalid_yaml, context
        )

        assert result["valid"] is False
        assert "Loss function validation errors" in result["error"]
        assert "logits" in result["error"]
        assert "prediction" in result["error"]

    @pytest.mark.asyncio
    async def test_validate_loss_function_valid_logits_reference(self, model_generator):
        """Test validation passes when loss correctly references 'logits' output."""
        # YAML with BCEWithLogitsLoss correctly using 'logits' output
        valid_yaml = """
inputs:
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
  - name: probabilities
    type: torch.nn.functional.sigmoid
    inputs:
      input: linear.output
outputs:
  logits: linear.output
  probabilities: probabilities.output
training:
  loss:
    type: torch.nn.BCEWithLogitsLoss
    inputs:
      input: logits
      target: target
  optimizer:
    type: torch.optim.Adam
    lr: 0.001
  epochs: 10
  batch_size: 32
"""

        context = {
            "data_profile": {
                "feature_columns": [
                    {"name": "feature1", "type": "REAL"},
                    {"name": "feature2", "type": "REAL"},
                ]
            },
            "available_components": {
                "node_types": [
                    "torch.nn.Linear",
                    "torch.nn.functional.sigmoid",
                ]
            },
        }

        result = await model_generator._validate_model_comprehensive(
            valid_yaml, context
        )

        assert result["valid"] is True
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_validate_loss_function_valid_prediction_reference(
        self, model_generator
    ):
        """Test validation passes when loss correctly references 'prediction' output."""
        # YAML with MSELoss using 'prediction' output
        valid_yaml = """
inputs:
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
outputs:
  prediction: linear.output
training:
  loss:
    type: torch.nn.functional.mse_loss
    inputs:
      input: prediction
      target: target
  optimizer:
    type: torch.optim.Adam
    lr: 0.001
  epochs: 10
  batch_size: 32
"""

        context = {
            "data_profile": {
                "feature_columns": [
                    {"name": "feature1", "type": "REAL"},
                    {"name": "feature2", "type": "REAL"},
                ]
            },
            "available_components": {"node_types": ["torch.nn.Linear"]},
        }

        result = await model_generator._validate_model_comprehensive(
            valid_yaml, context
        )

        assert result["valid"] is True
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_validate_loss_function_multiple_outputs(self, model_generator):
        """Test validation with multiple outputs where loss references one."""
        # YAML with multiple outputs, loss references one of them
        valid_yaml = """
inputs:
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
    type: torch.nn.functional.sigmoid
    inputs:
      input: linear.output
outputs:
  score: linear.output
  probability: sigmoid.output
  confidence: sigmoid.output
training:
  loss:
    type: torch.nn.functional.mse_loss
    inputs:
      input: score
      target: target
  optimizer:
    type: torch.optim.Adam
    lr: 0.001
  epochs: 10
  batch_size: 32
"""

        context = {
            "data_profile": {
                "feature_columns": [
                    {"name": "feature1", "type": "REAL"},
                    {"name": "feature2", "type": "REAL"},
                ]
            },
            "available_components": {
                "node_types": [
                    "torch.nn.Linear",
                    "torch.nn.functional.sigmoid",
                ]
            },
        }

        result = await model_generator._validate_model_comprehensive(
            valid_yaml, context
        )

        assert result["valid"] is True
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_validate_loss_function_cross_entropy_with_logits(
        self, model_generator
    ):
        """Test validation for CrossEntropyLoss with logits output."""
        # YAML with CrossEntropyLoss correctly using 'logits' output
        valid_yaml = """
inputs:
  features:
    dtype: float32
    shape: [null, 2]
    columns: [feature1, feature2]
graph:
  - name: linear
    type: torch.nn.Linear
    params:
      in_features: 2
      out_features: 5
    inputs:
      input: features
  - name: probabilities
    type: torch.nn.functional.softmax
    params:
      dim: 1
    inputs:
      input: linear.output
outputs:
  logits: linear.output
  probabilities: probabilities.output
training:
  loss:
    type: torch.nn.functional.cross_entropy
    inputs:
      input: logits
      target: target
  optimizer:
    type: torch.optim.Adam
    lr: 0.001
  epochs: 10
  batch_size: 32
"""

        context = {
            "data_profile": {
                "feature_columns": [
                    {"name": "feature1", "type": "REAL"},
                    {"name": "feature2", "type": "REAL"},
                ]
            },
            "available_components": {
                "node_types": [
                    "torch.nn.Linear",
                    "torch.nn.functional.softmax",
                ]
            },
        }

        result = await model_generator._validate_model_comprehensive(
            valid_yaml, context
        )

        assert result["valid"] is True
        assert result["error"] is None
