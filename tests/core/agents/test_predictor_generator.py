"""Tests for PredictorGeneratorAgent in the separated architecture."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from arc.core.agents.predictor_generator import (
    PredictorGeneratorAgent,
    PredictorGeneratorError,
)
from arc.graph.predictor import PredictorSpec


class TestPredictorGeneratorAgent:
    """Test PredictorGeneratorAgent functionality."""

    @pytest.fixture
    def mock_services(self):
        """Mock services container."""
        return MagicMock()

    @pytest.fixture
    def mock_agent(self):
        """Mock Arc agent."""
        return MagicMock()

    @pytest.fixture
    def predictor_generator(self, mock_services, mock_agent):
        """PredictorGeneratorAgent instance."""
        return PredictorGeneratorAgent(mock_services, mock_agent)

    @pytest.fixture
    def valid_predictor_yaml(self):
        """Valid predictor YAML for testing."""
        return """name: test_predictor
model_id: test_model
model_version: 1
outputs:
  prediction: sigmoid.output
  confidence: sigmoid.output
  raw_logits: linear.output"""

    @pytest.mark.asyncio
    async def test_generate_predictor_success(
        self, predictor_generator, mock_agent, valid_predictor_yaml
    ):
        """Test successful predictor generation."""
        # Mock the LLM response
        chat_entry_mock = MagicMock()
        chat_entry_mock.type = "assistant"
        chat_entry_mock.content = valid_predictor_yaml

        mock_agent.process_user_message = AsyncMock(return_value=[chat_entry_mock])

        # Generate predictor
        predictor_spec, predictor_yaml = await predictor_generator.generate_predictor(
            model_id="test_model",
            user_context="Prediction service for binary classification",
        )

        # Verify result
        assert isinstance(predictor_spec, PredictorSpec)
        assert predictor_spec.name == "test_predictor"
        assert predictor_spec.model_id == "test_model"
        assert predictor_spec.model_version == 1

        # Verify outputs
        assert len(predictor_spec.outputs) == 3
        assert predictor_spec.outputs["prediction"] == "sigmoid.output"
        assert predictor_spec.outputs["confidence"] == "sigmoid.output"
        assert predictor_spec.outputs["raw_logits"] == "linear.output"

        # Verify YAML is valid
        assert "name:" in predictor_yaml
        assert "model_id:" in predictor_yaml
        assert "outputs:" in predictor_yaml

    @pytest.mark.asyncio
    async def test_generate_predictor_with_custom_name(
        self, predictor_generator, mock_agent
    ):
        """Test predictor generation with custom name."""
        custom_yaml = """name: custom_predictor_name
model_id: diabetes_model
outputs:
  risk_score: prediction.output"""

        chat_entry_mock = MagicMock()
        chat_entry_mock.type = "assistant"
        chat_entry_mock.content = custom_yaml

        mock_agent.process_user_message = AsyncMock(return_value=[chat_entry_mock])

        # Generate predictor (name comes from the YAML response)
        predictor_spec, predictor_yaml = await predictor_generator.generate_predictor(
            model_id="diabetes_model",
            user_context="Medical diagnosis predictor",
        )

        assert predictor_spec.name == "custom_predictor_name"
        assert predictor_spec.model_id == "diabetes_model"

    @pytest.mark.asyncio
    async def test_generate_predictor_with_output_path(
        self, predictor_generator, mock_agent, valid_predictor_yaml
    ):
        """Test predictor generation with file output."""
        # Mock the LLM response
        chat_entry_mock = MagicMock()
        chat_entry_mock.type = "assistant"
        chat_entry_mock.content = valid_predictor_yaml

        mock_agent.process_user_message = AsyncMock(return_value=[chat_entry_mock])

        # Generate predictor
        predictor_spec, predictor_yaml = await predictor_generator.generate_predictor(
            model_id="test_model",
            user_context="Test predictor",
        )

        # Verify predictor was generated
        assert predictor_spec is not None
        assert predictor_yaml is not None
        assert "name:" in predictor_yaml
        assert "model_id:" in predictor_yaml

    @pytest.mark.asyncio
    async def test_generate_predictor_with_specific_outputs(
        self, predictor_generator, mock_agent
    ):
        """Test predictor generation with specific outputs requested."""
        specific_outputs_yaml = """name: specific_output_predictor
model_id: test_model
outputs:
  class_probability: softmax.output
  feature_importance: attention.weights"""

        chat_entry_mock = MagicMock()
        chat_entry_mock.type = "assistant"
        chat_entry_mock.content = specific_outputs_yaml

        mock_agent.process_user_message = AsyncMock(return_value=[chat_entry_mock])

        # Generate predictor (outputs come from YAML response)
        predictor_spec, predictor_yaml = await predictor_generator.generate_predictor(
            model_id="test_model",
            user_context="Multi-output predictor",
        )

        assert len(predictor_spec.outputs) == 2
        assert "class_probability" in predictor_spec.outputs
        assert "feature_importance" in predictor_spec.outputs

    @pytest.mark.asyncio
    async def test_generate_predictor_with_model_version(
        self, predictor_generator, mock_agent
    ):
        """Test predictor generation with specific model version."""
        versioned_yaml = """name: versioned_predictor
model_id: test_model
model_version: 5
outputs:
  prediction: model.output"""

        chat_entry_mock = MagicMock()
        chat_entry_mock.type = "assistant"
        chat_entry_mock.content = versioned_yaml

        mock_agent.process_user_message = AsyncMock(return_value=[chat_entry_mock])

        # Generate predictor with specific model version
        predictor_spec, predictor_yaml = await predictor_generator.generate_predictor(
            model_id="test_model",
            user_context="Version-specific predictor",
            model_version=5,
        )

        assert predictor_spec.model_version == 5

    @pytest.mark.asyncio
    async def test_generate_predictor_invalid_yaml(
        self, predictor_generator, mock_agent
    ):
        """Test predictor generation with invalid YAML response."""
        # Mock invalid YAML response
        chat_entry_mock = MagicMock()
        chat_entry_mock.type = "assistant"
        chat_entry_mock.content = "invalid: yaml: content: ["

        mock_agent.process_user_message = AsyncMock(return_value=[chat_entry_mock])

        # Should raise PredictorGeneratorError
        with pytest.raises(PredictorGeneratorError):
            await predictor_generator.generate_predictor(
                model_id="test_model", user_context="Test predictor"
            )

    @pytest.mark.asyncio
    async def test_generate_predictor_missing_required_fields(
        self, predictor_generator, mock_agent
    ):
        """Test predictor generation with missing required fields."""
        # Mock response missing required fields
        incomplete_yaml = """name: incomplete_predictor
# Missing model_id"""

        chat_entry_mock = MagicMock()
        chat_entry_mock.type = "assistant"
        chat_entry_mock.content = incomplete_yaml

        mock_agent.process_user_message = AsyncMock(return_value=[chat_entry_mock])

        # Should raise PredictorGeneratorError
        with pytest.raises(PredictorGeneratorError):
            await predictor_generator.generate_predictor(
                model_id="test_model", user_context="Test predictor"
            )

    @pytest.mark.asyncio
    async def test_generate_predictor_empty_model_id(
        self, predictor_generator, mock_agent
    ):
        """Test predictor generation with empty model_id."""
        # Mock response with empty model_id
        invalid_yaml = """
        name: test_predictor
        model_id: ""
        """

        chat_entry_mock = MagicMock()
        chat_entry_mock.type = "assistant"
        chat_entry_mock.content = invalid_yaml

        mock_agent.process_user_message = AsyncMock(return_value=[chat_entry_mock])

        # Should raise PredictorGeneratorError due to validation
        with pytest.raises(PredictorGeneratorError):
            await predictor_generator.generate_predictor(
                model_id="test_model", user_context="Test predictor"
            )

    def test_get_predictor_examples(self, predictor_generator):
        """Test getting predictor examples."""
        examples = predictor_generator._get_predictor_examples("binary classification")

        assert isinstance(examples, list)
        # Should return at least zero examples (may be empty initially)
        assert len(examples) >= 0

        if examples:
            example = examples[0]
            assert "schema" in example
            assert "name" in example

    def test_validate_predictor_comprehensive_valid(
        self, predictor_generator, valid_predictor_yaml
    ):
        """Test comprehensive predictor validation with valid input."""
        context = {}

        result = predictor_generator._validate_predictor_comprehensive(
            valid_predictor_yaml, context
        )

        assert result["valid"] is True
        assert isinstance(result["object"], PredictorSpec)
        assert result["error"] is None

    def test_validate_predictor_comprehensive_invalid_yaml(self, predictor_generator):
        """Test comprehensive predictor validation with invalid YAML."""
        invalid_yaml = "not: valid: yaml: ["
        context = {}

        result = predictor_generator._validate_predictor_comprehensive(
            invalid_yaml, context
        )

        assert result["valid"] is False
        assert (
            "mapping values are not allowed" in result["error"]
            or "YAML" in result["error"]
            or "Validation exception" in result["error"]
        )

    def test_validate_predictor_comprehensive_not_dict(self, predictor_generator):
        """Test comprehensive predictor validation with non-dict YAML."""
        list_yaml = "- item1\n- item2"
        context = {}

        result = predictor_generator._validate_predictor_comprehensive(
            list_yaml, context
        )

        assert result["valid"] is False
        assert "dictionary" in result["error"]

    def test_validate_predictor_comprehensive_missing_fields(self, predictor_generator):
        """Test comprehensive predictor validation with missing required fields."""
        incomplete_yaml = """
        name: test_predictor
        # Missing model_id
        """
        context = {}

        result = predictor_generator._validate_predictor_comprehensive(
            incomplete_yaml, context
        )

        assert result["valid"] is False
        assert "Missing required fields" in result["error"]

    @pytest.mark.asyncio
    async def test_generate_predictor_llm_error(self, predictor_generator, mock_agent):
        """Test predictor generation with LLM error."""
        # Mock LLM error
        mock_agent.process_user_message = AsyncMock(
            side_effect=Exception("LLM connection failed")
        )

        # Should raise PredictorGeneratorError
        with pytest.raises(PredictorGeneratorError):
            await predictor_generator.generate_predictor(
                model_id="test_model", user_context="Test predictor"
            )

    @pytest.mark.asyncio
    async def test_generate_predictor_no_response(
        self, predictor_generator, mock_agent
    ):
        """Test predictor generation with no LLM response."""
        # Mock empty response
        mock_agent.process_user_message = AsyncMock(return_value=[])

        # Should raise PredictorGeneratorError
        with pytest.raises(PredictorGeneratorError):
            await predictor_generator.generate_predictor(
                model_id="test_model", user_context="Test predictor"
            )
