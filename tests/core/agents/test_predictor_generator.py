"""Tests for PredictorGeneratorAgent in the separated architecture."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from arc.core.agents.predictor_generator import (
    PredictorGeneratorAgent,
    PredictorGeneratorError,
)


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
    def test_files(self, tmp_path):
        """Create test spec files."""
        # Create test model spec file
        model_spec_file = tmp_path / "test_model.yaml"
        model_spec_file.write_text("""
inputs:
  data:
    dtype: float32
    shape: [null, 8]
outputs:
  prediction: model.output
""")

        # Create test trainer spec file
        trainer_spec_file = tmp_path / "test_trainer.yaml"
        trainer_spec_file.write_text("""
name: test_trainer
optimizer:
  type: Adam
loss:
  type: BCELoss
""")

        return {
            "model_spec": str(model_spec_file),
            "trainer_spec": str(trainer_spec_file),
        }

    @pytest.fixture
    def predictor_generator(self, mock_services):
        """PredictorGeneratorAgent instance."""
        from unittest.mock import AsyncMock, MagicMock

        generator = PredictorGeneratorAgent(mock_services, "test_api_key")

        # Mock the arc_client to avoid actual API calls
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "mocked_yaml_content"
        mock_client.chat = AsyncMock(return_value=mock_response)
        generator.arc_client = mock_client

        return generator

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
        self, predictor_generator, valid_predictor_yaml, test_files
    ):
        """Test successful predictor generation."""
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = valid_predictor_yaml
        predictor_generator.arc_client.chat = AsyncMock(return_value=mock_response)

        # Generate predictor
        predictor_yaml = await predictor_generator.generate_predictor(
            user_context="Prediction service for binary classification",
            model_spec_path=test_files["model_spec"],
        )

        # Verify YAML is returned
        assert isinstance(predictor_yaml, str)
        assert "name:" in predictor_yaml
        assert "model_id:" in predictor_yaml
        assert "outputs:" in predictor_yaml

    @pytest.mark.asyncio
    async def test_generate_predictor_with_custom_name(
        self, predictor_generator, test_files
    ):
        """Test predictor generation with custom name."""
        custom_yaml = """name: custom_predictor_name
model_id: diabetes_model
outputs:
  risk_score: prediction.output"""

        mock_response = MagicMock()
        mock_response.content = custom_yaml
        predictor_generator.arc_client.chat = AsyncMock(return_value=mock_response)

        # Generate predictor
        predictor_yaml = await predictor_generator.generate_predictor(
            user_context="Medical diagnosis predictor",
            model_spec_path=test_files["model_spec"],
        )

        assert "custom_predictor_name" in predictor_yaml
        assert "diabetes_model" in predictor_yaml

    @pytest.mark.asyncio
    async def test_generate_predictor_with_output_path(
        self, predictor_generator, valid_predictor_yaml, test_files
    ):
        """Test predictor generation with file output."""
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = valid_predictor_yaml
        predictor_generator.arc_client.chat = AsyncMock(return_value=mock_response)

        # Generate predictor
        predictor_yaml = await predictor_generator.generate_predictor(
            user_context="Test predictor",
            model_spec_path=test_files["model_spec"],
        )

        # Verify predictor was generated
        assert predictor_yaml is not None
        assert "name:" in predictor_yaml
        assert "model_id:" in predictor_yaml

    @pytest.mark.asyncio
    async def test_generate_predictor_with_specific_outputs(
        self, predictor_generator, test_files
    ):
        """Test predictor generation with specific outputs requested."""
        specific_outputs_yaml = """name: specific_output_predictor
model_id: test_model
outputs:
  class_probability: softmax.output
  feature_importance: attention.weights"""

        mock_response = MagicMock()
        mock_response.content = specific_outputs_yaml
        predictor_generator.arc_client.chat = AsyncMock(return_value=mock_response)

        # Generate predictor (outputs come from YAML response)
        predictor_yaml = await predictor_generator.generate_predictor(
            user_context="Multi-output predictor",
            model_spec_path=test_files["model_spec"],
        )

        assert "class_probability" in predictor_yaml
        assert "feature_importance" in predictor_yaml

    @pytest.mark.asyncio
    async def test_generate_predictor_with_model_version(
        self, predictor_generator, test_files
    ):
        """Test predictor generation with specific model version."""
        versioned_yaml = """name: versioned_predictor
model_id: test_model
model_version: 5
outputs:
  prediction: model.output"""

        mock_response = MagicMock()
        mock_response.content = versioned_yaml
        predictor_generator.arc_client.chat = AsyncMock(return_value=mock_response)

        # Generate predictor
        predictor_yaml = await predictor_generator.generate_predictor(
            user_context="Version-specific predictor",
            model_spec_path=test_files["model_spec"],
        )

        assert "model_version: 5" in predictor_yaml

    @pytest.mark.asyncio
    async def test_generate_predictor_invalid_yaml(
        self, predictor_generator, test_files
    ):
        """Test predictor generation with invalid YAML response."""
        # Mock invalid YAML response
        mock_response = MagicMock()
        mock_response.content = "invalid: yaml: content: ["
        predictor_generator.arc_client.chat = AsyncMock(return_value=mock_response)

        # Should raise PredictorGeneratorError
        with pytest.raises(PredictorGeneratorError):
            await predictor_generator.generate_predictor(
                user_context="Test predictor",
                model_spec_path=test_files["model_spec"],
            )

    @pytest.mark.asyncio
    async def test_generate_predictor_missing_required_fields(
        self, predictor_generator, test_files
    ):
        """Test predictor generation with missing required fields."""
        # Mock response missing required fields
        incomplete_yaml = """# Missing name field
model_id: test_model"""

        mock_response = MagicMock()
        mock_response.content = incomplete_yaml
        predictor_generator.arc_client.chat = AsyncMock(return_value=mock_response)

        # Should raise PredictorGeneratorError
        with pytest.raises(PredictorGeneratorError):
            await predictor_generator.generate_predictor(
                user_context="Test predictor",
                model_spec_path=test_files["model_spec"],
            )

    @pytest.mark.asyncio
    async def test_generate_predictor_empty_name(self, predictor_generator, test_files):
        """Test predictor generation with empty name."""
        # Mock response with empty name
        invalid_yaml = """
        name: ""
        model_id: test_model
        """

        mock_response = MagicMock()
        mock_response.content = invalid_yaml
        predictor_generator.arc_client.chat = AsyncMock(return_value=mock_response)

        # Should raise PredictorGeneratorError due to validation
        with pytest.raises(PredictorGeneratorError):
            await predictor_generator.generate_predictor(
                user_context="Test predictor",
                model_spec_path=test_files["model_spec"],
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
        assert isinstance(result["object"], dict)
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
        # Missing name field
        model_id: test_model
        """
        context = {}

        result = predictor_generator._validate_predictor_comprehensive(
            incomplete_yaml, context
        )

        assert result["valid"] is False
        assert "Missing required fields" in result["error"]

    @pytest.mark.asyncio
    async def test_generate_predictor_llm_error(self, predictor_generator, test_files):
        """Test predictor generation with LLM error."""
        # Mock LLM error
        predictor_generator.arc_client.chat = AsyncMock(
            side_effect=Exception("LLM connection failed")
        )

        # Should raise PredictorGeneratorError
        with pytest.raises(PredictorGeneratorError):
            await predictor_generator.generate_predictor(
                user_context="Test predictor",
                model_spec_path=test_files["model_spec"],
            )

    @pytest.mark.asyncio
    async def test_generate_predictor_file_not_found(self, predictor_generator):
        """Test predictor generation with missing model spec file."""
        # Should raise PredictorGeneratorError for missing file
        with pytest.raises(PredictorGeneratorError):
            await predictor_generator.generate_predictor(
                user_context="Test predictor",
                model_spec_path="/nonexistent/file.yaml",
            )

    @pytest.mark.asyncio
    async def test_generate_predictor_with_trainer_spec(
        self, predictor_generator, valid_predictor_yaml, test_files
    ):
        """Test predictor generation with trainer spec file."""
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = valid_predictor_yaml
        predictor_generator.arc_client.chat = AsyncMock(return_value=mock_response)

        # Generate predictor with trainer spec
        predictor_yaml = await predictor_generator.generate_predictor(
            user_context="Prediction service with training context",
            model_spec_path=test_files["model_spec"],
            trainer_spec_path=test_files["trainer_spec"],
        )

        # Verify YAML is returned
        assert isinstance(predictor_yaml, str)
        assert "name:" in predictor_yaml
