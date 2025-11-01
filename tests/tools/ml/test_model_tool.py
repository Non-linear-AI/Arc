"""Test ml_model tool JSON output structure."""

import json
from unittest.mock import MagicMock

import pytest

from arc.tools.ml.model_tool import MLModelTool


class TestMLModelToolJSONOutput:
    """Test ml_model tool JSON output structure."""

    @pytest.fixture
    def mock_tool(self):
        """Create a mock MLModelTool instance."""
        mock_services = MagicMock()
        mock_runtime = MagicMock()
        mock_ui = MagicMock()
        tool = MLModelTool(
            mock_services,
            runtime=mock_runtime,
            api_key="test_api_key",
            base_url=None,
            model=None,
            ui_interface=mock_ui,
        )
        return tool

    def test_build_model_result_accepted_with_training(self, mock_tool):
        """Test _build_model_result returns valid JSON for accepted with training."""
        model_spec = {
            "inputs": {"features": ["age", "income"]},
            "graph": {"layers": [{"type": "dense", "units": 64}]},
            "outputs": {"prediction": "float"},
            "loss": "binary_crossentropy",
            "training": {"epochs": 10, "batch_size": 32},
        }

        result = mock_tool._build_model_result(
            status="accepted",
            model_id="diabetes-classifier-v1",
            model_spec_dict=model_spec,
            training_job_id="job_xyz789",
            train_table="diabetes_training_data",
            training_status="submitted",
        )

        # Should be valid JSON
        parsed = json.loads(result)

        # Check structure
        assert parsed["status"] == "accepted"
        assert parsed["model_id"] == "diabetes-classifier-v1"
        assert parsed["model_spec"] == model_spec
        assert parsed["training"]["status"] == "submitted"
        assert parsed["training"]["job_id"] == "job_xyz789"
        assert parsed["training"]["train_table"] == "diabetes_training_data"

    def test_build_model_result_cancelled(self, mock_tool):
        """Test _build_model_result returns valid JSON for cancelled status."""
        model_spec = {"inputs": {}, "graph": {}, "outputs": {}}

        result = mock_tool._build_model_result(
            status="cancelled",
            model_id="model-cancelled",
            model_spec_dict=model_spec,
            training_status="not_started",
        )

        # Should be valid JSON
        parsed = json.loads(result)

        # Check structure
        assert parsed["status"] == "cancelled"
        assert parsed["model_id"] == "model-cancelled"
        assert parsed["model_spec"] == model_spec
        assert parsed["training"]["status"] == "not_started"
        assert (
            "job_id" not in parsed["training"]
        )  # Should not have job_id when cancelled

    def test_build_model_result_training_failed(self, mock_tool):
        """Test _build_model_result returns valid JSON for training failure."""
        result = mock_tool._build_model_result(
            status="accepted",
            model_id="model-v1",
            model_spec_dict={},
            training_status="failed",
            training_error="Training job failed due to invalid data",
        )

        # Should be valid JSON
        parsed = json.loads(result)

        # Check structure
        assert parsed["status"] == "accepted"
        assert parsed["training"]["status"] == "failed"
        assert parsed["training"]["error"] == "Training job failed due to invalid data"

    def test_build_model_result_not_started(self, mock_tool):
        """Test _build_model_result returns valid JSON for not_started training."""
        result = mock_tool._build_model_result(
            status="accepted",
            model_id="model-v1",
            model_spec_dict={},
            training_status="not_started",
        )

        # Should be valid JSON
        parsed = json.loads(result)

        # Check structure
        assert parsed["status"] == "accepted"
        assert parsed["training"]["status"] == "not_started"
        assert "job_id" not in parsed["training"]

    def test_build_model_result_default_empty_dict(self, mock_tool):
        """Test _build_model_result uses empty dict for None model_spec."""
        result = mock_tool._build_model_result(
            status="accepted",
            model_id="model-v1",
            model_spec_dict=None,
            training_status="not_started",
        )

        # Should be valid JSON
        parsed = json.loads(result)

        # Check structure - should have empty dict for model_spec
        assert parsed["model_spec"] == {}

    def test_build_model_result_json_formatting(self, mock_tool):
        """Test _build_model_result returns compact JSON."""
        result = mock_tool._build_model_result(
            status="accepted",
            model_id="model-v1",
            model_spec_dict={"inputs": {"test": "value"}},
            training_status="not_started",
        )

        # Should be compact (no newlines or indentation)
        assert "\n" not in result
        assert result.startswith("{")
        assert result.endswith("}")

    def test_json_output_has_required_fields(self, mock_tool):
        """Test JSON output always has required fields."""
        result = mock_tool._build_model_result(
            status="accepted",
            model_id="model-v1",
        )

        parsed = json.loads(result)

        # Required top-level fields
        assert "status" in parsed
        assert "model_id" in parsed
        assert "model_spec" in parsed
        assert "training" in parsed

        # Required training fields
        assert "status" in parsed["training"]

    def test_json_output_status_values(self, mock_tool):
        """Test JSON output accepts valid status values."""
        for status in ["accepted", "cancelled"]:
            result = mock_tool._build_model_result(
                status=status,
                model_id="model-v1",
            )
            parsed = json.loads(result)
            assert parsed["status"] == status

    def test_json_output_training_status_values(self, mock_tool):
        """Test JSON output accepts valid training status values."""
        for training_status in ["submitted", "failed", "not_started"]:
            result = mock_tool._build_model_result(
                status="accepted",
                model_id="model-v1",
                training_status=training_status,
            )
            parsed = json.loads(result)
            assert parsed["training"]["status"] == training_status

    def test_json_output_optional_fields(self, mock_tool):
        """Test JSON output includes optional fields only when provided."""
        # Without optional fields
        result = mock_tool._build_model_result(
            status="accepted",
            model_id="model-v1",
            training_status="not_started",
        )
        parsed = json.loads(result)
        assert "job_id" not in parsed["training"]
        assert "train_table" not in parsed["training"]
        assert "error" not in parsed["training"]

        # With all optional fields
        result = mock_tool._build_model_result(
            status="accepted",
            model_id="model-v1",
            training_job_id="job_123",
            train_table="train_data",
            training_status="failed",
            training_error="Error message",
        )
        parsed = json.loads(result)
        assert parsed["training"]["job_id"] == "job_123"
        assert parsed["training"]["train_table"] == "train_data"
        assert parsed["training"]["error"] == "Error message"
