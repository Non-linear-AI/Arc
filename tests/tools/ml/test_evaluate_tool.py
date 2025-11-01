"""Test ml_evaluate tool JSON output structure."""

import json
from unittest.mock import MagicMock

import pytest

from arc.tools.ml.evaluate_tool import MLEvaluateTool


class TestMLEvaluateToolJSONOutput:
    """Test ml_evaluate tool JSON output structure."""

    @pytest.fixture
    def mock_tool(self):
        """Create a mock MLEvaluateTool instance."""
        mock_services = MagicMock()
        mock_runtime = MagicMock()
        mock_ui = MagicMock()
        tool = MLEvaluateTool(
            mock_services,
            runtime=mock_runtime,
            ui_interface=mock_ui,
        )
        return tool

    def test_build_evaluate_result_accepted_with_job(self, mock_tool):
        """Test _build_evaluate_result returns valid JSON for accepted with job."""
        result = mock_tool._build_evaluate_result(
            status="accepted",
            evaluator_id="diabetes-classifier-v1_evaluator-v1",
            evaluation_job_id="job_abc123",
            model_id="diabetes-classifier-v1",
            dataset="diabetes_test_data",
            evaluation_status="submitted",
        )

        # Should be valid JSON
        parsed = json.loads(result)

        # Check structure
        assert parsed["status"] == "accepted"
        assert parsed["evaluator_id"] == "diabetes-classifier-v1_evaluator-v1"
        assert parsed["evaluation"]["status"] == "submitted"
        assert parsed["evaluation"]["job_id"] == "job_abc123"
        assert parsed["evaluation"]["model_id"] == "diabetes-classifier-v1"
        assert parsed["evaluation"]["dataset"] == "diabetes_test_data"

    def test_build_evaluate_result_cancelled(self, mock_tool):
        """Test _build_evaluate_result returns valid JSON for cancelled status."""
        result = mock_tool._build_evaluate_result(
            status="cancelled",
            evaluator_id="evaluator-cancelled",
            evaluation_status="not_started",
        )

        # Should be valid JSON
        parsed = json.loads(result)

        # Check structure
        assert parsed["status"] == "cancelled"
        assert parsed["evaluator_id"] == "evaluator-cancelled"
        assert parsed["evaluation"]["status"] == "not_started"
        assert (
            "job_id" not in parsed["evaluation"]
        )  # Should not have job_id when cancelled

    def test_build_evaluate_result_evaluation_failed(self, mock_tool):
        """Test _build_evaluate_result returns valid JSON for evaluation failure."""
        result = mock_tool._build_evaluate_result(
            status="accepted",
            evaluator_id="evaluator-v1",
            evaluation_status="failed",
            evaluation_error="Evaluation failed due to missing data",
        )

        # Should be valid JSON
        parsed = json.loads(result)

        # Check structure
        assert parsed["status"] == "accepted"
        assert parsed["evaluation"]["status"] == "failed"
        assert parsed["evaluation"]["error"] == "Evaluation failed due to missing data"

    def test_build_evaluate_result_not_started(self, mock_tool):
        """Test _build_evaluate_result returns valid JSON for not_started evaluation."""
        result = mock_tool._build_evaluate_result(
            status="accepted",
            evaluator_id="evaluator-v1",
            evaluation_status="not_started",
        )

        # Should be valid JSON
        parsed = json.loads(result)

        # Check structure
        assert parsed["status"] == "accepted"
        assert parsed["evaluation"]["status"] == "not_started"
        assert "job_id" not in parsed["evaluation"]

    def test_build_evaluate_result_json_formatting(self, mock_tool):
        """Test _build_evaluate_result returns compact JSON."""
        result = mock_tool._build_evaluate_result(
            status="accepted",
            evaluator_id="evaluator-v1",
            evaluation_job_id="job_123",
            evaluation_status="submitted",
        )

        # Should be compact (no newlines or indentation)
        assert "\n" not in result
        assert result.startswith("{")
        assert result.endswith("}")

    def test_json_output_has_required_fields(self, mock_tool):
        """Test JSON output always has required fields."""
        result = mock_tool._build_evaluate_result(
            status="accepted",
            evaluator_id="evaluator-v1",
        )

        parsed = json.loads(result)

        # Required top-level fields
        assert "status" in parsed
        assert "evaluator_id" in parsed
        assert "evaluation" in parsed

        # Required evaluation fields
        assert "status" in parsed["evaluation"]

    def test_json_output_status_values(self, mock_tool):
        """Test JSON output accepts valid status values."""
        for status in ["accepted", "cancelled"]:
            result = mock_tool._build_evaluate_result(
                status=status,
                evaluator_id="evaluator-v1",
            )
            parsed = json.loads(result)
            assert parsed["status"] == status

    def test_json_output_evaluation_status_values(self, mock_tool):
        """Test JSON output accepts valid evaluation status values."""
        for evaluation_status in ["submitted", "failed", "not_started"]:
            result = mock_tool._build_evaluate_result(
                status="accepted",
                evaluator_id="evaluator-v1",
                evaluation_status=evaluation_status,
            )
            parsed = json.loads(result)
            assert parsed["evaluation"]["status"] == evaluation_status

    def test_json_output_optional_fields(self, mock_tool):
        """Test JSON output includes optional fields only when provided."""
        # Without optional fields
        result = mock_tool._build_evaluate_result(
            status="accepted",
            evaluator_id="evaluator-v1",
            evaluation_status="not_started",
        )
        parsed = json.loads(result)
        assert "job_id" not in parsed["evaluation"]
        assert "model_id" not in parsed["evaluation"]
        assert "dataset" not in parsed["evaluation"]
        assert "error" not in parsed["evaluation"]

        # With all optional fields
        result = mock_tool._build_evaluate_result(
            status="accepted",
            evaluator_id="evaluator-v1",
            evaluation_job_id="job_123",
            model_id="model-v1",
            dataset="test_data",
            evaluation_status="failed",
            evaluation_error="Error message",
        )
        parsed = json.loads(result)
        assert parsed["evaluation"]["job_id"] == "job_123"
        assert parsed["evaluation"]["model_id"] == "model-v1"
        assert parsed["evaluation"]["dataset"] == "test_data"
        assert parsed["evaluation"]["error"] == "Error message"

    def test_no_duplicate_run_id(self, mock_tool):
        """Test that run_id is not included in evaluation output (only job_id)."""
        result = mock_tool._build_evaluate_result(
            status="accepted",
            evaluator_id="evaluator-v1",
            evaluation_job_id="job_123",
            evaluation_status="submitted",
        )

        parsed = json.loads(result)

        # Should have job_id, not run_id
        assert "job_id" in parsed["evaluation"]
        assert "run_id" not in parsed["evaluation"]
