"""Tests for label encoder unknown category handling."""

import pytest
import torch

from arc.ml.data import DataProcessor, ProcessorError


class TestLabelEncoderUnknownCategories:
    """Test label encoder handling of unknown categories."""

    def test_label_encoder_basic_fit_and_transform(self):
        """Test basic label encoding workflow."""
        processor = DataProcessor(ml_data_service=None)

        # Fit encoder
        fit_result = processor._execute_operator(
            "fit.label_encoder", {"x": ["cat", "dog", "bird", "cat", "dog"]}
        )
        state = fit_result["state"]

        # Verify vocabulary
        assert "vocabulary" in state
        assert len(state["vocabulary"]) == 3
        assert "cat" in state["vocabulary"]
        assert "dog" in state["vocabulary"]
        assert "bird" in state["vocabulary"]

        # Transform with known categories
        transform_result = processor._execute_operator(
            "transform.label_encode",
            {"x": ["cat", "dog", "bird"], "state": state},
        )

        output = transform_result["output"]
        assert output.dtype == torch.long
        assert len(output) == 3

    def test_label_encoder_unknown_category_with_default_handling(self):
        """Test that unknown categories use unknown_value when provided."""
        processor = DataProcessor(ml_data_service=None)

        # Fit with known categories
        fit_result = processor._execute_operator(
            "fit.label_encoder",
            {"x": ["cat", "dog", "bird"]},
        )
        state = fit_result["state"]

        # Transform with unknown category - should use unknown_value
        transform_result = processor._execute_operator(
            "transform.label_encode",
            {
                "x": ["cat", "elephant", "dog"],  # "elephant" is unknown
                "state": state,
                "unknown_value": -1,  # Use -1 for unknown categories
            },
        )

        output = transform_result["output"]
        assert len(output) == 3
        # Check that unknown value is used
        assert output[1].item() == -1

    def test_label_encoder_unknown_category_error_mode(self):
        """Test that unknown categories raise error when handle_unknown='error'."""
        processor = DataProcessor(ml_data_service=None)

        # Fit with known categories
        fit_result = processor._execute_operator(
            "fit.label_encoder",
            {"x": ["cat", "dog", "bird"]},
        )
        state = fit_result["state"]

        # Transform with unknown category - should raise error
        with pytest.raises(
            ProcessorError, match="Unknown category.*elephant.*not in vocabulary"
        ):
            processor._execute_operator(
                "transform.label_encode",
                {
                    "x": ["cat", "elephant", "dog"],
                    "state": state,
                    "handle_unknown": "error",  # Strict mode
                },
            )

    def test_label_encoder_unknown_category_ignore_mode(self):
        """Test that unknown categories are mapped to special index when handle_unknown='use_unknown_value'."""
        processor = DataProcessor(ml_data_service=None)

        # Fit with known categories
        fit_result = processor._execute_operator(
            "fit.label_encoder",
            {"x": ["cat", "dog", "bird"]},
        )
        state = fit_result["state"]

        # Transform with unknown category
        transform_result = processor._execute_operator(
            "transform.label_encode",
            {
                "x": ["cat", "unknown1", "dog", "unknown2"],
                "state": state,
                "handle_unknown": "use_unknown_value",
                "unknown_value": 999,
            },
        )

        output = transform_result["output"]
        assert len(output) == 4
        # Both unknowns should map to unknown_value
        assert output[1].item() == 999
        assert output[3].item() == 999

    def test_label_encoder_with_none_values(self):
        """Test label encoder with None/null values in data."""
        processor = DataProcessor(ml_data_service=None)

        # Fit with None values
        fit_result = processor._execute_operator(
            "fit.label_encoder",
            {"x": ["cat", None, "dog", "cat", None]},
        )
        state = fit_result["state"]

        # None should be in vocabulary
        assert None in state["vocabulary"]

        # Transform with None
        transform_result = processor._execute_operator(
            "transform.label_encode",
            {"x": ["cat", None, "dog"], "state": state},
        )

        output = transform_result["output"]
        assert len(output) == 3
        # None should have consistent encoding
        assert output[1] == state["vocabulary"][None]

    def test_label_encoder_vocabulary_sorted_order(self):
        """Test that vocabulary is built in deterministic sorted order."""
        processor = DataProcessor(ml_data_service=None)

        # Fit with unsorted data
        fit_result = processor._execute_operator(
            "fit.label_encoder",
            {"x": ["zebra", "apple", "mango", "banana"]},
        )
        state = fit_result["state"]

        vocab = state["vocabulary"]
        # Vocabulary should be sorted (None first if present, then alphabetically)
        keys = [k for k in vocab if k is not None]
        assert keys == sorted(keys)

    def test_label_encoder_default_unknown_value(self):
        """Test that unknown categories use vocab_size as default unknown_value."""
        processor = DataProcessor(ml_data_service=None)

        # Fit with 3 categories
        fit_result = processor._execute_operator(
            "fit.label_encoder",
            {"x": ["cat", "dog", "bird"]},
        )
        state = fit_result["state"]
        vocab_size = state["vocab_size"]  # Should be 3

        # Transform with unknown category (no explicit unknown_value)
        transform_result = processor._execute_operator(
            "transform.label_encode",
            {
                "x": ["cat", "elephant"],
                "state": state,
                # No unknown_value specified - should default to vocab_size
            },
        )

        output = transform_result["output"]
        # Unknown should map to vocab_size (= 3 in this case)
        assert output[1].item() == vocab_size
