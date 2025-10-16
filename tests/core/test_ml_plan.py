"""Test ML plan data structure and integration."""

from datetime import datetime

import pytest

from arc.core.ml_plan import MLPlan


class TestMLPlan:
    """Test suite for ML plan data structure."""

    @pytest.fixture
    def ml_plan_dict(self):
        """Mock ML plan dictionary."""
        return {
            "summary": "Binary classification for churn prediction",
            "feature_engineering": """
        Recommended feature engineering:
        1. Standardize numerical features (age, tenure, monthly_charges)
        2. Create interaction features: tenure * monthly_charges
        3. One-hot encode categorical variables
        4. Generate aggregation features from transaction history
        """,
            "model_architecture_and_loss": "MLP architecture with 2-3 hidden layers",
            "training_configuration": "Adam optimizer with learning rate 0.001",
            "evaluation": "AUC-ROC as primary metric",
            "version": 1,
            "stage": "initial",
            "created_at": datetime.now().isoformat(),
        }

    def test_create_ml_plan_from_dict(self, ml_plan_dict):
        """Test that MLPlan can be created from dictionary."""
        plan = MLPlan.from_dict(ml_plan_dict)

        assert plan.summary == ml_plan_dict["summary"]
        assert plan.feature_engineering == ml_plan_dict["feature_engineering"]
        assert plan.model_architecture_and_loss == ml_plan_dict["model_architecture_and_loss"]
        assert plan.training_configuration == ml_plan_dict["training_configuration"]
        assert plan.evaluation == ml_plan_dict["evaluation"]
        assert plan.version == 1
        assert plan.stage == "initial"

    def test_extract_feature_engineering(self, ml_plan_dict):
        """Test that feature_engineering field can be extracted."""
        plan = MLPlan.from_dict(ml_plan_dict)
        feature_engineering = plan.feature_engineering

        assert feature_engineering is not None
        assert len(feature_engineering) > 0
        assert "Standardize numerical features" in feature_engineering

    def test_extract_model_architecture(self, ml_plan_dict):
        """Test that model_architecture_and_loss field can be extracted."""
        plan = MLPlan.from_dict(ml_plan_dict)
        architecture = plan.model_architecture_and_loss

        assert architecture is not None
        assert len(architecture) > 0
        assert "MLP architecture" in architecture

    def test_ml_plan_to_dict(self, ml_plan_dict):
        """Test that MLPlan can be serialized to dictionary."""
        plan = MLPlan.from_dict(ml_plan_dict)
        result_dict = plan.to_dict()

        assert result_dict["summary"] == ml_plan_dict["summary"]
        assert result_dict["feature_engineering"] == ml_plan_dict["feature_engineering"]
        assert result_dict["version"] == 1
        assert result_dict["stage"] == "initial"

    def test_tool_extraction_pattern(self, ml_plan_dict):
        """Test the extraction pattern used in data process tool."""
        # Simulate what the data process tool does
        ml_plan_feature_engineering = None
        if ml_plan_dict:
            plan = MLPlan.from_dict(ml_plan_dict)
            ml_plan_feature_engineering = plan.feature_engineering

        assert ml_plan_feature_engineering is not None
        assert len(ml_plan_feature_engineering) > 0

    def test_model_tool_extraction_pattern(self, ml_plan_dict):
        """Test the extraction pattern used in ML model tool."""
        # Simulate what the ML model tool does
        ml_plan_architecture = None
        if ml_plan_dict:
            plan = MLPlan.from_dict(ml_plan_dict)
            ml_plan_architecture = plan.model_architecture_and_loss

        assert ml_plan_architecture is not None
        assert len(ml_plan_architecture) > 0
