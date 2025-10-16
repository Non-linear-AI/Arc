"""Test ML plan template rendering for data process agent."""

from pathlib import Path

import jinja2
import pytest


class TestDataProcessTemplateRendering:
    """Test suite for data process template rendering with ML plan."""

    @pytest.fixture
    def template_env(self):
        """Create Jinja2 environment with data process templates."""
        template_path = Path("src/arc/core/agents/data_process/templates")
        if not template_path.exists():
            pytest.skip(f"Template directory not found: {template_path}")

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_path)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return env

    @pytest.fixture
    def schema_info(self):
        """Mock schema information."""
        return {
            "database": "user",
            "tables": [
                {
                    "name": "customers",
                    "columns": [
                        {"name": "customer_id", "type": "INTEGER", "nullable": False},
                        {"name": "tenure", "type": "INTEGER", "nullable": True},
                        {"name": "monthly_charges", "type": "REAL", "nullable": True},
                    ],
                }
            ],
            "total_tables": 1,
        }

    @pytest.fixture
    def ml_plan_feature_engineering(self):
        """Mock ML plan feature engineering guidance."""
        return """
    Feature engineering strategy:
    - Normalize all numerical features using StandardScaler
    - Create interaction features between tenure and monthly charges
    - Generate time-based features from customer signup date
    - Encode categorical variables using one-hot encoding
    """

    def test_template_with_ml_plan(self, template_env, schema_info, ml_plan_feature_engineering):
        """Test that template correctly includes ML plan guidance."""
        template = template_env.get_template("prompt.j2")

        prompt = template.render(
            user_context="Create customer features for churn prediction",
            schema_info=schema_info,
            target_tables=["customers"],
            existing_yaml=None,
            editing_instructions=None,
            ml_plan_feature_engineering=ml_plan_feature_engineering,
        )

        # Verify ML plan section appears
        assert "ML PLAN FEATURE ENGINEERING GUIDANCE" in prompt
        assert ml_plan_feature_engineering.strip() in prompt
        assert "CRITICAL" in prompt
        assert "Follow this strategy closely" in prompt

    def test_template_without_ml_plan(self, template_env, schema_info):
        """Test that template correctly omits ML plan section when not provided."""
        template = template_env.get_template("prompt.j2")

        prompt = template.render(
            user_context="Create customer features for churn prediction",
            schema_info=schema_info,
            target_tables=["customers"],
            existing_yaml=None,
            editing_instructions=None,
            ml_plan_feature_engineering=None,
        )

        # Verify ML plan section does NOT appear
        assert "ML PLAN FEATURE ENGINEERING GUIDANCE" not in prompt

        # Verify the prompt still has essential sections
        assert "TASK" in prompt
        assert "Available Tables" in prompt
        assert "customers" in prompt

    def test_template_structure_without_ml_plan(self, template_env, schema_info):
        """Test that without ML plan, TASK section directly follows with Available Tables."""
        template = template_env.get_template("prompt.j2")

        prompt = template.render(
            user_context="Create customer features",
            schema_info=schema_info,
            target_tables=["customers"],
            existing_yaml=None,
            editing_instructions=None,
            ml_plan_feature_engineering=None,
        )

        lines = prompt.split("\n")

        # Find TASK and Available Tables sections
        task_idx = None
        available_tables_idx = None

        for i, line in enumerate(lines):
            if "## TASK" in line:
                task_idx = i
            elif "## Available Tables" in line:
                available_tables_idx = i

        assert task_idx is not None, "TASK section should exist"
        assert available_tables_idx is not None, "Available Tables section should exist"

        # Without ML plan, Available Tables should come relatively soon after TASK
        # (allowing for a few lines of context)
        assert (
            available_tables_idx - task_idx < 10
        ), "Available Tables should follow TASK section closely when no ML plan"
