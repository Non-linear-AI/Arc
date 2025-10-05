"""Tests for DataSourceSpec YAML conversion functionality."""

import pytest

from arc.graph.features.data_source import DataSourceSpec, DataSourceStep


class TestDataSourceSpecYAMLConversion:
    """Test DataSourceSpec to/from YAML conversion."""

    @pytest.fixture
    def complex_sql_spec(self):
        """Create a DataSourceSpec with complex multi-line SQL."""
        steps = [
            DataSourceStep(
                name="customer_metrics",
                depends_on=["customers", "orders", "payments"],
                # Intentionally not formatted
                sql="SELECT column0,\n       case when column1 = 0 then null\n            else column1 end as column1,\n       case when column2 = 0 then null\n            else column2 end as column2,\n       case when column3 = 0 then null\n            else column3 end as column3,\n       case when column4 = 0 then null\n            else column4 end as column4,\n       case when column5 = 0.0 then null\n            else column5 end as column5,\n       column6,\n       column7,\n       column8\nFROM   iris_raw",  # noqa: E501
            ),
            DataSourceStep(
                name="validate_cleaned_data",
                depends_on=["clean_physiological_data"],
                # Intentionally not formatted
                sql="SELECT column0,\n       column1,\n       column2,\n       column3,\n       column4,\n       column5,\n       column6,\n       column7,\n       column8\nFROM   validate_cleaned_data\nWHERE  column0 >= 0\n   and column0 <= 20\n   and (column1 is null\n    or column1 between 50\n   and 300)\n   and (column2 is null\n    or column2 between 30\n   and 180)\n   and (column3 is null\n    or column3 between 10\n   and 80)\n   and (column4 is null\n    or column4 between 20\n   and 900)\n   and (column5 is null\n    or column5 between 15.0\n   and 60.0)\n   and column6 >= 0.0\n   and column6 <= 2.5\n   and column7 >= 20\n   and column7 <= 100\n   and column8 in (0, 1)",  # noqa: E501
            ),
        ]
        return DataSourceSpec(
            steps=steps, outputs=["customer_metrics"], vars={"min_date": "2023-01-01"}
        )

    def test_to_yaml_formats_sql_and_adds_spacing(self, complex_sql_spec):
        """Test that to_yaml always formats SQL and adds proper spacing."""
        yaml_content = complex_sql_spec.to_yaml()

        # Verify YAML is generated
        assert yaml_content
        assert "data_source:" in yaml_content

        # Verify spacing between steps - should have blank line between steps
        lines = yaml_content.splitlines()
        step_indices = [
            i for i, line in enumerate(lines) if line.strip().startswith("- name:")
        ]
        if len(step_indices) > 1:
            # Check there's a blank line before the second step
            assert lines[step_indices[1] - 1].strip() == ""

        # Verify spacing before sections - check for blank lines before sections
        assert "\n  outputs:" in yaml_content
        assert "\n  vars:" in yaml_content

        # Verify SQL formatting - should use literal block style
        assert "sql: |-" in yaml_content or "sql: |" in yaml_content

        # Verify SQL is formatted in readable multi-line format
        # Check for SELECT keyword on its own line
        assert "SELECT\n" in yaml_content or "      SELECT\n" in yaml_content
        # Check for FROM keyword on its own line
        assert "FROM iris_raw" in yaml_content or "      FROM iris_raw" in yaml_content
        # Verify columns are indented properly (multi-line format)
        assert "column0," in yaml_content
