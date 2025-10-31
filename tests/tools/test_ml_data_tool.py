"""Test ml_data tool JSON output structure."""

import json
from unittest.mock import MagicMock

import pytest

from arc.tools.ml_data import MLDataTool


class TestMLDataToolJSONOutput:
    """Test ml_data tool JSON output structure."""

    @pytest.fixture
    def mock_tool(self):
        """Create a mock MLDataTool instance."""
        mock_services = MagicMock()
        mock_services.query = MagicMock()
        mock_services.ml_data = MagicMock()
        tool = MLDataTool(mock_services, api_key="test_api_key")
        return tool

    def test_build_data_result_accepted(self, mock_tool):
        """Test _build_data_result returns valid JSON for accepted status."""
        result = mock_tool._build_data_result(
            status="accepted",
            data_processing_id="data_abc123",
            output_tables=["table1", "table2"],
            sql_operations=[
                "CREATE TABLE table1 AS SELECT...",
                "CREATE TABLE table2 AS SELECT...",
            ],
        )

        # Should be valid JSON
        parsed = json.loads(result)

        # Check structure
        assert parsed["status"] == "accepted"
        assert parsed["data_processing_id"] == "data_abc123"
        assert "execution" in parsed
        assert parsed["execution"]["output_tables"] == ["table1", "table2"]
        assert len(parsed["execution"]["sql_operations"]) == 2

    def test_build_data_result_cancelled(self, mock_tool):
        """Test _build_data_result returns valid JSON for cancelled status."""
        result = mock_tool._build_data_result(
            status="cancelled",
            data_processing_id="cancelled_xyz789",
            output_tables=[],
            sql_operations=[],
        )

        # Should be valid JSON
        parsed = json.loads(result)

        # Check structure
        assert parsed["status"] == "cancelled"
        assert parsed["data_processing_id"] == "cancelled_xyz789"
        assert "execution" in parsed
        assert parsed["execution"]["output_tables"] == []
        assert parsed["execution"]["sql_operations"] == []

    def test_build_data_result_default_empty_lists(self, mock_tool):
        """Test _build_data_result uses empty lists for None values."""
        result = mock_tool._build_data_result(
            status="accepted",
            data_processing_id="data_123",
            output_tables=None,
            sql_operations=None,
        )

        # Should be valid JSON
        parsed = json.loads(result)

        # Check structure - should have empty lists
        assert parsed["execution"]["output_tables"] == []
        assert parsed["execution"]["sql_operations"] == []

    def test_build_data_result_json_formatting(self, mock_tool):
        """Test _build_data_result returns compact JSON."""
        result = mock_tool._build_data_result(
            status="accepted",
            data_processing_id="data_123",
            output_tables=["table1"],
            sql_operations=["CREATE TABLE table1"],
        )

        # Should be compact (no newlines or indentation)
        assert "\n" not in result
        assert result.startswith("{")
        assert result.endswith("}")

    def test_json_output_has_required_fields(self, mock_tool):
        """Test JSON output always has required fields."""
        result = mock_tool._build_data_result(
            status="accepted",
            data_processing_id="data_123",
        )

        parsed = json.loads(result)

        # Required top-level fields
        assert "status" in parsed
        assert "data_processing_id" in parsed
        assert "execution" in parsed

        # Required execution fields
        assert "output_tables" in parsed["execution"]
        assert "sql_operations" in parsed["execution"]

    def test_json_output_status_values(self, mock_tool):
        """Test JSON output accepts valid status values."""
        for status in ["accepted", "cancelled"]:
            result = mock_tool._build_data_result(
                status=status,
                data_processing_id="data_123",
            )
            parsed = json.loads(result)
            assert parsed["status"] == status

    def test_parse_sql_string_to_array(self, mock_tool):
        """Test parsing SQL string with step markers into array."""
        sql_string = """BEGIN TRANSACTION;

-- Step 1/3: drop_old_table (execute)
DROP TABLE IF EXISTS "my_table";

-- Step 2/3: create_view (view)
DROP TABLE IF EXISTS "my_view";
DROP VIEW IF EXISTS "my_view";
CREATE VIEW "my_view" AS (SELECT * FROM source);

-- Step 3/3: create_table (table)
DROP VIEW IF EXISTS "my_table";
DROP TABLE IF EXISTS "my_table";
CREATE TABLE "my_table" AS (SELECT * FROM my_view);

COMMIT;"""

        result = mock_tool._build_data_result(
            status="accepted",
            data_processing_id="data_123",
            sql_operations=sql_string,  # Pass as string
        )

        parsed = json.loads(result)
        sql_ops = parsed["execution"]["sql_operations"]

        # Should have 3 operations
        assert len(sql_ops) == 3

        # Check each operation is clean (no step markers, no transaction)
        assert sql_ops[0] == 'DROP TABLE IF EXISTS "my_table";'
        assert "DROP TABLE IF EXISTS" in sql_ops[1]
        assert "CREATE VIEW" in sql_ops[1]
        assert "-- Step" not in sql_ops[1]
        assert "BEGIN TRANSACTION" not in sql_ops[2]
        assert "COMMIT" not in sql_ops[2]
