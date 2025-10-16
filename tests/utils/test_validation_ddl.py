"""Test DDL validation in arc.utils.validation."""

import pytest

from arc.utils.validation import validate_sql_syntax


class TestDDLValidation:
    """Test suite for DDL statement validation."""

    def test_drop_table_with_hyphen(self):
        """Test that DROP TABLE with hyphens in name passes validation."""
        sql = "DROP TABLE IF EXISTS my-eval-v15_predictions"
        errors = validate_sql_syntax(sql)
        assert errors == [], f"DROP TABLE with hyphen should pass, got errors: {errors}"

    def test_drop_table_without_if_exists(self):
        """Test DROP TABLE without IF EXISTS clause."""
        sql = "DROP TABLE diabetes_evaluator-v1_predictions"
        errors = validate_sql_syntax(sql)
        assert errors == [], f"DROP TABLE should pass, got errors: {errors}"

    def test_drop_view_with_hyphen(self):
        """Test DROP VIEW with hyphens in name."""
        sql = "DROP VIEW IF EXISTS my-view-name"
        errors = validate_sql_syntax(sql)
        assert errors == [], f"DROP VIEW should pass, got errors: {errors}"

    def test_alter_table(self):
        """Test ALTER TABLE statement."""
        sql = "ALTER TABLE my-table ADD COLUMN new_col INT"
        errors = validate_sql_syntax(sql)
        assert errors == [], f"ALTER TABLE should pass, got errors: {errors}"

    def test_truncate_table(self):
        """Test TRUNCATE TABLE statement."""
        sql = "TRUNCATE TABLE my-data-table"
        errors = validate_sql_syntax(sql)
        assert errors == [], f"TRUNCATE TABLE should pass, got errors: {errors}"

    def test_select_with_nonexistent_table(self):
        """Test that SELECT with nonexistent table passes (catalog error ignored)."""
        sql = "SELECT * FROM nonexistent_table"
        errors = validate_sql_syntax(sql)
        # Catalog errors (table not found) should be ignored
        assert errors == [], f"SELECT should pass even with missing table, got: {errors}"

    def test_invalid_select_syntax(self):
        """Test that invalid SELECT syntax is caught."""
        sql = "SELECT FROM WHERE"
        errors = validate_sql_syntax(sql)
        assert len(errors) > 0, "Invalid SELECT syntax should be caught"
        assert any("syntax error" in err.lower() for err in errors)

    @pytest.mark.parametrize(
        "sql",
        [
            "DROP TABLE IF EXISTS diabetes_evaluator-v1_predictions",
            "DROP TABLE IF EXISTS pidd-evaluation-v1_predictions",
            "DROP TABLE IF EXISTS my-eval-v27_predictions",
        ],
    )
    def test_multiple_drop_statements(self, sql):
        """Test various DROP TABLE statements with different name formats."""
        errors = validate_sql_syntax(sql)
        assert errors == [], f"DROP TABLE should pass for: {sql}, got errors: {errors}"
