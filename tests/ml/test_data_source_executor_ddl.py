"""Test DDL statement handling in data source executor."""

import pytest

from arc.ml.data_source_executor import _quote_ddl_identifiers


class TestDDLIdentifierQuoting:
    """Test suite for DDL identifier quoting."""

    def test_drop_table_with_hyphen_if_exists(self):
        """Test DROP TABLE IF EXISTS with hyphen."""
        sql = "DROP TABLE IF EXISTS my-eval-v15_predictions"
        expected = 'DROP TABLE IF EXISTS "my-eval-v15_predictions"'
        result = _quote_ddl_identifiers(sql)
        assert result == expected, f"Expected: {expected}, Got: {result}"

    def test_drop_table_with_hyphen_no_if_exists(self):
        """Test DROP TABLE without IF EXISTS."""
        sql = "DROP TABLE diabetes_evaluator-v1_predictions"
        expected = 'DROP TABLE "diabetes_evaluator-v1_predictions"'
        result = _quote_ddl_identifiers(sql)
        assert result == expected, f"Expected: {expected}, Got: {result}"

    def test_drop_view_with_hyphen(self):
        """Test DROP VIEW with hyphen."""
        sql = "DROP VIEW IF EXISTS my-view-name"
        expected = 'DROP VIEW IF EXISTS "my-view-name"'
        result = _quote_ddl_identifiers(sql)
        assert result == expected, f"Expected: {expected}, Got: {result}"

    def test_alter_table_with_hyphen(self):
        """Test ALTER TABLE with hyphen."""
        sql = "ALTER TABLE my-table ADD COLUMN new_col INT"
        expected = 'ALTER TABLE "my-table" ADD COLUMN new_col INT'
        result = _quote_ddl_identifiers(sql)
        assert result == expected, f"Expected: {expected}, Got: {result}"

    def test_truncate_table_with_hyphen(self):
        """Test TRUNCATE TABLE with hyphen."""
        sql = "TRUNCATE TABLE my-data-table"
        expected = 'TRUNCATE TABLE "my-data-table"'
        result = _quote_ddl_identifiers(sql)
        assert result == expected, f"Expected: {expected}, Got: {result}"

    def test_simple_table_name(self):
        """Test table name without special chars still gets quoted."""
        sql = "DROP TABLE simple_table"
        expected = 'DROP TABLE "simple_table"'
        result = _quote_ddl_identifiers(sql)
        assert result == expected, f"Expected: {expected}, Got: {result}"

    def test_already_quoted(self):
        """Test that already quoted identifiers are preserved."""
        sql = 'DROP TABLE "already-quoted"'
        expected = 'DROP TABLE "already-quoted"'
        result = _quote_ddl_identifiers(sql)
        assert result == expected, f"Expected: {expected}, Got: {result}"

    @pytest.mark.parametrize(
        "input_sql,expected",
        [
            (
                "DROP TABLE my-eval-v15_predictions",
                'DROP TABLE "my-eval-v15_predictions"',
            ),
            (
                "DROP VIEW test-view",
                'DROP VIEW "test-view"',
            ),
            (
                "ALTER TABLE my-table RENAME TO new-table",
                'ALTER TABLE "my-table" RENAME TO new-table',
            ),
        ],
    )
    def test_various_ddl_statements(self, input_sql, expected):
        """Test various DDL statements with different formats."""
        result = _quote_ddl_identifiers(input_sql)
        assert result == expected, f"Input: {input_sql}, Expected: {expected}, Got: {result}"
