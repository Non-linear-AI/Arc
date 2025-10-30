"""Test plan_executions table schema."""

import json
from datetime import datetime

import pytest

from arc.database.duckdb import DuckDBDatabase


@pytest.fixture
def test_db():
    """Create test database."""
    db = DuckDBDatabase(":memory:")
    db.init_schema()
    yield db
    db.close()


def test_plan_executions_table_exists(test_db):
    """Verify plan_executions table is created."""
    result = test_db.query("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name = 'plan_executions'
    """)

    assert len(result.rows) == 1
    assert result.rows[0]["table_name"] == "plan_executions"


def test_plan_executions_columns(test_db):
    """Verify plan_executions has correct columns."""
    result = test_db.query("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'plan_executions'
        ORDER BY ordinal_position
    """)

    columns = {row["column_name"]: row["data_type"] for row in result.rows}

    # Verify required columns exist
    assert "id" in columns
    assert "plan_id" in columns
    assert "step_type" in columns
    assert "status" in columns
    assert "started_at" in columns
    assert "completed_at" in columns
    assert "context" in columns
    assert "outputs" in columns
    assert "error_message" in columns
    assert "created_at" in columns


def test_plan_executions_indexes(test_db):
    """Verify indexes are created."""
    result = test_db.query("""
        SELECT index_name
        FROM duckdb_indexes()
        WHERE table_name = 'plan_executions'
    """)

    index_names = [row["index_name"] for row in result.rows]

    # Verify indexes exist
    assert "idx_plan_executions_plan" in index_names
    assert "idx_plan_executions_step_type" in index_names


def test_insert_plan_execution(test_db):
    """Test inserting a plan execution record."""
    # First create a plan (to satisfy foreign key)
    test_db.execute("""
        INSERT INTO plans (plan_id, version, user_context, source_tables, plan_yaml, status)
        VALUES ('test_plan', 1, 'test context', 'test_table', 'test yaml', 'active')
    """)

    # Insert plan execution
    now = datetime.now()
    test_db.execute(
        """
        INSERT INTO plan_executions
        (id, plan_id, step_type, status, started_at, completed_at, context, outputs)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        [
            "exec_123",
            "test_plan",
            "data_processing",
            "completed",
            now,
            now,
            "CREATE TABLE test AS SELECT 1",
            json.dumps([{"name": "test", "row_count": 1}]),
        ],
    )

    # Verify it was inserted
    result = test_db.query("SELECT * FROM plan_executions WHERE id = 'exec_123'")

    assert len(result.rows) == 1
    row = result.rows[0]
    assert row["id"] == "exec_123"
    assert row["plan_id"] == "test_plan"
    assert row["step_type"] == "data_processing"
    assert row["status"] == "completed"
    assert "CREATE TABLE" in row["context"]

    # Verify outputs is valid JSON
    outputs = json.loads(row["outputs"])
    assert len(outputs) == 1
    assert outputs[0]["name"] == "test"


def test_query_by_plan_id(test_db):
    """Test querying executions by plan_id."""
    # Create plan
    test_db.execute("""
        INSERT INTO plans (plan_id, version, user_context, source_tables, plan_yaml, status)
        VALUES ('test_plan', 1, 'test', 'table', 'yaml', 'active')
    """)

    # Insert multiple executions
    now = datetime.now()
    for i in range(3):
        test_db.execute(
            """
            INSERT INTO plan_executions
            (id, plan_id, step_type, status, started_at, context, outputs)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            [
                f"exec_{i}",
                "test_plan",
                "data_processing",
                "completed",
                now,
                f"SQL {i}",
                "[]",
            ],
        )

    # Query by plan_id
    result = test_db.query("""
        SELECT id FROM plan_executions
        WHERE plan_id = 'test_plan'
        ORDER BY id
    """)

    assert len(result.rows) == 3
    assert result.rows[0]["id"] == "exec_0"
    assert result.rows[1]["id"] == "exec_1"
    assert result.rows[2]["id"] == "exec_2"


def test_query_by_step_type(test_db):
    """Test querying executions by step_type."""
    # Create plan
    test_db.execute("""
        INSERT INTO plans (plan_id, version, user_context, source_tables, plan_yaml, status)
        VALUES ('test_plan', 1, 'test', 'table', 'yaml', 'active')
    """)

    # Insert executions of different types
    now = datetime.now()
    test_db.execute(
        """
        INSERT INTO plan_executions
        (id, plan_id, step_type, status, started_at, context, outputs)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        ["exec_data", "test_plan", "data_processing", "completed", now, "SQL", "[]"],
    )

    test_db.execute(
        """
        INSERT INTO plan_executions
        (id, plan_id, step_type, status, started_at, context, outputs)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        ["exec_train", "test_plan", "training", "completed", now, "YAML", "[]"],
    )

    # Query by step_type
    result = test_db.query("""
        SELECT id FROM plan_executions
        WHERE plan_id = 'test_plan' AND step_type = 'data_processing'
    """)

    assert len(result.rows) == 1
    assert result.rows[0]["id"] == "exec_data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
