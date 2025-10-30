"""Test PlanExecutionService."""

import time

import pytest

from arc.database import DatabaseManager
from arc.database.services.plan_execution_service import PlanExecutionService


@pytest.fixture
def db_manager(tmp_path):
    """Create test database manager with schema."""
    system_db = tmp_path / "system.db"
    user_db = tmp_path / "user.db"
    with DatabaseManager(str(system_db), str(user_db)) as manager:
        yield manager


@pytest.fixture
def service(db_manager):
    """Create service instance."""
    return PlanExecutionService(db_manager)


@pytest.fixture
def test_plan(db_manager):
    """Create a test plan."""
    db_manager.system_execute("""
        INSERT INTO plans (plan_id, version, user_context, source_tables, plan_yaml, status)
        VALUES ('test_plan', 1, 'test context', 'test_table', 'test yaml', 'active')
    """)
    return "test_plan"


def test_store_execution(service, test_plan):
    """Test storing an execution record."""
    service.store_execution(
        execution_id="exec_123",
        plan_id=test_plan,
        step_type="data_processing",
        context="CREATE TABLE test AS SELECT 1",
        outputs=[
            {
                "name": "test",
                "row_count": 1,
                "columns": [{"name": "col1", "type": "INTEGER"}],
            }
        ],
    )

    # Verify it was stored
    exec_record = service.get_execution("exec_123")
    assert exec_record is not None
    assert exec_record["id"] == "exec_123"
    assert exec_record["plan_id"] == test_plan
    assert exec_record["step_type"] == "data_processing"
    assert exec_record["status"] == "completed"
    assert "CREATE TABLE" in exec_record["context"]
    assert len(exec_record["outputs"]) == 1
    assert exec_record["outputs"][0]["name"] == "test"


def test_store_execution_with_error(service, test_plan):
    """Test storing a failed execution."""
    service.store_execution(
        execution_id="exec_fail",
        plan_id=test_plan,
        step_type="training",
        context="model: test",
        outputs=[],
        status="failed",
        error_message="Out of memory",
    )

    exec_record = service.get_execution("exec_fail")
    assert exec_record is not None
    assert exec_record["status"] == "failed"
    assert exec_record["error_message"] == "Out of memory"


def test_get_execution_not_found(service):
    """Test getting non-existent execution."""
    result = service.get_execution("nonexistent")
    assert result is None


def test_get_latest_execution(service, test_plan):
    """Test getting the latest execution of a step type."""
    # Store multiple executions
    service.store_execution(
        execution_id="exec_1",
        plan_id=test_plan,
        step_type="data_processing",
        context="SQL 1",
        outputs=[],
    )

    time.sleep(0.01)  # Ensure different timestamps

    service.store_execution(
        execution_id="exec_2",
        plan_id=test_plan,
        step_type="data_processing",
        context="SQL 2",
        outputs=[],
    )

    time.sleep(0.01)

    service.store_execution(
        execution_id="exec_3",
        plan_id=test_plan,
        step_type="data_processing",
        context="SQL 3",
        outputs=[],
    )

    # Get latest
    latest = service.get_latest_execution(test_plan, "data_processing")
    assert latest is not None
    assert latest["id"] == "exec_3"
    assert "SQL 3" in latest["context"]


def test_get_latest_execution_by_type(service, test_plan):
    """Test getting latest execution filters by step type."""
    service.store_execution(
        execution_id="exec_data",
        plan_id=test_plan,
        step_type="data_processing",
        context="SQL",
        outputs=[],
    )

    time.sleep(0.01)

    service.store_execution(
        execution_id="exec_train",
        plan_id=test_plan,
        step_type="training",
        context="YAML",
        outputs=[],
    )

    # Get latest data processing
    latest_data = service.get_latest_execution(test_plan, "data_processing")
    assert latest_data["id"] == "exec_data"

    # Get latest training
    latest_train = service.get_latest_execution(test_plan, "training")
    assert latest_train["id"] == "exec_train"


def test_get_latest_execution_not_found(service, test_plan):
    """Test getting latest execution when none exist."""
    result = service.get_latest_execution(test_plan, "data_processing")
    assert result is None


def test_get_all_executions(service, test_plan):
    """Test getting all executions for a plan."""
    # Store executions of different types
    service.store_execution(
        execution_id="exec_1",
        plan_id=test_plan,
        step_type="data_processing",
        context="SQL",
        outputs=[],
    )

    time.sleep(0.01)

    service.store_execution(
        execution_id="exec_2",
        plan_id=test_plan,
        step_type="training",
        context="YAML",
        outputs=[],
    )

    time.sleep(0.01)

    service.store_execution(
        execution_id="exec_3",
        plan_id=test_plan,
        step_type="evaluation",
        context="Eval config",
        outputs=[],
    )

    # Get all executions
    all_execs = service.get_all_executions(test_plan)
    assert len(all_execs) == 3
    assert all_execs[0]["id"] == "exec_1"
    assert all_execs[1]["id"] == "exec_2"
    assert all_execs[2]["id"] == "exec_3"


def test_get_all_executions_empty(service, test_plan):
    """Test getting all executions when none exist."""
    result = service.get_all_executions(test_plan)
    assert result == []


def test_outputs_json_serialization(service, test_plan):
    """Test that outputs are properly JSON serialized/deserialized."""
    complex_outputs = [
        {
            "name": "table1",
            "type": "table",
            "row_count": 100,
            "columns": [
                {"name": "col1", "type": "INTEGER"},
                {"name": "col2", "type": "VARCHAR"},
            ],
        },
        {
            "name": "table2",
            "type": "view",
            "row_count": 50,
            "columns": [
                {"name": "feature1", "type": "DOUBLE"},
                {"name": "feature2", "type": "DOUBLE"},
            ],
        },
    ]

    service.store_execution(
        execution_id="exec_complex",
        plan_id=test_plan,
        step_type="data_processing",
        context="Complex SQL",
        outputs=complex_outputs,
    )

    exec_record = service.get_execution("exec_complex")
    assert exec_record is not None
    assert len(exec_record["outputs"]) == 2
    assert exec_record["outputs"][0]["name"] == "table1"
    assert exec_record["outputs"][0]["type"] == "table"
    assert len(exec_record["outputs"][0]["columns"]) == 2
    assert exec_record["outputs"][1]["name"] == "table2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
