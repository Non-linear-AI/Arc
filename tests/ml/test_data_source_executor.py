"""Test data_source_executor module."""

import pytest

from arc.database.manager import DatabaseManager
from arc.graph.features.data_source import DataSourceSpec, DataSourceStep
from arc.ml.data_source_executor import execute_data_source_pipeline


@pytest.fixture
def db_manager(tmp_path):
    """Create database manager with test databases."""
    system_db = tmp_path / "system.db"
    user_db = tmp_path / "user.db"
    with DatabaseManager(str(system_db), str(user_db)) as manager:
        yield manager


@pytest.mark.asyncio
async def test_execute_pipeline_captures_sql(db_manager):
    """Test that SQL is captured from pipeline execution."""
    # Create a simple data source spec
    spec = DataSourceSpec(
        name="test_pipeline",
        steps=[
            DataSourceStep(
                name="test_table",
                depends_on=[],
                type="table",
                sql="SELECT 1 as id, 'test' as value",
            )
        ],
        outputs=["test_table"],
        vars={},
    )

    # Execute pipeline
    result = await execute_data_source_pipeline(
        spec=spec,
        target_db="user",
        db_manager=db_manager,
    )

    # Verify SQL was captured
    assert result.sql is not None
    assert "test_table" in result.sql
    assert "SELECT 1 as id" in result.sql
    assert "CREATE TABLE" in result.sql
    # Should contain actual transaction markers
    assert "BEGIN TRANSACTION" in result.sql
    assert "COMMIT" in result.sql


@pytest.mark.asyncio
async def test_execute_pipeline_captures_outputs(db_manager):
    """Test that output schemas and row counts are captured."""
    # Create a data source spec with outputs
    spec = DataSourceSpec(
        name="test_output_pipeline",
        steps=[
            DataSourceStep(
                name="test_output",
                depends_on=[],
                type="table",
                sql="SELECT 1 as id, 'test' as name, 42 as value",
            )
        ],
        outputs=["test_output"],
        vars={},
    )

    # Execute pipeline
    result = await execute_data_source_pipeline(
        spec=spec,
        target_db="user",
        db_manager=db_manager,
    )

    # Verify outputs were captured
    assert result.outputs is not None
    assert len(result.outputs) == 1

    output = result.outputs[0]
    assert output["name"] == "test_output"
    assert output["type"] == "table"
    assert output["row_count"] == 1
    assert len(output["columns"]) == 3

    # Verify column details
    col_names = [col["name"] for col in output["columns"]]
    assert "id" in col_names
    assert "name" in col_names
    assert "value" in col_names


@pytest.mark.asyncio
async def test_execute_pipeline_multiple_steps(db_manager):
    """Test SQL capture with multiple steps."""
    spec = DataSourceSpec(
        name="multi_step_pipeline",
        steps=[
            DataSourceStep(
                name="step1",
                depends_on=[],
                type="table",
                sql="SELECT 1 as id",
            ),
            DataSourceStep(
                name="step2",
                depends_on=["step1"],
                type="view",
                sql="SELECT * FROM step1 WHERE id > 0",
            ),
            DataSourceStep(
                name="final_output",
                depends_on=["step2"],
                type="table",
                sql="SELECT * FROM step2",
            ),
        ],
        outputs=["final_output"],
        vars={},
    )

    result = await execute_data_source_pipeline(
        spec=spec,
        target_db="user",
        db_manager=db_manager,
    )

    # Verify SQL contains all steps
    assert result.sql is not None
    assert "step1" in result.sql
    assert "step2" in result.sql
    assert "final_output" in result.sql

    # Verify step comments
    assert "Step 1/3" in result.sql
    assert "Step 2/3" in result.sql
    assert "Step 3/3" in result.sql

    # Verify outputs
    assert len(result.outputs) == 1
    assert result.outputs[0]["name"] == "final_output"


@pytest.mark.asyncio
async def test_execute_pipeline_multiple_outputs(db_manager):
    """Test capturing multiple output tables."""
    spec = DataSourceSpec(
        name="multi_output_pipeline",
        steps=[
            DataSourceStep(
                name="output1",
                depends_on=[],
                type="table",
                sql="SELECT 1 as id",
            ),
            DataSourceStep(
                name="output2",
                depends_on=[],
                type="table",
                sql="SELECT 2 as id",
            ),
        ],
        outputs=["output1", "output2"],
        vars={},
    )

    result = await execute_data_source_pipeline(
        spec=spec,
        target_db="user",
        db_manager=db_manager,
    )

    # Verify both outputs captured
    assert len(result.outputs) == 2
    assert result.outputs[0]["name"] == "output1"
    assert result.outputs[1]["name"] == "output2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
