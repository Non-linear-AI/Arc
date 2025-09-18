import pytest

from src.arc.graph import ArcGraph, ArcGraphValidator, GraphValidationError
from src.arc.graph.validator import validate_graph_dict


class MockDatabase:
    """Mock database for testing runtime validation."""

    def __init__(self, schema: dict[str, str]):
        self.schema = schema

    def query(self, sql: str):
        """Mock query method."""

        class MockResult:
            def __init__(self, rows):
                self.rows = rows

            def __iter__(self):
                return iter(self.rows)

        if sql.startswith("DESCRIBE"):
            rows = [
                {"column_name": col, "column_type": dtype}
                for col, dtype in self.schema.items()
            ]
            return MockResult(rows)
        return MockResult([])


def test_validator_rejects_missing_fields():
    bad = {"model_name": "x"}  # missing required sections
    with pytest.raises(GraphValidationError):
        validate_graph_dict(bad)


def test_validator_requires_model_and_trainer_sections():
    data = {
        "version": "0.1",
        "model_name": "m",
        "features": {"feature_columns": ["a"], "processors": []},
        # missing model and trainer
    }
    with pytest.raises(GraphValidationError):
        validate_graph_dict(data)


def test_validator_accepts_simple_graph():
    data = {
        "version": "0.1",
        "model_name": "m",
        "features": {"feature_columns": ["a"], "processors": []},
        "model": {
            "inputs": {"features": {"dtype": "float32", "shape": [None, 4]}},
            "graph": [
                {"name": "linear", "type": "core.Linear", "params": {"in": 4}},
            ],
            "outputs": {"y": "linear.output"},
        },
        "trainer": {
            "optimizer": {"type": "AdamW", "config": {"learning_rate": 0.001}},
            "loss": {"type": "core.MSELoss", "inputs": {"pred": "model.y"}},
        },
    }
    validate_graph_dict(data)  # should not raise


class TestArcGraphValidator:
    """Test the class-based validator."""

    def test_static_validation_with_variables(self):
        """Test static validation with vars, states, and tensors."""
        data = {
            "version": "0.1",
            "model_name": "test_model",
            "features": {
                "feature_columns": ["x1", "x2"],
                "processors": [
                    {
                        "name": "inspect_features",
                        "op": "inspect.feature_stats",
                        "train_only": True,
                        "inputs": {"columns": "feature_columns"},
                        "outputs": {"vars.n_features": "n_features"},
                    },
                    {
                        "name": "learn_scaler",
                        "op": "fit.standard_scaler",
                        "train_only": True,
                        "inputs": {"x": "feature_columns"},
                        "outputs": {"states.scaler": "state"},
                    },
                    {
                        "name": "apply_scaling",
                        "op": "transform.standard_scaler",
                        "inputs": {"x": "feature_columns", "state": "states.scaler"},
                        "outputs": {"tensors.features": "output"},
                    },
                ],
            },
            "model": {
                "inputs": {
                    "features": {"dtype": "float32", "shape": [None, "vars.n_features"]}
                },
                "graph": [
                    {
                        "name": "linear",
                        "type": "core.Linear",
                        "params": {"in_features": "vars.n_features", "out_features": 1},
                    },
                ],
                "outputs": {"prediction": "linear.output"},
            },
            "trainer": {
                "optimizer": {"type": "AdamW"},
                "loss": {
                    "type": "core.MSELoss",
                    "inputs": {"pred": "model.prediction"},
                },
            },
        }

        graph = ArcGraph.from_dict(data)
        validator = ArcGraphValidator()

        # Should pass static validation
        validator.validate_static(graph)

    def test_static_validation_undefined_variable(self):
        """Test that undefined variable references are caught."""
        data = {
            "version": "0.1",
            "model_name": "test_model",
            "features": {"feature_columns": ["x1"], "processors": []},
            "model": {
                "inputs": {
                    "features": {
                        "dtype": "float32",
                        "shape": [None, "vars.undefined"],
                    }  # undefined variable
                },
                "graph": [
                    {"name": "linear", "type": "core.Linear"},
                ],
                "outputs": {"prediction": "linear.output"},
            },
            "trainer": {
                "optimizer": {"type": "AdamW"},
                "loss": {"type": "core.MSELoss"},
            },
        }

        graph = ArcGraph.from_dict(data)
        validator = ArcGraphValidator()

        with pytest.raises(GraphValidationError, match="undefined variable"):
            validator.validate_static(graph)

    def test_static_validation_undefined_node_reference(self):
        """Test that undefined node references are caught."""
        data = {
            "version": "0.1",
            "model_name": "test_model",
            "features": {"feature_columns": ["x1"], "processors": []},
            "model": {
                "inputs": {"features": {"dtype": "float32", "shape": [None, 1]}},
                "graph": [
                    {
                        "name": "linear",
                        "type": "core.Linear",
                        "inputs": {"input": "undefined_node.output"},
                    },  # undefined node
                ],
                "outputs": {"prediction": "linear.output"},
            },
            "trainer": {
                "optimizer": {"type": "AdamW"},
                "loss": {"type": "core.MSELoss"},
            },
        }

        graph = ArcGraph.from_dict(data)
        validator = ArcGraphValidator()

        with pytest.raises(GraphValidationError, match="undefined source"):
            validator.validate_static(graph)

    def test_runtime_validation_missing_columns(self):
        """Test runtime validation catches missing columns."""
        data = {
            "version": "0.1",
            "model_name": "test_model",
            "features": {
                "feature_columns": ["x1", "missing_col"],  # missing_col doesn't exist
                "processors": [],
            },
            "model": {
                "inputs": {"features": {"dtype": "float32", "shape": [None, 2]}},
                "graph": [
                    {"name": "linear", "type": "core.Linear"},
                ],
                "outputs": {"prediction": "linear.output"},
            },
            "trainer": {
                "optimizer": {"type": "AdamW"},
                "loss": {"type": "core.MSELoss"},
            },
        }

        graph = ArcGraph.from_dict(data)
        mock_db = MockDatabase({"x1": "FLOAT"})  # missing missing_col
        validator = ArcGraphValidator(database=mock_db)

        with pytest.raises(GraphValidationError, match="Missing feature columns"):
            validator.validate_runtime(graph, "test_table")

    def test_runtime_validation_success(self):
        """Test successful runtime validation."""
        data = {
            "version": "0.1",
            "model_name": "test_model",
            "features": {
                "feature_columns": ["x1", "x2"],
                "target_columns": ["y"],
                "processors": [],
            },
            "model": {
                "inputs": {"features": {"dtype": "float32", "shape": [None, 2]}},
                "graph": [
                    {"name": "linear", "type": "core.Linear"},
                ],
                "outputs": {"prediction": "linear.output"},
            },
            "trainer": {
                "optimizer": {"type": "AdamW"},
                "loss": {"type": "core.MSELoss"},
            },
        }

        graph = ArcGraph.from_dict(data)
        mock_db = MockDatabase({"x1": "FLOAT", "x2": "FLOAT", "y": "FLOAT"})
        validator = ArcGraphValidator(database=mock_db)

        # Should pass without error
        validator.validate_runtime(graph, "test_table")

    def test_runtime_validation_no_database(self):
        """Test that runtime validation requires database."""
        data = {
            "version": "0.1",
            "model_name": "test_model",
            "features": {"feature_columns": ["x1"], "processors": []},
            "model": {
                "inputs": {"features": {"dtype": "float32", "shape": [None, 1]}},
                "graph": [{"name": "linear", "type": "core.Linear"}],
                "outputs": {"prediction": "linear.output"},
            },
            "trainer": {
                "optimizer": {"type": "AdamW"},
                "loss": {"type": "core.MSELoss"},
            },
        }

        graph = ArcGraph.from_dict(data)
        validator = ArcGraphValidator()  # No database

        with pytest.raises(RuntimeError, match="Database connection required"):
            validator.validate_runtime(graph, "test_table")
