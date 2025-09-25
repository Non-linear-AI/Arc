"""Tests for the spec-driven features processor pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest
import torch

from arc.database.base import Database, QueryResult
from arc.ml.data import DataProcessor
from arc.ml.processors.base import ProcessorError


@dataclass
class _FakeDB(Database):
    """Minimal in-memory Database to support simple SELECT queries.

    Supports queries of the form:
        SELECT col1, col2, ... FROM table
    """

    rows: list[dict[str, Any]]

    def query(self, sql: str) -> QueryResult:
        # Very naive parser sufficient for tests
        sql_upper = sql.upper()
        assert sql_upper.startswith("SELECT ") and " FROM " in sql_upper
        select_part, from_part = sql.split(" FROM ", 1)
        cols_part = select_part[len("SELECT ") :]
        cols = [c.strip() for c in cols_part.split(",")]

        # Project requested columns from stored rows
        projected: list[dict[str, Any]] = []
        for row in self.rows:
            projected.append({c: row.get(c) for c in cols})

        return QueryResult(rows=projected)

    # Unused in tests
    def execute(self, sql: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover
        pass

    def init_schema(self) -> None:  # pragma: no cover
        pass


class _FakeMLDataService:
    """Minimal fake MLDataService for testing."""

    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def dataset_exists(self, _dataset_name: str) -> bool:
        return True  # Always pretend dataset exists for testing

    def get_features_as_tensors(
        self,
        dataset_name: str,  # noqa: ARG002
        feature_columns: list[str],
        target_columns: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Convert fake data to tensors."""
        # Extract feature data
        feature_data = []
        for row in self.rows:
            feature_row = [row[col] for col in feature_columns]
            feature_data.append(feature_row)

        features = torch.tensor(feature_data, dtype=torch.float32)

        # Extract target data if specified
        targets = None
        if target_columns:
            target_data = []
            for row in self.rows:
                target_row = [row[col] for col in target_columns]
                target_data.append(target_row)
            targets = torch.tensor(target_data, dtype=torch.float32)
            if len(target_columns) == 1:
                targets = targets.squeeze(1)  # Remove extra dimension for single target

        return features, targets

    def get_data(
        self,
        dataset_name: str,  # noqa: ARG002
        columns: list[str] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Get data as pandas DataFrame."""
        if columns is None:
            # If no columns specified, get all columns
            if not self.rows:
                return pd.DataFrame()
            columns = list(self.rows[0].keys())

        # Filter data by columns
        filtered_data = []
        for row in self.rows:
            filtered_row = {col: row[col] for col in columns if col in row}
            filtered_data.append(filtered_row)

        df = pd.DataFrame(filtered_data)

        if limit is not None:
            df = df.head(limit)

        return df


def test_run_feature_pipeline_numeric_and_hash_ops():
    # Sample table rows
    rows = [
        {"age": 30, "bmi": 22.5, "glucose_level": 120.0, "outcome": 0, "country": "US"},
        {"age": 45, "bmi": 27.1, "glucose_level": 99.0, "outcome": 1, "country": "CA"},
        {"age": 52, "bmi": 31.2, "glucose_level": 150.0, "outcome": 1, "country": "US"},
        {"age": 28, "bmi": 20.0, "glucose_level": 85.0, "outcome": 0, "country": "FR"},
    ]

    ml_data_service = _FakeMLDataService(rows=rows)
    dp = DataProcessor(ml_data_service=ml_data_service)

    features_spec = {
        "feature_columns": ["age", "bmi", "glucose_level"],
        "target_columns": ["outcome"],
        "processors": [
            {
                "name": "assemble_raw_features",
                "op": "transform.assemble_vector",
                "inputs": {"columns": "feature_columns"},
                "outputs": {"tensors.raw_features": "output"},
            },
            {
                "name": "get_feature_count",
                "op": "inspect.feature_stats",
                "train_only": True,
                "inputs": {"tensor": "tensors.raw_features"},
                "outputs": {"vars.n_features": "n_features"},
            },
            {
                "name": "learn_scaler_state",
                "op": "fit.standard_scaler",
                "train_only": True,
                "inputs": {"x": "tensors.raw_features"},
                "outputs": {"states.scaler_params": "state"},
            },
            {
                "name": "apply_scaling",
                "op": "transform.standard_scaler",
                "inputs": {
                    "x": "tensors.raw_features",
                    "state": "states.scaler_params",
                },
                "outputs": {"tensors.features": "output"},
            },
            {
                "name": "hash_country",
                "op": "transform.hash_bucket",
                "inputs": {"x": "country", "num_buckets": 50},
                "outputs": {"tensors.country_id": "output"},
            },
        ],
    }

    # Training run (executes inspect.* and fit.*)
    ctx_train, X_train, y_train = dp.run_feature_pipeline(
        table_name="t", features_spec=features_spec, training=True
    )

    assert isinstance(X_train, torch.Tensor)
    assert X_train is not None and X_train.shape == (len(rows), 3)
    assert isinstance(y_train, torch.Tensor) and y_train.shape == (len(rows),)
    assert ctx_train["vars"]["n_features"] == 3
    assert set(ctx_train["states"]["scaler_params"].keys()) == {"mean", "std"}
    country_id = ctx_train["tensors"]["country_id"]
    assert isinstance(country_id, torch.Tensor)
    assert country_id.dtype == torch.long and country_id.shape == (len(rows),)

    # Prediction run (uses saved states/vars; skips inspect/fit)
    init_ctx = {
        "vars": {"n_features": ctx_train["vars"]["n_features"]},
        "states": {"scaler_params": ctx_train["states"]["scaler_params"]},
    }
    ctx_pred, X_pred, y_pred = dp.run_feature_pipeline(
        table_name="t",
        features_spec=features_spec,
        training=False,
        initial_context=init_ctx,
    )

    assert y_pred is not None and torch.allclose(y_train, y_pred)
    assert X_pred is not None and torch.allclose(X_train, X_pred)
    assert torch.equal(ctx_pred["tensors"]["country_id"], country_id)


def test_inspect_fit_train_only_and_prediction_requires_state():
    # Simple numeric rows
    rows = [
        {"f1": 1.0, "f2": 2.0},
        {"f1": 3.0, "f2": 4.0},
    ]
    ml_data_service = _FakeMLDataService(rows=rows)
    dp = DataProcessor(ml_data_service=ml_data_service)

    spec = {
        "feature_columns": ["f1", "f2"],
        "processors": [
            {
                "name": "assemble",
                "op": "transform.assemble_vector",
                "inputs": {"columns": "feature_columns"},
                "outputs": {"tensors.raw": "output"},
            },
            {
                "name": "inspect_features",
                "op": "inspect.feature_stats",
                "train_only": True,
                "inputs": {"tensor": "tensors.raw"},
                "outputs": {"vars.n": "n_features"},
            },
            {
                "name": "fit_scaler",
                "op": "fit.standard_scaler",
                "train_only": True,
                "inputs": {"x": "tensors.raw"},
                "outputs": {"states.scaler": "state"},
            },
            {
                "name": "scale",
                "op": "transform.standard_scaler",
                "inputs": {"x": "tensors.raw", "state": "states.scaler"},
                "outputs": {"tensors.features": "output"},
            },
        ],
    }

    # Training executes inspect/fit and produces vars/states
    ctx_train, X_train, _ = dp.run_feature_pipeline("t", spec, training=True)
    assert ctx_train["vars"]["n"] == 2
    assert set(ctx_train["states"]["scaler"].keys()) == {"mean", "std"}
    assert isinstance(X_train, torch.Tensor)

    # Prediction without providing state should fail (transform depends on it)
    with pytest.raises(ProcessorError):
        dp.run_feature_pipeline("t", spec, training=False)

    # Provide saved state/vars -> should succeed and match training output
    init_ctx = {
        "vars": {"n": ctx_train["vars"]["n"]},
        "states": {"scaler": ctx_train["states"]["scaler"]},
    }
    ctx_pred, X_pred, _ = dp.run_feature_pipeline(
        "t", spec, training=False, initial_context=init_ctx
    )
    assert torch.allclose(X_pred, X_train)
    # inspect/fit not executed in prediction; provided values remain unchanged
    assert ctx_pred["vars"]["n"] == 2
    assert set(ctx_pred["states"]["scaler"].keys()) == {"mean", "std"}


def test_sequence_shift_and_hash_and_pad_ops():
    ml_data_service = _FakeMLDataService(rows=[{"dummy": 1}])
    dp = DataProcessor(ml_data_service=ml_data_service)

    # sequence_shift on 2D tensor
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    out_left = dp._op_transform_sequence_shift(
        {"x": x, "direction": "left", "fill_value": 0}
    )["output"]
    out_right = dp._op_transform_sequence_shift(
        {"x": x, "direction": "right", "fill_value": 9}
    )["output"]
    assert torch.equal(out_left, torch.tensor([[2, 3, 0], [5, 6, 0]], dtype=torch.long))
    assert torch.equal(
        out_right, torch.tensor([[9, 1, 2], [9, 4, 5]], dtype=torch.long)
    )

    # hash_and_pad on list of lists of strings
    seqs = [["a", "b", "c"], ["x"], []]
    r1 = dp._op_transform_hash_and_pad(
        {"x": seqs, "num_buckets": 10, "max_length": 4, "fill_value": 0}
    )["output"]
    r2 = dp._op_transform_hash_and_pad(
        {"x": seqs, "num_buckets": 10, "max_length": 4, "fill_value": 0}
    )["output"]
    assert r1.shape == (3, 4)
    assert r1.dtype == torch.long
    # Deterministic hashing
    assert torch.equal(r1, r2)
    # Padded positions are equal to fill_value
    assert r1[1, 1:].tolist() == [0, 0, 0]
    assert r1[2].tolist() == [0, 0, 0, 0]
