"""Tests for ML Data Service high-level API."""

import pandas as pd
import pytest
import torch

from arc.database.manager import DatabaseManager
from arc.database.services import DatasetInfo, MLDataService


@pytest.fixture
def db_manager():
    """Create in-memory database manager for testing."""
    return DatabaseManager(system_db_path=":memory:", user_db_path=":memory:")


@pytest.fixture
def ml_data_service(db_manager):
    """Create ML data service for testing."""
    return MLDataService(db_manager)


@pytest.fixture
def sample_dataset(ml_data_service):
    """Create sample dataset with test data."""
    # Create test table
    create_sql = """
    CREATE TABLE test_dataset (
        id INTEGER PRIMARY KEY,
        feature1 DOUBLE,
        feature2 DOUBLE,
        category VARCHAR(10),
        target INTEGER
    )
    """
    ml_data_service.db_manager.user_execute(create_sql)

    # Insert test data
    insert_sql = """
    INSERT INTO test_dataset VALUES
        (1, 1.0, 2.0, 'A', 0),
        (2, 2.0, 4.0, 'B', 1),
        (3, 3.0, 6.0, 'A', 0),
        (4, 4.0, 8.0, 'C', 1),
        (5, 5.0, 10.0, 'B', 1)
    """
    ml_data_service.db_manager.user_execute(insert_sql)

    return "test_dataset"


class TestMLDataService:
    """Test cases for ML Data Service high-level API."""

    def test_list_datasets(self, ml_data_service, sample_dataset):
        """Test listing available datasets."""
        datasets = ml_data_service.list_datasets()
        assert sample_dataset in datasets

    def test_dataset_exists(self, ml_data_service, sample_dataset):
        """Test checking if dataset exists."""
        assert ml_data_service.dataset_exists(sample_dataset)
        assert not ml_data_service.dataset_exists("nonexistent_dataset")

    def test_get_dataset_info(self, ml_data_service, sample_dataset):
        """Test getting dataset information."""
        info = ml_data_service.get_dataset_info(sample_dataset)

        assert isinstance(info, DatasetInfo)
        assert info.name == sample_dataset
        assert info.row_count == 5
        assert len(info.columns) == 5

        # Check column names
        assert "id" in info.column_names
        assert "feature1" in info.column_names
        assert "feature2" in info.column_names
        assert "category" in info.column_names
        assert "target" in info.column_names

        # Check numeric vs categorical columns
        assert "feature1" in info.numeric_columns
        assert "feature2" in info.numeric_columns
        assert "category" in info.categorical_columns

    def test_get_dataset_info_nonexistent(self, ml_data_service):
        """Test getting info for nonexistent dataset returns None."""
        info = ml_data_service.get_dataset_info("nonexistent")
        assert info is None

    def test_get_features_and_targets(self, ml_data_service, sample_dataset):
        """Test extracting features and targets as DataFrames."""
        features_df, targets_df = ml_data_service.get_features_and_targets(
            sample_dataset,
            feature_columns=["feature1", "feature2"],
            target_columns=["target"],
        )

        assert isinstance(features_df, pd.DataFrame)
        assert isinstance(targets_df, pd.DataFrame)

        assert len(features_df) == 5
        assert len(targets_df) == 5
        assert list(features_df.columns) == ["feature1", "feature2"]
        assert list(targets_df.columns) == ["target"]

        # Check data values
        assert features_df.iloc[0]["feature1"] == 1.0
        assert targets_df.iloc[0]["target"] == 0

    def test_get_features_and_targets_no_targets(self, ml_data_service, sample_dataset):
        """Test extracting only features without targets."""
        features_df, targets_df = ml_data_service.get_features_and_targets(
            sample_dataset, feature_columns=["feature1", "feature2"]
        )

        assert isinstance(features_df, pd.DataFrame)
        assert targets_df is None
        assert len(features_df) == 5

    def test_get_features_and_targets_with_limit(self, ml_data_service, sample_dataset):
        """Test extracting features with row limit."""
        features_df, targets_df = ml_data_service.get_features_and_targets(
            sample_dataset,
            feature_columns=["feature1"],
            target_columns=["target"],
            limit=3,
        )

        assert len(features_df) == 3
        assert len(targets_df) == 3

    def test_get_features_and_targets_invalid_dataset(self, ml_data_service):
        """Test that invalid dataset raises ValueError."""
        with pytest.raises(ValueError, match="Dataset 'nonexistent' does not exist"):
            ml_data_service.get_features_and_targets(
                "nonexistent", feature_columns=["col1"]
            )

    def test_get_features_and_targets_invalid_columns(
        self, ml_data_service, sample_dataset
    ):
        """Test that invalid columns raise ValueError."""
        with pytest.raises(ValueError, match="Feature column 'invalid_col' not found"):
            ml_data_service.get_features_and_targets(
                sample_dataset, feature_columns=["invalid_col"]
            )

    def test_get_features_as_tensors(self, ml_data_service, sample_dataset):
        """Test extracting features and targets as PyTorch tensors."""
        features_tensor, targets_tensor = ml_data_service.get_features_as_tensors(
            sample_dataset,
            feature_columns=["feature1", "feature2"],
            target_columns=["target"],
        )

        assert isinstance(features_tensor, torch.Tensor)
        assert isinstance(targets_tensor, torch.Tensor)

        assert features_tensor.shape == (5, 2)  # 5 samples, 2 features
        assert targets_tensor.shape == (5,)  # 5 targets, squeezed from (5, 1)

        # Check data types
        assert features_tensor.dtype == torch.float32
        assert targets_tensor.dtype == torch.float32

        # Check values
        assert features_tensor[0, 0].item() == 1.0  # First feature of first sample
        assert targets_tensor[0].item() == 0.0  # First target

    def test_get_features_as_tensors_multiple_targets(
        self, ml_data_service, sample_dataset
    ):
        """Test tensors with multiple target columns (not squeezed)."""
        features_tensor, targets_tensor = ml_data_service.get_features_as_tensors(
            sample_dataset,
            feature_columns=["feature1"],
            target_columns=["target", "id"],  # Multiple targets
        )

        assert features_tensor.shape == (5, 1)
        assert targets_tensor.shape == (5, 2)  # Not squeezed

    def test_get_column_statistics(self, ml_data_service, sample_dataset):
        """Test getting column statistics."""
        stats = ml_data_service.get_column_statistics(sample_dataset, "feature1")

        assert stats is not None
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0
        assert stats["count"] == 5
        assert stats["total_rows"] == 5

    def test_get_column_statistics_nonexistent_dataset(self, ml_data_service):
        """Test column stats for nonexistent dataset returns None."""
        stats = ml_data_service.get_column_statistics("nonexistent", "col")
        assert stats is None

    def test_get_unique_values(self, ml_data_service, sample_dataset):
        """Test getting unique values from categorical column."""
        unique_values = ml_data_service.get_unique_values(sample_dataset, "category")

        assert len(unique_values) == 3
        assert set(unique_values) == {"A", "B", "C"}

    def test_get_unique_values_with_limit(self, ml_data_service, sample_dataset):
        """Test getting unique values with limit."""
        unique_values = ml_data_service.get_unique_values(
            sample_dataset, "category", limit=2
        )

        assert len(unique_values) <= 2

    def test_sample_data(self, ml_data_service, sample_dataset):
        """Test sampling data from dataset."""
        sample_df = ml_data_service.sample_data(sample_dataset, limit=3)

        assert isinstance(sample_df, pd.DataFrame)
        assert len(sample_df) == 3
        assert "feature1" in sample_df.columns
        assert "category" in sample_df.columns

    def test_sample_data_specific_columns(self, ml_data_service, sample_dataset):
        """Test sampling specific columns."""
        sample_df = ml_data_service.sample_data(
            sample_dataset, columns=["feature1", "target"], limit=2
        )

        assert len(sample_df) == 2
        assert list(sample_df.columns) == ["feature1", "target"]

    def test_sample_data_nonexistent_dataset(self, ml_data_service):
        """Test sampling nonexistent dataset returns empty DataFrame."""
        sample_df = ml_data_service.sample_data("nonexistent")
        assert isinstance(sample_df, pd.DataFrame)
        assert len(sample_df) == 0

    def test_validate_columns(self, ml_data_service, sample_dataset):
        """Test column validation."""
        validation = ml_data_service.validate_columns(
            sample_dataset, ["feature1", "invalid_col", "target"]
        )

        assert validation["feature1"] is True
        assert validation["invalid_col"] is False
        assert validation["target"] is True

    def test_validate_columns_nonexistent_dataset(self, ml_data_service):
        """Test column validation for nonexistent dataset."""
        validation = ml_data_service.validate_columns("nonexistent", ["col1", "col2"])

        assert validation["col1"] is False
        assert validation["col2"] is False

    def test_dataset_info_properties(self, ml_data_service, sample_dataset):
        """Test DatasetInfo properties."""
        info = ml_data_service.get_dataset_info(sample_dataset)

        # Test column_names property
        assert "id" in info.column_names
        assert "category" in info.column_names

        # Test numeric_columns property
        numeric_cols = info.numeric_columns
        assert "feature1" in numeric_cols
        assert "feature2" in numeric_cols
        assert "id" in numeric_cols
        assert "target" in numeric_cols
        assert "category" not in numeric_cols

        # Test categorical_columns property
        categorical_cols = info.categorical_columns
        assert "category" in categorical_cols
        assert "feature1" not in categorical_cols

    def test_table_name_validation(self, ml_data_service):
        """Test table name validation."""
        # Valid names
        assert ml_data_service._is_valid_table_name("valid_table")
        assert ml_data_service._is_valid_table_name("table123")

        # Invalid names
        assert not ml_data_service._is_valid_table_name("")
        assert not ml_data_service._is_valid_table_name("123invalid")
        assert not ml_data_service._is_valid_table_name("table;drop")
        assert not ml_data_service._is_valid_table_name("system.table")

    def test_column_name_validation(self, ml_data_service):
        """Test column name validation."""
        # Valid names
        assert ml_data_service._is_valid_column_name("valid_column")
        assert ml_data_service._is_valid_column_name("col123")

        # Invalid names
        assert not ml_data_service._is_valid_column_name("")
        assert not ml_data_service._is_valid_column_name("123invalid")
        assert not ml_data_service._is_valid_column_name("col;drop")

    # Phase 1: Categorical Column Validation Tests

    def test_get_features_as_tensors_rejects_categorical_columns(
        self, ml_data_service, sample_dataset
    ):
        """Test that categorical columns are rejected when converting to tensors."""
        with pytest.raises(
            ValueError,
            match=r"Cannot convert categorical column 'category'.*VARCHAR",
        ):
            ml_data_service.get_features_as_tensors(
                sample_dataset,
                feature_columns=["feature1", "category"],  # category is VARCHAR
                target_columns=["target"],
            )

    def test_get_features_as_tensors_rejects_multiple_categorical_columns(
        self, ml_data_service
    ):
        """Test that multiple categorical columns are all reported in error."""
        # Create dataset with multiple categorical columns
        create_sql = """
        CREATE TABLE multi_cat_dataset (
            id INTEGER,
            num_feature DOUBLE,
            cat1 VARCHAR(10),
            cat2 TEXT,
            target INTEGER
        )
        """
        ml_data_service.db_manager.user_execute(create_sql)

        ml_data_service.db_manager.user_execute(
            "INSERT INTO multi_cat_dataset VALUES (1, 1.0, 'A', 'X', 0)"
        )

        # Should raise error mentioning first categorical column
        with pytest.raises(ValueError, match="Cannot convert categorical column"):
            ml_data_service.get_features_as_tensors(
                "multi_cat_dataset",
                feature_columns=["num_feature", "cat1", "cat2"],
                target_columns=["target"],
            )

    def test_get_features_as_tensors_allows_categorical_targets(self, ml_data_service):
        """Test that categorical columns are allowed as targets (will fail at conversion)."""
        # Create dataset with categorical target
        create_sql = """
        CREATE TABLE cat_target_dataset (
            id INTEGER,
            feature DOUBLE,
            cat_target VARCHAR(10)
        )
        """
        ml_data_service.db_manager.user_execute(create_sql)

        ml_data_service.db_manager.user_execute(
            "INSERT INTO cat_target_dataset VALUES (1, 1.0, 'Class_A')"
        )

        # Target validation should fail at tensor conversion (not at our validation)
        with pytest.raises(ValueError, match="Failed to convert data to tensors"):
            ml_data_service.get_features_as_tensors(
                "cat_target_dataset",
                feature_columns=["feature"],
                target_columns=["cat_target"],  # categorical target
            )

    def test_get_features_as_tensors_accepts_all_numeric_features(
        self, ml_data_service, sample_dataset
    ):
        """Test that all-numeric features work correctly."""
        features_tensor, targets_tensor = ml_data_service.get_features_as_tensors(
            sample_dataset,
            feature_columns=["feature1", "feature2", "id"],  # all numeric
            target_columns=["target"],
        )

        assert isinstance(features_tensor, torch.Tensor)
        assert features_tensor.shape == (5, 3)  # 5 samples, 3 features
        assert features_tensor.dtype == torch.float32

    def test_get_features_as_tensors_error_message_quality(
        self, ml_data_service, sample_dataset
    ):
        """Test that error message provides helpful guidance."""
        with pytest.raises(ValueError) as exc_info:
            ml_data_service.get_features_as_tensors(
                sample_dataset,
                feature_columns=["category"],  # VARCHAR column
                target_columns=["target"],
            )

        error_msg = str(exc_info.value)
        # Check error message contains helpful information
        assert "category" in error_msg  # mentions column name
        assert "VARCHAR" in error_msg or "categorical" in error_msg  # mentions type
        assert (
            "encoding" in error_msg.lower() or "embed" in error_msg.lower()
        )  # suggests solution
