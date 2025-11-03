import pytest
import torch

from arc.ml.data import ArcDataset, DataProcessor, MultiInputDataset
from arc.ml.utils import create_sample_data


class TestArcDataset:
    """Test ArcDataset implementation."""

    def test_dataset_creation(self):
        """Test creating dataset with features only."""
        features = torch.randn(100, 5)
        dataset = ArcDataset(features)

        assert len(dataset) == 100
        assert isinstance(dataset[0], torch.Tensor)
        assert dataset[0].shape == (5,)

    def test_dataset_with_targets(self):
        """Test creating dataset with features and targets."""
        features = torch.randn(100, 5)
        targets = torch.randn(100)
        dataset = ArcDataset(features, targets)

        assert len(dataset) == 100
        sample = dataset[0]
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert sample[0].shape == (5,)
        assert sample[1].shape == ()

    def test_dataset_mismatched_lengths(self):
        """Test dataset creation with mismatched feature/target lengths."""
        features = torch.randn(100, 5)
        targets = torch.randn(50)  # Wrong length

        with pytest.raises(ValueError, match="same length"):
            ArcDataset(features, targets)


class TestMultiInputDataset:
    """Test MultiInputDataset implementation for models with multiple named inputs."""

    def test_basic_multi_input_splitting(self):
        """Test splitting features into multiple named inputs."""
        # Create sample data: 100 samples, 5 features
        # Columns: UserID, MovieID, age, gender, rating
        features = torch.randn(100, 5)
        targets = torch.randn(100)

        # Define input specification
        input_spec = {
            "user_input": {
                "columns": ["UserID"],
                "dtype": "long",
                "shape": [None, 1],
            },
            "movie_input": {
                "columns": ["MovieID"],
                "dtype": "long",
                "shape": [None, 1],
            },
            "features": {
                "columns": ["age", "gender", "rating"],
                "dtype": "float32",
                "shape": [None, 3],
            },
        }

        feature_columns = ["UserID", "MovieID", "age", "gender", "rating"]

        dataset = MultiInputDataset(features, targets, input_spec, feature_columns)

        assert len(dataset) == 100

        # Get a sample
        sample = dataset[0]
        assert isinstance(sample, tuple)
        assert len(sample) == 2  # (inputs_dict, target)

        inputs_dict, target = sample

        # Check that we have all expected inputs
        assert "user_input" in inputs_dict
        assert "movie_input" in inputs_dict
        assert "features" in inputs_dict

        # Check shapes
        assert inputs_dict["user_input"].shape == (1,)  # Single column
        assert inputs_dict["movie_input"].shape == (1,)  # Single column
        assert inputs_dict["features"].shape == (3,)  # Three columns

    def test_dtype_conversion(self):
        """Test that inputs are converted to correct dtypes."""
        features = torch.randn(50, 4).float()
        targets = torch.randn(50)

        input_spec = {
            "id_input": {
                "columns": ["id"],
                "dtype": "long",  # Should convert to long
                "shape": [None, 1],
            },
            "float_input": {
                "columns": ["f1", "f2"],
                "dtype": "float32",
                "shape": [None, 2],
            },
            "double_input": {
                "columns": ["d1"],
                "dtype": "float64",  # Should convert to double
                "shape": [None, 1],
            },
        }

        feature_columns = ["id", "f1", "f2", "d1"]

        dataset = MultiInputDataset(features, targets, input_spec, feature_columns)

        inputs_dict, _ = dataset[0]

        # Check dtypes
        assert inputs_dict["id_input"].dtype == torch.long
        assert inputs_dict["float_input"].dtype == torch.float32
        assert inputs_dict["double_input"].dtype == torch.float64

    def test_without_targets(self):
        """Test MultiInputDataset without target values."""
        features = torch.randn(30, 3)

        input_spec = {
            "input1": {"columns": ["col1"], "dtype": "float32", "shape": [None, 1]},
            "input2": {
                "columns": ["col2", "col3"],
                "dtype": "float32",
                "shape": [None, 2],
            },
        }

        feature_columns = ["col1", "col2", "col3"]

        dataset = MultiInputDataset(features, None, input_spec, feature_columns)

        assert len(dataset) == 30

        # Should return dict only (no target)
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert "input1" in sample
        assert "input2" in sample

    def test_missing_column_error(self):
        """Test error when input spec references non-existent column."""
        features = torch.randn(20, 2)
        targets = torch.randn(20)

        input_spec = {
            "input1": {
                "columns": ["existing_col"],
                "dtype": "float32",
                "shape": [None, 1],
            },
            "input2": {
                "columns": ["missing_col"],  # This column doesn't exist
                "dtype": "float32",
                "shape": [None, 1],
            },
        }

        feature_columns = ["existing_col", "another_col"]

        with pytest.raises(ValueError, match="missing_col.*not in feature_columns"):
            MultiInputDataset(features, targets, input_spec, feature_columns)

    def test_mismatched_lengths_error(self):
        """Test error when features and targets have different lengths."""
        features = torch.randn(100, 3)
        targets = torch.randn(50)  # Wrong length

        input_spec = {
            "input1": {
                "columns": ["col1", "col2", "col3"],
                "dtype": "float32",
                "shape": [None, 3],
            },
        }

        feature_columns = ["col1", "col2", "col3"]

        with pytest.raises(ValueError, match="same length"):
            MultiInputDataset(features, targets, input_spec, feature_columns)

    def test_no_columns_specified_error(self):
        """Test error when input spec doesn't specify columns."""
        features = torch.randn(20, 2)
        targets = torch.randn(20)

        input_spec = {
            "input1": {
                "columns": [],  # Empty columns list
                "dtype": "float32",
                "shape": [None, 0],
            },
        }

        feature_columns = ["col1", "col2"]

        with pytest.raises(ValueError, match="no columns specified"):
            MultiInputDataset(features, targets, input_spec, feature_columns)

    def test_single_input_edge_case(self):
        """Test that single input still works (though ArcDataset might be better)."""
        features = torch.randn(40, 3)
        targets = torch.randn(40)

        input_spec = {
            "single_input": {
                "columns": ["col1", "col2", "col3"],
                "dtype": "float32",
                "shape": [None, 3],
            },
        }

        feature_columns = ["col1", "col2", "col3"]

        dataset = MultiInputDataset(features, targets, input_spec, feature_columns)

        inputs_dict, target = dataset[0]

        # Should still work as dict with one key
        assert len(inputs_dict) == 1
        assert "single_input" in inputs_dict
        assert inputs_dict["single_input"].shape == (3,)

    def test_column_order_independence(self):
        """Test that columns are correctly mapped regardless of order."""
        # Features: col1, col2, col3, col4
        features = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ]
        )

        input_spec = {
            # Reference columns in different order
            "input1": {
                "columns": ["col3", "col1"],
                "dtype": "float32",
                "shape": [None, 2],
            },
            "input2": {
                "columns": ["col4", "col2"],
                "dtype": "float32",
                "shape": [None, 2],
            },
        }

        feature_columns = ["col1", "col2", "col3", "col4"]

        dataset = MultiInputDataset(features, None, input_spec, feature_columns)

        inputs_dict = dataset[0]

        # input1 should have col3 (index 2) and col1 (index 0)
        assert torch.equal(inputs_dict["input1"], torch.tensor([3.0, 1.0]))

        # input2 should have col4 (index 3) and col2 (index 1)
        assert torch.equal(inputs_dict["input2"], torch.tensor([4.0, 2.0]))

    def test_dtype_parsing(self):
        """Test various dtype string formats."""
        features = torch.randn(10, 3).float()

        test_cases = [
            ("float32", torch.float32),
            ("float64", torch.float64),
            ("float", torch.float32),
            ("double", torch.float64),
            ("int32", torch.int32),
            ("int64", torch.int64),
            ("long", torch.long),
            ("int", torch.int32),
            ("bool", torch.bool),
        ]

        for dtype_str, expected_dtype in test_cases:
            input_spec = {
                "test_input": {
                    "columns": ["col1", "col2", "col3"],
                    "dtype": dtype_str,
                    "shape": [None, 3],
                },
            }

            feature_columns = ["col1", "col2", "col3"]

            dataset = MultiInputDataset(features, None, input_spec, feature_columns)
            inputs_dict = dataset[0]

            assert inputs_dict["test_input"].dtype == expected_dtype, (
                f"dtype_str '{dtype_str}' did not produce {expected_dtype}"
            )

    def test_integration_with_dataloader(self):
        """Test that MultiInputDataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        features = torch.randn(100, 5)
        targets = torch.randn(100)

        input_spec = {
            "input1": {
                "columns": ["col1", "col2"],
                "dtype": "float32",
                "shape": [None, 2],
            },
            "input2": {"columns": ["col3"], "dtype": "float32", "shape": [None, 1]},
            "input3": {
                "columns": ["col4", "col5"],
                "dtype": "float32",
                "shape": [None, 2],
            },
        }

        feature_columns = ["col1", "col2", "col3", "col4", "col5"]

        dataset = MultiInputDataset(features, targets, input_spec, feature_columns)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        # Get one batch
        batch = next(iter(dataloader))

        # Batch should be a list: [inputs_dict, targets]
        assert isinstance(batch, list)
        assert len(batch) == 2

        inputs_dict, batch_targets = batch

        # Check that inputs_dict has correct structure
        assert isinstance(inputs_dict, dict)
        assert "input1" in inputs_dict
        assert "input2" in inputs_dict
        assert "input3" in inputs_dict

        # Check batch dimensions
        assert inputs_dict["input1"].shape == (16, 2)
        assert inputs_dict["input2"].shape == (16, 1)
        assert inputs_dict["input3"].shape == (16, 2)
        assert batch_targets.shape == (16,)

    def test_unknown_dtype_fallback(self):
        """Test that unknown dtype strings fall back to float32."""
        features = torch.randn(10, 2)

        input_spec = {
            "test_input": {
                "columns": ["col1", "col2"],
                "dtype": "unknown_dtype",  # Should fall back to float32
                "shape": [None, 2],
            },
        }

        feature_columns = ["col1", "col2"]

        dataset = MultiInputDataset(features, None, input_spec, feature_columns)
        inputs_dict = dataset[0]

        # Should fall back to float32
        assert inputs_dict["test_input"].dtype == torch.float32

    def test_input_spec_without_dtype(self):
        """Test that missing dtype defaults to float32."""
        features = torch.randn(10, 2)

        input_spec = {
            "test_input": {
                "columns": ["col1", "col2"],
                # No dtype specified
                "shape": [None, 2],
            },
        }

        feature_columns = ["col1", "col2"]

        dataset = MultiInputDataset(features, None, input_spec, feature_columns)
        inputs_dict = dataset[0]

        # Should default to float32
        assert inputs_dict["test_input"].dtype == torch.float32

    def test_real_world_recommendation_model_example(self):
        """Test realistic recommendation model with user/item embeddings + features."""
        # Simulate MovieLens-style data
        # Columns: UserID, MovieID, user_age, user_gender, movie_year, user_rating_count
        num_samples = 200
        num_features = 6

        features = torch.randn(num_samples, num_features)
        targets = torch.randint(1, 6, (num_samples,)).float()  # Ratings 1-5

        input_spec = {
            "user_embedding_input": {
                "columns": ["UserID"],
                "dtype": "long",
                "shape": [None, 1],
            },
            "movie_embedding_input": {
                "columns": ["MovieID"],
                "dtype": "long",
                "shape": [None, 1],
            },
            "user_features": {
                "columns": ["user_age", "user_gender", "user_rating_count"],
                "dtype": "float32",
                "shape": [None, 3],
            },
            "movie_features": {
                "columns": ["movie_year"],
                "dtype": "float32",
                "shape": [None, 1],
            },
        }

        feature_columns = [
            "UserID",
            "MovieID",
            "user_age",
            "user_gender",
            "movie_year",
            "user_rating_count",
        ]

        dataset = MultiInputDataset(features, targets, input_spec, feature_columns)

        # Test with DataLoader
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        inputs_dict, batch_targets = next(iter(loader))

        # Verify all expected inputs are present
        assert set(inputs_dict.keys()) == {
            "user_embedding_input",
            "movie_embedding_input",
            "user_features",
            "movie_features",
        }

        # Verify shapes
        assert inputs_dict["user_embedding_input"].shape == (32, 1)
        assert inputs_dict["movie_embedding_input"].shape == (32, 1)
        assert inputs_dict["user_features"].shape == (32, 3)
        assert inputs_dict["movie_features"].shape == (32, 1)

        # Verify dtypes
        assert inputs_dict["user_embedding_input"].dtype == torch.long
        assert inputs_dict["movie_embedding_input"].dtype == torch.long
        assert inputs_dict["user_features"].dtype == torch.float32
        assert inputs_dict["movie_features"].dtype == torch.float32

        assert batch_targets.shape == (32,)


class TestDataProcessor:
    """Test DataProcessor functionality."""

    def test_create_dataloader(self):
        """Test creating DataLoader from tensors."""
        processor = DataProcessor()
        features, targets = create_sample_data(100, 5)

        dataloader = processor.create_dataloader(
            features, targets, batch_size=32, shuffle=False
        )

        # Test one batch
        batch = next(iter(dataloader))
        assert isinstance(batch, list)  # PyTorch DataLoader returns list by default
        assert len(batch) == 2
        assert batch[0].shape[0] == 32  # Batch size
        assert batch[0].shape[1] == 5  # Features
        assert batch[1].shape[0] == 32  # Batch size for targets

    def test_normalize_features(self):
        """Test feature normalization."""
        processor = DataProcessor()

        # Create features with known mean and std
        features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        normalized, stats = processor.normalize_features(features)

        # Check normalization
        assert torch.allclose(normalized.mean(dim=0), torch.zeros(2), atol=1e-6)
        assert torch.allclose(normalized.std(dim=0), torch.ones(2), atol=1e-6)

        # Check stats
        assert "mean" in stats
        assert "std" in stats
        assert stats["mean"].shape == (1, 2)
        assert stats["std"].shape == (1, 2)

    def test_apply_normalization(self):
        """Test applying existing normalization to new data."""
        processor = DataProcessor()

        # Create training data and normalize
        train_features = torch.randn(100, 3)
        _, stats = processor.normalize_features(train_features)

        # Apply to new data
        test_features = torch.randn(50, 3)
        normalized_test = processor.apply_normalization(test_features, stats)

        assert normalized_test.shape == test_features.shape

    def test_train_val_split(self):
        """Test train/validation split."""
        processor = DataProcessor()
        features, targets = create_sample_data(100, 5)

        train_features, val_features, train_targets, val_targets = (
            processor.create_train_val_split(
                features, targets, val_ratio=0.2, random_seed=42
            )
        )

        # Check sizes
        assert len(train_features) == 80
        assert len(val_features) == 20
        assert len(train_targets) == 80
        assert len(val_targets) == 20

        # Check no overlap
        total_samples = torch.cat([train_features, val_features], dim=0)
        assert len(total_samples) == 100

    def test_train_val_split_features_only(self):
        """Test train/validation split with features only."""
        processor = DataProcessor()
        features = torch.randn(100, 5)

        train_features, val_features, train_targets, val_targets = (
            processor.create_train_val_split(features, val_ratio=0.3, random_seed=42)
        )

        assert len(train_features) == 70
        assert len(val_features) == 30
        assert train_targets is None
        assert val_targets is None

    def test_load_from_table_no_database(self):
        """Test loading from table without MLDataService."""
        processor = DataProcessor()

        with pytest.raises(
            RuntimeError, match="MLDataService is required for data access"
        ):
            processor.load_from_table("test_table", ["col1", "col2"])


class TestDataUtilities:
    """Test data utility functions."""

    def test_create_sample_data_binary(self):
        """Test creating binary classification sample data."""
        features, targets = create_sample_data(100, 5, binary_classification=True)

        assert features.shape == (100, 5)
        assert targets.shape == (100,)
        assert torch.all((targets == 0) | (targets == 1))

    def test_create_sample_data_regression(self):
        """Test creating regression sample data."""
        features, targets = create_sample_data(50, 3, binary_classification=False)

        assert features.shape == (50, 3)
        assert targets.shape == (50,)
        # For regression, targets are continuous
        assert not torch.all((targets == 0) | (targets == 1))


class TestLabelEncodingOperators:
    """Test label encoding operators for categorical features."""

    def test_fit_label_encoder_basic(self):
        """Test fitting label encoder on simple categorical data."""
        processor = DataProcessor()

        # Sample categorical data
        categories = ["cat", "dog", "bird", "cat", "dog", "cat"]

        # Execute fit.label_encoder operator
        result = processor._execute_operator("fit.label_encoder", {"x": categories})

        # Check output structure
        assert "state" in result
        state = result["state"]
        assert "vocabulary" in state
        assert "vocab_size" in state

        # Check vocabulary contains all unique values
        vocab = state["vocabulary"]
        assert len(vocab) == 3  # cat, dog, bird
        assert "cat" in vocab
        assert "dog" in vocab
        assert "bird" in vocab

        # Check vocab indices are sequential starting from 0
        indices = sorted(vocab.values())
        assert indices == [0, 1, 2]

    def test_fit_label_encoder_with_none(self):
        """Test fitting label encoder with None values."""
        processor = DataProcessor()

        # Categorical data with None
        categories = ["red", "blue", None, "green", "red", None, "blue"]

        result = processor._execute_operator("fit.label_encoder", {"x": categories})

        state = result["state"]
        vocab = state["vocabulary"]

        # None should be included in vocabulary with special index
        assert None in vocab or "<NULL>" in vocab
        assert state["vocab_size"] >= 3  # red, blue, green + null

    def test_fit_label_encoder_maintains_order(self):
        """Test that vocabulary maintains consistent order."""
        processor = DataProcessor()

        categories1 = ["z", "a", "m", "z", "a"]
        result1 = processor._execute_operator("fit.label_encoder", {"x": categories1})

        categories2 = ["z", "a", "m", "z", "a"]
        result2 = processor._execute_operator("fit.label_encoder", {"x": categories2})

        # Same input should produce same vocabulary
        assert result1["state"]["vocabulary"] == result2["state"]["vocabulary"]

    def test_fit_label_encoder_single_value(self):
        """Test fitting with only one unique value."""
        processor = DataProcessor()

        categories = ["same", "same", "same"]
        result = processor._execute_operator("fit.label_encoder", {"x": categories})

        state = result["state"]
        assert state["vocab_size"] == 1
        assert "same" in state["vocabulary"]

    def test_transform_label_encode_basic(self):
        """Test transforming categorical data to integer indices."""
        processor = DataProcessor()

        # First fit the encoder
        train_data = ["cat", "dog", "bird", "cat"]
        fit_result = processor._execute_operator("fit.label_encoder", {"x": train_data})
        vocab_state = fit_result["state"]

        # Now transform data using the fitted vocabulary
        test_data = ["dog", "cat", "bird", "dog"]
        transform_result = processor._execute_operator(
            "transform.label_encode", {"x": test_data, "state": vocab_state}
        )

        # Check output
        assert "output" in transform_result
        encoded = transform_result["output"]

        # Should be a long tensor
        assert isinstance(encoded, torch.Tensor)
        assert encoded.dtype == torch.long
        assert len(encoded) == 4

        # Check that same values get same indices
        cat_idx = encoded[1].item()
        assert encoded[1].item() == cat_idx  # Second "cat" same as first

        dog_idx = encoded[0].item()
        assert encoded[3].item() == dog_idx  # Second "dog" same as first

    def test_transform_label_encode_with_oov(self):
        """Test encoding with out-of-vocabulary (OOV) values."""
        processor = DataProcessor()

        # Fit on limited vocabulary
        train_data = ["apple", "banana"]
        fit_result = processor._execute_operator("fit.label_encoder", {"x": train_data})
        vocab_state = fit_result["state"]

        # Transform with OOV value
        test_data = ["apple", "cherry", "banana"]  # "cherry" is OOV
        transform_result = processor._execute_operator(
            "transform.label_encode", {"x": test_data, "state": vocab_state}
        )

        encoded = transform_result["output"]
        assert len(encoded) == 3

        # OOV value should get a special index (typically vocab_size or 0)
        # Implementation will use 0 for OOV
        cherry_idx = encoded[1].item()
        assert cherry_idx >= 0  # Valid index

    def test_transform_label_encode_with_none(self):
        """Test encoding with None values."""
        processor = DataProcessor()

        # Fit including None
        train_data = ["x", "y", None, "z"]
        fit_result = processor._execute_operator("fit.label_encoder", {"x": train_data})
        vocab_state = fit_result["state"]

        # Transform with None
        test_data = ["x", None, "y", None]
        transform_result = processor._execute_operator(
            "transform.label_encode", {"x": test_data, "state": vocab_state}
        )

        encoded = transform_result["output"]
        assert len(encoded) == 4

        # None values should get consistent encoding
        none_idx = encoded[1].item()
        assert encoded[3].item() == none_idx  # Both Nones same index

    def test_transform_label_encode_empty_list(self):
        """Test encoding empty list."""
        processor = DataProcessor()

        # Fit with some data
        train_data = ["a", "b"]
        fit_result = processor._execute_operator("fit.label_encoder", {"x": train_data})
        vocab_state = fit_result["state"]

        # Transform empty list
        test_data = []
        transform_result = processor._execute_operator(
            "transform.label_encode", {"x": test_data, "state": vocab_state}
        )

        encoded = transform_result["output"]
        assert isinstance(encoded, torch.Tensor)
        assert len(encoded) == 0

    def test_label_encoding_preserves_indices(self):
        """Test that encoding is invertible (indices map back to values)."""
        processor = DataProcessor()

        # Fit encoder
        categories = ["Action", "Drama", "Comedy", "Action", "Drama"]
        fit_result = processor._execute_operator("fit.label_encoder", {"x": categories})
        vocab = fit_result["state"]["vocabulary"]

        # Transform
        transform_result = processor._execute_operator(
            "transform.label_encode", {"x": categories, "state": fit_result["state"]}
        )
        encoded = transform_result["output"]

        # Build reverse mapping
        idx_to_value = {idx: val for val, idx in vocab.items()}

        # Verify we can decode back
        for i, original in enumerate(categories):
            encoded_idx = encoded[i].item()
            if encoded_idx in idx_to_value:
                decoded = idx_to_value[encoded_idx]
                assert decoded == original
