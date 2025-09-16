import pytest
import torch

from src.arc.ml.data import ArcDataset, DataProcessor
from src.arc.ml.utils import create_sample_data


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
        """Test loading from table without database."""
        processor = DataProcessor()

        with pytest.raises(RuntimeError, match="Database connection required"):
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
