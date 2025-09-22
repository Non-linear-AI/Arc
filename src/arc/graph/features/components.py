"""Core feature processing components for Arc-Graph."""

from __future__ import annotations

from typing import Any, Protocol


class FeatureProcessor(Protocol):
    """Protocol for feature processors."""

    def fit(self, data: Any) -> None:
        """Fit the processor to data."""
        ...

    def transform(self, data: Any) -> Any:
        """Transform data using fitted processor."""
        ...

    def fit_transform(self, data: Any) -> Any:
        """Fit and transform data in one step."""
        ...


# Placeholder processor classes - these would be implemented based on actual
# requirements
class StandardNormalizationProcessor:
    """Standard normalization (z-score) processor."""

    def __init__(self, **params):
        self.params = params
        self.mean_ = None
        self.std_ = None

    def fit(self, data: Any) -> None:
        """Fit normalization parameters."""
        pass  # Implementation would compute mean and std

    def transform(self, data: Any) -> Any:
        """Apply standard normalization."""
        pass  # Implementation would apply (x - mean) / std

    def fit_transform(self, data: Any) -> Any:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)


class MinMaxNormalizationProcessor:
    """Min-max normalization processor."""

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0), **params):
        self.feature_range = feature_range
        self.params = params
        self.min_ = None
        self.max_ = None

    def fit(self, data: Any) -> None:
        """Fit normalization parameters."""
        pass  # Implementation would compute min and max

    def transform(self, data: Any) -> Any:
        """Apply min-max normalization."""
        pass  # Implementation would apply scaling

    def fit_transform(self, data: Any) -> Any:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)


class RobustNormalizationProcessor:
    """Robust normalization using median and IQR."""

    def __init__(self, **params):
        self.params = params
        self.median_ = None
        self.iqr_ = None

    def fit(self, data: Any) -> None:
        """Fit normalization parameters."""
        pass  # Implementation would compute median and IQR

    def transform(self, data: Any) -> Any:
        """Apply robust normalization."""
        pass  # Implementation would apply (x - median) / IQR

    def fit_transform(self, data: Any) -> Any:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)


class OneHotEncodingProcessor:
    """One-hot encoding processor for categorical variables."""

    def __init__(self, handle_unknown: str = "ignore", **params):
        self.handle_unknown = handle_unknown
        self.params = params
        self.categories_ = None

    def fit(self, data: Any) -> None:
        """Fit encoding parameters."""
        pass  # Implementation would discover categories

    def transform(self, data: Any) -> Any:
        """Apply one-hot encoding."""
        pass  # Implementation would create binary columns

    def fit_transform(self, data: Any) -> Any:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)


class LabelEncodingProcessor:
    """Label encoding processor for categorical variables."""

    def __init__(self, **params):
        self.params = params
        self.classes_ = None
        self.mapping_ = None

    def fit(self, data: Any) -> None:
        """Fit encoding parameters."""
        pass  # Implementation would create label mapping

    def transform(self, data: Any) -> Any:
        """Apply label encoding."""
        pass  # Implementation would map categories to integers

    def fit_transform(self, data: Any) -> Any:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)


class TargetEncodingProcessor:
    """Target encoding processor for categorical variables."""

    def __init__(self, smoothing: float = 1.0, **params):
        self.smoothing = smoothing
        self.params = params
        self.target_mean_ = None
        self.category_means_ = None

    def fit(self, data: Any, target: Any = None) -> None:
        """Fit encoding parameters."""
        pass  # Implementation would compute target statistics

    def transform(self, data: Any) -> Any:
        """Apply target encoding."""
        pass  # Implementation would replace categories with target means

    def fit_transform(self, data: Any, target: Any = None) -> Any:
        """Fit and transform in one step."""
        self.fit(data, target)
        return self.transform(data)


# Core processor registry
CORE_PROCESSORS = {
    # Normalization processors
    "StandardNormalization": StandardNormalizationProcessor,
    "MinMaxNormalization": MinMaxNormalizationProcessor,
    "RobustNormalization": RobustNormalizationProcessor,
    # Encoding processors
    "OneHotEncoding": OneHotEncodingProcessor,
    "LabelEncoding": LabelEncodingProcessor,
    "TargetEncoding": TargetEncodingProcessor,
    # Aliases for common operations
    "ZScoreNormalization": StandardNormalizationProcessor,
    "Standardization": StandardNormalizationProcessor,
    "Normalization": MinMaxNormalizationProcessor,
}


def get_processor_class(processor_type: str) -> type[FeatureProcessor]:
    """Get feature processor class by type name.

    Args:
        processor_type: Processor type name (e.g., "StandardNormalization")

    Returns:
        Feature processor class

    Raises:
        ValueError: If processor type is not supported
    """
    if processor_type not in CORE_PROCESSORS:
        raise ValueError(f"Unsupported processor type: {processor_type}")

    return CORE_PROCESSORS[processor_type]


def validate_processor_params(processor_type: str, params: dict[str, Any]) -> bool:
    """Validate processor parameters for a given processor type.

    Args:
        processor_type: Processor type name
        params: Parameters dictionary

    Returns:
        True if parameters are valid

    Raises:
        ValueError: If parameters are invalid
    """
    if processor_type == "MinMaxNormalization":
        if "feature_range" in params:
            feature_range = params["feature_range"]
            if not isinstance(feature_range, (list, tuple)) or len(feature_range) != 2:
                raise ValueError(
                    f"MinMaxNormalization feature_range must be a tuple of 2 values, "
                    f"got: {feature_range}"
                )
            if feature_range[0] >= feature_range[1]:
                raise ValueError(
                    f"MinMaxNormalization feature_range min must be < max, "
                    f"got: {feature_range}"
                )

    elif processor_type == "OneHotEncoding":
        if "handle_unknown" in params:
            handle_unknown = params["handle_unknown"]
            if handle_unknown not in ("ignore", "error"):
                raise ValueError(
                    f"OneHotEncoding handle_unknown must be 'ignore' or 'error', "
                    f"got: {handle_unknown}"
                )

    elif processor_type == "TargetEncoding" and "smoothing" in params:
        smoothing = params["smoothing"]
        if not isinstance(smoothing, (int, float)) or smoothing < 0:
            raise ValueError(
                f"TargetEncoding smoothing must be non-negative, got: {smoothing}"
            )

    return True


def get_supported_processor_types() -> list[str]:
    """Get list of all supported processor types.

    Returns:
        List of supported processor type names
    """
    return list(CORE_PROCESSORS.keys())
