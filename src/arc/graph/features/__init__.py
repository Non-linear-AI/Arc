"""Features specification and components for Arc-Graph."""

from .components import CORE_PROCESSORS, get_processor_class
from .spec import FeatureSpec, ProcessorConfig
from .validator import FeaturesValidationError, validate_features_dict

__all__ = [
    "ProcessorConfig",
    "FeatureSpec",
    "get_processor_class",
    "CORE_PROCESSORS",
    "validate_features_dict",
    "FeaturesValidationError",
]
