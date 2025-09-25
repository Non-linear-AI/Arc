"""Features specification and components for Arc-Graph."""

from arc.graph.features.components import CORE_PROCESSORS, get_processor_class
from arc.graph.features.spec import FeatureSpec, ProcessorConfig
from arc.graph.features.validator import FeaturesValidationError, validate_features_dict

__all__ = [
    "ProcessorConfig",
    "FeatureSpec",
    "get_processor_class",
    "CORE_PROCESSORS",
    "validate_features_dict",
    "FeaturesValidationError",
]
