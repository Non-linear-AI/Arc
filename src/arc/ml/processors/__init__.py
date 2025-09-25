"""Data processors for Arc Graph ML pipeline.

This module provides extensible data processing capabilities through
a plugin-based architecture. Processors can learn parameters from
the entire database and apply transformations consistently.
"""

from arc.ml.processors.base import ProcessorError, StatefulProcessor
from arc.ml.processors.builtin import (
    CategoricalEncodingProcessor,
    MinMaxNormalizationProcessor,
    RobustNormalizationProcessor,
    StandardNormalizationProcessor,
)

__all__ = [
    "StatefulProcessor",
    "ProcessorError",
    "StandardNormalizationProcessor",
    "MinMaxNormalizationProcessor",
    "RobustNormalizationProcessor",
    "CategoricalEncodingProcessor",
]
