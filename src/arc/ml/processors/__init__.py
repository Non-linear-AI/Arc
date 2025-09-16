"""Data processors for Arc Graph ML pipeline.

This module provides extensible data processing capabilities through
a plugin-based architecture. Processors can learn parameters from
the entire database and apply transformations consistently.
"""

from .base import ProcessorError, StatefulProcessor
from .builtin import (
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
