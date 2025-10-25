"""Machine learning tool implementations.

This module provides a backward-compatibility facade for the ML tools package.
All tool classes have been moved to the arc.tools.ml package for better organization.

Import from this module will continue to work, but new code should import directly
from arc.tools.ml:
    from arc.tools.ml import MLPlanTool, MLModelTool, MLTrainTool, etc.
"""

# Re-export all ML tool classes from the new package structure
# Re-export helper functions for backward compatibility
# (These are internal helpers but may be used by existing code)
from arc.tools.ml._helpers import (
    _as_optional_float,
    _as_optional_int,
    _as_string_list,
)
from arc.tools.ml.evaluate import MLEvaluateTool
from arc.tools.ml.evaluator_generator import MLEvaluatorGeneratorTool
from arc.tools.ml.model import MLModelTool
from arc.tools.ml.plan import MLPlanTool
from arc.tools.ml.train import MLTrainTool

__all__ = [
    # Main tool classes
    "MLPlanTool",
    "MLModelTool",
    "MLTrainTool",
    "MLEvaluateTool",
    "MLEvaluatorGeneratorTool",
    # Helper functions (private but re-exported for compatibility)
    "_as_optional_int",
    "_as_optional_float",
    "_as_string_list",
]
