"""Base classes for tool implementations."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Result of tool execution."""

    success: bool
    output: str | None = None
    error: str | None = None
    recovery_actions: str | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def success_result(
        cls, output: str = "Success", metadata: dict[str, Any] | None = None
    ) -> "ToolResult":
        """Create a successful result."""
        return cls(success=True, output=output, metadata=metadata)

    @classmethod
    def error_result(
        cls, error: str, recovery_actions: str | None = None
    ) -> "ToolResult":
        """Create an error result."""
        return cls(success=False, error=error, recovery_actions=recovery_actions)


class BaseTool(ABC):
    """Base class for all tools.

    Tools can optionally use built-in error handling by implementing _execute_impl
    instead of execute. This provides automatic:
    - Logging of execution start/end
    - Error handling with recovery suggestions
    - Execution time tracking
    """

    def __init__(self):
        """Initialize base tool with logger."""
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments.

        Subclasses should implement this method directly, OR implement
        _execute_impl() to get automatic error handling and logging.
        """
        pass
