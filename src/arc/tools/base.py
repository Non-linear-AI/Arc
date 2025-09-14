"""Base classes for tool implementations."""

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

    @classmethod
    def success_result(cls, output: str = "Success") -> "ToolResult":
        """Create a successful result."""
        return cls(success=True, output=output)

    @classmethod
    def error_result(cls, error: str, recovery_actions: str | None = None) -> "ToolResult":
        """Create an error result."""
        return cls(success=False, error=error, recovery_actions=recovery_actions)


class BaseTool(ABC):
    """Base class for all tools."""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments."""
        pass
