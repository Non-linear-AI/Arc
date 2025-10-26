"""Base classes for tool implementations."""

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
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

    @contextmanager
    def _section_printer(
        self,
        ui_interface,
        title: str,
        color: str = "magenta",
        metadata: list[str] | None = None,
    ):
        """Context manager for consistent section printing across tools.

        This provides automatic section management with proper visual hierarchy
        (indentation via add_dot=True) and spacing.

        Args:
            ui_interface: UI interface instance (or None for non-UI contexts)
            title: Section title text
            color: Section color (default: magenta)
            metadata: Optional list of metadata strings to append to title

        Yields:
            Section printer if UI available, None otherwise

        Example:
            with self._section_printer(self.ui, "ML Model", metadata=[plan_id]) as p:
                if p:
                    p.print("Processing...")
                # ... do work ...
                if p:
                    p.print("✓ Complete")
                # Section cleanup happens automatically
        """
        if ui_interface:
            # Use add_dot=True for proper visual hierarchy (indentation)
            with ui_interface._printer.section(color=color, add_dot=True) as printer:
                # Build title with metadata
                full_title = title
                if metadata:
                    full_title += f" [dim]({' • '.join(metadata)})[/dim]"
                printer.print(full_title)

                yield printer
        else:
            # No UI - yield None, no-op
            yield None

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments.

        Subclasses should implement this method directly, OR implement
        _execute_impl() to get automatic error handling and logging.
        """
        pass
