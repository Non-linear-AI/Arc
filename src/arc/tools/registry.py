"""Tool registry for dynamic tool dispatch and management."""

import asyncio
import json
import logging
from typing import Any

from arc.core.client import ArcToolCall
from arc.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing and dispatching tool executions.

    Provides centralized tool management with:
    - Dynamic tool registration
    - Centralized parameter parsing and validation
    - Timeout support
    - Consistent error handling
    """

    def __init__(self, default_timeout: int = 300):
        """Initialize the tool registry.

        Args:
            default_timeout: Default timeout in seconds for tool execution
        """
        self.tools: dict[str, BaseTool] = {}
        self.default_timeout = default_timeout
        self._logger = logger

    def register(self, name: str, tool: BaseTool) -> None:
        """Register a tool with the registry.

        Args:
            name: Tool name (must match tool name in tools.yaml)
            tool: Tool instance implementing BaseTool interface

        Raises:
            ValueError: If tool name already registered
        """
        if name in self.tools:
            raise ValueError(f"Tool '{name}' is already registered")

        if not isinstance(tool, BaseTool):
            raise TypeError(f"Tool must implement BaseTool interface, got {type(tool)}")

        self.tools[name] = tool
        self._logger.debug(f"Registered tool: {name}")

    def unregister(self, name: str) -> None:
        """Unregister a tool from the registry.

        Args:
            name: Tool name to unregister
        """
        if name in self.tools:
            del self.tools[name]
            self._logger.debug(f"Unregistered tool: {name}")

    def is_registered(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: Tool name to check

        Returns:
            True if tool is registered, False otherwise
        """
        return name in self.tools

    def get_registered_tools(self) -> list[str]:
        """Get list of all registered tool names.

        Returns:
            List of registered tool names
        """
        return list(self.tools.keys())

    async def execute(
        self,
        tool_call: ArcToolCall,
        timeout: int | None = None,
    ) -> ToolResult:
        """Execute a tool call with timeout and error handling.

        Args:
            tool_call: Tool call to execute
            timeout: Optional timeout override (uses default_timeout if None)

        Returns:
            ToolResult with execution result or error
        """
        # Check if tool exists
        if tool_call.name not in self.tools:
            available_tools = ", ".join(self.get_registered_tools())
            return ToolResult.error_result(
                f"Unknown tool: {tool_call.name}. Available tools: {available_tools}"
            )

        # Parse and validate arguments
        try:
            args = json.loads(tool_call.arguments)
        except json.JSONDecodeError as e:
            return ToolResult.error_result(
                f"Invalid JSON arguments for tool '{tool_call.name}': {e}"
            )

        if not isinstance(args, dict):
            return ToolResult.error_result(
                f"Tool arguments must be a JSON object, got {type(args).__name__}"
            )

        # Execute tool with timeout
        tool = self.tools[tool_call.name]
        timeout_value = timeout if timeout is not None else self.default_timeout

        try:
            result = await asyncio.wait_for(
                tool.execute(**args),
                timeout=timeout_value,
            )
            return result
        except TimeoutError:
            # Treat timeouts as cancellations - don't retry, ask user instead
            return ToolResult(
                success=False,
                error=f"Tool '{tool_call.name}' execution timed out after {timeout_value}s. "
                      "This operation took longer than expected.",
                metadata={"cancelled": True, "reason": "timeout"},
            )
        except TypeError as e:
            # Handle parameter validation errors
            return ToolResult.error_result(
                f"Invalid parameters for tool '{tool_call.name}': {e}"
            )
        except Exception as e:
            self._logger.exception(f"Tool '{tool_call.name}' execution failed")
            return ToolResult.error_result(
                f"Tool '{tool_call.name}' execution error: {str(e)}"
            )

    def validate_against_yaml_tools(self, yaml_tool_names: list[str]) -> dict[str, Any]:
        """Validate that registered tools match tools defined in YAML.

        Args:
            yaml_tool_names: List of tool names from tools.yaml

        Returns:
            Dictionary with validation results:
            - valid: bool - True if validation passed
            - missing_in_registry: list[str] - Tools in YAML but not registered
            - extra_in_registry: list[str] - Tools registered but not in YAML
            - error: str | None - Error message if validation failed
        """
        yaml_tools = set(yaml_tool_names)
        registered_tools = set(self.get_registered_tools())

        missing = yaml_tools - registered_tools
        extra = registered_tools - yaml_tools

        valid = len(missing) == 0 and len(extra) == 0

        result = {
            "valid": valid,
            "missing_in_registry": sorted(missing),
            "extra_in_registry": sorted(extra),
            "error": None,
        }

        if not valid:
            error_parts = []
            if missing:
                error_parts.append(
                    f"Tools in YAML but not registered: {', '.join(sorted(missing))}"
                )
            if extra:
                error_parts.append(
                    f"Tools registered but not in YAML: {', '.join(sorted(extra))}"
                )
            result["error"] = "; ".join(error_parts)

        return result
