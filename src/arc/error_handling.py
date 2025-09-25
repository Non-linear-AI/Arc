"""Error handling and recovery system for Arc CLI."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any

from arc.tools.base import ToolResult


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better handling."""

    NETWORK = "network"
    FILE_SYSTEM = "file_system"
    PERMISSION = "permission"
    VALIDATION = "validation"
    API = "api"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for errors."""

    function_name: str
    arguments: dict[str, Any]
    timestamp: float
    user_action: str = ""
    file_path: str | None = None
    line_number: int | None = None


@dataclass
class RecoveryAction:
    """Represents a recovery action for an error."""

    name: str
    description: str
    action: Callable
    auto_execute: bool = False
    confidence: float = 1.0  # 0.0 to 1.0


class ArcError(Exception):
    """Base exception class for Arc CLI with enhanced context."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: ErrorContext | None = None,
        original_error: Exception | None = None,
        recovery_actions: list[RecoveryAction] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context
        self.original_error = original_error
        self.recovery_actions = recovery_actions or []

        # Auto-classify error if not provided
        if category == ErrorCategory.UNKNOWN and original_error:
            self.category = self._classify_error(original_error)

    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Automatically classify error based on type and message."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        if (
            "network" in error_str
            or "connection" in error_str
            or "timeout" in error_str
        ):
            return ErrorCategory.NETWORK
        elif (
            "permission" in error_str or "access" in error_str or "denied" in error_str
        ):
            return ErrorCategory.PERMISSION
        elif "file" in error_str or "directory" in error_str or "path" in error_str:
            return ErrorCategory.FILE_SYSTEM
        elif (
            "api" in error_str
            or "http" in error_str
            or error_type in ["httperror", "apierror"]
        ):
            return ErrorCategory.API
        elif "timeout" in error_str or error_type == "timeouterror":
            return ErrorCategory.TIMEOUT
        elif (
            "memory" in error_str
            or "resource" in error_str
            or error_type == "memoryerror"
        ):
            return ErrorCategory.RESOURCE
        elif (
            "validation" in error_str
            or "invalid" in error_str
            or error_type in ["valueerror", "typeerror"]
        ):
            return ErrorCategory.VALIDATION
        else:
            return ErrorCategory.UNKNOWN

    def add_recovery_action(self, action: RecoveryAction):
        """Add a recovery action to this error."""
        self.recovery_actions.append(action)

    def get_user_message(self) -> str:
        """Get user-friendly error message."""
        base_message = self.message

        if self.category == ErrorCategory.PERMISSION:
            return (
                f"Permission denied: {base_message}\n💡 Try running with "
                "appropriate permissions or check file ownership."
            )
        elif self.category == ErrorCategory.NETWORK:
            return (
                f"Network error: {base_message}\n"
                "💡 Check your internet connection and try again."
            )
        elif self.category == ErrorCategory.FILE_SYSTEM:
            return (
                f"File system error: {base_message}\n💡 Check if the "
                "file/directory exists and you have the right permissions."
            )
        elif self.category == ErrorCategory.API:
            return (
                f"API error: {base_message}\n"
                "💡 Check your API credentials and network connection."
            )
        elif self.category == ErrorCategory.TIMEOUT:
            return (
                f"Operation timed out: {base_message}\n💡 The operation took too "
                "long. Try again or check system resources."
            )
        elif self.category == ErrorCategory.VALIDATION:
            return (
                f"Validation error: {base_message}\n"
                "💡 Please check your input parameters."
            )
        else:
            return f"Error: {base_message}"


class ErrorHandler:
    """Enhanced error handling with recovery strategies."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_history: list[ArcError] = []
        self.recovery_strategies: dict[ErrorCategory, list[Callable]] = {
            ErrorCategory.NETWORK: [self._retry_with_backoff, self._check_connectivity],
            ErrorCategory.FILE_SYSTEM: [
                self._create_missing_directories,
                self._check_permissions,
            ],
            ErrorCategory.PERMISSION: [self._suggest_permission_fix],
            ErrorCategory.API: [self._retry_with_backoff, self._check_api_key],
            ErrorCategory.TIMEOUT: [self._increase_timeout_and_retry],
            ErrorCategory.VALIDATION: [self._suggest_input_correction],
        }

    def handle_error(
        self,
        error: Exception,
        context: ErrorContext | None = None,
        auto_recover: bool = True,
    ) -> ToolResult:
        """Handle an error with appropriate recovery strategies."""
        # Convert to ArcError if needed
        if isinstance(error, ArcError):
            arc_error = error
        else:
            arc_error = ArcError(
                message=str(error), context=context, original_error=error
            )

        # Log the error
        self._log_error(arc_error)

        # Add to history
        self.error_history.append(arc_error)

        # Generate recovery actions
        self._generate_recovery_actions(arc_error)

        # Attempt auto-recovery if enabled
        if auto_recover:
            recovery_result = self._attempt_recovery(arc_error)
            if recovery_result:
                return recovery_result

        # Return error result with recovery suggestions
        return ToolResult.error_result(
            arc_error.get_user_message(),
            recovery_actions=self._format_recovery_actions(arc_error.recovery_actions),
        )

    def _log_error(self, error: ArcError):
        """Log error with appropriate level."""
        log_message = f"[{error.category.value}] {error.message}"

        if error.context:
            log_message += f" (in {error.context.function_name})"

        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, exc_info=error.original_error)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, exc_info=error.original_error)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

    def _generate_recovery_actions(self, error: ArcError):
        """Generate appropriate recovery actions for the error."""
        if error.category in self.recovery_strategies:
            for strategy in self.recovery_strategies[error.category]:
                try:
                    action = strategy(error)
                    if action:
                        error.add_recovery_action(action)
                except Exception as e:
                    self.logger.debug(f"Error generating recovery action: {e}")

    def _attempt_recovery(self, error: ArcError) -> ToolResult | None:
        """Attempt automatic recovery using available actions."""
        for action in error.recovery_actions:
            if action.auto_execute and action.confidence > 0.7:
                try:
                    result = action.action()
                    if result and isinstance(result, ToolResult) and result.success:
                        self.logger.info(
                            f"Auto-recovery successful using: {action.name}"
                        )
                        return result
                except Exception as e:
                    self.logger.debug(f"Recovery action {action.name} failed: {e}")

        return None

    def _format_recovery_actions(self, actions: list[RecoveryAction]) -> str:
        """Format recovery actions for user display."""
        if not actions:
            return ""

        formatted = "\n\n🔧 Suggested fixes:\n"
        for i, action in enumerate(actions, 1):
            confidence_indicator = (
                "🟢"
                if action.confidence > 0.8
                else "🟡"
                if action.confidence > 0.5
                else "🟠"
            )
            formatted += f"  {i}. {confidence_indicator} {action.description}\n"

        return formatted

    # Recovery strategy implementations
    def _retry_with_backoff(self, _error: ArcError) -> RecoveryAction | None:
        """Generate retry action with exponential backoff."""

        async def retry_action():
            # Implementation would retry the original operation
            # This is a placeholder for the actual retry logic
            await asyncio.sleep(1)
            return ToolResult.success_result("Retry completed")

        return RecoveryAction(
            name="retry_with_backoff",
            description="Retry the operation with exponential backoff",
            action=lambda: asyncio.create_task(retry_action()),
            auto_execute=True,
            confidence=0.6,
        )

    def _create_missing_directories(self, error: ArcError) -> RecoveryAction | None:
        """Generate action to create missing directories."""
        if error.context and error.context.file_path:
            from pathlib import Path

            def create_dirs():
                try:
                    Path(error.context.file_path).parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    return ToolResult.success_result("Created missing directories")
                except Exception as e:
                    return ToolResult.error_result(f"Failed to create directories: {e}")

            return RecoveryAction(
                name="create_directories",
                description="Create missing parent directories",
                action=create_dirs,
                auto_execute=True,
                confidence=0.8,
            )

        return None

    def _check_permissions(self, _error: ArcError) -> RecoveryAction | None:
        """Generate action to check and suggest permission fixes."""

        def check_perms():
            # Placeholder for permission checking logic
            return ToolResult.success_result("Permission check completed")

        return RecoveryAction(
            name="check_permissions",
            description="Check file/directory permissions",
            action=check_perms,
            auto_execute=False,
            confidence=0.7,
        )

    def _suggest_permission_fix(self, _error: ArcError) -> RecoveryAction | None:
        """Suggest permission fix commands."""

        def suggest_fix():
            suggestion = "Try running: chmod +r <file> or sudo <command>"
            return ToolResult.success_result(f"Permission fix suggestion: {suggestion}")

        return RecoveryAction(
            name="suggest_permission_fix",
            description="Get permission fix suggestions",
            action=suggest_fix,
            auto_execute=False,
            confidence=0.5,
        )

    def _check_connectivity(self, _error: ArcError) -> RecoveryAction | None:
        """Check network connectivity."""

        def check_network():
            # Placeholder for network connectivity check
            return ToolResult.success_result("Network connectivity check completed")

        return RecoveryAction(
            name="check_connectivity",
            description="Check network connectivity",
            action=check_network,
            auto_execute=True,
            confidence=0.6,
        )

    def _check_api_key(self, _error: ArcError) -> RecoveryAction | None:
        """Check API key validity."""

        def check_key():
            # Placeholder for API key validation
            return ToolResult.success_result("API key validation completed")

        return RecoveryAction(
            name="check_api_key",
            description="Validate API key configuration",
            action=check_key,
            auto_execute=False,
            confidence=0.8,
        )

    def _increase_timeout_and_retry(self, _error: ArcError) -> RecoveryAction | None:
        """Increase timeout and retry operation."""

        async def timeout_retry():
            await asyncio.sleep(2)  # Longer wait
            return ToolResult.success_result("Timeout retry completed")

        return RecoveryAction(
            name="increase_timeout_retry",
            description="Retry with increased timeout",
            action=lambda: asyncio.create_task(timeout_retry()),
            auto_execute=True,
            confidence=0.7,
        )

    def _suggest_input_correction(self, error: ArcError) -> RecoveryAction | None:
        """Suggest input corrections."""

        def suggest_correction():
            # Analyze error message for common validation issues
            suggestions = []
            error_msg = error.message.lower()

            if "invalid" in error_msg:
                suggestions.append("Check input format and try again")
            if "required" in error_msg:
                suggestions.append("Make sure all required parameters are provided")
            if "type" in error_msg:
                suggestions.append("Check data types of input parameters")

            return ToolResult.success_result(
                f"Input suggestions: {'; '.join(suggestions)}"
            )

        return RecoveryAction(
            name="suggest_input_correction",
            description="Get input correction suggestions",
            action=suggest_correction,
            auto_execute=False,
            confidence=0.6,
        )

    def get_error_stats(self) -> dict[str, Any]:
        """Get error statistics."""
        if not self.error_history:
            return {"total_errors": 0}

        stats = {
            "total_errors": len(self.error_history),
            "by_category": {},
            "by_severity": {},
            "recent_errors": len(list(self.error_history[-10:])),  # Last 10
        }

        for error in self.error_history:
            category = error.category.value
            severity = error.severity.value

            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
            stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1

        return stats


# Decorator for enhanced error handling
def with_error_handling(
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    auto_recover: bool = True,
):
    """Decorator for enhanced error handling."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            error_handler = ErrorHandler()

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    function_name=func.__name__,
                    arguments={"args": args, "kwargs": kwargs},
                    timestamp=asyncio.get_event_loop().time(),
                )

                arc_error = ArcError(
                    message=str(e),
                    category=category,
                    severity=severity,
                    context=context,
                    original_error=e,
                )

                return error_handler.handle_error(arc_error, auto_recover=auto_recover)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            error_handler = ErrorHandler()

            try:
                return func(*args, **kwargs)
            except Exception as e:
                import time

                context = ErrorContext(
                    function_name=func.__name__,
                    arguments={"args": args, "kwargs": kwargs},
                    timestamp=time.time(),
                )

                arc_error = ArcError(
                    message=str(e),
                    category=category,
                    severity=severity,
                    context=context,
                    original_error=e,
                )

                return error_handler.handle_error(arc_error, auto_recover=auto_recover)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Global error handler instance
error_handler = ErrorHandler()
