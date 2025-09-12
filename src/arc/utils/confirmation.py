"""User confirmation utilities."""


class ConfirmationService:
    """Service for handling user confirmations."""

    def __init__(self):
        self.session_flags: dict[str, bool] = {}

    def set_session_flag(self, flag: str, value: bool) -> None:
        """Set a session flag for auto-approval."""
        self.session_flags[flag] = value

    def should_auto_approve(self, operation_type: str) -> bool:
        """Check if operation should be auto-approved."""
        return self.session_flags.get("allOperations", False) or self.session_flags.get(
            operation_type, False
        )

    async def request_confirmation(
        self,
        operation: str,  # noqa: ARG002
        details: str,  # noqa: ARG002
        operation_type: str = "general",
    ) -> bool:
        """Request user confirmation for an operation."""
        if self.should_auto_approve(operation_type):
            return True

        # In a real implementation, this would prompt the user
        # For now, auto-approve all operations
        return True
