"""Confirmation service for user operations.

Always uses the UI's Printer (prompt_toolkit) for interactive confirmations.
If no UI is present (e.g., headless), auto-approves silently.
"""

from typing import Any, Optional


class ConfirmationResult:
    """Result of a confirmation request."""

    def __init__(
        self, confirmed: bool, dont_ask_again: bool = False, feedback: str = ""
    ):
        self.confirmed = confirmed
        self.dont_ask_again = dont_ask_again
        self.feedback = feedback


class ConfirmationService:
    """Service for requesting user confirmations."""

    _instance: Optional["ConfirmationService"] = None

    def __init__(self):
        self.session_flags = {
            "file_operations": False,
            "bash_commands": False,
            "all_operations": False,
        }
        self._ui: Any | None = None  # Injected UI for prompting (InteractiveInterface)

    @classmethod
    def get_instance(cls) -> "ConfirmationService":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def request_confirmation(
        self,
        operation: str,
        operation_type: str = "file",
        content: str = "",
    ) -> ConfirmationResult:
        """Request user confirmation for an operation."""

        # Check session flags
        if (
            self.session_flags["all_operations"]
            or (operation_type == "file" and self.session_flags["file_operations"])
            or (operation_type == "bash" and self.session_flags["bash_commands"])
        ):
            return ConfirmationResult(confirmed=True)

        # Show confirmation dialog
        return await self._show_confirmation_dialog(operation, content)

    async def _show_confirmation_dialog(
        self, operation: str, content: str
    ) -> ConfirmationResult:
        """Show the confirmation dialog."""

        # UI is required for interactive confirmations; if missing, auto-approve
        ui = self._ui
        if ui is None or getattr(ui, "_printer", None) is None:
            return ConfirmationResult(confirmed=True)

        # Start a section for the entire tool call (confirmation + result)
        # Manually manage section lifecycle instead of using context manager
        # because we need to keep the section open across two separate methods:
        # 1. confirmation.py starts section and shows confirmation prompt
        # 2. console.py continues section and shows tool result
        # This creates a single unified section for the entire tool call

        # Determine shape and title based on operation type
        if "bash" in operation.lower():
            shape = "▶"
            title = "Run"
        elif "create" in operation.lower():
            shape = "■"
            title = "Create"
        elif "edit" in operation.lower():
            shape = "■"
            title = "Edit"
        else:
            # Fallback: extract first word from operation
            shape = "▸"
            title = operation.split()[0] if operation else "Action"

        # Start section but don't close it (manual lifecycle management)
        section = ui._printer.section(shape=shape)
        p = section.__enter__()

        try:
            # Store section and printer so console.py can continue it
            ui._printer._active_confirmation_section = section
            ui._printer._active_confirmation_printer = p

            # Show title
            p.print(title)

            # Show confirmation details
            if content:
                lines = content.split("\n")
                for line in lines[:3]:  # Show up to 3 lines
                    if line.strip():
                        p.print(f"[dim]{line}[/dim]")
                if len(lines) > 3:
                    p.print(f"[dim]… +{len(lines) - 3} more lines[/dim]")

            # Choices match the main prompt behavior; Esc cancels globally
            options = [
                ("yes", "Yes"),
                ("yes_session", "Yes, and don't ask again this session"),
                ("no", "No"),
            ]

            try:
                # Escape watcher suspension happens automatically in get_choice_async()
                selection = await ui._printer.get_choice_async(options, default="yes")
            finally:
                # Always reset state regardless of outcome
                ui._printer.add_separator("space")
                # Reset prompt session to ensure consistent state after nested prompt
                ui._printer.reset_prompt_session()

            # Handle ESC as cancellation (same as "no")
            if selection == "__esc__":
                return ConfirmationResult(confirmed=False, feedback="Cancelled by user")

            if selection == "yes":
                return ConfirmationResult(confirmed=True)
            if selection == "yes_session":
                if "bash" in operation.lower():
                    self.session_flags["bash_commands"] = True
                elif (
                    "file" in operation.lower()
                    or "create" in operation.lower()
                    or "edit" in operation.lower()
                ):
                    self.session_flags["file_operations"] = True
                return ConfirmationResult(confirmed=True, dont_ask_again=True)
            # selection == "no"
            return ConfirmationResult(
                confirmed=False, feedback="User denied permission for this operation"
            )
        except Exception:
            # On any error, ensure section is closed properly
            section.__exit__(None, None, None)
            raise

    def set_ui(self, ui: Any) -> None:
        """Inject the UI object to enable prompt_toolkit-backed input."""
        self._ui = ui

    def reset_session(self):
        """Reset all session flags."""
        self.session_flags = {
            "file_operations": False,
            "bash_commands": False,
            "all_operations": False,
        }

    def get_session_flags(self) -> dict[str, bool]:
        """Get current session flags."""
        return self.session_flags.copy()

    def set_session_flag(self, flag: str, value: bool) -> None:
        """Set a session flag for auto-approval."""
        self.session_flags[flag] = value

    def should_auto_approve(self, operation_type: str) -> bool:
        """Check if operation should be auto-approved."""
        return self.session_flags.get(
            "all_operations", False
        ) or self.session_flags.get(operation_type, False)
