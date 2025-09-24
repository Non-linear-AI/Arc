"""Confirmation service for user operations.

Always uses the UI's Printer (prompt_toolkit) for interactive confirmations.
If no UI is present (e.g., headless), auto-approves silently.
"""

from typing import Any, Optional
from prompt_toolkit.shortcuts.choice_input import ChoiceInput


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
        target: str,
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
        return await self._show_confirmation_dialog(operation, target, content)

    async def _show_confirmation_dialog(
        self, operation: str, target: str, content: str
    ) -> ConfirmationResult:
        """Show the confirmation dialog."""

        # UI is required for interactive confirmations; if missing, auto-approve
        ui = self._ui
        if ui is None or getattr(ui, "_printer", None) is None:
            return ConfirmationResult(confirmed=True)

        # Add spacing before confirmation and show header
        with ui._printer.section(color="magenta") as p:
            p.print(f"{operation}({target})")
            if content:
                p.print("  [dim]⎿ Requesting user confirmation[/dim]")
                lines = content.split("\n")
                p.print(f"  [dim]⎿ {lines[0]}[/dim]")
                if len(lines) > 1:
                    for line in lines[1:3]:  # Show up to 2 more lines
                        if line.strip():
                            p.print(f"      [dim]{line}[/dim]")
                    if len(lines) > 3:
                        p.print(f"      [dim]… +{len(lines) - 3} more lines[/dim]")
            else:
                p.print("  [dim]⎿ Requesting user confirmation[/dim]")
        # Separate options from header and present a choice prompt
        ui._printer.add_separator("space")
        with ui._printer.section(color="blue", add_dot=False) as p:
            p.print("Do you want to proceed with this operation?")
        

        # Get user choice (prefer UI's prompt_toolkit path if available)
        # Use prompt_toolkit's choice selector for responsive selection.
        # Mark input as active to pause ESC watcher.
        options = [
            ("yes", "Yes"),
            ("yes_session", "Yes, and don't ask again this session"),
            ("no", "No"),
        ]
        try:
            # Best-effort: signal that input is active to pause ESC watcher.
            ui._printer._input_active = True  # noqa: SLF001 (internal, controlled)
            selection = await ChoiceInput(
                message="Use arrows/enter to select:",
                options=options,
                default="yes",
                show_frame=False,
            ).prompt_async()
        finally:
            ui._printer._input_active = False  # noqa: SLF001

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
        return ConfirmationResult(confirmed=False, feedback="Operation cancelled by user")

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
