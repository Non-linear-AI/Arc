"""Confirmation service for user operations."""


from typing import Any, Optional

from rich.console import Console

_fallback_console = Console()


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

        # Select output target
        ui = self._ui
        use_ui = ui is not None and getattr(ui, "_printer", None) is not None

        # Add spacing before confirmation and show header
        if use_ui:
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
                            p.print(
                                f"      [dim]… +{len(lines) - 3} more lines[/dim]"
                            )
                else:
                    p.print("  [dim]⎿ Requesting user confirmation[/dim]")
            # Separate options from header
            ui._printer.add_separator("space")
            with ui._printer.section(color="blue", add_dot=False) as p:
                p.print("Do you want to proceed with this operation?")
                p.print("")
                p.print("  1. Yes")
                p.print("  2. Yes, and don't ask again this session")
                p.print("  3. No")
                p.print("")
                p.print("[dim]Enter option number (1-3):[/dim]")
        else:
            # Fallback console rendering
            _fallback_console.print()
            _fallback_console.print(
                f"[magenta]⏺[/magenta] [white]{operation}({target})[/white]"
            )
            if content:
                _fallback_console.print("  [dim]⎿ Requesting user confirmation[/dim]")
                lines = content.split("\n")
                _fallback_console.print(f"  [dim]⎿ {lines[0]}[/dim]")
                if len(lines) > 1:
                    for line in lines[1:3]:  # Show up to 2 more lines
                        if line.strip():
                            _fallback_console.print(f"      [dim]{line}[/dim]")
                    if len(lines) > 3:
                        _fallback_console.print(
                            f"      [dim]… +{len(lines) - 3} more lines[/dim]"
                        )
            else:
                _fallback_console.print("  [dim]⎿ Requesting user confirmation[/dim]")
            _fallback_console.print()
            _fallback_console.print("Do you want to proceed with this operation?")
            _fallback_console.print()
            _fallback_console.print(
                "  1. Yes\n  2. Yes, and don't ask again this session\n  3. No"
            )
            _fallback_console.print()
            _fallback_console.print("[dim]Enter option number (1-3):[/dim]")
        

        # Get user choice (prefer UI's prompt_toolkit path if available)
        while True:
            try:
                if use_ui:
                    user_inp = await ui._printer.get_input_async("(1): ")
                else:
                    user_inp = _fallback_console.input("(1): ")

                choice = (user_inp or "1").strip()
                choice_num = int(choice)

                if choice_num == 1:
                    return ConfirmationResult(confirmed=True)
                elif choice_num == 2:
                    # Set appropriate session flag
                    if "bash" in operation.lower():
                        self.session_flags["bash_commands"] = True
                    elif (
                        "file" in operation.lower()
                        or "create" in operation.lower()
                        or "edit" in operation.lower()
                    ):
                        self.session_flags["file_operations"] = True
                    return ConfirmationResult(confirmed=True, dont_ask_again=True)
                elif choice_num == 3:
                    return ConfirmationResult(
                        confirmed=False, feedback="Operation cancelled by user"
                    )
                else:
                    if use_ui:
                        ui._printer.print(
                            "[red]Please enter a number between 1 and 3[/red]"
                        )
                    else:
                        _fallback_console.print(
                            "[red]Please enter a number between 1 and 3[/red]"
                        )

            except (ValueError, KeyboardInterrupt, EOFError):
                if use_ui:
                    ui._printer.print("[red]Please enter a valid number (1-3)[/red]")
                else:
                    _fallback_console.print(
                        "[red]Please enter a valid number (1-3)[/red]"
                    )

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
