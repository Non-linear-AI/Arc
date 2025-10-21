"""Interactive ML Plan confirmation workflow for Arc ML tools.

This module provides components for interactive ML plan presentation and confirmation,
including markdown rendering and user options for approval, feedback, or questions.
"""

from __future__ import annotations

import termios
from typing import TYPE_CHECKING

from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel

if TYPE_CHECKING:
    from arc.core.ml_plan import MLPlan
    from arc.ui.console import InteractiveInterface


class MLPlanConfirmationWorkflow:
    """Manages interactive confirmation workflow for ML plans."""

    def __init__(self, ui_interface: InteractiveInterface | None = None):
        """Initialize ML plan confirmation workflow.

        Args:
            ui_interface: Optional UI interface for interactive prompts
        """
        self.ui = ui_interface

    async def run_workflow(
        self,
        plan: MLPlan,
        _is_revision: bool = False,
    ) -> dict:
        """Run interactive confirmation workflow for an ML plan.

        Args:
            plan: The ML plan to confirm
            _is_revision: Whether this is a plan revision (reserved for future use)

        Returns:
            Dict with 'choice' and optional 'feedback':
                - choice: 'accept', 'accept_all', 'feedback', 'cancel'
                - feedback: User's feedback text (only if choice='feedback')
        """
        if not self.ui:
            # Headless mode - auto-approve
            return {"choice": "accept"}

        # Display the plan with markdown rendering
        plan_markdown = plan.format_for_display()
        self._show_markdown(plan_markdown)

        # Define options for up/down selection
        options = [
            ("accept", "Accept and proceed"),
            ("accept_all", "Accept all (don't ask again this session)"),
            ("feedback", "Provide feedback to revise"),
            ("cancel", "Cancel"),
        ]

        try:
            # Get user choice with arrow key selection
            choice = await self.ui._printer.get_choice_async(options, default="accept")

            # Validate choice
            valid_choices = {"accept", "accept_all", "feedback", "cancel"}
            if choice not in valid_choices:
                raise ValueError(
                    f"Invalid workflow choice: '{choice}'. "
                    f"Expected one of: {valid_choices}"
                )

            if choice == "accept":
                return {"choice": "accept"}
            elif choice == "accept_all":
                return {"choice": "accept_all"}
            elif choice == "feedback":
                # Ask for feedback
                self.ui._printer.reset_prompt_session()
                feedback = await self.ui.get_user_input_async(
                    "Please describe your feedback:"
                )
                return {"choice": "feedback", "feedback": feedback}
            elif choice == "cancel":
                return {"choice": "cancel"}

        finally:
            # Reset prompt state
            self.ui._printer.reset_prompt_session()

    def _show_markdown(self, markdown_text: str):
        """Display markdown-formatted text using Rich with panel border.

        Args:
            markdown_text: Markdown-formatted text to display
        """
        if not self.ui:
            return

        # Use Rich's Markdown renderer with left alignment inside a panel
        md = Markdown(markdown_text, justify="left")
        panel = Panel(md, border_style="color(245)", expand=False)
        # Add 2-space left padding to match section indentation
        padded_panel = Padding(panel, (0, 0, 0, 2))
        self.ui._printer.console.print(padded_panel)
        self.ui._printer.console.print()  # Add blank line

        # Synchronize terminal output to prevent race condition with prompt_toolkit
        self.ui._printer.console.file.flush()
        try:
            if hasattr(self.ui._printer.console.file, "fileno"):
                termios.tcdrain(self.ui._printer.console.file.fileno())
        except (OSError, AttributeError):
            # Graceful fallback for non-TTY environments
            pass
