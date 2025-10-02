"""Interactive ML Plan confirmation workflow for Arc ML tools.

This module provides components for interactive ML plan presentation and confirmation,
including markdown rendering and user options for approval, feedback, or questions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.markdown import Markdown
from rich.panel import Panel

if TYPE_CHECKING:
    from arc.core.ml_plan import MLPlan
    from arc.ui.console import InteractiveInterface


class MLPlanConfirmationWorkflow:
    """Manages interactive confirmation workflow for ML plans."""

    def __init__(self, ui_interface: "InteractiveInterface | None" = None):
        """Initialize ML plan confirmation workflow.

        Args:
            ui_interface: Optional UI interface for interactive prompts
        """
        self.ui = ui_interface

    async def run_workflow(
        self,
        plan: "MLPlan",
        is_revision: bool = False,
    ) -> str:
        """Run interactive confirmation workflow for an ML plan.

        Args:
            plan: The ML plan to confirm
            is_revision: Whether this is a plan revision

        Returns:
            User's response to the plan
        """
        if not self.ui:
            # Headless mode - auto-approve
            return "Looks good, please proceed."

        # Display the plan with markdown rendering
        plan_markdown = plan.format_for_display()
        self._show_markdown(plan_markdown)

        try:
            # Ask one simple open question
            user_response = await self.ui.get_user_input_async(
                "What do you think of this plan?"
            )
            return user_response

        finally:
            # Reset prompt state
            self.ui._printer.add_separator("space")
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
        self.ui._printer.console.print(panel)
        self.ui._printer.console.print()  # Add blank line
