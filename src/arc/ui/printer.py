"""Printer: centralizes console output, streaming, and input handling.

- Provides output sections that automatically insert a trailing blank line.
- Provides streaming helpers for incremental output.
- Delegates actual rendering to rich.console.Console, but keeps formatting
  decisions out of CLI business logic.
"""

from contextlib import contextmanager, suppress

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel


class Printer:
    def __init__(self):
        self.console = Console()
        self._is_streaming = False
        self._section_started = False

    @contextmanager
    def output_section(self, separator_style: str = "blank"):
        """Group an output block and add a separator line after it."""
        try:
            yield self
        finally:
            with suppress(Exception):
                self.add_separator(separator_style)

    @contextmanager
    def section(
        self,
        color: str = "cyan",
        add_dot: bool = True,
        streaming: bool = False,
        prefix: str = "",
        separator_style: str = "blank",
    ):
        """Unified context manager for both regular and streaming output with
        automatic dot prefix and line separation.

        Args:
            color: Color for the dot prefix
            add_dot: Whether to add a dot prefix (can be disabled for welcome messages)
            streaming: Whether this section supports streaming output
            prefix: Optional prefix text (like for assistant responses, only used
                when streaming=True)
            separator_style: Style of separator after section ("blank", "line",
                "dots", "space")
        """
        self._section_started = False

        class UnifiedSectionPrinter:
            def __init__(
                self, printer, color, add_dot, streaming, prefix, separator_style
            ):
                self.printer = printer
                self.color = color
                self.add_dot = add_dot
                self.streaming = streaming
                self.prefix = prefix
                self.separator_style = separator_style
                self._streaming_started = False
                self.live: Live | None = None
                self._current_text: str = ""
                self._finalized: bool = False
                self._cursor_markup: str = "[dim]█[/dim]"

            def print(self, *args, **kwargs):
                # Check if we're printing a Rich object (Table, Panel, etc.)
                is_rich_object = args and hasattr(args[0], "__rich_console__")

                # Add dot prefix on first print call within this section
                # (but not for Rich objects)
                if (
                    not self.printer._section_started
                    and self.add_dot
                    and not is_rich_object
                ):
                    # Extract first line and add dot prefix
                    if args:
                        first_arg = str(args[0])
                        lines = first_arg.split("\n")
                        if lines:
                            # Add dot prefix to first line
                            prefixed_first = (
                                f"[{self.color}]⏺[/{self.color}] {lines[0]}"
                            )
                            if len(lines) > 1:
                                # Reconstruct with remaining lines
                                remaining_lines = "\n".join(lines[1:])
                                new_args = (
                                    prefixed_first + "\n" + remaining_lines,
                                ) + args[1:]
                            else:
                                new_args = (prefixed_first,) + args[1:]
                            self.printer.console.print(*new_args, **kwargs)
                        else:
                            self.printer.console.print(*args, **kwargs)
                    else:
                        self.printer.console.print(*args, **kwargs)
                    self.printer._section_started = True
                else:
                    # Regular print for subsequent calls or Rich objects
                    # (which handle their own formatting)
                    if not self.printer._section_started:
                        self.printer._section_started = True
                    self.printer.console.print(*args, **kwargs)

            def print_panel(self, panel, **kwargs):
                """Print a Rich Panel object with appropriate handling.

                Args:
                    panel: Rich Panel object to print
                    **kwargs: Additional arguments passed to console.print()
                """
                # For panels, we don't add dots - they have their own visual structure
                # Just ensure this counts as the section being started
                if not self.printer._section_started:
                    self.printer._section_started = True
                self.printer.console.print(panel, **kwargs)

            def stream_text(self, text: str, end: str = ""):
                """Stream text output (only available when streaming=True)."""
                if not self.streaming:
                    raise ValueError("stream_text() only available when streaming=True")

                if not self._streaming_started:
                    self._start_streaming()

                # Accumulate and update the live render inline with the dot prefix
                self._current_text += text + end
                if self.live:
                    prefix = f"[{self.color}]⏺[/{self.color}] {self.prefix}"
                    update_str = f"{prefix}{self._current_text}{self._cursor_markup}"
                    self.live.update(update_str)

            def _start_streaming(self):
                if not self._streaming_started and self.add_dot:
                    # Create a live region so we can replace content later cleanly
                    initial = f"[{self.color}]⏺[/{self.color}] {self.prefix}"
                    self.live = Live(
                        initial,
                        console=self.printer.console,
                        refresh_per_second=24,
                        transient=False,
                    )
                    self.live.start()
                    self._streaming_started = True
                    self.printer._section_started = True

            def finalize_to_markdown_panel(self, full_text: str):
                """Replace live content with markdown inside a light border panel."""
                panel = Panel(Markdown(full_text), border_style="color(245)")
                if self.live is not None:
                    with suppress(Exception):
                        self.live.update(panel)
                        self.live.stop()
                        self._finalized = True
                else:
                    # If live wasn't started, just print the panel once
                    self.printer.console.print(panel)

        section_printer = UnifiedSectionPrinter(
            self, color, add_dot, streaming, prefix, separator_style
        )
        try:
            yield section_printer
        finally:
            try:
                # For streaming sections, ensure live is stopped and cursor removed
                if (
                    streaming
                    and section_printer._streaming_started
                    and getattr(section_printer, "live", None) is not None
                ):
                    with suppress(Exception):
                        if not section_printer._finalized:
                            # Update one last time without the cursor
                            prefix = (
                                f"[{section_printer.color}]⏺[/{section_printer.color}] "
                                f"{section_printer.prefix}"
                            )
                            final_text = f"{prefix}{section_printer._current_text}"
                            section_printer.live.update(final_text)
                        section_printer.live.stop()
                # Add a separator after section using the configurable style
                self.add_separator(separator_style)
            except Exception:
                pass

    def print(self, *args, **kwargs) -> None:
        """Direct print passthrough to the underlying console.

        Use this for simple output that doesn't need section management.
        For organized output with dots and separation, use section() context manager.
        """
        self.console.print(*args, **kwargs)

    def add_separator(self, style: str = "blank") -> None:
        """Add a separator line with configurable style.

        Args:
            style: Type of separator - "blank" (default), "line", "dots", etc.
        """
        if style == "blank":
            self.console.print("")
        elif style == "line":
            self.console.print("─" * 50, style="dim")
        elif style == "dots":
            self.console.print("⋯" * 25, style="dim")
        elif style == "space":
            self.console.print(" ")
        else:
            # Default to blank for unknown styles
            self.console.print("")

    def get_input(self, prompt: str = "") -> str:
        # No spinner; just return input via rich console
        return self.console.input(prompt)

    def clear(self) -> None:
        """Clear the screen without any section management."""
        self.console.clear()

    def show_message(
        self, message: str, style: str | None = None, use_section: bool = True
    ) -> None:
        """Show a message, optionally with section management.

        Args:
            message: The message to display
            style: Optional Rich style
            use_section: Whether to use section context (adds separator line)
        """
        if use_section:
            with self.output_section() as p:
                if style:
                    p.print(message, style=style)
                else:
                    p.print(message)
        else:
            if style:
                self.console.print(message, style=style)
            else:
                self.console.print(message)

    def cleanup(self) -> None:
        # No background resources to cleanup now
        pass
