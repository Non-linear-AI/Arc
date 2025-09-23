"""Printer: centralizes console output, streaming, and input handling.

- Provides output sections that automatically insert a trailing blank line.
- Provides streaming helpers for incremental output.
- Delegates actual rendering to rich.console.Console, but keeps formatting
  decisions out of CLI business logic.
- Enhanced with prompt_toolkit for advanced input handling including history,
  cursor movement, and interruption support.
"""

import sys
import time
from contextlib import contextmanager, suppress
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel


class Printer:
    def __init__(self):
        self.console = Console()
        self._is_streaming = False
        self._section_started = False

        # Advanced input handling
        self._prompt_session = None
        self._prompt_enabled = True
        self._last_interrupt_time = None
        self._interrupt_count = 0
        self._agent_callback = None

        # Set up history file path
        self._history_file = Path.home() / ".arc_history"

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
        """Enhanced input method with history, cursor movement, and interruption."""
        if not self._prompt_enabled or not self._is_real_terminal():
            # Fallback to basic console input if prompt_toolkit is disabled
            # or we're not in a real terminal
            return self.console.input(prompt)

        # Check if we're in an async context - if so, fall back to basic input
        # to avoid event loop conflicts
        try:
            import asyncio
            asyncio.get_running_loop()
            # We're in an async context, use basic input to avoid conflicts
            return self.console.input(prompt)
        except RuntimeError:
            # No event loop running, safe to use prompt_toolkit
            pass

        try:
            if not self._prompt_session:
                self._prompt_session = self._create_prompt_session()

            return self._prompt_session.prompt(prompt)
        except KeyboardInterrupt:
            return self._handle_keyboard_interrupt()
        except EOFError:
            # Handle Ctrl+D (EOF)
            raise SystemExit("EOF received, exiting...") from None

    async def get_input_async(self, prompt: str = "") -> str:
        """Async version of get_input for use in async contexts.

        Uses prompt_toolkit's async API when in a proper terminal,
        falls back gracefully otherwise.
        """
        if not self._prompt_enabled or not self._is_real_terminal():
            return self.console.input(prompt)

        try:
            if not self._prompt_session:
                self._prompt_session = self._create_prompt_session()

            # Use prompt_toolkit's async API properly
            from prompt_toolkit.application import create_app_session
            from prompt_toolkit.input import create_input
            from prompt_toolkit.output import create_output

            # Create a new app session for async compatibility
            with create_app_session(input=create_input(), output=create_output()):
                result = await self._prompt_session.prompt_async(prompt)
                return result

        except KeyboardInterrupt:
            return self._handle_keyboard_interrupt()
        except EOFError:
            # Handle Ctrl+D (EOF)
            raise SystemExit("EOF received, exiting...") from None
        except Exception:
            # If prompt_toolkit fails, fall back to basic input
            return self.console.input(prompt)

    def _is_real_terminal(self) -> bool:
        """Check if we're running in a real terminal that supports advanced input."""
        return (
            sys.stdin.isatty() and
            sys.stdout.isatty() and
            hasattr(sys.stdin, 'fileno')
        )

    def _create_prompt_session(self) -> PromptSession:
        """Create a PromptSession with advanced key bindings and history."""
        kb = self._create_key_bindings()

        return PromptSession(
            history=FileHistory(str(self._history_file)),
            key_bindings=kb,
            multiline=False,
            wrap_lines=True,
            mouse_support=False,  # Keep it simple for CLI
            complete_style='column',
        )

    def _create_key_bindings(self) -> KeyBindings:
        """Create comprehensive key bindings for input handling."""
        kb = KeyBindings()

        # History navigation (Ctrl+Up/Down like aider)
        @kb.add('c-up')
        def _(event):
            """Navigate backward through history"""
            event.current_buffer.history_backward()

        @kb.add('c-down')
        def _(event):
            """Navigate forward through history"""
            event.current_buffer.history_forward()

        # Alternative history navigation (Up/Down arrows)
        @kb.add('up')
        def _(event):
            """Navigate backward through history with up arrow"""
            event.current_buffer.history_backward()

        @kb.add('down')
        def _(event):
            """Navigate forward through history with down arrow"""
            event.current_buffer.history_forward()

        # Cursor movement
        @kb.add('left')
        def _(event):
            """Move cursor left"""
            event.current_buffer.cursor_left()

        @kb.add('right')
        def _(event):
            """Move cursor right"""
            event.current_buffer.cursor_right()

        # Word movement (Ctrl+Left/Right)
        @kb.add('c-left')
        def _(event):
            """Move cursor to beginning of previous word"""
            event.current_buffer.cursor_left(count=event.current_buffer.document.find_previous_word_beginning())

        @kb.add('c-right')
        def _(event):
            """Move cursor to end of next word"""
            event.current_buffer.cursor_right(count=event.current_buffer.document.find_next_word_ending())

        # Line navigation
        @kb.add('home')
        @kb.add('c-a')
        def _(event):
            """Move cursor to beginning of line"""
            event.current_buffer.cursor_position = 0

        @kb.add('end')
        @kb.add('c-e')
        def _(event):
            """Move cursor to end of line"""
            event.current_buffer.cursor_position = len(event.current_buffer.text)

        # Text deletion
        @kb.add('c-k')
        def _(event):
            """Delete from cursor to end of line"""
            buffer = event.current_buffer
            buffer.delete(count=len(buffer.text) - buffer.cursor_position)

        @kb.add('c-u')
        def _(event):
            """Delete from beginning of line to cursor"""
            buffer = event.current_buffer
            buffer.delete(count=-buffer.cursor_position)

        @kb.add('c-w')
        def _(event):
            """Delete previous word"""
            buffer = event.current_buffer
            pos = buffer.document.find_previous_word_beginning()
            if pos:
                buffer.delete(count=pos)

        # Clear line
        @kb.add('c-l')
        def _(event):
            """Clear the screen"""
            event.app.output.clear()

        # Clear current input (like ESC in some shells)
        @kb.add('escape')
        def _(event):
            """Clear current input line"""
            event.current_buffer.reset()

        return kb

    def _handle_keyboard_interrupt(self) -> str:
        """Handle Ctrl+C with double-press exit pattern (from aider)."""
        now = time.time()

        # If we have an agent callback, call it for cancellation
        if self._agent_callback:
            with suppress(Exception):
                self._agent_callback()

        # Double Ctrl+C within 2 seconds exits
        if self._last_interrupt_time and now - self._last_interrupt_time < 2:
            self.console.print("\n\n^C KeyboardInterrupt - Exiting...", style="red")
            raise SystemExit(1)

        # First Ctrl+C just shows message and resets input
        self.console.print(
            "\n\n^C Press Ctrl+C again within 2 seconds to exit", style="yellow"
        )
        self._last_interrupt_time = now
        return ""  # Return empty string to continue input loop

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

    def set_prompt_enabled(self, enabled: bool) -> None:
        """Enable or disable prompt_toolkit features (for compatibility)."""
        self._prompt_enabled = enabled

    def setup_agent_interrupt_handler(self, agent_callback=None) -> None:
        """Set up interrupt handling for agent execution contexts.

        This method sets up a more compatible interrupt handling approach
        that works with asyncio event loops.

        Args:
            agent_callback: Optional callback to call when interrupt is detected
        """
        # Store the callback for later use
        self._agent_callback = agent_callback

        # For asyncio compatibility, we'll handle interrupts at the input level
        # rather than using signal handlers during async operations
        self.console.print(
            "[dim]Interrupt handling active: Ctrl+C to cancel operations[/dim]"
        )

    def restore_default_interrupt_handler(self) -> None:
        """Restore default Ctrl+C handling."""
        # Reset agent callback
        self._agent_callback = None

    def cleanup(self) -> None:
        """Cleanup resources and restore default handlers."""
        self.restore_default_interrupt_handler()
