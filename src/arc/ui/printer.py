"""Printer: centralizes console output, streaming, and input handling.

- Provides output sections that automatically insert a trailing blank line.
- Provides streaming helpers for incremental output.
- Delegates actual rendering to rich.console.Console, but keeps formatting
  decisions out of CLI business logic.
- Enhanced with prompt_toolkit for advanced input handling including history,
  cursor movement, and interruption support.
"""

import sys
from contextlib import contextmanager, suppress
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel


class ArcCompleter(Completer):
    """Custom completer for Arc CLI commands with context-aware suggestions."""

    def __init__(self):
        # Base commands
        self.base_commands = [
            "/help",
            "/stats",
            "/performance",
            "/tree",
            "/config",
            "/clear",
            "/exit",
            "/quit",
        ]

        # SQL commands
        self.sql_commands = ["/sql use system", "/sql use user", "/sql"]

        # ML base commands
        self.ml_base_commands = [
            "/ml generate-model",
            "/ml generate-trainer",
            "/ml generate-predictor",
            "/ml create-model",
            "/ml train",
            "/ml predict",
            "/ml jobs",
        ]

        # ML subcommands for jobs
        self.ml_jobs_subcommands = ["list", "status"]

        # Command-specific parameters with descriptions
        self.ml_command_params = {
            "generate-model": [
                ("--name", "Model name (required)"),
                ("--context", "Model description and context (required)"),
                ("--data-table", "Database table name for data (required)"),
                ("--model", "AI model to use (optional)"),
            ],
            "generate-trainer": [
                ("--name", "Trainer name (required)"),
                ("--context", "Training context and requirements (required)"),
                ("--model-spec", "Path to model specification file (required)"),
                ("--model", "AI model to use (optional)"),
            ],
            "generate-predictor": [
                ("--model-spec", "Path to model specification file (required)"),
                ("--context", "Prediction requirements and use case (required)"),
                ("--trainer-spec", "Path to trainer specification file (optional)"),
                ("--output", "Output file path (optional)"),
                ("--model", "AI model to use (optional)"),
            ],
            "create-model": [
                ("--name", "Model name (required)"),
                ("--schema", "Path to model schema file (required)"),
            ],
            "train": [
                ("--model", "Model name (required)"),
                ("--data", "Data table name (required)"),
            ],
            "predict": [
                ("--model", "Model name (required)"),
                ("--data", "Input data table name (required)"),
                ("--output", "Output table name (required)"),
            ],
        }

        # All available options for fallback completion
        all_options = set()
        for params in self.ml_command_params.values():
            all_options.update(param[0] for param in params)
        self.common_options = list(all_options)

        # Create word completer for simple cases
        all_completions = (
            self.base_commands
            + self.sql_commands
            + self.ml_base_commands
            + self.common_options
        )
        self.word_completer = WordCompleter(all_completions, ignore_case=True)

    def get_completions(self, document, _complete_event):
        """Generate completions based on current input context."""
        text = document.text.lower()
        word_before_cursor = document.get_word_before_cursor().lower()
        text_parts = text.split()

        # Handle different completion contexts
        if len(text_parts) >= 2 and text_parts[0] == "/ml" and text_parts[1] == "jobs":
            # Complete ML jobs subcommands when we have "/ml jobs"
            # Show options when there's a trailing space or typing third word
            if len(text_parts) >= 3 or text.endswith(" "):
                for subcommand in self.ml_jobs_subcommands:
                    if subcommand.startswith(word_before_cursor):
                        yield Completion(
                            subcommand,
                            start_position=-len(word_before_cursor),
                            display_meta="ML jobs command",
                        )
            # If user typed exactly "/ml jobs" + tab, show subcommands
            elif len(text_parts) == 2 and text.strip() == "/ml jobs":
                for subcommand in self.ml_jobs_subcommands:
                    yield Completion(
                        " " + subcommand,  # Add space before subcommand
                        start_position=0,
                        display_meta="ML jobs command",
                    )

        elif text.startswith("/ml ") and len(text_parts) >= 2:
            # Check if we're completing parameters for a specific ML command
            if len(text_parts) >= 3 and text_parts[1] in self.ml_command_params:
                # Complete parameters for the specific ML command
                command = text_parts[1]
                available_params = self.ml_command_params[command]

                # Check which parameters are already used in the command
                used_params = set()
                for i in range(2, len(text_parts)):
                    if text_parts[i].startswith("--"):
                        used_params.add(text_parts[i])

                # Suggest unused parameters
                if word_before_cursor.startswith("--") or text.endswith(" "):
                    for param, description in available_params:
                        if param not in used_params and param.startswith(
                            word_before_cursor
                        ):
                            yield Completion(
                                param,
                                start_position=-len(word_before_cursor),
                                display_meta=description,
                            )

            # If we have "/ml <subcommand>" and valid command, show parameters
            elif len(text_parts) == 2 and text_parts[1] in self.ml_command_params:
                command = text_parts[1]
                available_params = self.ml_command_params[command]
                for param, description in available_params:
                    yield Completion(
                        " " + param,  # Add space before parameter
                        start_position=0,
                        display_meta=description,
                    )

            else:
                # Complete ML subcommands
                ml_subcommands = [
                    ("generate-model", "Generate ML model specification"),
                    ("generate-trainer", "Generate training configuration"),
                    ("generate-predictor", "Generate prediction service"),
                    ("create-model", "Create model from schema"),
                    ("train", "Start training job"),
                    ("predict", "Run prediction"),
                    ("jobs", "Manage ML jobs"),
                ]
                for subcommand, description in ml_subcommands:
                    if subcommand.startswith(word_before_cursor):
                        yield Completion(
                            subcommand,
                            start_position=-len(word_before_cursor),
                            display_meta=description,
                        )

        elif text.startswith("/sql ") and len(text.split()) >= 2:
            # Complete SQL specific options
            sql_options = [
                ("use system", "Switch to system database"),
                ("use user", "Switch to user database"),
            ]
            remaining_text = text[5:].strip()  # Remove '/sql '
            for option, description in sql_options:
                if option.startswith(remaining_text):
                    yield Completion(
                        option,
                        start_position=-len(remaining_text),
                        display_meta=description,
                    )

        elif text.startswith("/") and len(text.split()) == 1:
            # Complete base commands
            all_commands = [
                ("/help", "Show help information"),
                ("/ml", "Machine learning commands"),
                ("/sql", "Database operations"),
                ("/stats", "Show statistics"),
                ("/performance", "Performance metrics"),
                ("/tree", "Directory structure"),
                ("/config", "Configuration"),
                ("/clear", "Clear screen"),
                ("/exit", "Exit application"),
                ("/quit", "Exit application"),
            ]
            for command, description in all_commands:
                if command.startswith(text):
                    yield Completion(
                        command, start_position=-len(text), display_meta=description
                    )

        elif word_before_cursor.startswith("--"):
            # Complete command options
            for option in self.common_options:
                if option.startswith(word_before_cursor):
                    yield Completion(
                        option,
                        start_position=-len(word_before_cursor),
                        display_meta="Command option",
                    )

        # If no specific context matches, don't suggest anything to avoid clutter
        # This prevents random word completion when typing regular text


class Printer:
    def __init__(self):
        self.console = Console()
        self._is_streaming = False
        self._section_started = False

        # Advanced input handling
        self._prompt_session = None
        self._prompt_enabled = True

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
            # Ctrl+C exits the application
            raise SystemExit(0) from None
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
            # Ctrl+C exits the application
            raise SystemExit(0) from None
        except EOFError:
            # Handle Ctrl+D (EOF)
            raise SystemExit("EOF received, exiting...") from None
        except Exception:
            # If prompt_toolkit fails, fall back to basic input
            return self.console.input(prompt)

    def _is_real_terminal(self) -> bool:
        """Check if we're running in a real terminal that supports advanced input."""
        return (
            sys.stdin.isatty() and sys.stdout.isatty() and hasattr(sys.stdin, "fileno")
        )

    def _create_prompt_session(self) -> PromptSession:
        """Create a PromptSession with advanced key bindings and history."""
        kb = self._create_key_bindings()
        completer = ArcCompleter()

        # Create Arc CLI-styled completion menu to match the existing design
        arc_style = Style.from_dict(
            {
                # Completion menu styling - subtle, professional look
                # Dark background, light text (VS Code-like)
                "completion-menu.completion": "bg:#1e1e1e fg:#d4d4d4",
                # Cyan-600 selection (matches Arc cyan theme)
                "completion-menu.completion.current": "bg:#0e7490 fg:#ffffff bold",
                # Dimmed text for descriptions (matches "dim" style)
                "completion-menu.meta.completion": "bg:#1e1e1e fg:#888888",
                # Light cyan for selected description
                "completion-menu.meta.completion.current": "bg:#0e7490 fg:#f0f9ff",
                # Consistent with dim text
                "completion-menu.multi-column-meta": "bg:#1e1e1e fg:#888888",
                "completion-menu.scrollbar": "bg:#404040",  # Subtle scrollbar
                "completion-menu": "bg:#1e1e1e",  # Overall menu background
                # Keep input text styling clean
                "": "",  # Use terminal defaults for input text
            }
        )

        return PromptSession(
            history=FileHistory(str(self._history_file)),
            key_bindings=kb,
            completer=completer,
            multiline=False,
            wrap_lines=True,
            mouse_support=False,  # Keep it simple for CLI
            complete_style="column",
            style=arc_style,
        )

    def _create_key_bindings(self) -> KeyBindings:
        """Create comprehensive key bindings for input handling."""
        kb = KeyBindings()

        # History navigation (Ctrl+Up/Down like aider)
        @kb.add("c-up")
        def _(event):
            """Navigate backward through history"""
            event.current_buffer.history_backward()

        @kb.add("c-down")
        def _(event):
            """Navigate forward through history"""
            event.current_buffer.history_forward()

        # Let prompt_toolkit handle up/down arrows automatically:
        # - When completions are shown: navigate completions
        # - When no completions: navigate history
        # This provides better UX than forcing history navigation

        # Cursor movement
        @kb.add("left")
        def _(event):
            """Move cursor left"""
            event.current_buffer.cursor_left()

        @kb.add("right")
        def _(event):
            """Move cursor right"""
            event.current_buffer.cursor_right()

        # Word movement (Ctrl+Left/Right)
        @kb.add("c-left")
        def _(event):
            """Move cursor to beginning of previous word"""
            # document.find_previous_word_beginning() returns a negative offset
            event.current_buffer.cursor_position += (
                event.current_buffer.document.find_previous_word_beginning()
            )

        @kb.add("c-right")
        def _(event):
            """Move cursor to end of next word"""
            event.current_buffer.cursor_position += (
                event.current_buffer.document.find_next_word_ending()
            )

        # Line navigation
        @kb.add("home")
        @kb.add("c-a")
        def _(event):
            """Move cursor to beginning of line"""
            event.current_buffer.cursor_position = 0

        @kb.add("end")
        @kb.add("c-e")
        def _(event):
            """Move cursor to end of line"""
            event.current_buffer.cursor_position = len(event.current_buffer.text)

        # Text deletion
        @kb.add("c-k")
        def _(event):
            """Delete from cursor to end of line"""
            buffer = event.current_buffer
            buffer.delete(count=len(buffer.text) - buffer.cursor_position)

        @kb.add("c-u")
        def _(event):
            """Delete from beginning of line to cursor"""
            buffer = event.current_buffer
            buffer.delete(count=-buffer.cursor_position)

        # ESC: interrupt current input and return control to caller
        @kb.add("escape")
        def _(event):
            event.app.exit(result="")

        return kb

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
