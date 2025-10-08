"""Enhanced UX components for Arc CLI."""

import asyncio
import sys
import threading
from collections.abc import Callable
from contextlib import contextmanager, suppress
from typing import Any

from rich import box
from rich.align import Align
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from arc.database import QueryResult
from arc.ui.printer import Printer


class InteractiveInterface:
    """Console facade: formats content; delegates printing/streaming to Printer."""

    def __init__(self):
        self._printer = Printer()

    def show_welcome(self, _model: str, _directory: str):
        """Display a centered ASCII banner in an 80-char panel."""
        banner = (
            " ▓▓▓▓▓╗   ▓▓▓▓▓▓╗    ▓▓▓▓▓▓╗\n"
            "▓▓╔══▓▓╗  ▓▓╔══▓▓╗  ▓▓╔════╝\n"
            "▓▓▓▓▓▓▓║  ▓▓▓▓▓▓╔╝  ▓▓║\n"
            "▓▓╔══▓▓║  ▓▓╔══▓▓╗  ▓▓║\n"
            "▓▓║  ▓▓║  ▓▓║  ╚▓▓╗ ╚▓▓▓▓▓▓╗\n"
            "╚═╝  ╚═╝  ╚═╝   ╚═╝  ╚═════╝\n"
            " From Question to Prediction\n"
        )

        panel = Panel(
            Align.center(banner, style="cyan"),
            border_style="cyan",
            box=box.DOUBLE,
            padding=(1, 0, 0, 0),
            width=80,
        )
        self._printer.print(Align.left(panel))

        # Single concise hint
        self._printer.print(" Use /help for more information. Press Esc to interrupt.")
        self._printer.add_separator()

    # Lightweight ESC watcher used during streaming (no prompt active)
    class _EscWatcher:
        def __init__(
            self,
            loop: asyncio.AbstractEventLoop | None = None,
            event: asyncio.Event | None = None,
            is_input_active: Callable[[], bool] | None = None,
        ):
            self._pressed = threading.Event()
            self._stop = threading.Event()
            self._thread: threading.Thread | None = None
            self._fd = None
            self._orig_attrs = None
            self._loop = loop
            self._event = event
            self._is_input_active = is_input_active

        def start(self):
            if not sys.stdin.isatty():
                return
            try:
                import select
                import termios
                import tty

                self._termios = termios  # store for stop()
                self._select = select
                self._fd = sys.stdin.fileno()
                self._orig_attrs = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
                # Keep terminal echo enabled to avoid interfering with prompts

                def _run():
                    try:
                        while not self._stop.is_set():
                            # If an input prompt is active, don't consume stdin
                            if self._is_input_active and self._is_input_active():
                                # Sleep briefly to yield
                                self._stop.wait(0.05)
                                continue

                            r, _, _ = self._select.select([sys.stdin], [], [], 0.05)
                            if r:
                                ch = sys.stdin.read(1)
                                if ch == "\x1b":  # ESC
                                    self._pressed.set()
                                    # Notify asyncio side immediately if available
                                    if (
                                        self._loop is not None
                                        and self._event is not None
                                    ):
                                        with suppress(Exception):
                                            self._loop.call_soon_threadsafe(
                                                self._event.set
                                            )
                                    break
                    except Exception:
                        pass

                self._thread = threading.Thread(target=_run, daemon=True)
                self._thread.start()
            except Exception:
                # Best-effort only; if unavailable, do nothing
                pass

        def is_pressed(self) -> bool:
            return self._pressed.is_set()

        def stop(self):
            try:
                self._stop.set()
                if self._thread:
                    self._thread.join(timeout=0.2)
                if self._orig_attrs is not None and self._fd is not None:
                    self._termios.tcsetattr(
                        self._fd, self._termios.TCSADRAIN, self._orig_attrs
                    )
            except Exception:
                pass

    @contextmanager
    def escape_watcher(self):
        """Start an ESC watcher during streaming and restore terminal state on exit.

        Yields an object with:
          - is_pressed(): bool (thread-side immediate flag)
          - event: asyncio.Event (fires promptly on ESC)
        """
        loop = asyncio.get_running_loop()
        event = asyncio.Event()

        class EscHandle:
            def __init__(self, watcher, event):
                self._watcher = watcher
                self.event = event

            def is_pressed(self) -> bool:
                return self._watcher.is_pressed()

        watcher = self._EscWatcher(
            loop, event, is_input_active=self._printer.is_input_active
        )
        watcher.start()
        # Track active watcher so we can suspend it during interactive menus
        self._active_watcher = watcher
        try:
            yield EscHandle(watcher, event)
        finally:
            watcher.stop()
            if getattr(self, "_active_watcher", None) is watcher:
                self._active_watcher = None

    def suspend_escape(self) -> None:
        """Stop any active ESC watcher to avoid stealing input (e.g., menus)."""
        watcher = getattr(self, "_active_watcher", None)
        if watcher is not None:
            with suppress(Exception):
                watcher.stop()
            self._active_watcher = None

    def trigger_escape(self) -> None:
        """Programmatically trigger an ESC event to cancel the current task."""
        watcher = getattr(self, "_active_watcher", None)
        if watcher is not None:
            if hasattr(watcher, "_event") and watcher._event is not None:
                watcher._event.set()
            if hasattr(watcher, "_pressed") and watcher._pressed is not None:
                watcher._pressed.set()

    def show_commands(self) -> None:
        """Display available slash commands in a concise list."""
        with self._printer.section(color="blue") as p:
            p.print("How to Use Arc")
            p.print(
                "  [dim]Ask questions in natural language or use slash commands "
                "below.[/dim]"
            )
            p.print(
                "  [dim]Examples: 'analyze my data', 'help me train a model', "
                "'/config'[/dim]"
            )

            p.print()
            p.print("  System Commands")
            commands = [
                ("/help", "Show available commands and features"),
                ("/config", "View or edit configuration"),
                ("/report", "Report a bug or feedback on GitHub"),
                (
                    "/sql use [system|user] | /sql <query>",
                    "Switch database or execute SQL query ",
                ),
                ("/clear", "Clear the screen"),
                ("/exit", "Exit the application"),
            ]
            for cmd, desc in commands:
                p.print(f"  - [cyan]{cmd}[/cyan]: {desc}")

            p.print()
            p.print("  ML Commands")
            ml_commands = [
                (
                    "/ml plan --context DESC --data-table TABLE --target-column COL",
                    "Create ML workflow plan (feature engineering, architecture, "
                    "training, evaluation)",
                ),
                (
                    "/ml revise-plan --feedback FEEDBACK",
                    "Revise the current ML plan based on feedback or training results",
                ),
                (
                    "/ml generate-model --name NAME [--context DESC] "
                    "--data-table TABLE [--target-column COL] [--plan-id PLAN_ID]",
                    "Generate an Arc-Graph model specification "
                    "(--context optional with --plan-id)",
                ),
                (
                    "/ml generate-trainer --name NAME --context DESC --model-spec FILE",
                    "Generate an Arc-Graph trainer specification",
                ),
                (
                    "/ml generate-predictor --model-id ID --context DESC",
                    "Generate an Arc-Graph predictor specification",
                ),
                (
                    "/ml create-model --name NAME --schema PATH",
                    "Register an Arc-Graph model",
                ),
                (
                    "/ml data-processing --yaml PATH",
                    "Process data based on the Arc-formatYAML specification",
                ),
                ("/ml train --model NAME --data TABLE", "Launch a training job"),
                (
                    "/ml predict --model NAME --data TABLE --output TABLE",
                    "Run inference and save predictions",
                ),
                ("/ml jobs list", "Show recent ML jobs"),
                ("/ml jobs status JOB_ID", "Inspect an individual job"),
            ]
            for cmd, desc in ml_commands:
                p.print(f"  - [cyan]{cmd}[/cyan]: {desc}")

    def _action_label(self, tool_name: str) -> str:
        mapping = {
            "view_file": "Read",
            "create_file": "Create",
            "str_replace_editor": "Update",
            "bash": "Bash",
            "search": "Search",
            "create_todo_list": "Create Plan",
            "update_todo_list": "Update Plan",
            "database_query": "SQL Query",
            "schema_discovery": "Schema Discovery",
            "ml_create_model": "Register Model",
            "ml_train": "Train Model",
            "ml_predict": "Predict",
            "ml_model_generator": "Model Generator",
            "ml_trainer_generator": "Trainer Generator",
            "ml_predictor_generator": "Predictor Generator",
            "data_processor_generator": "Data Processor Generator",
        }
        # Also handle MCP-prefixed tools nicely
        if tool_name.startswith("mcp__"):
            parts = tool_name.split("__")
            if len(parts) >= 3:
                server = parts[1]
                actual = " ".join(parts[2:]).replace("_", " ")
                return f"{server.title()}({actual})"
        return mapping.get(tool_name, tool_name)

    def _get_dot_color(self, tool_name: str) -> str:
        """Get color for the dot based on semantic action type.

        Color scheme:
        - Blue: System operations, configuration, databases
        - Green: Success operations, ML training/prediction
        - Yellow: File operations, search, user attention
        - Red: System commands, potentially risky operations
        - Default: Neutral tool output, informational
        """
        if tool_name in ["create_todo_list", "update_todo_list"]:
            return "blue"  # Planning/system operations
        elif tool_name in ["bash"]:
            return "red"  # System commands (potentially risky)
        elif tool_name in ["search"]:
            return "yellow"  # Search operations (attention/discovery)
        elif tool_name in ["view_file", "create_file", "str_replace_editor"]:
            return "yellow"  # File operations (user attention needed)
        elif tool_name in ["database_query", "schema_discovery"]:
            return "blue"  # Database/system operations
        elif tool_name in [
            "ml_create_model",
            "ml_train",
            "ml_predict",
            "ml_model_generator",
            "ml_trainer_generator",
            "ml_predictor_generator",
        ]:
            return "green"  # ML operations (success/completion focused)
        elif tool_name in ["data_processor_generator"]:
            return "bright_yellow"
        else:
            return "white"  # Default/neutral informational output

    def show_tool_execution(self, _tool_name: str, _args: dict[str, Any]):
        """Show tool execution line that will be replaced with result."""
        # Don't show anything here - we'll show the result directly
        self._working_active = True

    def show_tool_result(self, tool_name: str, result, _execution_time: float):
        """Print one tool result as a single output section."""
        label = self._action_label(tool_name)

        if self._working_active:
            self._working_active = False

        content = result.output if result.success else result.error
        content = content or ""

        dot_color = self._get_dot_color(tool_name)
        with self._printer.section(color=dot_color) as p:
            if (
                tool_name in ["create_todo_list", "update_todo_list"]
                and content.strip()
            ):
                self._print_todo_with_inline_progress(label, content, printer=p)
            else:
                p.print(f"{label}")
                if content.strip():
                    self._print_details_block(content, printer=p)

    def _print_todo_with_inline_progress(
        self,
        label: str,
        content: str,
        printer: Any | None = None,
    ) -> None:
        """Print todo with progress bar inline with the action label."""
        lines = content.splitlines()
        if not lines:
            return

        # Find the progress bar line and extract it
        progress_line = None
        todo_items = []

        for line in lines:
            line = line.strip()
            if line.startswith("📋"):
                # Extract just the progress bar part
                if "[" in line and "]" in line:
                    start = line.find("[")
                    end = line.find("]") + 1
                    progress_part = line[start:end]
                    # Also get the ratio part
                    ratio_part = line.split("]")[-1].strip()
                    progress_line = f"{progress_part} {ratio_part}"
            elif line.startswith("└"):
                todo_items.append(line)

        target = printer if printer else self._printer
        if progress_line:
            target.print(f"{label} {progress_line}")
        for item in todo_items:
            target.print(f"  {item}")

    def _print_todo_content(self, content: str) -> None:
        """Print todo content with progress bar format."""
        lines = content.splitlines()
        if not lines:
            return

        # Print the todo content directly without modification
        for line in lines:
            if line.strip():
                with self._printer.section(add_dot=False) as p:
                    p.print(f"  {line}")

    def _print_details_block(
        self, content: str, _max_lines: int = 8, printer: Any | None = None
    ) -> None:
        """Print details block matching the exact format from the example."""
        lines = content.splitlines()
        if not lines:
            return

        # Show first line with ⎿ marker
        first = lines[0].rstrip()
        target = printer if printer else self._printer
        target.print(f"  [dim]⎿ {first}[/dim]")

        # Show up to 2 more lines with proper indentation
        rest = lines[1:5]  # Only show 2 more lines max
        for ln in rest:
            if ln.strip():  # Skip empty lines
                target.print(f"     [dim]{ln.rstrip()}[/dim]")

        # Show ellipsis if there are more lines
        if len(lines) > 5:
            remaining = len(lines) - 5
            target.print(f"     [dim]… +{remaining} lines (ctrl+r to expand)[/dim]")

    def show_user_message(self, content: str):
        """Clear the input line and show user message inside a light border."""
        text = content.strip()
        if not text:
            return

        # Calculate how many lines to clear (prompt + any multiline input)
        lines = text.split("\n")
        lines_to_clear = len(lines)

        # Use ANSI escape sequences to move cursor and clear lines
        # Move cursor up to the beginning of the prompt line
        print(f"\033[{lines_to_clear}A\r", end="", flush=True)

        # Clear each line from current position to end of line
        for i in range(lines_to_clear):
            print("\033[K", end="", flush=True)  # Clear to end of line
            if i < lines_to_clear - 1:
                print(
                    "\033[1B\r", end="", flush=True
                )  # Move down one line and to start

        # Move cursor back to the start position
        if lines_to_clear > 1:
            print(f"\033[{lines_to_clear - 1}A\r", end="", flush=True)

        # Render the user message in soft purple-gray (no border)
        if lines:
            self._printer.print(
                f"[color(245)]>[/color(245)] [color(245)]{lines[0]}[/color(245)]"
            )
            for ln in lines[1:]:
                self._printer.print(f"  [color(245)]{ln}[/color(245)]")
            self._printer.add_separator()

    def show_assistant_step(self, content: str):
        """Render assistant thoughts as a cyan dot step with the content."""
        text = content.strip()
        if not text:
            return

        # Render each line with a single cyan dot header once, then plain lines
        lines = text.split("\n")
        if lines:
            with self._printer.section(color="cyan") as p:
                p.print(f"{lines[0]}")
                for ln in lines[1:]:
                    p.print(f"  {ln}")

    @contextmanager
    def assistant_response(self):
        """Context manager for streaming assistant responses.

        Usage:
            with ui.assistant_response() as stream:
                stream.stream_text("Hello ")
                stream.stream_text("world!")
        """
        streaming_context = self._printer.section(color="cyan", streaming=True)
        stream_printer = streaming_context.__enter__()
        try:
            yield stream_printer
        finally:
            streaming_context.__exit__(None, None, None)

    @contextmanager
    def stream_response(self, start_time):
        """Context manager for handling an entire streaming response workflow.

        Usage:
            with ui.stream_response(start_time) as handler:
                async for chunk in agent.process_user_message_stream(user_input):
                    handler.handle_chunk(chunk)
        """
        import json
        import time

        class StreamResponseHandler:
            def __init__(self, ui, start_time):
                self.ui = ui
                self.start_time = start_time
                self.assistant_context = None
                self.stream = None
                self._buffer = ""

            def handle_chunk(self, chunk):
                if chunk.type == "content" and chunk.content:
                    # Start streaming context on first content chunk
                    if self.assistant_context is None:
                        self.assistant_context = self.ui.assistant_response()
                        self.stream = self.assistant_context.__enter__()
                    # Accumulate for final markdown rendering
                    self._buffer += chunk.content
                    # Stream content immediately as it arrives
                    self.stream.stream_text(chunk.content)

                elif chunk.type == "tool_calls" and chunk.tool_call:
                    # Finish any ongoing assistant streaming before tool execution
                    self._finish_assistant_streaming(final=False)

                    args = {}
                    if chunk.tool_call and chunk.tool_call.arguments:
                        try:
                            args = json.loads(chunk.tool_call.arguments)
                        except json.JSONDecodeError:
                            args = {"raw_arguments": chunk.tool_call.arguments}

                    tool_name = (
                        chunk.tool_call.name if chunk.tool_call else "Unknown Tool"
                    )
                    self.ui.show_tool_execution(tool_name, args)

                elif (
                    chunk.type == "tool_result"
                    and chunk.tool_result
                    and chunk.tool_call
                ):
                    # Show tool result as one output section
                    tool_time = time.time() - self.start_time
                    self.ui.show_tool_result(
                        chunk.tool_call.name, chunk.tool_result, tool_time
                    )

                elif chunk.type == "error":
                    # Finish any ongoing assistant streaming before showing error
                    self._finish_assistant_streaming(final=False)
                    self.ui.show_system_error(chunk.content)

                elif chunk.type == "done":
                    # Finish any ongoing assistant streaming
                    self._finish_assistant_streaming(final=True)

            def _finish_assistant_streaming(self, final: bool = False):
                """Finish streaming; if final=True, clear and markdown-render it.

                When final=False (e.g., before a tool call), we simply close the
                streaming context and keep what was streamed as-is, without
                clearing or re-rendering.
                """
                if self.assistant_context is not None:
                    # If final, replace the live region with markdown panel before exit
                    if final and self._buffer and self.stream is not None:
                        with suppress(Exception):
                            self.stream.finalize_to_markdown_panel(self._buffer)

                    # Close the streaming section context
                    self.assistant_context.__exit__(None, None, None)

                    # Reset state; only keep buffer if not final (but we don't use it)
                    self.assistant_context = None
                    self.stream = None
                    self._buffer = "" if final else ""

        handler = StreamResponseHandler(self, start_time)
        try:
            yield handler
        finally:
            # Ensure cleanup even if an exception occurs (no final rendering here)
            handler._finish_assistant_streaming(final=False)

    def _format_args(self, args: dict[str, Any]) -> str:
        """Format tool arguments for display."""
        if not args:
            return "none"

        formatted = []
        for key, value in args.items():
            if isinstance(value, str) and len(value) > 50:
                value = value[:50] + "..."
            formatted.append(f"{key}={value}")

        return ", ".join(formatted)

    def prompt_confirmation(self, message: str) -> bool:
        """Confirmation prompt using prompt_toolkit via Printer."""
        while True:
            try:
                resp = self._printer.get_input(f"🤔 {message} [y/N]: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                return False

            if resp in ("y", "yes"):
                return True
            if resp in ("n", "no", ""):
                return False
            self._printer.print("Please enter 'y' or 'n'.")

    def confirm(self, message: str, default: bool = False) -> bool:
        """Confirmation prompt with configurable default.

        Args:
            message: The confirmation message to display
            default: Default value if user just presses Enter (True=yes, False=no)

        Returns:
            True if user confirms, False otherwise
        """
        prompt_suffix = " [Y/n]: " if default else " [y/N]: "
        while True:
            try:
                resp = (
                    self._printer.get_input(f"🤔 {message}{prompt_suffix}")
                    .strip()
                    .lower()
                )
            except (KeyboardInterrupt, EOFError):
                return False

            if resp in ("y", "yes"):
                return True
            if resp in ("n", "no"):
                return False
            if resp == "":
                return default
            self._printer.print("Please enter 'y' or 'n'.")

    async def confirm_async(self, message: str, default: bool = False) -> bool:
        """Async confirmation prompt with configurable default.

        Args:
            message: The confirmation message to display
            default: Default value if user just presses Enter (True=yes, False=no)

        Returns:
            True if user confirms, False otherwise
        """
        prompt_suffix = " [Y/n]: " if default else " [y/N]: "
        while True:
            try:
                resp = (
                    (
                        await self._printer.get_input_async(
                            f"🤔 {message}{prompt_suffix}"
                        )
                    )
                    .strip()
                    .lower()
                )
            except (KeyboardInterrupt, EOFError):
                return False

            if resp in ("y", "yes"):
                return True
            if resp in ("n", "no"):
                return False
            if resp == "":
                return default
            self._printer.print("Please enter 'y' or 'n'.")

    def prompt_input(self, message: str, default: str | None = None) -> str:
        """Simple input prompt using prompt_toolkit via Printer."""
        try:
            value = self._printer.get_input(
                f"💭 {message}{' [' + default + ']' if default else ''}: "
            )
        except (KeyboardInterrupt, EOFError):
            return default or ""
        value = value.strip()
        return value if value else (default or "")

    def show_code_diff(self, old_code: str, new_code: str, language: str = "python"):
        """Show code diff with syntax highlighting."""
        with self._printer.section(color="yellow") as p:
            p.print("📝 [bold]Code Changes:[/bold]")

        # Show old code
        if old_code:
            with self._printer.section(color="red") as p:
                p.print_panel(
                    Panel(
                        Syntax(
                            old_code, language, theme="github-dark", line_numbers=True
                        ),
                        title="🔴 Before",
                        border_style="red",
                    )
                )

        # Show new code
        if new_code:
            with self._printer.section(color="green") as p:
                p.print_panel(
                    Panel(
                        Syntax(
                            new_code, language, theme="github-dark", line_numbers=True
                        ),
                        title="🟢 After",
                        border_style="green",
                    )
                )

    def show_streaming_response(self, content: str):
        """Show streaming response with typing effect."""
        with self._printer.section(color="cyan", streaming=True) as stream:
            for char in content:
                stream.stream_text(char, end="")

    def show_sql_result(
        self,
        result: QueryResult,
        target_db: str,
        execution_time: float | None = None,
    ) -> None:
        """Display SQL query results in a formatted table."""
        db_label = "System DB" if target_db == "system" else "User DB"
        header = f"SQL Query ({db_label})"
        if execution_time is not None:
            header += f" - {execution_time:.3f}s"

        with self._printer.section(color="blue") as p:
            # Header
            p.print(f"{header}")

            if result.empty():
                p.print("\n [dim]No results found[/dim]")
                return

            # Build clean table for panel display
            table = Table(
                show_header=True,
                header_style="bold",
                border_style="color(240)",
                box=box.HORIZONTALS,
            )

            # Add columns from first row
            first_row = result.first()
            if first_row:
                for column_name in first_row:
                    table.add_column(str(column_name), no_wrap=False)

            # Add data rows (limit to avoid overwhelming output)
            max_rows = 100
            for row_count, row in enumerate(result):
                if row_count >= max_rows:
                    table.add_row(*["..." for _ in first_row], style="dim")
                    break

                # Convert all values to strings and handle None
                row_values = []
                for value in row.values():
                    if value is None:
                        row_values.append("[dim]NULL[/dim]")
                    elif isinstance(value, (dict, list)):
                        # Format JSON-like objects
                        import json

                        try:
                            row_values.append(
                                json.dumps(value, indent=None, separators=(",", ":"))
                            )
                        except (TypeError, ValueError):
                            row_values.append(str(value))
                    else:
                        row_values.append(str(value))

                table.add_row(*row_values)

            # Show the table and summary in a compact panel
            p.print(table)
            total_rows = result.count()
            if total_rows > max_rows:
                p.print(f" [dim]Showing {max_rows} of {total_rows} rows[/dim]")
            else:
                row_text = "row" if total_rows == 1 else "rows"
                p.print(f" [dim]{total_rows} {row_text} returned[/dim]")

    # System and misc helpers using Printer
    def show_system_error(self, message: str) -> None:
        self._printer.show_message(f"❌ {message}", style="red", use_section=False)

    def show_system_success(self, message: str) -> None:
        self._printer.show_message(f"✅ {message}", use_section=False)

    def show_warning(self, message: str) -> None:
        self._printer.show_message(f"⚠️ {message}", style="yellow", use_section=False)

    def show_goodbye(self) -> None:
        self._printer.show_message("👋 Goodbye!", style="cyan", use_section=False)

    def show_info(self, message: str) -> None:
        self._printer.show_message(message, use_section=False)

    def clear_screen(self) -> None:
        self._printer.clear()

    def show_config_panel(self, config_text: str) -> None:
        with self._printer.section(color="blue") as p:
            p.print_panel(
                Panel(
                    config_text,
                    expand=False,
                    border_style="color(240)",
                    title="Configuration (edit via /config; env vars override)",
                )
            )

    def show_table(self, title: str, columns: list[str], rows: list[list[str]]) -> None:
        table = Table(title=title, box=box.SIMPLE_HEAVY)
        for col in columns:
            table.add_column(col)

        if not rows:
            table.add_row(*(["-"] * len(columns)))
        else:
            for row in rows:
                table.add_row(*row)

        with self._printer.section(color="blue") as p:
            p.print(table)

    def show_key_values(self, title: str, pairs: list[list[str]]) -> None:
        table = Table(title=title, box=box.SIMPLE)
        table.add_column("Field", style="bold")
        table.add_column("Value")

        for pair in pairs:
            if len(pair) >= 2:
                table.add_row(pair[0], pair[1])

        with self._printer.section(color="blue") as p:
            p.print(table)

    def get_user_input(self, prompt: str = "\n> ") -> str:
        return self._printer.get_input(prompt).strip()

    async def get_user_input_async(self, prompt: str = "\n> ") -> str:
        """Async version of get_user_input for use in async contexts."""
        result = await self._printer.get_input_async(prompt)
        return result.strip()

    def cleanup(self) -> None:
        self._printer.cleanup()
