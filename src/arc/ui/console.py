"""Enhanced UX components for Arc CLI."""

from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rich import box
from rich.align import Align
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from ..database import QueryResult
from .printer import Printer


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
        self._printer.print(" Use /help for more information.")
        self._printer.add_separator()

    def show_commands(self) -> None:
        """Display available slash commands in a concise list."""
        with self._printer.section(color="blue") as p:
            p.print("[bold]System Commands[/bold]")
            p.print(
                "  [dim]Commands require '/' prefix. "
                "Regular text without '/' is sent to the AI.[/dim]"
            )
            commands = [
                ("/help", "Show available commands and features"),
                ("/stats", "Show editing strategy statistics"),
                ("/performance", "Show performance metrics and cache statistics"),
                ("/tree", "Show directory structure"),
                ("/config", "View current configuration"),
                (
                    "/sql [system|user] <query>",
                    "Execute SQL query (system: read-only, user: full access)",
                ),
                ("/clear", "Clear the screen"),
                ("/exit or /quit", "Exit the application"),
            ]
            for cmd, desc in commands:
                p.print(f"  • [bold cyan]{cmd}[/bold cyan]: {desc}")

    def _action_label(self, tool_name: str) -> str:
        mapping = {
            "view_file": "Read",
            "create_file": "Create",
            "str_replace_editor": "Update",
            "bash": "Bash",
            "search": "Search",
            "create_todo_list": "Create Plan",
            "update_todo_list": "Update Plan",
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
        """Get color for the dot based on action type."""
        if tool_name in ["create_todo_list", "update_todo_list"]:
            return "blue"  # Plan operations
        elif tool_name in ["bash"]:
            return "red"  # System operations
        elif tool_name in ["search"]:
            return "yellow"  # Search operations
        elif tool_name in ["view_file", "create_file", "str_replace_editor"]:
            return "magenta"  # File operations
        else:
            return "cyan"  # Default/messages

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
                self._print_todo_with_inline_progress(
                    label, content, dot_color, printer=p
                )
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
        rest = lines[1:3]  # Only show 2 more lines max
        for ln in rest:
            if ln.strip():  # Skip empty lines
                target.print(f"     [dim]{ln.rstrip()}[/dim]")

        # Show ellipsis if there are more lines
        if len(lines) > 3:
            remaining = len(lines) - 3
            target.print(f"     [dim]… +{remaining} lines (ctrl+r to expand)[/dim]")

    def show_user_message(self, content: str):
        """Clear the input line and redisplay user message in light gray."""
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

        # Render the user message in soft purple-gray
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

            def handle_chunk(self, chunk):
                if chunk.type == "content" and chunk.content:
                    # Start streaming context on first content chunk
                    if self.assistant_context is None:
                        self.assistant_context = self.ui.assistant_response()
                        self.stream = self.assistant_context.__enter__()
                    # Stream content immediately as it arrives
                    self.stream.stream_text(chunk.content)

                elif chunk.type == "tool_calls" and chunk.tool_call:
                    # Finish any ongoing assistant streaming before tool execution
                    self._finish_assistant_streaming()

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
                    self._finish_assistant_streaming()
                    self.ui.show_system_error(chunk.content)

                elif chunk.type == "done":
                    # Finish any ongoing assistant streaming
                    self._finish_assistant_streaming()

            def _finish_assistant_streaming(self):
                """Helper to finish assistant streaming context."""
                if self.assistant_context is not None:
                    self.assistant_context.__exit__(None, None, None)
                    self.assistant_context = None
                    self.stream = None

        handler = StreamResponseHandler(self, start_time)
        try:
            yield handler
        finally:
            # Ensure cleanup even if an exception occurs
            handler._finish_assistant_streaming()

    def show_edit_summary(self, strategy_stats: dict[str, dict[str, Any]]):
        """Show editing strategy statistics."""
        stats_table = Table(title="📊 Editing Strategy Performance")
        stats_table.add_column("Strategy", style="bold")
        stats_table.add_column("Success", style="green")
        stats_table.add_column("Failures", style="red")
        stats_table.add_column("Success Rate", style="cyan")
        stats_table.add_column("Total Ops", style="yellow")

        for strategy_name, stats in strategy_stats.items():
            success_rate = f"{stats['success_rate']:.1%}"
            stats_table.add_row(
                strategy_name.replace("_", " ").title(),
                str(stats["success_count"]),
                str(stats["failure_count"]),
                success_rate,
                str(stats["total_operations"]),
            )

        with self._printer.section(color="blue") as p:
            p.print(stats_table)

    def show_performance_metrics(
        self, metrics: dict[str, Any], error_stats: dict[str, Any] | None = None
    ):
        """Show performance metrics dashboard."""
        perf_table = Table(title="🚀 Performance Metrics")
        perf_table.add_column("Metric", style="bold cyan")
        perf_table.add_column("Value", style="yellow")
        perf_table.add_column("Description", style="dim")

        perf_table.add_row(
            "Cache Hit Rate",
            f"{metrics.get('cache_hit_rate', 0):.1%}",
            "Percentage of requests served from cache",
        )
        perf_table.add_row(
            "Cache Hits",
            str(metrics.get("cache_hits", 0)),
            "Number of successful cache retrievals",
        )
        perf_table.add_row(
            "Cache Misses",
            str(metrics.get("cache_misses", 0)),
            "Number of cache misses requiring computation",
        )
        perf_table.add_row(
            "Avg Response Time",
            f"{metrics.get('avg_response_time', 0):.3f}s",
            "Average time per request",
        )
        perf_table.add_row(
            "Total Requests",
            str(metrics.get("total_requests", 0)),
            "Total number of processed requests",
        )
        perf_table.add_row(
            "File Operations",
            str(metrics.get("file_operations", 0)),
            "Number of file operations performed",
        )
        perf_table.add_row(
            "Tool Executions",
            str(metrics.get("tool_executions", 0)),
            "Number of tool executions",
        )
        perf_table.add_row(
            "Memory Cache Size",
            str(metrics.get("memory_cache_size", 0)),
            "Number of items in memory cache",
        )
        perf_table.add_row(
            "File Cache Size",
            str(metrics.get("file_cache_size", 0)),
            "Number of items in persistent cache",
        )

        with self._printer.section(color="green") as p:
            p.print(perf_table)

        # Show error statistics if available
        if error_stats and error_stats.get("total_errors", 0) > 0:
            error_table = Table(title="⚠️ Error Statistics")
            error_table.add_column("Category", style="bold red")
            error_table.add_column("Count", style="yellow")

            for category, count in error_stats.get("by_category", {}).items():
                error_table.add_row(category.replace("_", " ").title(), str(count))

            with self._printer.section(color="red") as p:
                p.print(error_table)

    def show_file_tree(self, directory: str, max_depth: int = 3):
        """Display file tree for current directory."""
        try:
            tree = Tree(f"📁 {directory}")
            self._build_tree(Path(directory), tree, max_depth, 0)

            with self._printer.section(color="blue") as p:
                p.print_panel(Panel(tree))
        except Exception as e:
            with self._printer.section(color="red") as p:
                p.print(f"❌ Error building file tree: {e}")

    def _build_tree(self, path: Path, tree: Tree, max_depth: int, current_depth: int):
        """Recursively build file tree."""
        if current_depth >= max_depth:
            return

        try:
            items = sorted(path.iterdir())[:20]  # Limit to 20 items per directory

            for item in items:
                if item.is_dir():
                    branch = tree.add(f"📁 {item.name}/")
                    if current_depth < max_depth - 1:
                        self._build_tree(item, branch, max_depth, current_depth + 1)
                else:
                    # Add file with size info
                    size = item.stat().st_size
                    size_str = self._format_file_size(size)
                    tree.add(f"📄 {item.name} ({size_str})")
        except PermissionError:
            tree.add("❌ Permission denied")

    def _format_file_size(self, size: int) -> str:
        """Format file size in human readable form."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

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
        """Enhanced confirmation prompt."""
        return Confirm.ask(f"🤔 {message}")

    def prompt_input(self, message: str, default: str | None = None) -> str:
        """Enhanced input prompt."""
        return Prompt.ask(f"💭 {message}", default=default)

    def show_code_diff(self, old_code: str, new_code: str, language: str = "python"):
        """Show code diff with syntax highlighting."""
        with self._printer.section(color="yellow") as p:
            p.print("📝 [bold]Code Changes:[/bold]")

        # Show old code
        if old_code:
            with self._printer.section(color="red") as p:
                p.print_panel(
                    Panel(
                        Syntax(old_code, language, theme="monokai", line_numbers=True),
                        title="🔴 Before",
                        border_style="red",
                    )
                )

        # Show new code
        if new_code:
            with self._printer.section(color="green") as p:
                p.print_panel(
                    Panel(
                        Syntax(new_code, language, theme="monokai", line_numbers=True),
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
        query: str,
        execution_time: float | None = None,
    ) -> None:
        """Display SQL query results in a formatted table."""
        # Show query header
        db_label = "System DB" if target_db == "system" else "User DB"
        header = f"🗃️ SQL Query ({db_label})"
        if execution_time is not None:
            header += f" - {execution_time:.3f}s"

        self.console.print(f"\n[bold cyan]{header}[/bold cyan]")

        # Show the query in a code block
        self.console.print(
            Panel(
                Syntax(query.strip(), "sql", theme="monokai", word_wrap=True),
                title="Query",
                border_style="blue",
                padding=(0, 1),
            )
        )

        if result.empty():
            self.console.print(
                Panel(
                    "[dim]No results found[/dim]", border_style="yellow", padding=(0, 1)
                )
            )
            return

        # Create Rich table
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)

        # Add columns from first row
        first_row = result.first()
        if first_row:
            for column_name in first_row:
                table.add_column(str(column_name), style="cyan", no_wrap=False)

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

        # Show the table
        self.console.print(table)

        # Show result summary
        total_rows = result.count()
        if total_rows > max_rows:
            self.console.print(f"\n[dim]Showing {max_rows} of {total_rows} rows[/dim]")
        else:
            row_text = "row" if total_rows == 1 else "rows"
            self.console.print(f"\n[dim]{total_rows} {row_text} returned[/dim]")

    # System and misc helpers using Printer
    def show_system_error(self, message: str) -> None:
        self._printer.show_message(f"❌ {message}", style="red", use_section=False)

    def show_system_success(self, message: str) -> None:
        self._printer.show_message(f"✅ {message}", use_section=False)

    def show_goodbye(self) -> None:
        self._printer.show_message("👋 Goodbye!", style="cyan", use_section=False)

    def show_info(self, message: str) -> None:
        self._printer.show_message(message, use_section=False)

    def clear_screen(self) -> None:
        self._printer.clear()

    def show_config_panel(self, config_text: str) -> None:
        with self._printer.section(color="blue") as p:
            p.print_panel(Panel(config_text))

    def get_user_input(self, prompt: str = "\n[bold green]>[/bold green] ") -> str:
        return self._printer.get_input(prompt).strip()

    def cleanup(self) -> None:
        self._printer.cleanup()
