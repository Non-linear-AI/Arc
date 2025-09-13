"""Enhanced UX components for Arc CLI."""

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rich import box
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

console = Console()


class ProgressTracker:
    """Track and display progress for various operations."""

    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        )
        self.tasks = {}

    @contextmanager
    def track(self, description: str, total: int | None = None):
        """Context manager for tracking progress."""
        with self.progress:
            task_id = self.progress.add_task(description, total=total or 100)
            self.tasks[description] = task_id

            class TaskUpdater:
                def __init__(self, progress, task_id):
                    self.progress = progress
                    self.task_id = task_id

                def update(self, advance: int = 1, description: str | None = None):
                    self.progress.update(
                        self.task_id,
                        advance=advance,
                        description=description
                        or self.progress.tasks[self.task_id].description,
                    )

                def complete(self):
                    self.progress.update(self.task_id, completed=True)

            yield TaskUpdater(self.progress, task_id)


class InteractiveInterface:
    """Enhanced interactive interface for Arc CLI."""

    def __init__(self):
        self.console = console
        self.progress_tracker = ProgressTracker()
        self._working_active = False
        self._spinner_thread = None
        self._spinner_stop = False
        self._spinner_frames = ["/", "-", "\\", "|"]
        self._spinner_frame_index = 0

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
        self.console.print(Align.left(panel))

        # Single concise hint
        self.console.print("\n Use /help for more information.\n")

    def show_typing_indicator(self, message: str = "Arc is thinking"):
        """Show typing indicator for AI responses."""
        return Live(
            f"[dim]{message}...[/dim]", refresh_per_second=4, console=self.console
        )

    def show_commands(self) -> None:
        """Display available slash commands in a concise list."""
        self.console.print("\n[bold]System Commands[/bold]")
        self.console.print(
            "  [dim]Commands require '/' prefix. Regular text without '/' is sent "
            "to the AI.[/dim]"
        )
        commands = [
            ("/help", "Show available commands and features"),
            ("/stats", "Show editing strategy statistics"),
            ("/performance", "Show performance metrics and cache statistics"),
            ("/tree", "Show directory structure"),
            ("/config", "View current configuration"),
            ("/clear", "Clear the screen"),
            ("/exit or /quit", "Exit the application"),
        ]
        for cmd, desc in commands:
            self.console.print(f"  • [bold cyan]{cmd}[/bold cyan]: {desc}")

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
            return "green"  # System operations
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
        """Clean tool result display matching the example format."""
        label = self._action_label(tool_name)

        # Clear working active flag
        if self._working_active:
            self._working_active = False

        # Format the result content
        content = result.output if result.success else result.error
        if content is None:
            content = ""

        # Add spacing before every action
        self.console.print()
        
        # Get color for this action type
        dot_color = self._get_dot_color(tool_name)
        
        # Special handling for todo operations - show progress bar inline
        if tool_name in ["create_todo_list", "update_todo_list"] and content.strip():
            self._print_todo_with_inline_progress(label, content, dot_color)
        else:
            # Header line as a step - colored dot and tool name
            self.console.print(f"[{dot_color}]⏺[/{dot_color}] [white]{label}[/white]")
            
            # Show details if there's content
            if content.strip():
                self._print_details_block(content)

    def _print_todo_with_inline_progress(self, label: str, content: str, dot_color: str = "blue") -> None:
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
        
        # Print header with inline progress
        if progress_line:
            self.console.print(f"[{dot_color}]⏺[/{dot_color}] [white]{label}[/white] {progress_line}")
        
        # Print todo items
        for item in todo_items:
            self.console.print(f"  {item}")

    def _print_todo_content(self, content: str) -> None:
        """Print todo content with progress bar format."""
        lines = content.splitlines()
        if not lines:
            return

        # Print the todo content directly without modification
        for line in lines:
            if line.strip():
                self.console.print(f"  {line}")

    def _print_details_block(self, content: str, _max_lines: int = 5) -> None:
        """Print details block matching the exact format from the example."""
        lines = content.splitlines()
        if not lines:
            return

        # Show first line with ⎿ marker
        first = lines[0].rstrip()
        self.console.print(f"  [dim]⎿ {first}[/dim]")

        # Show up to 2 more lines with proper indentation
        rest = lines[1:3]  # Only show 2 more lines max
        for ln in rest:
            if ln.strip():  # Skip empty lines
                self.console.print(f"     [dim]{ln.rstrip()}[/dim]")

        # Show ellipsis if there are more lines
        if len(lines) > 3:
            remaining = len(lines) - 3
            self.console.print(
                f"     [dim]… +{remaining} lines (ctrl+r to expand)[/dim]"
            )

    def show_assistant_step(self, content: str):
        """Render assistant thoughts as a cyan dot step with the content."""
        text = content.strip()
        if not text:
            return
        
        # Add spacing before assistant messages
        self.console.print()
        
        # Render each line with a single cyan dot header once, then plain lines
        lines = text.split("\n")
        if lines:
            self.console.print(f"[cyan]⏺[/cyan] [white]{lines[0]}[/white]")
            for ln in lines[1:]:
                self.console.print(f"  [white]{ln}[/white]")

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

        self.console.print(stats_table)

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

        self.console.print(perf_table)

        # Show error statistics if available
        if error_stats and error_stats.get("total_errors", 0) > 0:
            error_table = Table(title="⚠️ Error Statistics")
            error_table.add_column("Category", style="bold red")
            error_table.add_column("Count", style="yellow")

            for category, count in error_stats.get("by_category", {}).items():
                error_table.add_row(category.replace("_", " ").title(), str(count))

            self.console.print(error_table)

    def show_file_tree(self, directory: str, max_depth: int = 3):
        """Display file tree for current directory."""
        try:
            tree = Tree(f"📁 {directory}")
            self._build_tree(Path(directory), tree, max_depth, 0)

            self.console.print(
                Panel(tree, title="📁 Directory Structure", border_style="blue")
            )
        except Exception as e:
            self.console.print(f"❌ Error building file tree: {e}")

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
        self.console.print("\n📝 [bold]Code Changes:[/bold]")

        # Show old code
        if old_code:
            self.console.print(
                Panel(
                    Syntax(old_code, language, theme="monokai", line_numbers=True),
                    title="🔴 Before",
                    border_style="red",
                )
            )

        # Show new code
        if new_code:
            self.console.print(
                Panel(
                    Syntax(new_code, language, theme="monokai", line_numbers=True),
                    title="🟢 After",
                    border_style="green",
                )
            )

    def show_streaming_response(self, content: str):
        """Show streaming response with typing effect."""
        for char in content:
            self.console.print(char, end="")
            time.sleep(0.01)  # Simulate typing
        self.console.print()  # New line at end


class SmartOutputFormatter:
    """Intelligently format different types of content."""

    @staticmethod
    def format_content(content: str, content_type: str | None = None) -> Any:
        """Smart content formatting based on content type detection."""
        if not content:
            return content

        # Detect content type if not provided
        if content_type is None:
            content_type = SmartOutputFormatter._detect_content_type(content)

        if content_type == "code":
            language = SmartOutputFormatter._detect_language(content)
            return Syntax(content, language, theme="monokai", line_numbers=True)
        elif content_type == "markdown":
            return Markdown(content)
        elif content_type == "json":
            import json

            try:
                parsed = json.loads(content)
                formatted = json.dumps(parsed, indent=2)
                return Syntax(formatted, "json", theme="monokai")
            except (json.JSONDecodeError, ValueError):
                return content
        elif content_type == "table":
            return SmartOutputFormatter._create_table_from_text(content)
        else:
            return content

    @staticmethod
    def _detect_content_type(content: str) -> str:
        """Detect content type from content."""
        content_lower = content.lower().strip()

        if content.startswith("```") or any(
            keyword in content_lower
            for keyword in ["def ", "class ", "import ", "function", "var ", "const "]
        ):
            return "code"
        elif content.startswith("#") or "**" in content or "*" in content[:10]:
            return "markdown"
        elif content.strip().startswith("{") or content.strip().startswith("["):
            return "json"
        elif "|" in content and content.count("|") > 2:
            return "table"
        else:
            return "text"

    @staticmethod
    def _detect_language(content: str) -> str:
        """Detect programming language from code content."""
        if "def " in content or "import " in content or "class " in content:
            return "python"
        elif "function" in content or "const " in content or "let " in content:
            return "javascript"
        elif "#include" in content or "int main" in content:
            return "c"
        elif "public class" in content or "System.out" in content:
            return "java"
        else:
            return "text"

    @staticmethod
    def _create_table_from_text(content: str) -> Table:
        """Create Rich table from text content."""
        lines = content.strip().split("\n")
        if len(lines) < 2:
            return content

        # Try to parse as table
        table = Table()

        # Use first line as headers
        headers = [col.strip() for col in lines[0].split("|")]
        for header in headers:
            if header:  # Skip empty columns
                table.add_column(header)

        # Add rows
        for line in lines[1:]:
            if "|" in line:
                columns = [col.strip() for col in line.split("|")]
                if len(columns) >= len(headers):
                    table.add_row(*columns[: len(headers)])

        return table
