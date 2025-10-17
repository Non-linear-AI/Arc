"""File editing and viewing tools."""

from pathlib import Path

from arc.editing import EditInstruction, EditorManager
from arc.tools.base import BaseTool, ToolResult
from arc.utils.confirmation import ConfirmationService
from arc.utils.performance import cached, performance_manager, timed


class FileEditorTool(BaseTool):
    """Base tool for file operations: view, create, and edit files."""

    def __init__(self):
        super().__init__()
        self.editor_manager = EditorManager()
        self.confirmation_service = ConfirmationService.get_instance()

    async def execute(self, **kwargs) -> ToolResult:
        """Execute file operation. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement execute()")

    @cached(ttl=60, use_file_cache=False)  # Cache for 1 minute
    @timed("file_view_time")
    async def view_file(
        self, path: str, start_line: int | None = None, end_line: int | None = None
    ) -> ToolResult:
        """View file contents or list directory."""
        try:
            file_path = Path(path)

            if not file_path.exists():
                return ToolResult.error_result(f"Path does not exist: {path}")

            # If it's a directory, list contents (no emojis, concise header)
            if file_path.is_dir():
                try:
                    items = []
                    for item in sorted(file_path.iterdir()):
                        if item.is_dir():
                            items.append(f"{item.name}/")
                        else:
                            size = item.stat().st_size
                            items.append(f"{item.name} ({size} bytes)")

                    if not items:
                        return ToolResult.success_result(f"Directory '{path}' is empty")

                    header = f"Directory contents of {path}:"
                    return ToolResult.success_result(header + "\n" + "\n".join(items))
                except PermissionError:
                    return ToolResult.error_result(
                        f"Permission denied accessing directory: {path}"
                    )

            # If it's a file, read contents
            if file_path.is_file():
                try:
                    with open(file_path, encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()

                    # Apply line range if specified
                    if start_line is not None or end_line is not None:
                        start_idx = (start_line - 1) if start_line is not None else 0
                        end_idx = end_line if end_line is not None else len(lines)

                        start_idx = max(0, start_idx)
                        end_idx = min(len(lines), end_idx)

                        lines = lines[start_idx:end_idx]

                        # Add line numbers
                        numbered_lines = []
                        for i, line in enumerate(lines, start=start_idx + 1):
                            numbered_lines.append(f"{i:4d}: {line.rstrip()}")

                        content = "\n".join(numbered_lines)
                        return ToolResult.success_result(
                            f"File '{path}' (lines {start_idx + 1}-{end_idx}):\n"
                            f"{content}"
                        )
                    else:
                        # Show full file with line numbers
                        numbered_lines = []
                        for i, line in enumerate(lines, start=1):
                            numbered_lines.append(f"{i:4d}: {line.rstrip()}")

                        content = "\n".join(numbered_lines)
                        return ToolResult.success_result(f"File '{path}':\n{content}")

                except (UnicodeDecodeError, PermissionError) as e:
                    return ToolResult.error_result(
                        f"Cannot read file '{path}': {str(e)}"
                    )

            return ToolResult.error_result(
                f"'{path}' is not a regular file or directory"
            )

        except Exception as e:
            return ToolResult.error_result(f"Error accessing '{path}': {str(e)}")

    @timed("file_create_time")
    async def create_file(self, path: str, content: str) -> ToolResult:
        """Create a new file with given content using multi-strategy editing."""
        try:
            # Check if file already exists
            if Path(path).exists():
                return ToolResult.error_result(
                    f"File already exists: {path}. "
                    f"Use edit_file to modify existing files."
                )

            # Request confirmation from user
            session_flags = self.confirmation_service.get_session_flags()
            if (
                not session_flags["file_operations"]
                and not session_flags["all_operations"]
            ):
                # Preview content for confirmation (first few lines)
                content_lines = content.split("\n")
                preview = "\n".join(content_lines[:10])
                if len(content_lines) > 10:
                    preview += f"\n... +{len(content_lines) - 10} more lines"

                confirmation_result = await self.confirmation_service.request_confirmation(  # noqa
                    operation="Create file",
                    target=path,
                    operation_type="file",
                    content=f"Creating new file: {path}\nContent preview:\n{preview}",
                )

                if not confirmation_result.confirmed:
                    return ToolResult.error_result(
                        confirmation_result.feedback
                        or "User denied permission to create files"
                    )

            # Create edit instruction for new file
            instruction = EditInstruction(
                file_path=path, new_content=content, create_if_missing=True
            )

            # Apply edit using strategy manager
            result = await self.editor_manager.apply_edit(instruction)

            if result.success:
                strategy_info = f" (strategy: {result.strategy_used})"
                return ToolResult.success_result(f"{result.message}{strategy_info}")
            else:
                return ToolResult.error_result(
                    f"File creation failed: {result.message}"
                )

        except Exception as e:
            return ToolResult.error_result(f"Failed to create file '{path}': {str(e)}")

    @timed("file_edit_time")
    async def edit_file(
        self, path: str, old_str: str, new_str: str, replace_all: bool = False
    ) -> ToolResult:
        """Edit an existing file by replacing text using EXACT string matching.

        This follows Claude Code's approach: exact match only, no fuzzy matching.
        If the string is not found or appears multiple times, returns a clear error.
        """
        try:
            file_path = Path(path)

            # Check if file exists
            if not file_path.exists():
                return ToolResult.error_result(
                    f"File does not exist: {path}. Use create_file for new files."
                )

            if not file_path.is_file():
                return ToolResult.error_result(f"Path is not a file: {path}")

            # Read current content
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                return ToolResult.error_result(
                    f"Cannot read file '{path}': not a text file"
                )
            except PermissionError:
                return ToolResult.error_result(
                    f"Permission denied reading file: {path}"
                )

            # Count occurrences for exact match
            count = content.count(old_str)

            if count == 0:
                # Provide helpful error message
                return ToolResult.error_result(
                    f"String not found in {path}. "
                    f"The old_str must match exactly (including whitespace). "
                    f"Use view_file to see the current content and try again."
                )

            if count > 1 and not replace_all:
                return ToolResult.error_result(
                    f"String appears {count} times in {path}. "
                    f"Either:\n"
                    f"1. Provide more context in old_str to make it unique, or\n"
                    f"2. Set replace_all=true to replace all occurrences"
                )

            # Request confirmation from user
            session_flags = self.confirmation_service.get_session_flags()
            if (
                not session_flags["file_operations"]
                and not session_flags["all_operations"]
            ):
                confirmation_result = await self.confirmation_service.request_confirmation(  # noqa
                    operation="Edit file",
                    target=path,
                    operation_type="file",
                    content=(
                        f"Editing file: {path}\n"
                        f"Replace {count} occurrence(s)\n"
                        f"Old: {old_str[:100]}"
                        f"{'...' if len(old_str) > 100 else ''}\n"
                        f"New: {new_str[:100]}{'...' if len(new_str) > 100 else ''}"
                    ),
                )

                if not confirmation_result.confirmed:
                    return ToolResult.error_result(
                        confirmation_result.feedback
                        or "User denied permission to edit files"
                    )

            # Perform replacement
            if replace_all:
                new_content = content.replace(old_str, new_str)
            else:
                # Replace only first occurrence
                new_content = content.replace(old_str, new_str, 1)

            # Write back to file
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
            except PermissionError:
                return ToolResult.error_result(
                    f"Permission denied writing to file: {path}"
                )

            return ToolResult.success_result(
                f"Successfully replaced {count} occurrence(s) in {path}"
            )

        except Exception as e:
            return ToolResult.error_result(f"Failed to edit file '{path}': {str(e)}")


class ViewFileTool(FileEditorTool):
    """Tool for viewing file or directory contents."""

    async def execute(self, **kwargs) -> ToolResult:
        """Execute view_file operation."""
        performance_manager.metrics["file_operations"] += 1
        return await self.view_file(
            kwargs["path"], kwargs.get("start_line"), kwargs.get("end_line")
        )


class CreateFileTool(FileEditorTool):
    """Tool for creating new files."""

    async def execute(self, **kwargs) -> ToolResult:
        """Execute create_file operation."""
        performance_manager.metrics["file_operations"] += 1
        return await self.create_file(kwargs["path"], kwargs["content"])


class EditFileTool(FileEditorTool):
    """Tool for editing existing files."""

    async def execute(self, **kwargs) -> ToolResult:
        """Execute edit_file operation."""
        performance_manager.metrics["file_operations"] += 1
        return await self.edit_file(
            kwargs["path"],
            kwargs["old_str"],
            kwargs["new_str"],
            kwargs.get("replace_all", False),
        )
