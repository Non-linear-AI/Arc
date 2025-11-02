"""File editing and viewing tools."""

import json
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

    def _build_file_operation_result(
        self,
        status: str,
        operation_type: str,
        file_path: str,
        success_message: str | None = None,
        error_message: str | None = None,
        **details,
    ) -> str:
        """Build structured JSON result for file operations.

        Args:
            status: "completed", "cancelled", or "failed"
            operation_type: "create_file" or "edit_file"
            file_path: Path to the file
            success_message: Success message if completed
            error_message: Error message if failed
            **details: Additional operation-specific details

        Returns:
            JSON string with structured file operation result
        """
        result = {
            "status": status,
            "operation": {
                "type": operation_type,
                "file_path": file_path,
                **details,
            },
        }

        if success_message:
            result["operation"]["message"] = success_message
        if error_message:
            result["operation"]["error"] = error_message

        return json.dumps(result)

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
                    # User cancelled - return structured JSON with success=True
                    output_json = self._build_file_operation_result(
                        status="cancelled",
                        operation_type="create_file",
                        file_path=path,
                        content_preview=preview,
                    )
                    return ToolResult(
                        success=True,
                        output=output_json,
                        metadata={"cancelled": True},
                    )

            # Create edit instruction for new file
            instruction = EditInstruction(
                file_path=path, new_content=content, create_if_missing=True
            )

            # Apply edit using strategy manager
            result = await self.editor_manager.apply_edit(instruction)

            if result.success:
                # Success - return structured JSON
                output_json = self._build_file_operation_result(
                    status="completed",
                    operation_type="create_file",
                    file_path=path,
                    success_message=result.message,
                    strategy_used=result.strategy_used,
                )
                return ToolResult(
                    success=True,
                    output=output_json,
                    metadata={"strategy": result.strategy_used},
                )
            else:
                # Failed - return structured JSON with error
                output_json = self._build_file_operation_result(
                    status="failed",
                    operation_type="create_file",
                    file_path=path,
                    error_message=f"File creation failed: {result.message}",
                )
                return ToolResult(
                    success=False,
                    output=output_json,
                    metadata={"error": "creation_failed"},
                )

        except Exception as e:
            # Exception - return structured JSON with error
            output_json = self._build_file_operation_result(
                status="failed",
                operation_type="create_file",
                file_path=path,
                error_message=f"Failed to create file: {str(e)}",
            )
            return ToolResult(
                success=False,
                output=output_json,
                metadata={"error": "exception"},
            )

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
                output_json = self._build_file_operation_result(
                    status="failed",
                    operation_type="edit_file",
                    file_path=path,
                    error_message="File does not exist. Use create_file for new files.",
                )
                return ToolResult(
                    success=False,
                    output=output_json,
                    metadata={"error": "file_not_found"},
                )

            if not file_path.is_file():
                output_json = self._build_file_operation_result(
                    status="failed",
                    operation_type="edit_file",
                    file_path=path,
                    error_message="Path is not a file",
                )
                return ToolResult(
                    success=False,
                    output=output_json,
                    metadata={"error": "not_a_file"},
                )

            # Read current content
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                output_json = self._build_file_operation_result(
                    status="failed",
                    operation_type="edit_file",
                    file_path=path,
                    error_message="Cannot read file: not a text file",
                )
                return ToolResult(
                    success=False,
                    output=output_json,
                    metadata={"error": "unicode_decode_error"},
                )
            except PermissionError:
                output_json = self._build_file_operation_result(
                    status="failed",
                    operation_type="edit_file",
                    file_path=path,
                    error_message="Permission denied reading file",
                )
                return ToolResult(
                    success=False,
                    output=output_json,
                    metadata={"error": "permission_denied_read"},
                )

            # Count occurrences for exact match
            count = content.count(old_str)

            if count == 0:
                # String not found - return structured JSON
                output_json = self._build_file_operation_result(
                    status="failed",
                    operation_type="edit_file",
                    file_path=path,
                    error_message=(
                        "String not found in file. "
                        "The old_str must match exactly (including whitespace). "
                        "Use view_file to see the current content and try again."
                    ),
                )
                return ToolResult(
                    success=False,
                    output=output_json,
                    metadata={"error": "string_not_found"},
                )

            if count > 1 and not replace_all:
                # Multiple occurrences without replace_all - return structured JSON
                output_json = self._build_file_operation_result(
                    status="failed",
                    operation_type="edit_file",
                    file_path=path,
                    error_message=(
                        f"String appears {count} times in file. "
                        f"Either provide more context in old_str to make it unique, "
                        f"or set replace_all=true to replace all occurrences."
                    ),
                    occurrence_count=count,
                )
                return ToolResult(
                    success=False,
                    output=output_json,
                    metadata={"error": "multiple_occurrences", "count": count},
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
                    # User cancelled - return structured JSON with success=True
                    output_json = self._build_file_operation_result(
                        status="cancelled",
                        operation_type="edit_file",
                        file_path=path,
                        old_string_preview=old_str[:100] + ("..." if len(old_str) > 100 else ""),
                        new_string_preview=new_str[:100] + ("..." if len(new_str) > 100 else ""),
                        occurrence_count=count,
                    )
                    return ToolResult(
                        success=True,
                        output=output_json,
                        metadata={"cancelled": True},
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
                # Permission error - return structured JSON
                output_json = self._build_file_operation_result(
                    status="failed",
                    operation_type="edit_file",
                    file_path=path,
                    error_message=f"Permission denied writing to file: {path}",
                )
                return ToolResult(
                    success=False,
                    output=output_json,
                    metadata={"error": "permission_denied"},
                )

            # Success - return structured JSON
            output_json = self._build_file_operation_result(
                status="completed",
                operation_type="edit_file",
                file_path=path,
                success_message=f"Successfully replaced {count} occurrence(s)",
                occurrence_count=count,
                replace_all=replace_all,
            )
            return ToolResult(
                success=True,
                output=output_json,
                metadata={"occurrences_replaced": count},
            )

        except Exception as e:
            # Exception - return structured JSON
            output_json = self._build_file_operation_result(
                status="failed",
                operation_type="edit_file",
                file_path=path,
                error_message=f"Failed to edit file: {str(e)}",
            )
            return ToolResult(
                success=False,
                output=output_json,
                metadata={"error": "exception"},
            )


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
