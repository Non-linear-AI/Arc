"""Bash command execution tool."""

import asyncio
import json
import os

from arc.tools.base import BaseTool, ToolResult
from arc.utils.confirmation import ConfirmationService


class BashTool(BaseTool):
    """Tool for executing bash commands."""

    def __init__(self):
        self._current_directory = os.getcwd()
        self.confirmation_service = ConfirmationService.get_instance()

    def get_current_directory(self) -> str:
        """Get the current working directory."""
        return self._current_directory

    def _build_bash_result(
        self,
        status: str,
        command: str,
        working_directory: str,
        exit_code: int | None = None,
        stdout: str = "",
        stderr: str = "",
        error_message: str | None = None,
    ) -> str:
        """Build structured JSON result for bash tool.

        Args:
            status: "completed", "cancelled", or "failed"
            command: The bash command that was executed
            working_directory: Directory where command was executed
            exit_code: Command exit code (None if not executed)
            stdout: Standard output from command
            stderr: Standard error from command
            error_message: Error message if execution failed

        Returns:
            JSON string with structured bash execution result
        """
        result = {
            "status": status,
            "execution": {
                "command": command,
                "working_directory": working_directory,
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
            },
        }

        if error_message:
            result["execution"]["error"] = error_message

        return json.dumps(result)

    async def execute(self, command: str, timeout: int = 60) -> ToolResult:
        """Execute a bash command."""
        try:
            # Request confirmation from user
            session_flags = self.confirmation_service.get_session_flags()
            if (
                not session_flags["bash_commands"]
                and not session_flags["all_operations"]
            ):
                confirmation_result = (
                    await self.confirmation_service.request_confirmation(
                        operation="Run bash command",
                        target=command,
                        operation_type="bash",
                        content=f"Command: {command}\n"
                        f"Working directory: {self._current_directory}",
                    )
                )

                if not confirmation_result.confirmed:
                    # User cancelled - return structured JSON with success=True
                    output_json = self._build_bash_result(
                        status="cancelled",
                        command=command,
                        working_directory=self._current_directory,
                        exit_code=None,
                        stdout="",
                        stderr="",
                    )
                    return ToolResult(
                        success=True,
                        output=output_json,
                        metadata={"cancelled": True},
                    )
            # Create the process
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._current_directory,
            )

            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                # Timeout is a failure - return structured JSON
                output_json = self._build_bash_result(
                    status="failed",
                    command=command,
                    working_directory=self._current_directory,
                    exit_code=None,
                    stdout="",
                    stderr="",
                    error_message=f"Command timed out after {timeout} seconds",
                )
                return ToolResult(
                    success=False,
                    output=output_json,
                    metadata={"error": "timeout"},
                )

            # Decode output
            stdout_str = stdout.decode("utf-8", errors="replace").strip()
            stderr_str = stderr.decode("utf-8", errors="replace").strip()

            # Update current directory if cd was used
            if command.strip().startswith("cd "):
                await self._update_current_directory()

            # Return result as structured JSON
            if process.returncode == 0:
                # Success - return completed status
                output_json = self._build_bash_result(
                    status="completed",
                    command=command,
                    working_directory=self._current_directory,
                    exit_code=process.returncode,
                    stdout=stdout_str,
                    stderr=stderr_str,
                )
                return ToolResult(
                    success=True,
                    output=output_json,
                    metadata={"exit_code": process.returncode},
                )
            else:
                # Non-zero exit code - return failed status
                output_json = self._build_bash_result(
                    status="failed",
                    command=command,
                    working_directory=self._current_directory,
                    exit_code=process.returncode,
                    stdout=stdout_str,
                    stderr=stderr_str,
                    error_message=f"Command failed with exit code {process.returncode}",
                )
                return ToolResult(
                    success=False,
                    output=output_json,
                    metadata={"exit_code": process.returncode, "error": "non_zero_exit"},
                )

        except Exception as e:
            # Exception during execution - return failed status
            output_json = self._build_bash_result(
                status="failed",
                command=command,
                working_directory=self._current_directory,
                exit_code=None,
                stdout="",
                stderr="",
                error_message=f"Failed to execute command: {str(e)}",
            )
            return ToolResult(
                success=False,
                output=output_json,
                metadata={"error": "exception"},
            )

    async def _update_current_directory(self) -> None:
        """Update the current directory tracking."""
        try:
            process = await asyncio.create_subprocess_shell(
                "pwd",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._current_directory,
            )
            stdout, _ = await process.communicate()
            if process.returncode == 0:
                new_dir = stdout.decode("utf-8").strip()
                if os.path.exists(new_dir):
                    self._current_directory = new_dir
        except Exception:
            # If we can't update, keep the current directory as is
            pass
