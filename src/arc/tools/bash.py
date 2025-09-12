"""Bash command execution tool."""

import asyncio
import os

from .base import BaseTool, ToolResult


class BashTool(BaseTool):
    """Tool for executing bash commands."""

    def __init__(self):
        self._current_directory = os.getcwd()

    def get_current_directory(self) -> str:
        """Get the current working directory."""
        return self._current_directory

    async def execute(self, command: str, timeout: int = 60) -> ToolResult:
        """Execute a bash command."""
        try:
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
                return ToolResult.error_result(
                    f"Command timed out after {timeout} seconds"
                )

            # Decode output
            stdout_str = stdout.decode("utf-8", errors="replace").strip()
            stderr_str = stderr.decode("utf-8", errors="replace").strip()

            # Update current directory if cd was used
            if command.strip().startswith("cd "):
                await self._update_current_directory()

            # Return result
            if process.returncode == 0:
                output = stdout_str if stdout_str else "Command executed successfully"
                if stderr_str:
                    output += f"\n[stderr]: {stderr_str}"
                return ToolResult.success_result(output)
            else:
                error_msg = (
                    stderr_str
                    if stderr_str
                    else f"Command failed with exit code {process.returncode}"
                )
                if stdout_str:
                    error_msg += f"\n[stdout]: {stdout_str}"
                return ToolResult.error_result(error_msg)

        except Exception as e:
            return ToolResult.error_result(f"Failed to execute command: {str(e)}")

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
