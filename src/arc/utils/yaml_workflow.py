"""Generic interactive YAML confirmation workflow for Arc generators.

This module provides reusable components for interactive YAML generation workflows,
including state management, user confirmation, and editing capabilities.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any


class YamlStateManager:
    """Manages YAML state using temporary files to avoid repeated content copying."""

    def __init__(self, yaml_suffix: str = ".yaml", prefix: str = "arc_"):
        """Initialize YAML state manager.

        Args:
            yaml_suffix: File suffix for temporary YAML files
            prefix: Prefix for temporary file names
        """
        self.temp_file = None
        self.context = {}
        self.conversation_history = None
        self.yaml_suffix = yaml_suffix
        self.prefix = prefix

    def save_yaml(
        self,
        yaml_content: str,
        context: dict[str, Any],
        conversation_history: list[dict[str, str]] | None = None,
    ) -> Path:
        """Save YAML content to temporary file and store context for editing.

        Args:
            yaml_content: YAML content to save
            context: Arbitrary context dictionary for future editing
            conversation_history: Optional conversation history for iterative editing

        Returns:
            Path to the temporary file
        """
        if not self.temp_file:
            # Create temp file with configured suffix and prefix
            name_part = context.get("name", "spec")
            self.temp_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
                mode="w+",
                suffix=self.yaml_suffix,
                prefix=f"{self.prefix}{name_part}_",
                delete=False,
            )

        # Write YAML content to file
        self.temp_file.seek(0)
        self.temp_file.truncate()
        self.temp_file.write(yaml_content)
        self.temp_file.flush()

        # Store context and conversation history for future editing
        self.context = context.copy()
        self.conversation_history = (
            conversation_history.copy() if conversation_history else None
        )

        return Path(self.temp_file.name)

    def get_yaml(self) -> str:
        """Read current YAML content from file.

        Returns:
            Current YAML content as string
        """
        if not self.temp_file:
            return ""

        self.temp_file.seek(0)
        return self.temp_file.read()

    def get_context(self) -> dict[str, Any]:
        """Get stored context dictionary.

        Returns:
            Copy of stored context dictionary
        """
        return self.context.copy()

    def get_conversation_history(self) -> list[dict[str, str]] | None:
        """Get stored conversation history.

        Returns:
            Copy of stored conversation history or None
        """
        return self.conversation_history.copy() if self.conversation_history else None

    def cleanup(self):
        """Clean up temporary file and reset state."""
        if self.temp_file:
            try:
                self.temp_file.close()
                os.unlink(self.temp_file.name)
            except (OSError, AttributeError):
                pass
            finally:
                self.temp_file = None
                self.context = {}
                self.conversation_history = None


class YamlEditorHelper:
    """Helper utilities for launching external editors and handling YAML editing."""

    @staticmethod
    def detect_editor() -> str | None:
        """Detect available system editor.

        Returns:
            Editor command if found, None otherwise
        """
        # Check environment variables first
        editor = os.environ.get("EDITOR")
        if editor:
            return editor

        # Try common editors in order of preference
        import shutil

        common_editors = ["code", "nano", "vim", "vi", "emacs"]
        for ed in common_editors:
            if shutil.which(ed):
                return ed

        return None

    @staticmethod
    async def edit_with_system_editor(
        yaml_content: str,
        header_comment: str = "# Edit and save to confirm changes\n\n",
        yaml_suffix: str = ".yaml",
    ) -> str | None:
        """Launch system editor for YAML editing.

        Args:
            yaml_content: YAML content to edit
            header_comment: Optional header comment to add to file
            yaml_suffix: File suffix for temporary file

        Returns:
            Edited YAML content if successful, None if cancelled or failed
        """
        editor = YamlEditorHelper.detect_editor()
        if not editor:
            return None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=yaml_suffix, prefix="arc_edit_", delete=False
            ) as f:
                # Add helpful header
                f.write(header_comment + yaml_content)
                temp_path = f.name

            # Create subprocess without redirecting stdout/stderr
            # so editor can interact with terminal
            process = await asyncio.create_subprocess_exec(editor, temp_path)

            await process.wait()

            if process.returncode == 0:
                with open(temp_path) as f:
                    content = f.read()
                    # Strip header if present
                    if content.startswith(header_comment):
                        content = content[len(header_comment) :]
                    return content.strip()
            else:
                return None

        except Exception:
            return None
        finally:
            # Cleanup temp file
            if "temp_path" in locals():
                with contextlib.suppress(OSError):
                    os.unlink(temp_path)


class YamlConfirmationWorkflow:
    """Interactive YAML confirmation workflow with editing support.

    This workflow provides a reusable pattern for:
    - Displaying YAML preview to users
    - Offering choices: save, edit with AI, edit manually, or cancel
    - Validating YAML after edits
    - Managing state across editing iterations
    """

    def __init__(
        self,
        validator_func: Callable[[str], list[str]],
        editor_func: Callable[
            [str, str, dict[str, Any], list[dict[str, str]] | None],
            Awaitable[tuple[str | None, list[dict[str, str]] | None]],
        ],
        ui_interface,
        yaml_type_name: str = "specification",
        yaml_suffix: str = ".yaml",
    ):
        """Initialize confirmation workflow.

        Args:
            validator_func: Function that takes YAML string and returns list of errors
            editor_func: Async function that takes (yaml, feedback, context,
                        conversation_history) and returns tuple of (edited_yaml,
                        updated_history)
            ui_interface: InteractiveInterface instance for user interaction
            yaml_type_name: Display name for YAML type (e.g., "model", "trainer")
            yaml_suffix: File suffix for temporary files
        """
        self.validator = validator_func
        self.editor = editor_func
        self.ui = ui_interface
        self.yaml_type_name = yaml_type_name
        self.yaml_suffix = yaml_suffix
        self.state_manager = YamlStateManager(
            yaml_suffix=yaml_suffix, prefix=f"arc_{yaml_type_name}_"
        )

    async def run_workflow(
        self,
        initial_yaml: str,
        context: dict[str, Any],
        output_path: str | None = None,
        initial_conversation_history: list[dict[str, str]] | None = None,
    ) -> tuple[bool, str]:
        """Run interactive confirmation workflow.

        Args:
            initial_yaml: Initial YAML content to confirm
            context: Context dictionary for editing (passed to editor_func)
            output_path: Optional output file path for display
            initial_conversation_history: Optional conversation history from
                initial generation

        Returns:
            Tuple of (approved: bool, final_yaml: str)
        """
        yaml_content = initial_yaml

        # Save initial state with conversation history
        self.state_manager.save_yaml(
            yaml_content, context, initial_conversation_history
        )

        while True:
            # Display preview
            await self._display_preview(yaml_content, output_path)

            # Get user choice
            options = [
                (
                    "save",
                    f"Accept - Save and use this {self.yaml_type_name}",
                ),
                ("edit_ai", "Iterate - Provide feedback to refine details"),
                ("edit_manual", "Edit manually - Modify in text editor"),
                ("cancel", f"Cancel - Discard this {self.yaml_type_name}"),
            ]

            if self.ui:
                try:
                    # Escape watcher suspension happens automatically in
                    # get_choice_async()
                    choice = await self.ui._printer.get_choice_async(
                        options, default="save"
                    )
                finally:
                    # Reset prompt session after choice
                    self.ui._printer.reset_prompt_session()
            else:
                # Fallback for non-UI usage
                choice = self._get_choice_fallback(options)

            if choice == "save":
                return True, yaml_content

            elif choice == "edit_ai":
                # AI-assisted editing with conversation history
                conversation_history = self.state_manager.get_conversation_history()
                edited_yaml, updated_history = await self._edit_with_ai(
                    yaml_content, context, conversation_history
                )
                if edited_yaml is None:
                    continue  # Edit cancelled, show confirmation again

                # Validate edited YAML
                errors = self.validator(edited_yaml)
                if errors:
                    error_msg = (
                        f"❌ Validation error in AI-edited {self.yaml_type_name}: "
                        f"{'; '.join(errors)}"
                    )
                    instruction_msg = "Please try different feedback or cancel."
                    if self.ui:
                        self.ui.show_system_error(error_msg)
                        self.ui.show_info(instruction_msg)
                    continue

                yaml_content = edited_yaml
                # Save updated state with new conversation history
                self.state_manager.save_yaml(yaml_content, context, updated_history)
                # Reset prompt session after nested AI editing prompts
                if self.ui:
                    self.ui._printer.reset_prompt_session()
                continue

            elif choice == "edit_manual":
                # Manual editing with external editor
                edited_yaml = await self._edit_manually(yaml_content)
                if edited_yaml is None:
                    continue  # Edit cancelled, show confirmation again

                # Validate edited YAML
                errors = self.validator(edited_yaml)
                if errors:
                    error_msg = (
                        f"❌ Validation error in edited {self.yaml_type_name}: "
                        f"{'; '.join(errors)}"
                    )
                    instruction_msg = "Please edit again or cancel."
                    if self.ui:
                        self.ui.show_system_error(error_msg)
                        self.ui.show_info(instruction_msg)
                    continue

                yaml_content = edited_yaml
                self.state_manager.save_yaml(yaml_content, context)
                # Reset prompt session after manual editing
                if self.ui:
                    self.ui._printer.reset_prompt_session()
                continue

            elif choice == "cancel" or choice == "__esc__":
                return False, yaml_content

            else:
                # Invalid choice
                if self.ui:
                    self.ui.show_system_error("❌ Invalid choice. Please try again.")
                continue

    async def _display_preview(
        self, yaml_content: str, output_path: str | None = None
    ) -> None:
        """Display formatted YAML preview.

        Args:
            yaml_content: YAML content to display
            output_path: Optional output file path for diff display
        """
        # Display YAML with diff support (title already shown by tool before generation)
        if self.ui:
            self.ui._printer.display_yaml_with_diff(yaml_content, output_path)

    async def _edit_with_ai(
        self,
        yaml_content: str,
        context: dict[str, Any],
        conversation_history: list[dict[str, str]] | None = None,
    ) -> tuple[str | None, list[dict[str, str]] | None]:
        """Collect user feedback and edit YAML with AI.

        Args:
            yaml_content: Current YAML content
            context: Context dictionary
            conversation_history: Optional conversation history for
                continuing conversation

        Returns:
            Tuple of (edited_yaml, updated_conversation_history) or
            (None, None) if cancelled
        """
        # Collect user feedback
        if self.ui:
            self.ui.show_info(
                f"🤖 Describe the changes you want to make to the "
                f"{self.yaml_type_name}:"
            )
            self.ui.show_info(
                "Examples: 'add dropout layers', 'change to 5 classes', "
                "'use different activation'"
            )

            try:
                feedback = await self.ui._printer.get_input_async(
                    "What changes do you want? "
                )

                if not feedback.strip():
                    if self.ui:
                        self.ui.show_system_error(
                            "❌ No feedback provided. Edit cancelled."
                        )
                    return None, None

            except Exception as e:
                if self.ui:
                    self.ui.show_system_error(f"❌ Error collecting feedback: {e}")
                return None, None
        else:
            # Fallback for non-UI usage
            feedback = input("What changes do you want? ").strip()
            if not feedback:
                return None, None

        # Show feedback confirmation
        if self.ui:
            self.ui.show_info(f"🔄 AI will apply: {feedback}")

        # Call the editor function with conversation history
        try:
            edited_yaml, updated_history = await self.editor(
                yaml_content, feedback, context, conversation_history
            )

            if edited_yaml and self.ui:
                self.ui.show_info("✅ AI has applied your requested changes.")

            return edited_yaml, updated_history

        except Exception as e:
            error_msg = f"❌ AI editing failed: {str(e)}"
            if self.ui:
                self.ui.show_system_error(error_msg)
            return None, None

    async def _edit_manually(self, yaml_content: str) -> str | None:
        """Launch external editor for manual YAML editing.

        Args:
            yaml_content: Current YAML content

        Returns:
            Edited YAML or None if cancelled or failed
        """
        editor = YamlEditorHelper.detect_editor()

        if not editor:
            if self.ui:
                self.ui.show_system_error(
                    "❌ No text editor found. Please set $EDITOR or install nano/vim."
                )
                self.ui.show_info("💡 You can:")
                self.ui.show_info(
                    "   1. Set EDITOR environment variable: export EDITOR=nano"
                )
                self.ui.show_info("   2. Install a text editor: brew install nano")
                self.ui.show_info("   3. Cancel this edit and try again")
            return None

        if self.ui:
            self.ui.show_info(f"🔧 Opening YAML in {editor}...")

        header = (
            f"# Arc-Graph {self.yaml_type_name.title()} Specification\n"
            f"# Edit and save to confirm changes\n\n"
        )

        edited = await YamlEditorHelper.edit_with_system_editor(
            yaml_content, header, self.yaml_suffix
        )

        if edited is None and self.ui:
            self.ui.show_system_error("❌ Editor failed or was cancelled")

        return edited

    def _get_choice_fallback(self, options: list[tuple[str, str]]) -> str:
        """Fallback choice selection for non-UI usage.

        Args:
            options: List of (key, label) tuples

        Returns:
            Selected option key
        """
        fallback_text = f"\n Confirm {self.yaml_type_name} specification"
        for i, (_, label) in enumerate(options, 1):
            fallback_text += f"\n  {i}. {label}"

        print(fallback_text)
        choice_input = input(f"Enter choice (1-{len(options)}): ").strip()

        choice_map = {str(i): key for i, (key, _) in enumerate(options, 1)}
        return choice_map.get(choice_input, "cancel")

    def cleanup(self):
        """Clean up temporary files and state."""
        self.state_manager.cleanup()
