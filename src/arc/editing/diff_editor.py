"""Diff-based editing strategy inspired by Aider's patch format."""

import re
from pathlib import Path

from arc.editing.base import EditInstruction, EditResult, EditStrategy


class DiffEditor(EditStrategy):
    """Strategy for applying unified diff patches."""

    def __init__(self):
        super().__init__("diff_patch")

    async def can_handle(self, instruction: EditInstruction) -> bool:
        """Check if this strategy can handle the instruction."""
        # Check if instruction contains diff-like content
        content = instruction.search_text or instruction.new_content or ""
        return (
            "@@" in content
            or content.startswith("---")
            or content.startswith("+++")
            or ("<<<<<<< SEARCH" in content and ">>>>>>> REPLACE" in content)
        )

    async def apply_edit(self, instruction: EditInstruction) -> EditResult:
        """Apply diff-based edits."""
        try:
            file_path = Path(instruction.file_path)

            if not file_path.exists():
                return EditResult.failure_result(
                    f"File does not exist: {instruction.file_path}", strategy=self.name
                )

            content = file_path.read_text(encoding="utf-8")

            # Try different diff formats
            if "<<<<<<< SEARCH" in (instruction.search_text or ""):
                result = await self._apply_search_replace_block(content, instruction)
            elif "@@" in (instruction.search_text or ""):
                result = await self._apply_unified_diff(content, instruction)
            else:
                result = await self._apply_simple_diff(content, instruction)

            if result.success:
                file_path.write_text(result.new_content, encoding="utf-8")
                return EditResult.success_result(
                    f"Applied diff patch to {instruction.file_path}",
                    changes=result.changes,
                    strategy=self.name,
                )
            else:
                return EditResult.failure_result(
                    f"Failed to apply diff patch: {result.error}", strategy=self.name
                )

        except Exception as e:
            return EditResult.failure_result(
                f"Error applying diff patch: {str(e)}", error=str(e), strategy=self.name
            )

    async def _apply_search_replace_block(
        self, content: str, instruction: EditInstruction
    ) -> "DiffResult":
        """Apply Aider-style SEARCH/REPLACE blocks."""
        search_text = instruction.search_text

        # Parse SEARCH/REPLACE blocks
        blocks = re.findall(
            r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
            search_text,
            re.DOTALL,
        )

        if not blocks:
            return DiffResult(False, content, 0, "No SEARCH/REPLACE blocks found")

        new_content = content
        total_changes = 0

        for search, replace in blocks:
            if search in new_content:
                new_content = new_content.replace(search, replace, 1)
                total_changes += 1
            else:
                # Try fuzzy matching
                fuzzy_result = await self._fuzzy_replace_in_content(
                    new_content, search, replace
                )
                if fuzzy_result.success:
                    new_content = fuzzy_result.new_content
                    total_changes += fuzzy_result.changes

        success = total_changes > 0
        return DiffResult(
            success, new_content, total_changes, "" if success else "No matches found"
        )

    async def _apply_unified_diff(
        self, content: str, instruction: EditInstruction
    ) -> "DiffResult":
        """Apply unified diff format patches."""
        diff_text = instruction.search_text

        # Parse unified diff headers
        lines = diff_text.split("\n")
        hunks = []
        current_hunk = []

        for line in lines:
            if line.startswith("@@"):
                if current_hunk:
                    hunks.append(current_hunk)
                current_hunk = [line]
            elif line.startswith((" ", "-", "+")):
                current_hunk.append(line)

        if current_hunk:
            hunks.append(current_hunk)

        if not hunks:
            return DiffResult(False, content, 0, "No valid diff hunks found")

        # Apply hunks in reverse order to maintain line numbers
        content_lines = content.split("\n")
        total_changes = 0

        for hunk in reversed(hunks):
            result = self._apply_hunk(content_lines, hunk)
            if result.success:
                content_lines = result.new_content.split("\n")
                total_changes += result.changes

        success = total_changes > 0
        return DiffResult(
            success,
            "\n".join(content_lines),
            total_changes,
            "" if success else "No hunks applied",
        )

    def _apply_hunk(
        self, content_lines: list[str], hunk_lines: list[str]
    ) -> "DiffResult":
        """Apply a single diff hunk."""
        if not hunk_lines or not hunk_lines[0].startswith("@@"):
            return DiffResult(False, "\n".join(content_lines), 0, "Invalid hunk format")

        # Parse hunk header: @@ -start,count +start,count @@
        header = hunk_lines[0]
        match = re.match(r"@@ -(\d+),?(\d+)? \+(\d+),?(\d+)? @@", header)
        if not match:
            return DiffResult(False, "\n".join(content_lines), 0, "Invalid hunk header")

        old_start = int(match.group(1)) - 1  # Convert to 0-based

        # Apply changes
        new_lines = []
        old_idx = 0

        # Copy lines before the hunk
        new_lines.extend(content_lines[:old_start])

        # Apply hunk changes
        for line in hunk_lines[1:]:
            if line.startswith(" "):
                # Context line
                new_lines.append(line[1:])
                old_idx += 1
            elif line.startswith("-"):
                # Deleted line - skip it
                old_idx += 1
            elif line.startswith("+"):
                # Added line
                new_lines.append(line[1:])

        # Copy remaining lines
        new_lines.extend(content_lines[old_start + old_idx :])

        return DiffResult(True, "\n".join(new_lines), 1, "")

    async def _apply_simple_diff(
        self, content: str, instruction: EditInstruction
    ) -> "DiffResult":
        """Apply simple before/after diff."""
        if (
            instruction.old_content
            and instruction.new_content
            and instruction.old_content in content
        ):
            new_content = content.replace(
                instruction.old_content, instruction.new_content, 1
            )
            return DiffResult(True, new_content, 1, "")

        return DiffResult(False, content, 0, "Simple diff pattern not found")

    async def _fuzzy_replace_in_content(
        self, content: str, search: str, replace: str
    ) -> "DiffResult":
        """Fuzzy matching replacement for diff blocks."""
        lines = content.split("\n")
        search_lines = search.split("\n")

        # Find best matching sequence
        best_match_start = None
        best_ratio = 0.0

        for i in range(len(lines) - len(search_lines) + 1):
            segment = lines[i : i + len(search_lines)]
            ratio = self._calculate_similarity(
                "\n".join(search_lines), "\n".join(segment)
            )

            if ratio > best_ratio and ratio >= 0.75:  # 75% similarity threshold
                best_ratio = ratio
                best_match_start = i

        if best_match_start is not None:
            replacement_lines = replace.split("\n")
            new_lines = (
                lines[:best_match_start]
                + replacement_lines
                + lines[best_match_start + len(search_lines) :]
            )

            return DiffResult(True, "\n".join(new_lines), 1, "")

        return DiffResult(False, content, 0, "No fuzzy match found")

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text blocks."""
        import difflib

        return difflib.SequenceMatcher(None, text1.strip(), text2.strip()).ratio()


class DiffResult:
    """Internal result class for diff operations."""

    def __init__(self, success: bool, new_content: str, changes: int, error: str = ""):
        self.success = success
        self.new_content = new_content
        self.changes = changes
        self.error = error
