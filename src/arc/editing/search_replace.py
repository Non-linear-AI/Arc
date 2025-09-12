"""Search and replace editing strategy."""

from pathlib import Path

from .base import EditInstruction, EditResult, EditStrategy


class SearchReplaceEditor(EditStrategy):
    """Enhanced search and replace editor with fuzzy matching fallbacks."""

    def __init__(self):
        super().__init__("search_replace")

    async def can_handle(self, instruction: EditInstruction) -> bool:
        """Check if this strategy can handle the instruction."""
        return (
            instruction.search_text is not None
            and instruction.replacement_text is not None
            and Path(instruction.file_path).exists()
        )

    async def apply_edit(self, instruction: EditInstruction) -> EditResult:
        """Apply search and replace edit with fallbacks."""
        try:
            file_path = Path(instruction.file_path)

            if not file_path.exists():
                return EditResult.failure_result(
                    f"File does not exist: {instruction.file_path}", strategy=self.name
                )

            # Read file content
            content = file_path.read_text(encoding="utf-8", errors="replace")
            original_content = content

            # Strategy 1: Exact match
            if instruction.search_text in content:
                result = self._apply_exact_replacement(content, instruction)
                if result.success:
                    file_path.write_text(result.new_content, encoding="utf-8")
                    return EditResult.success_result(
                        f"Applied exact search/replace in {instruction.file_path}",
                        changes=result.changes,
                        strategy=self.name,
                    )

            # Strategy 2: Fuzzy matching fallback
            fuzzy_result = await self._apply_fuzzy_replacement(content, instruction)
            if fuzzy_result.success:
                file_path.write_text(fuzzy_result.new_content, encoding="utf-8")
                return EditResult.success_result(
                    f"Applied fuzzy search/replace in {instruction.file_path}",
                    changes=fuzzy_result.changes,
                    strategy=self.name,
                    fallback_used=True,
                )

            # Strategy 3: Line-by-line fuzzy matching
            line_result = await self._apply_line_fuzzy_replacement(content, instruction)
            if line_result.success:
                file_path.write_text(line_result.new_content, encoding="utf-8")
                return EditResult.success_result(
                    f"Applied line-based fuzzy replacement in {instruction.file_path}",
                    changes=line_result.changes,
                    strategy=self.name,
                    fallback_used=True,
                )

            return EditResult.failure_result(
                f"Could not find text to replace: '{instruction.search_text[:100]}...'",
                strategy=self.name,
            )

        except Exception as e:
            return EditResult.failure_result(
                f"Error applying search/replace: {str(e)}",
                error=str(e),
                strategy=self.name,
            )

    def _apply_exact_replacement(
        self, content: str, instruction: EditInstruction
    ) -> "ReplacementResult":
        """Apply exact string replacement."""
        search_text = instruction.search_text
        replacement_text = instruction.replacement_text

        if instruction.replace_all:
            new_content = content.replace(search_text, replacement_text)
            changes = content.count(search_text)
        else:
            new_content = content.replace(search_text, replacement_text, 1)
            changes = 1 if search_text in content else 0

        return ReplacementResult(
            success=changes > 0, new_content=new_content, changes=changes
        )

    async def _apply_fuzzy_replacement(
        self, content: str, instruction: EditInstruction
    ) -> "ReplacementResult":
        """Apply fuzzy matching replacement."""
        lines = content.split("\n")
        search_lines = instruction.search_text.split("\n")

        # Find best matching sequence
        best_match_start = None
        best_ratio = 0.0

        for i in range(len(lines) - len(search_lines) + 1):
            segment = lines[i : i + len(search_lines)]
            ratio = self._calculate_sequence_similarity(search_lines, segment)

            if ratio > best_ratio and ratio >= 0.7:  # 70% similarity threshold
                best_ratio = ratio
                best_match_start = i

        if best_match_start is not None:
            replacement_lines = instruction.replacement_text.split("\n")
            new_lines = (
                lines[:best_match_start]
                + replacement_lines
                + lines[best_match_start + len(search_lines) :]
            )

            return ReplacementResult(
                success=True, new_content="\n".join(new_lines), changes=1
            )

        return ReplacementResult(success=False, new_content=content, changes=0)

    async def _apply_line_fuzzy_replacement(
        self, content: str, instruction: EditInstruction
    ) -> "ReplacementResult":
        """Apply line-by-line fuzzy replacement."""
        lines = content.split("\n")
        search_lines = instruction.search_text.split("\n")

        if len(search_lines) == 1:
            # Single line replacement with fuzzy matching
            search_line = search_lines[0].strip()
            best_match_idx = None
            best_ratio = 0.0

            for i, line in enumerate(lines):
                ratio = self._calculate_line_similarity(search_line, line.strip())
                if (
                    ratio > best_ratio and ratio >= 0.8
                ):  # 80% similarity for single line
                    best_ratio = ratio
                    best_match_idx = i

            if best_match_idx is not None:
                lines[best_match_idx] = instruction.replacement_text
                return ReplacementResult(
                    success=True, new_content="\n".join(lines), changes=1
                )

        return ReplacementResult(success=False, new_content=content, changes=0)

    def _calculate_sequence_similarity(self, seq1: list[str], seq2: list[str]) -> float:
        """Calculate similarity between two sequences of lines."""
        if len(seq1) != len(seq2):
            return 0.0

        total_ratio = 0.0
        for line1, line2 in zip(seq1, seq2, strict=False):
            total_ratio += self._calculate_line_similarity(line1, line2)

        return total_ratio / len(seq1)

    def _calculate_line_similarity(self, line1: str, line2: str) -> float:
        """Calculate similarity between two lines."""
        import difflib

        return difflib.SequenceMatcher(None, line1.strip(), line2.strip()).ratio()


class ReplacementResult:
    """Internal result class for replacement operations."""

    def __init__(self, success: bool, new_content: str, changes: int):
        self.success = success
        self.new_content = new_content
        self.changes = changes
