"""Whole file editing strategy."""

from pathlib import Path

from .base import EditInstruction, EditResult, EditStrategy


class WholeFileEditor(EditStrategy):
    """Strategy for replacing entire file contents."""

    def __init__(self):
        super().__init__("whole_file")

    async def can_handle(self, instruction: EditInstruction) -> bool:
        """Check if this strategy can handle the instruction."""
        return (
            instruction.new_content is not None
            and len(instruction.new_content.strip()) > 0
        )

    async def apply_edit(self, instruction: EditInstruction) -> EditResult:
        """Replace entire file content."""
        try:
            file_path = Path(instruction.file_path)

            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if file exists and get size info
            if file_path.exists():
                old_size = file_path.stat().st_size
                action = "replaced"
            else:
                if not instruction.create_if_missing:
                    return EditResult.failure_result(
                        f"File does not exist and create_if_missing is False: "
                        f"{instruction.file_path}",
                        strategy=self.name,
                    )
                old_size = 0
                action = "created"

            # Write new content
            file_path.write_text(instruction.new_content, encoding="utf-8")
            new_size = file_path.stat().st_size

            # Calculate meaningful change metric
            size_change = new_size - old_size
            size_change_str = (
                f" ({size_change:+d} bytes)" if old_size > 0 else f" ({new_size} bytes)"
            )

            return EditResult.success_result(
                f"Successfully {action} {instruction.file_path}{size_change_str}",
                changes=1,
                strategy=self.name,
            )

        except Exception as e:
            return EditResult.failure_result(
                f"Failed to write file {instruction.file_path}: {str(e)}",
                error=str(e),
                strategy=self.name,
            )

    def estimate_tokens_saved(self, old_content: str, new_content: str) -> int:
        """Estimate tokens saved by using whole file vs incremental edits."""
        # Rough estimation: 1 token ≈ 4 characters
        old_tokens = len(old_content) // 4
        new_tokens = len(new_content) // 4

        # Whole file replacement is efficient when making large changes
        change_ratio = abs(new_tokens - old_tokens) / max(old_tokens, 1)

        if change_ratio > 0.5:  # More than 50% change
            return max(0, old_tokens - new_tokens)
        else:
            return 0  # Not efficient for small changes

    async def should_use_over_search_replace(
        self, instruction: EditInstruction
    ) -> bool:
        """Determine if whole file replacement is better than search/replace."""
        if not Path(instruction.file_path).exists():
            return True

        try:
            old_content = Path(instruction.file_path).read_text(encoding="utf-8")
            new_content = instruction.new_content or ""

            # Use whole file if:
            # 1. File is small (< 1KB)
            # 2. Making large changes (>50% different)
            # 3. Multiple search/replace operations would be needed

            file_size = len(old_content)
            if file_size < 1024:  # Small file
                return True

            # Calculate change percentage
            import difflib

            similarity = difflib.SequenceMatcher(None, old_content, new_content).ratio()

            return similarity < 0.5  # More than 50% different

        except Exception:
            return False  # Default to search/replace if we can't analyze
