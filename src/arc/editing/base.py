"""Base classes for editing strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EditResult:
    """Result of an edit operation."""

    success: bool
    message: str
    changes_made: int = 0
    strategy_used: str = ""
    fallback_used: bool = False
    error_details: str | None = None

    @classmethod
    def success_result(
        cls, message: str, changes: int = 1, strategy: str = ""
    ) -> "EditResult":
        """Create a successful edit result."""
        return cls(
            success=True, message=message, changes_made=changes, strategy_used=strategy
        )

    @classmethod
    def failure_result(
        cls, message: str, error: str = "", strategy: str = ""
    ) -> "EditResult":
        """Create a failed edit result."""
        return cls(
            success=False, message=message, strategy_used=strategy, error_details=error
        )


@dataclass
class EditInstruction:
    """Represents an edit instruction."""

    file_path: str
    old_content: str | None = None
    new_content: str | None = None
    search_text: str | None = None
    replacement_text: str | None = None
    line_number: int | None = None
    replace_all: bool = False
    create_if_missing: bool = False


class EditStrategy(ABC):
    """Abstract base class for editing strategies."""

    def __init__(self, name: str):
        self.name = name
        self.success_count = 0
        self.failure_count = 0

    @abstractmethod
    async def can_handle(self, instruction: EditInstruction) -> bool:
        """Check if this strategy can handle the given instruction."""
        pass

    @abstractmethod
    async def apply_edit(self, instruction: EditInstruction) -> EditResult:
        """Apply the edit instruction using this strategy."""
        pass

    def get_success_rate(self) -> float:
        """Get the success rate of this strategy."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    def record_result(self, result: EditResult) -> None:
        """Record the result for success rate tracking."""
        if result.success:
            self.success_count += 1
        else:
            self.failure_count += 1


class FuzzyMatcher:
    """Fuzzy matching utilities for fallback strategies."""

    @staticmethod
    def find_best_match(
        target: str, candidates: list[str], threshold: float = 0.8
    ) -> str | None:
        """Find the best fuzzy match from candidates."""
        import difflib

        best_match = None
        best_ratio = threshold

        for candidate in candidates:
            ratio = difflib.SequenceMatcher(
                None, target.lower(), candidate.lower()
            ).ratio()
            if ratio > best_ratio:
                best_match = candidate
                best_ratio = ratio

        return best_match

    @staticmethod
    def get_line_matches(
        target_lines: list[str], file_lines: list[str], threshold: float = 0.8
    ) -> list[int]:
        """Find line numbers that match target lines with fuzzy matching."""
        matches = []

        for target_line in target_lines:
            for i, file_line in enumerate(file_lines):
                if FuzzyMatcher._lines_similar(target_line, file_line, threshold):
                    matches.append(i)
                    break

        return matches

    @staticmethod
    def _lines_similar(line1: str, line2: str, threshold: float) -> bool:
        """Check if two lines are similar enough."""
        import difflib

        return (
            difflib.SequenceMatcher(None, line1.strip(), line2.strip()).ratio()
            >= threshold
        )
