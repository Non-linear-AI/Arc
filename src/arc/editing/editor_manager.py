"""Multi-strategy editor manager with intelligent strategy selection."""

import logging
from pathlib import Path
from typing import Any

from .base import EditInstruction, EditResult, EditStrategy
from .diff_editor import DiffEditor
from .search_replace import SearchReplaceEditor
from .whole_file import WholeFileEditor

logger = logging.getLogger(__name__)


class EditorManager:
    """Orchestrates multiple editing strategies with intelligent selection."""

    def __init__(self):
        self.strategies: list[EditStrategy] = [
            DiffEditor(),
            SearchReplaceEditor(),
            WholeFileEditor(),
        ]
        self._strategy_cache: dict[str, EditStrategy] = {}

    async def apply_edit(self, instruction: EditInstruction) -> EditResult:
        """Apply edit using the best available strategy."""

        # Strategy 1: Try cached strategy first (if we've seen this pattern before)
        cached_strategy = self._get_cached_strategy(instruction)
        if cached_strategy:
            if await cached_strategy.can_handle(instruction):
                result = await cached_strategy.apply_edit(instruction)
                cached_strategy.record_result(result)
                if result.success:
                    return result

        # Strategy 2: Try strategies in order of capability
        capable_strategies = []
        for strategy in self.strategies:
            if await strategy.can_handle(instruction):
                capable_strategies.append(strategy)

        if not capable_strategies:
            return EditResult.failure_result(
                f"No strategy can handle edit instruction for {instruction.file_path}",
                strategy="manager",
            )

        # Strategy 3: Select best strategy based on context
        best_strategy = await self._select_best_strategy(
            instruction, capable_strategies
        )
        result = await best_strategy.apply_edit(instruction)
        best_strategy.record_result(result)

        # Cache successful strategy for similar future edits
        if result.success:
            self._cache_strategy(instruction, best_strategy)

        # Strategy 4: Fallback to other strategies if primary fails
        if not result.success and len(capable_strategies) > 1:
            logger.info(
                f"Primary strategy {best_strategy.name} failed, trying fallbacks"
            )

            for fallback_strategy in capable_strategies:
                if fallback_strategy == best_strategy:
                    continue

                fallback_result = await fallback_strategy.apply_edit(instruction)
                fallback_strategy.record_result(fallback_result)

                if fallback_result.success:
                    fallback_result.fallback_used = True
                    fallback_result.message += f" (fallback: {fallback_strategy.name})"
                    return fallback_result

        return result

    async def _select_best_strategy(
        self, instruction: EditInstruction, strategies: list[EditStrategy]
    ) -> EditStrategy:
        """Select the best strategy for the given instruction."""
        if len(strategies) == 1:
            return strategies[0]

        # Prioritize based on instruction type and file context
        strategy_scores = {}

        for strategy in strategies:
            score = 0.0

            # Base score from success rate
            score += strategy.get_success_rate() * 0.3

            # Strategy-specific scoring
            if isinstance(strategy, DiffEditor):
                # Prefer for diff-like content
                content = instruction.search_text or instruction.new_content or ""
                if any(
                    marker in content
                    for marker in ["@@", "<<<<<<< SEARCH", "---", "+++"]
                ):
                    score += 0.5

            elif isinstance(strategy, WholeFileEditor):
                # Prefer for new files or major changes
                if instruction.create_if_missing:
                    score += 0.4
                elif instruction.new_content and await self._is_major_change(
                    instruction
                ):
                    score += 0.3

            elif isinstance(strategy, SearchReplaceEditor):
                # Good default for most text replacements
                if instruction.search_text and instruction.replacement_text:
                    score += 0.2

            strategy_scores[strategy] = score

        # Return strategy with highest score
        return max(strategy_scores.keys(), key=lambda s: strategy_scores[s])

    async def _is_major_change(self, instruction: EditInstruction) -> bool:
        """Determine if this is a major change that benefits from whole file replacement."""
        try:
            file_path = Path(instruction.file_path)
            if not file_path.exists() or not instruction.new_content:
                return False

            current_content = file_path.read_text(encoding="utf-8")
            new_content = instruction.new_content

            # Use difflib to calculate change ratio
            import difflib

            similarity = difflib.SequenceMatcher(
                None, current_content, new_content
            ).ratio()

            # Consider it major if less than 50% similar or file is small
            return similarity < 0.5 or len(current_content) < 1024

        except Exception:
            return False

    def _get_cached_strategy(self, instruction: EditInstruction) -> EditStrategy | None:
        """Get cached strategy for similar edit patterns."""
        # Simple cache key based on file extension and edit type
        file_path = Path(instruction.file_path)
        cache_key = f"{file_path.suffix}_{self._get_edit_type(instruction)}"
        return self._strategy_cache.get(cache_key)

    def _cache_strategy(
        self, instruction: EditInstruction, strategy: EditStrategy
    ) -> None:
        """Cache successful strategy for future similar edits."""
        file_path = Path(instruction.file_path)
        cache_key = f"{file_path.suffix}_{self._get_edit_type(instruction)}"
        self._strategy_cache[cache_key] = strategy

    def _get_edit_type(self, instruction: EditInstruction) -> str:
        """Classify the type of edit for caching purposes."""
        if instruction.new_content and not instruction.search_text:
            return "whole_file"
        elif instruction.search_text and "@@" in instruction.search_text:
            return "diff_patch"
        elif instruction.search_text and "<<<<<<< SEARCH" in instruction.search_text:
            return "search_replace_block"
        elif instruction.search_text and instruction.replacement_text:
            return "search_replace"
        else:
            return "unknown"

    async def apply_multiple_edits(
        self, instructions: list[EditInstruction]
    ) -> list[EditResult]:
        """Apply multiple edits efficiently."""
        # Group by file to optimize file I/O
        file_groups = {}
        for instruction in instructions:
            if instruction.file_path not in file_groups:
                file_groups[instruction.file_path] = []
            file_groups[instruction.file_path].append(instruction)

        results = []

        # Process each file's edits together
        for file_path, file_instructions in file_groups.items():
            if len(file_instructions) == 1:
                # Single edit - use normal process
                result = await self.apply_edit(file_instructions[0])
                results.append(result)
            else:
                # Multiple edits to same file - optimize
                file_results = await self._apply_batched_edits(file_instructions)
                results.extend(file_results)

        return results

    async def _apply_batched_edits(
        self, instructions: list[EditInstruction]
    ) -> list[EditResult]:
        """Apply multiple edits to the same file efficiently."""
        results = []

        # Try to batch compatible edits
        whole_file_edits = [
            i for i in instructions if i.new_content and not i.search_text
        ]
        search_replace_edits = [
            i for i in instructions if i.search_text and i.replacement_text
        ]

        if whole_file_edits:
            # If any whole file edit, use the last one (most complete)
            result = await self.apply_edit(whole_file_edits[-1])
            results.extend([result] * len(instructions))
        else:
            # Apply search/replace edits sequentially
            for instruction in instructions:
                result = await self.apply_edit(instruction)
                results.append(result)

        return results

    def get_strategy_stats(self) -> dict[str, dict[str, Any]]:
        """Get performance statistics for all strategies."""
        stats = {}

        for strategy in self.strategies:
            stats[strategy.name] = {
                "success_count": strategy.success_count,
                "failure_count": strategy.failure_count,
                "success_rate": strategy.get_success_rate(),
                "total_operations": strategy.success_count + strategy.failure_count,
            }

        return stats

    async def test_strategies(self, instruction: EditInstruction) -> dict[str, bool]:
        """Test which strategies can handle a given instruction (for debugging)."""
        results = {}

        for strategy in self.strategies:
            try:
                can_handle = await strategy.can_handle(instruction)
                results[strategy.name] = can_handle
            except Exception as e:
                results[strategy.name] = f"Error: {str(e)}"

        return results
