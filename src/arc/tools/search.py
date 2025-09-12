"""Search functionality for finding files and text content."""

import os
import re
from pathlib import Path
from typing import Any

from .base import BaseTool, ToolResult


class SearchTool(BaseTool):
    """Unified search tool for finding text content or files."""

    async def execute(
        self,
        query: str,
        search_type: str = "both",
        include_pattern: str | None = None,
        exclude_pattern: str | None = None,
        case_sensitive: bool = False,
        whole_word: bool = False,
        regex: bool = False,
        max_results: int = 50,
        file_types: list[str] | None = None,
        include_hidden: bool = False,
        **kwargs,
    ) -> ToolResult:
        """Execute search based on parameters."""
        try:
            results = []

            if search_type in ["files", "both"]:
                file_results = await self._search_files(
                    query,
                    include_pattern,
                    exclude_pattern,
                    case_sensitive,
                    max_results,
                    file_types,
                    include_hidden,
                )
                results.extend(file_results)

            if search_type in ["text", "both"]:
                text_results = await self._search_text_content(
                    query,
                    include_pattern,
                    exclude_pattern,
                    case_sensitive,
                    whole_word,
                    regex,
                    max_results,
                    file_types,
                    include_hidden,
                )
                results.extend(text_results)

            if not results:
                return ToolResult.success_result(
                    f"No results found for query: '{query}'"
                )

            # Format results
            formatted_results = self._format_results(results, max_results)

            return ToolResult.success_result(
                f"Search results for '{query}':\n{formatted_results}"
            )

        except Exception as e:
            return ToolResult.error_result(f"Search failed: {str(e)}")

    async def _search_files(
        self,
        query: str,
        include_pattern: str | None,
        exclude_pattern: str | None,
        case_sensitive: bool,
        max_results: int,
        file_types: list[str] | None,
        include_hidden: bool,
    ) -> list[dict[str, Any]]:
        """Search for files by name/path."""
        results = []
        search_pattern = query if case_sensitive else query.lower()

        try:
            for root, dirs, files in os.walk("."):
                # Filter hidden directories if not included
                if not include_hidden:
                    dirs[:] = [d for d in dirs if not d.startswith(".")]

                # Skip excluded patterns in directories
                if exclude_pattern:
                    dirs[:] = [
                        d for d in dirs if not self._matches_pattern(d, exclude_pattern)
                    ]

                for file in files:
                    # Skip hidden files if not included
                    if not include_hidden and file.startswith("."):
                        continue

                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, ".")

                    # Apply include/exclude patterns
                    if include_pattern and not self._matches_pattern(
                        file, include_pattern
                    ):
                        continue
                    if exclude_pattern and self._matches_pattern(file, exclude_pattern):
                        continue

                    # Apply file type filter
                    if file_types:
                        file_ext = Path(file).suffix.lstrip(".")
                        if file_ext not in file_types:
                            continue

                    # Check if filename matches search query
                    search_target = file if case_sensitive else file.lower()
                    if search_pattern in search_target:
                        results.append(
                            {
                                "type": "file",
                                "path": relative_path,
                                "name": file,
                                "match": f"Filename contains '{query}'",
                            }
                        )

                        if len(results) >= max_results:
                            return results

        except Exception:
            # Continue with what we found so far
            pass

        return results

    async def _search_text_content(
        self,
        query: str,
        include_pattern: str | None,
        exclude_pattern: str | None,
        case_sensitive: bool,
        whole_word: bool,
        regex: bool,
        max_results: int,
        file_types: list[str] | None,
        include_hidden: bool,
    ) -> list[dict[str, Any]]:
        """Search for text content within files."""
        results = []

        # Compile search pattern
        try:
            if regex:
                flags = 0 if case_sensitive else re.IGNORECASE
                pattern = re.compile(query, flags)
            elif whole_word:
                escaped_query = re.escape(query)
                flags = 0 if case_sensitive else re.IGNORECASE
                pattern = re.compile(rf"\b{escaped_query}\b", flags)
            else:
                if case_sensitive:
                    pattern = query
                else:
                    pattern = query.lower()
        except re.error as e:
            return [{"type": "error", "message": f"Invalid regex pattern: {str(e)}"}]

        try:
            for root, dirs, files in os.walk("."):
                # Filter hidden directories if not included
                if not include_hidden:
                    dirs[:] = [d for d in dirs if not d.startswith(".")]

                # Skip excluded patterns in directories
                if exclude_pattern:
                    dirs[:] = [
                        d for d in dirs if not self._matches_pattern(d, exclude_pattern)
                    ]

                for file in files:
                    # Skip hidden files if not included
                    if not include_hidden and file.startswith("."):
                        continue

                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, ".")

                    # Apply include/exclude patterns
                    if include_pattern and not self._matches_pattern(
                        file, include_pattern
                    ):
                        continue
                    if exclude_pattern and self._matches_pattern(file, exclude_pattern):
                        continue

                    # Apply file type filter
                    if file_types:
                        file_ext = Path(file).suffix.lstrip(".")
                        if file_ext not in file_types:
                            continue

                    # Search within file content
                    try:
                        with open(file_path, encoding="utf-8", errors="ignore") as f:
                            for line_no, line in enumerate(f, 1):
                                line_content = line.rstrip()

                                # Perform search based on pattern type
                                match_found = False
                                match_text = ""

                                if regex or whole_word:
                                    if pattern.search(line_content):
                                        match_found = True
                                        match_text = line_content.strip()
                                else:
                                    search_line = (
                                        line_content
                                        if case_sensitive
                                        else line_content.lower()
                                    )
                                    if pattern in search_line:
                                        match_found = True
                                        match_text = line_content.strip()

                                if match_found:
                                    results.append(
                                        {
                                            "type": "text",
                                            "path": relative_path,
                                            "line": line_no,
                                            "match": match_text,
                                            "context": f"Line {line_no}: {match_text}",
                                        }
                                    )

                                    if len(results) >= max_results:
                                        return results
                    except (UnicodeDecodeError, PermissionError, IsADirectoryError):
                        # Skip files that can't be read as text
                        continue

        except Exception:
            # Continue with what we found so far
            pass

        return results

    def _matches_pattern(self, text: str, pattern: str) -> bool:
        """Check if text matches a glob-like pattern."""
        import fnmatch

        return fnmatch.fnmatch(text, pattern)

    def _format_results(self, results: list[dict[str, Any]], max_results: int) -> str:
        """Format search results for display."""
        if not results:
            return "No results found."

        # Limit results
        display_results = results[:max_results]

        formatted = []
        file_results = [r for r in display_results if r["type"] == "file"]
        text_results = [r for r in display_results if r["type"] == "text"]

        if file_results:
            formatted.append("📁 File matches:")
            for result in file_results:
                formatted.append(f"  • {result['path']} - {result['match']}")

        if text_results:
            if file_results:
                formatted.append("")
            formatted.append("📝 Text matches:")
            for result in text_results:
                formatted.append(
                    f"  • {result['path']}:{result['line']} - {result['match']}"
                )

        if len(results) > max_results:
            formatted.append(
                f"\n... and {len(results) - max_results} more results (use max_results to see more)"
            )

        return "\n".join(formatted)
