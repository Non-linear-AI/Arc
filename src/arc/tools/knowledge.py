"""Knowledge reading tool for ML agents."""

from pathlib import Path

from arc.core.agents.shared.knowledge_loader import KnowledgeLoader
from arc.tools.base import BaseTool, ToolResult


class ListAvailableKnowledgeTool(BaseTool):
    """Tool for listing available knowledge documents."""

    def __init__(
        self,
        builtin_path: Path | None = None,
        user_path: Path | None = None,
    ):
        """Initialize list available knowledge tool.

        Args:
            builtin_path: Path to builtin knowledge (default: package resources)
            user_path: Path to user knowledge (default: ~/.arc/knowledge)
        """
        super().__init__()
        self.knowledge_loader = KnowledgeLoader(builtin_path, user_path)

    async def execute(self, **_kwargs) -> ToolResult:
        """Execute list_available_knowledge operation.

        Returns:
            ToolResult with categorized list of available knowledge documents
        """
        # Scan metadata for all knowledge documents
        metadata_map = self.knowledge_loader.scan_metadata()

        if not metadata_map:
            return ToolResult.success_result(
                "No knowledge documents available.",
                metadata={"knowledge_count": 0},
            )

        # Organize by phase
        phase_groups: dict[str, list[dict]] = {}
        for knowledge_id, metadata in metadata_map.items():
            # Each knowledge can belong to multiple phases
            for phase in metadata.phases:
                if phase not in phase_groups:
                    phase_groups[phase] = []
                phase_groups[phase].append(
                    {
                        "id": knowledge_id,
                        "name": metadata.name,
                        "description": metadata.description or "",
                    }
                )

        # Build categorized output
        lines = []

        # Define phase display order and titles
        phase_titles = {
            "data": "Data Processing",
            "model": "Model Architectures",
            "trainer": "Training Patterns",
            "evaluator": "Evaluation Patterns",
        }

        for phase in ["data", "model", "trainer", "evaluator"]:
            if phase not in phase_groups:
                continue

            # Add phase header
            title = phase_titles.get(phase, phase.title())
            lines.append(f"\n{title}:")

            # Sort items by ID within each phase
            items = sorted(phase_groups[phase], key=lambda x: x["id"])

            # Add each item
            for item in items:
                lines.append(f"• {item['id']} - {item['name']}")

        output = "\n".join(lines)

        return ToolResult.success_result(
            output,
            metadata={"knowledge_count": len(metadata_map)},
        )


class ReadKnowledgeTool(BaseTool):
    """Tool for reading domain knowledge documents."""

    def __init__(
        self,
        builtin_path: Path | None = None,
        user_path: Path | None = None,
    ):
        """Initialize read knowledge tool.

        Args:
            builtin_path: Path to builtin knowledge (default: package resources)
            user_path: Path to user knowledge (default: ~/.arc/knowledge)
        """
        super().__init__()
        self.knowledge_loader = KnowledgeLoader(builtin_path, user_path)

    async def execute(self, **kwargs) -> ToolResult:
        """Execute read_knowledge operation.

        Args:
            knowledge_id: ID of the knowledge to read

        Returns:
            ToolResult with knowledge summary and file link
        """
        knowledge_id = kwargs.get("knowledge_id")

        if not knowledge_id:
            return ToolResult.error_result(
                "Missing required parameter: knowledge_id",
                recovery_actions=(
                    "Specify which knowledge document to read (e.g., 'dcn', 'mlp')"
                ),
            )

        # Load the knowledge document
        content = self.knowledge_loader.load_knowledge(knowledge_id)

        if content is None:
            # Get available knowledge for helpful error message
            metadata_map = self.knowledge_loader.scan_metadata()
            available_ids = list(metadata_map.keys())

            error_msg = f"Knowledge '{knowledge_id}' not found"
            recovery = (
                f"Available knowledge documents: {', '.join(available_ids)}"
                if available_ids
                else (
                    "No knowledge documents available. "
                    "Add documents to ~/.arc/knowledge/"
                )
            )

            return ToolResult.error_result(error_msg, recovery_actions=recovery)

        # Get metadata for context
        metadata_map = self.knowledge_loader.scan_metadata()
        metadata = metadata_map.get(knowledge_id)

        # Get file path (prefer user path if exists, otherwise builtin)
        file_path = None
        user_file = self.knowledge_loader.user_path / f"{knowledge_id}.md"
        builtin_file = self.knowledge_loader.builtin_path / f"{knowledge_id}.md"

        if user_file.exists():
            file_path = user_file
        elif builtin_file.exists():
            file_path = builtin_file

        # Count lines in content
        line_count = len(content.split("\n"))

        # Split content into lines
        lines = content.split("\n")

        # Extract content summary (first few sections, stop at ## heading or ~200 chars)
        summary_lines = []
        char_count = 0
        max_chars = 200

        for line in lines[:20]:  # Look at first 20 lines max
            if line.strip().startswith("## ") and summary_lines:
                # Stop at second-level heading if we already have content
                break
            summary_lines.append(line)
            char_count += len(line)
            if char_count > max_chars and line.strip():
                # Stop after we've exceeded char limit at a natural break
                break

        summary = "\n".join(summary_lines).strip()

        # Build output with metadata + summary + file link
        output_parts = []

        if metadata:
            # Show name and description
            output_parts.append(f"{metadata.name}")
            if metadata.description:
                output_parts.append(f"{metadata.description}")

            # Show phases
            if metadata.phases:
                phases_str = ", ".join(metadata.phases)
                output_parts.append(f"Phases: {phases_str}")

        # Add content summary
        if summary:
            output_parts.append("")  # Blank line before summary
            output_parts.append(summary)
        elif not metadata:
            # Fallback if no metadata and no summary
            output_parts.append(knowledge_id)

        # Add file link
        output_parts.append("")  # Single blank line before file path
        if file_path:
            output_parts.append(f"📄 {file_path} ({line_count} lines)")
        else:
            output_parts.append(f"({line_count} lines)")

        output = "\n".join(output_parts)

        return ToolResult.success_result(
            output,
            metadata={
                "knowledge_id": knowledge_id,
                "file_path": str(file_path) if file_path else None,
                "line_count": line_count,
            },
        )
