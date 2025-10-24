"""Knowledge reading tool for ML agents."""

from pathlib import Path

from arc.core.agents.shared.knowledge_loader import KnowledgeLoader
from arc.tools.base import BaseTool, ToolResult


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
            phase: Optional phase (model, trainer, evaluator)

        Returns:
            ToolResult with knowledge content or error
        """
        knowledge_id = kwargs.get("knowledge_id")
        phase = kwargs.get("phase", "model")

        if not knowledge_id:
            return ToolResult.error_result(
                "Missing required parameter: knowledge_id",
                recovery_actions=(
                    "Specify which knowledge document to read "
                    "(e.g., 'dcn', 'feature-interaction')"
                ),
            )

        # Load the knowledge document
        content = self.knowledge_loader.load_knowledge(knowledge_id, phase)

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

        # Build header with metadata
        if metadata:
            header_parts = [f"▸ Using knowledge: {metadata.name} ({knowledge_id})"]
            if metadata.type:
                header_parts.append(f"Type: {metadata.type}")
            if metadata.description:
                header_parts.append(f"Description: {metadata.description}")
            header = " - ".join(header_parts) + "\n\n"
        else:
            header = f"▸ Using knowledge: {knowledge_id}\n\n"

        output = header + content

        return ToolResult.success_result(
            output,
            metadata={"knowledge_id": knowledge_id, "phase": phase},
        )
