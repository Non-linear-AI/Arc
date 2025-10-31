"""Knowledge loader for ML agents."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class KnowledgeMetadata:
    """Metadata for a knowledge document."""

    def __init__(self, knowledge_id: str, data: dict[str, Any]):
        """Initialize metadata from parsed YAML.

        Args:
            knowledge_id: The knowledge document ID (used as key in metadata.yaml)
            data: Dictionary containing metadata fields
        """
        self.id = knowledge_id
        self.name = data.get("name", "")
        self.description = data.get("description", "")
        self.phases = data.get("phases", ["model"])  # Default to model phase

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for LLM context."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "phases": self.phases,
        }

    def __str__(self) -> str:
        """String representation for LLM context."""
        phases_str = ", ".join(self.phases)
        return (
            f"- {self.id}\n"
            f"  Name: {self.name}\n"
            f"  Description: {self.description}\n"
            f"  Phases: {phases_str}"
        )


class KnowledgeLoader:
    """Centralized knowledge loading for all ML agents.

    Loads knowledge from two sources:
    1. Builtin knowledge (bundled with package)
    2. User knowledge (optional customizations at ~/.arc/knowledge/)

    User knowledge overrides builtin knowledge if same ID exists.
    """

    def __init__(
        self,
        builtin_path: Path | None = None,
        user_path: Path | None = None,
    ):
        """Initialize knowledge loader.

        Args:
            builtin_path: Path to builtin knowledge (default: package resources)
            user_path: Path to user knowledge (default: ~/.arc/knowledge)
        """
        if builtin_path is None:
            # Builtin knowledge bundled with package
            builtin_path = (
                Path(__file__).parent.parent.parent.parent / "resources" / "knowledge"
            )

        if user_path is None:
            # User knowledge for customizations
            user_path = Path.home() / ".arc" / "knowledge"

        self.builtin_path = Path(builtin_path)
        self.user_path = Path(user_path)
        self._metadata_cache: dict[str, KnowledgeMetadata] | None = None
        # Cache for knowledge content: key=knowledge_id, value=content
        self._content_cache: dict[str, str | None] = {}

    def scan_metadata(self) -> dict[str, KnowledgeMetadata]:
        """Scan all knowledge metadata files from builtin and user paths.

        User knowledge overrides builtin knowledge if same ID exists.

        Returns:
            Dictionary mapping knowledge_id to metadata
        """
        if self._metadata_cache is not None:
            return self._metadata_cache

        metadata_map = {}

        # First, scan builtin knowledge
        if self.builtin_path.exists():
            builtin_metadata = self._scan_metadata_from_path(
                self.builtin_path, "builtin"
            )
            metadata_map.update(builtin_metadata)
            logger.debug(f"Loaded {len(builtin_metadata)} builtin knowledge documents")
        else:
            logger.warning(
                f"Builtin knowledge path does not exist: {self.builtin_path}"
            )

        # Then, scan user knowledge (overrides builtin)
        if self.user_path.exists():
            user_metadata = self._scan_metadata_from_path(self.user_path, "user")
            # User knowledge overrides builtin
            for knowledge_id, metadata in user_metadata.items():
                if knowledge_id in metadata_map:
                    logger.info(
                        f"User knowledge '{knowledge_id}' overrides builtin knowledge"
                    )
                metadata_map[knowledge_id] = metadata
            logger.debug(f"Loaded {len(user_metadata)} user knowledge documents")

        self._metadata_cache = metadata_map
        return metadata_map

    def _scan_metadata_from_path(
        self, path: Path, source: str
    ) -> dict[str, KnowledgeMetadata]:
        """Scan metadata from a specific path.

        Reads the single metadata.yaml file at the base path containing all knowledge.

        Args:
            path: Base path to scan (should contain metadata.yaml)
            source: Source label ("builtin" or "user") for logging

        Returns:
            Dictionary mapping knowledge_id to metadata
        """
        metadata_map = {}
        metadata_file = path / "metadata.yaml"

        if not metadata_file.exists():
            logger.debug(f"No metadata.yaml at {path}")
            return metadata_map

        try:
            with open(metadata_file) as f:
                all_metadata = yaml.safe_load(f)

            if not isinstance(all_metadata, dict):
                logger.warning(
                    f"{source} metadata.yaml is not a dictionary: {metadata_file}"
                )
                return metadata_map

            # Each key is a knowledge_id, value is its metadata
            for knowledge_id, data in all_metadata.items():
                try:
                    metadata = KnowledgeMetadata(knowledge_id, data)
                    metadata_map[knowledge_id] = metadata
                except Exception as e:
                    logger.warning(
                        f"Failed to parse {source} knowledge '{knowledge_id}': {e}"
                    )
                    continue

        except Exception as e:
            logger.warning(
                f"Failed to load {source} metadata from {metadata_file}: {e}"
            )

        return metadata_map

    def get_metadata_list(self) -> list[KnowledgeMetadata]:
        """Get list of all knowledge metadata.

        Returns:
            List of metadata objects
        """
        metadata_map = self.scan_metadata()
        return list(metadata_map.values())

    def load_multiple(self, knowledge_ids: list[str]) -> list[dict[str, str]]:
        """Load multiple knowledge documents.

        Args:
            knowledge_ids: List of knowledge IDs to load

        Returns:
            List of dicts with keys: 'id', 'name', 'content'
            Only includes successfully loaded knowledge docs
        """
        results = []
        metadata_map = self.scan_metadata()

        for knowledge_id in knowledge_ids:
            content = self.load_knowledge(knowledge_id)
            if content:
                metadata = metadata_map.get(knowledge_id)
                results.append(
                    {
                        "id": knowledge_id,
                        "name": metadata.name if metadata else knowledge_id,
                        "content": content,
                    }
                )

        return results

    def load_knowledge(self, knowledge_id: str) -> str | None:
        """Load specific knowledge document with caching.

        Checks user path first, then builtin path.
        Results are cached to avoid repeated file I/O.

        Args:
            knowledge_id: ID of the knowledge to read (e.g., 'mlp', 'dcn')

        Returns:
            Content of the knowledge document, or None if not found
        """
        # Check cache first
        if knowledge_id in self._content_cache:
            return self._content_cache[knowledge_id]

        # Try user knowledge first (allows override)
        if self.user_path.exists():
            content = self._load_knowledge_from_path(self.user_path, knowledge_id)
            if content is not None:
                logger.debug(f"Loaded user knowledge: {knowledge_id}")
                # Cache the result
                self._content_cache[knowledge_id] = content
                return content

        # Fall back to builtin knowledge
        if self.builtin_path.exists():
            content = self._load_knowledge_from_path(self.builtin_path, knowledge_id)
            if content is not None:
                logger.debug(f"Loaded builtin knowledge: {knowledge_id}")
                # Cache the result
                self._content_cache[knowledge_id] = content
                return content

        # Knowledge not found is normal - agent will discover what's available via tools
        logger.debug(f"Knowledge not found: {knowledge_id}")
        # Cache the negative result to avoid repeated lookups
        self._content_cache[knowledge_id] = None
        return None

    def _load_knowledge_from_path(
        self, base_path: Path, knowledge_id: str
    ) -> str | None:
        """Load knowledge from a specific base path.

        Args:
            base_path: Base path to search (builtin or user)
            knowledge_id: Knowledge ID

        Returns:
            Knowledge content or None if not found
        """
        knowledge_file = base_path / f"{knowledge_id}.md"

        if not knowledge_file.exists():
            return None

        try:
            return knowledge_file.read_text()
        except Exception as e:
            logger.error(f"Failed to read {knowledge_file}: {e}")
            return None

    def get_available_phases(self, knowledge_id: str) -> list[str]:
        """Get list of available phases for a knowledge document.

        Reads phases from metadata.

        Args:
            knowledge_id: Knowledge document ID

        Returns:
            List of available phases (e.g., ["data", "model"])
        """
        metadata_map = self.scan_metadata()
        metadata = metadata_map.get(knowledge_id)

        if metadata:
            return metadata.phases

        return []

    def format_metadata_for_llm(self, allowed_phases: list[str] | None = None) -> str:
        """Format metadata as a string for LLM context, optionally filtering by phases.

        Args:
            allowed_phases: Optional list of phases to filter by
                           (e.g., ["data", "model"]).
                           If None, shows all knowledge documents.

        Returns:
            Formatted string listing available knowledge
        """
        metadata_list = self.get_metadata_list()

        if not metadata_list:
            return "No knowledge documents available."

        # Filter by allowed phases if specified
        if allowed_phases:
            filtered_list = []
            for metadata in metadata_list:
                # Include if any of the knowledge's phases overlap with allowed phases
                if any(phase in allowed_phases for phase in metadata.phases):
                    filtered_list.append(metadata)
            metadata_list = filtered_list

        if not metadata_list:
            phases_str = ", ".join(allowed_phases)
            return f"No knowledge documents available for phases: {phases_str}"

        lines = ["Available knowledge documents:\n"]

        for metadata in metadata_list:
            lines.append(str(metadata))
            lines.append("")

        return "\n".join(lines)
