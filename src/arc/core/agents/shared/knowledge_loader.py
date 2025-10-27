"""Knowledge loader for ML agents."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class KnowledgeMetadata:
    """Metadata for a knowledge document."""

    def __init__(self, data: dict[str, Any]):
        """Initialize metadata from parsed YAML."""
        self.id = data.get("id", "")
        self.name = data.get("name", "")
        self.type = data.get("type", "")
        self.description = data.get("description", "")
        self.keywords = data.get("keywords", [])
        self.problem_type = data.get("problem_type", "")
        self.recommended_patterns = data.get("recommended_patterns", [])
        self.related_knowledge = data.get("related_knowledge", [])
        self.phases = data.get("phases", [])
        self.complexity = data.get("complexity", "")
        self.domain = data.get("domain", "")
        self.data_types = data.get("data_types", [])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for LLM context."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "keywords": self.keywords,
            "problem_type": self.problem_type,
            "phases": self.phases,
            "complexity": self.complexity,
            "domain": self.domain,
        }

    def __str__(self) -> str:
        """String representation for LLM context."""
        return (
            f"- {self.id} (type: {self.type})\n"
            f"  {self.description}\n"
            f"  Keywords: {', '.join(self.keywords)}\n"
            f"  Phases: {', '.join(self.phases)}"
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
        # Cache for knowledge content: key=(knowledge_id, phase), value=content
        self._content_cache: dict[tuple[str, str], str | None] = {}

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

        Args:
            path: Path to scan
            source: Source label ("builtin" or "user") for logging

        Returns:
            Dictionary mapping knowledge_id to metadata
        """
        metadata_map = {}

        for knowledge_dir in path.iterdir():
            if not knowledge_dir.is_dir():
                continue

            metadata_file = knowledge_dir / "metadata.yaml"
            if not metadata_file.exists():
                logger.debug(f"No metadata.yaml in {knowledge_dir}")
                continue

            try:
                with open(metadata_file) as f:
                    data = yaml.safe_load(f)

                metadata = KnowledgeMetadata(data)
                metadata_map[metadata.id] = metadata

            except Exception as e:
                logger.warning(
                    f"Failed to load {source} metadata from {metadata_file}: {e}"
                )
                continue

        return metadata_map

    def get_metadata_list(self) -> list[KnowledgeMetadata]:
        """Get list of all knowledge metadata.

        Returns:
            List of metadata objects
        """
        metadata_map = self.scan_metadata()
        return list(metadata_map.values())

    def load_knowledge(self, knowledge_id: str, phase: str = "model") -> str | None:
        """Load specific knowledge document with caching.

        Checks user path first, then builtin path.
        Results are cached to avoid repeated file I/O.

        Args:
            knowledge_id: ID of the knowledge to read
                (e.g., 'dcn', 'feature-interaction')
            phase: Which phase guide to read (default: 'model')

        Returns:
            Content of the knowledge document, or None if not found
        """
        # Check cache first
        cache_key = (knowledge_id, phase)
        if cache_key in self._content_cache:
            return self._content_cache[cache_key]

        # Try user knowledge first (allows override)
        if self.user_path.exists():
            content = self._load_knowledge_from_path(
                self.user_path, knowledge_id, phase
            )
            if content is not None:
                logger.debug(f"Loaded user knowledge: {knowledge_id} ({phase})")
                # Cache the result
                self._content_cache[cache_key] = content
                return content

        # Fall back to builtin knowledge
        if self.builtin_path.exists():
            content = self._load_knowledge_from_path(
                self.builtin_path, knowledge_id, phase
            )
            if content is not None:
                logger.debug(f"Loaded builtin knowledge: {knowledge_id} ({phase})")
                # Cache the result
                self._content_cache[cache_key] = content
                return content

        # Knowledge not found is normal - agent will discover what's available via tools
        logger.debug(f"Knowledge not found: {knowledge_id} (phase: {phase})")
        # Cache the negative result to avoid repeated lookups
        self._content_cache[cache_key] = None
        return None

    def _load_knowledge_from_path(
        self, base_path: Path, knowledge_id: str, phase: str
    ) -> str | None:
        """Load knowledge from a specific base path.

        Args:
            base_path: Base path to search (builtin or user)
            knowledge_id: Knowledge ID
            phase: Phase to load (e.g., "general", "model", "train", "evaluate", "data")

        Returns:
            Knowledge content or None if not found
        """
        knowledge_dir = base_path / knowledge_id

        if not knowledge_dir.exists():
            return None

        # If phase is "general", go directly to guide.md
        if phase == "general":
            general_guide = knowledge_dir / "guide.md"
            if general_guide.exists():
                try:
                    return general_guide.read_text()
                except Exception as e:
                    logger.error(f"Failed to read {general_guide}: {e}")
                    return None
            return None

        # Try phase-specific guide first (e.g., model-guide.md)
        phase_guide = knowledge_dir / f"{phase}-guide.md"
        if phase_guide.exists():
            try:
                return phase_guide.read_text()
            except Exception as e:
                logger.error(f"Failed to read {phase_guide}: {e}")
                return None

        # Fall back to general guide.md
        general_guide = knowledge_dir / "guide.md"
        if general_guide.exists():
            try:
                return general_guide.read_text()
            except Exception as e:
                logger.error(f"Failed to read {general_guide}: {e}")
                return None

        return None

    def get_available_phases(self, knowledge_id: str) -> list[str]:
        """Get list of actually available phases for a knowledge document.

        Checks which guide files exist (both phase-specific and general).

        Args:
            knowledge_id: Knowledge document ID

        Returns:
            List of available phases (e.g., ["general", "model", "train"])
        """
        available_phases = []

        # Check both user and builtin paths
        for base_path in [self.user_path, self.builtin_path]:
            knowledge_dir = base_path / knowledge_id
            if not knowledge_dir.exists():
                continue

            # Check for general guide
            if (knowledge_dir / "guide.md").exists():
                if "general" not in available_phases:
                    available_phases.append("general")

            # Check for phase-specific guides
            for phase in ["model", "train", "evaluate", "data"]:
                phase_guide = knowledge_dir / f"{phase}-guide.md"
                if phase_guide.exists() and phase not in available_phases:
                    available_phases.append(phase)

        return available_phases

    def format_metadata_for_llm(self) -> str:
        """Format all metadata as a string for LLM context.

        Returns:
            Formatted string listing all available knowledge
        """
        metadata_list = self.get_metadata_list()

        if not metadata_list:
            return "No knowledge documents available."

        lines = ["Available knowledge documents (use read_knowledge tool to access):\n"]

        # Group by type
        by_type: dict[str, list[KnowledgeMetadata]] = {}
        for metadata in metadata_list:
            if metadata.type not in by_type:
                by_type[metadata.type] = []
            by_type[metadata.type].append(metadata)

        # Format each type group
        for knowledge_type in ["architecture", "pattern", "scenario"]:
            if knowledge_type not in by_type:
                continue

            lines.append(f"\n{knowledge_type.upper()}S:")
            for metadata in by_type[knowledge_type]:
                lines.append(str(metadata))
                lines.append("")

        return "\n".join(lines)
