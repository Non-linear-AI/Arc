"""Knowledge selection utilities for ML agents."""

from __future__ import annotations

from arc.core.agents.shared.knowledge_loader import KnowledgeLoader

# Singleton knowledge loader for keyword mapping
_knowledge_loader: KnowledgeLoader | None = None


def _get_knowledge_loader() -> KnowledgeLoader:
    """Get or create singleton knowledge loader."""
    global _knowledge_loader
    if _knowledge_loader is None:
        _knowledge_loader = KnowledgeLoader()
    return _knowledge_loader


def _build_keyword_mapping() -> dict[str, list[str]]:
    """Build keyword mapping from actual knowledge metadata.

    Returns:
        Dictionary mapping knowledge_id to list of keywords
    """
    loader = _get_knowledge_loader()
    metadata_map = loader.scan_metadata()

    keyword_mapping = {}
    for knowledge_id, metadata in metadata_map.items():
        # Use keywords from metadata.yaml
        keywords = [kw.lower() for kw in metadata.keywords]
        # Also add the knowledge name as a keyword
        if metadata.name:
            keywords.append(metadata.name.lower())
        # Add the ID itself as a keyword
        keywords.append(knowledge_id.lower())

        keyword_mapping[knowledge_id] = keywords

    return keyword_mapping


def extract_knowledge_ids_from_text(
    instruction: str | None = None,
    ml_plan_architecture: str | None = None,
) -> list[str]:
    """Extract knowledge IDs from instruction and ML Plan architecture text.

    This function looks for architecture keywords in the user instruction
    and ML Plan's architecture guidance to determine which knowledge documents
    should be loaded.

    Dynamically builds keyword mapping from available knowledge metadata,
    so only real knowledge IDs are returned.

    Args:
        instruction: User instruction text
        ml_plan_architecture: ML Plan's model_architecture_and_loss section

    Returns:
        List of knowledge IDs to load (e.g., ["dcn", "mlp"])
    """
    knowledge_ids = []

    # Combine instruction and ML plan text for searching
    search_text = ""
    if instruction:
        search_text += instruction.lower()
    if ml_plan_architecture:
        search_text += " " + ml_plan_architecture.lower()

    if not search_text:
        return knowledge_ids

    # Build keyword mapping from actual knowledge metadata
    architecture_keywords = _build_keyword_mapping()

    # Check each architecture's keywords
    for knowledge_id, keywords in architecture_keywords.items():
        for keyword in keywords:
            if keyword in search_text:
                if knowledge_id not in knowledge_ids:
                    knowledge_ids.append(knowledge_id)
                break

    # Default fallback: if nothing detected, use MLP if it exists
    if not knowledge_ids and "mlp" in architecture_keywords:
        knowledge_ids.append("mlp")

    return knowledge_ids
