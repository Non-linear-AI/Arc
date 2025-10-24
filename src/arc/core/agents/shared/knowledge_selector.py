"""Knowledge selection utilities for ML agents."""

from __future__ import annotations


def extract_knowledge_ids_from_text(
    instruction: str | None = None,
    ml_plan_architecture: str | None = None,
) -> list[str]:
    """Extract knowledge IDs from instruction and ML Plan architecture text.

    This function looks for architecture keywords in the user instruction
    and ML Plan's architecture guidance to determine which knowledge documents
    should be loaded.

    Args:
        instruction: User instruction text
        ml_plan_architecture: ML Plan's model_architecture_and_loss section

    Returns:
        List of knowledge IDs to load (e.g., ["dcn", "feature-interaction"])
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

    # Architecture keyword mapping
    # Each architecture has keywords that might appear in instruction/plan
    architecture_keywords = {
        "dcn": ["dcn", "deep & cross", "deep and cross", "cross network"],
        "mlp": ["mlp", "multi-layer perceptron", "feedforward", "dense layers"],
        "transformer": ["transformer", "attention", "self-attention", "multi-head"],
        "feature-interaction": [
            "feature interaction",
            "feature cross",
            "polynomial features",
            "crossed features",
        ],
    }

    # Check each architecture's keywords
    for knowledge_id, keywords in architecture_keywords.items():
        for keyword in keywords:
            if keyword in search_text:
                if knowledge_id not in knowledge_ids:
                    knowledge_ids.append(knowledge_id)
                break

    # Default fallback: if nothing detected, use MLP (most common)
    if not knowledge_ids:
        knowledge_ids.append("mlp")

    return knowledge_ids
