"""ML Plan data structure for managing machine learning workflow plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class MLPlan:
    """Represents an ML plan that evolves through the workflow.

    This plan is created by the problem analyzer and can be revised
    based on training results, evaluation metrics, or user feedback.
    """

    # Core plan content - Separated by workflow stage
    name: str
    data_plan: str  # Feature engineering guidance for ml_data
    model_plan: str  # Model architecture + training guidance for ml_model (unified)

    # Stage-specific knowledge recommendations
    # Maps workflow stage to list of knowledge IDs to preload for that stage
    knowledge: dict[str, list[str]] = field(default_factory=dict)
    # Example: {"data": ["feature_eng"], "model": ["mlp", "adam_optimizer"]}

    # Metadata
    version: int = 1
    stage: str = "initial"  # "initial", "post_training", "post_evaluation", etc.
    reason_for_update: str | None = None
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_analysis(
        cls,
        analysis: dict[str, Any],
        version: int = 1,
        stage: str = "initial",
        reason: str | None = None,
    ) -> MLPlan:
        """Create MLPlan from problem analyzer output.

        Args:
            analysis: Dictionary from problem analyzer with plan fields
            version: Plan version number
            stage: Current stage in the workflow
            reason: Reason for this plan version (for revisions)

        Returns:
            MLPlan instance
        """
        return cls(
            name=analysis.get("name", ""),
            data_plan=analysis.get("data_plan", ""),
            model_plan=analysis.get("model_plan", ""),
            knowledge=analysis.get("knowledge", {}),
            version=version,
            stage=stage,
            reason_for_update=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "data_plan": self.data_plan,
            "model_plan": self.model_plan,
            "knowledge": self.knowledge,
            "version": self.version,
            "stage": self.stage,
            "reason_for_update": self.reason_for_update,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MLPlan:
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            name=data["name"],
            data_plan=data["data_plan"],
            model_plan=data["model_plan"],
            knowledge=data.get("knowledge", {}),
            version=data.get("version", 1),
            stage=data.get("stage", "initial"),
            reason_for_update=data.get("reason_for_update"),
            created_at=created_at or datetime.now(),
        )

    def to_generation_context(self) -> dict[str, Any]:
        """Convert to context dict for generation tools.

        Returns the plan content split by workflow stage.
        """
        return {
            "name": self.name,
            "data_plan": self.data_plan,
            "model_plan": self.model_plan,
        }

    def format_for_display(self) -> str:
        """Format plan as readable markdown text with left-aligned headers.

        Returns:
            Formatted markdown string for display
        """
        # Technical decisions with bold labels instead of headers
        lines = []
        # Show plan name at the top with clear label
        lines.extend(
            [
                f"**Plan** {self.name}",
                "",
                "**Data Plan (Feature Engineering)**",
                self.data_plan,
                "",
                "**Model Plan (Architecture + Training)**",
                self.model_plan,
            ]
        )

        # Show knowledge at the end if present
        if self.knowledge:
            lines.append("")
            lines.append("**Knowledge Recommendations**")
            # Show stage-specific knowledge in a readable format
            for stage in ["data", "model"]:
                knowledge_ids = self.knowledge.get(stage, [])
                if knowledge_ids:
                    knowledge_str = ", ".join(knowledge_ids)
                    stage_name = stage.title()  # Capitalize first letter
                    lines.append(f"  • {stage_name}: {knowledge_str}")

        return "\n".join(lines)


@dataclass
class MLPlanDiff:
    """Represents differences between two ML plans."""

    data_plan_changed: bool = False
    model_plan_changed: bool = False

    # Store old/new values for changed sections
    old_data_plan: str = ""
    new_data_plan: str = ""
    old_model_plan: str = ""
    new_model_plan: str = ""

    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return self.data_plan_changed or self.model_plan_changed

    def format_for_display(self, new_plan: MLPlan) -> str:
        """Format diff as readable markdown with changes highlighted.

        Args:
            new_plan: The new plan being proposed

        Returns:
            Formatted markdown string showing changes
        """
        lines = [
            "**🔄 ML Plan Revision**",
            f"*Version {new_plan.version} • "
            f"{new_plan.stage.replace('_', ' ').title()}*",
        ]

        if new_plan.reason_for_update:
            lines.append(f"*Reason: {new_plan.reason_for_update}*")

        lines.append("")
        lines.append("---")
        lines.append("")

        # Data Plan
        if self.data_plan_changed:
            lines.extend(
                [
                    "**Data Plan (Feature Engineering)** *(Changed)*",
                    new_plan.data_plan,
                    "",
                    "<details><summary>Show previous version</summary>",
                    "",
                    self.old_data_plan,
                    "",
                    "</details>",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "**Data Plan (Feature Engineering)**",
                    new_plan.data_plan,
                    "",
                ]
            )

        # Model Plan
        if self.model_plan_changed:
            lines.extend(
                [
                    "**Model Plan (Architecture + Training)** *(Changed)*",
                    new_plan.model_plan,
                    "",
                    "<details><summary>Show previous version</summary>",
                    "",
                    self.old_model_plan,
                    "",
                    "</details>",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "**Model Plan (Architecture + Training)**",
                    new_plan.model_plan,
                    "",
                ]
            )

        # Add confirmation options
        lines.extend(
            [
                "---",
                "",
                "**How would you like to proceed?**",
                "",
                "- ✅ **Approve** - Generate model using the revised plan",
                "- 💬 **Feedback** - Request additional changes",
                "- ❓ **Questions** - Ask about the revisions",
            ]
        )

        return "\n".join(lines)


def compute_plan_diff(old_plan: dict | MLPlan, new_plan: MLPlan) -> MLPlanDiff:
    """Compute differences between two plans.

    Args:
        old_plan: Previous plan (dict or MLPlan)
        new_plan: New plan (MLPlan)

    Returns:
        MLPlanDiff object with changes
    """
    # Convert old_plan to MLPlan if it's a dict
    if isinstance(old_plan, dict):
        old_plan = MLPlan.from_dict(old_plan)

    diff = MLPlanDiff()

    # Text field changes
    if old_plan.data_plan != new_plan.data_plan:
        diff.data_plan_changed = True
        diff.old_data_plan = old_plan.data_plan
        diff.new_data_plan = new_plan.data_plan

    if old_plan.model_plan != new_plan.model_plan:
        diff.model_plan_changed = True
        diff.old_model_plan = old_plan.model_plan
        diff.new_model_plan = new_plan.model_plan

    return diff
