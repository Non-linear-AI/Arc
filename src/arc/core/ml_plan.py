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

    # Core plan content - Technical Decisions only
    name: str
    feature_engineering: str
    model_architecture_and_loss: str
    training_configuration: str
    evaluation: str

    # Optional knowledge recommendations
    # Agent can read knowledge but chooses which to recommend based on task
    recommended_knowledge_ids: list[str] = field(default_factory=list)

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
            feature_engineering=analysis.get("feature_engineering", ""),
            model_architecture_and_loss=analysis.get("model_architecture_and_loss", ""),
            training_configuration=analysis.get("training_configuration", ""),
            evaluation=analysis.get("evaluation", ""),
            recommended_knowledge_ids=analysis.get("recommended_knowledge_ids", []),
            version=version,
            stage=stage,
            reason_for_update=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "feature_engineering": self.feature_engineering,
            "model_architecture_and_loss": self.model_architecture_and_loss,
            "training_configuration": self.training_configuration,
            "evaluation": self.evaluation,
            "recommended_knowledge_ids": self.recommended_knowledge_ids,
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
            feature_engineering=data["feature_engineering"],
            model_architecture_and_loss=data["model_architecture_and_loss"],
            training_configuration=data["training_configuration"],
            evaluation=data["evaluation"],
            recommended_knowledge_ids=data.get("recommended_knowledge_ids", []),
            version=data.get("version", 1),
            stage=data.get("stage", "initial"),
            reason_for_update=data.get("reason_for_update"),
            created_at=created_at or datetime.now(),
        )

    def to_generation_context(self) -> dict[str, Any]:
        """Convert to context dict for model generation.

        This is the format expected by MLModelSpecGeneratorTool's
        analysis_result parameter.
        """
        return {
            "name": self.name,
            "feature_engineering": self.feature_engineering,
            "model_architecture_and_loss": self.model_architecture_and_loss,
            "training_configuration": self.training_configuration,
            "evaluation": self.evaluation,
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
                "**Feature Engineering**",
                self.feature_engineering,
                "",
                "**Model Architecture & Loss**",
                self.model_architecture_and_loss,
                "",
                "**Training Configuration**",
                self.training_configuration,
                "",
                "**Evaluation**",
                self.evaluation,
            ]
        )

        # Show knowledge at the end if present
        if self.recommended_knowledge_ids:
            knowledge_str = ", ".join(self.recommended_knowledge_ids)
            lines.extend(
                [
                    "",
                    "**Knowledge**",
                    knowledge_str,
                ]
            )

        return "\n".join(lines)


@dataclass
class MLPlanDiff:
    """Represents differences between two ML plans."""

    feature_engineering_changed: bool = False
    architecture_changed: bool = False
    training_config_changed: bool = False
    evaluation_changed: bool = False

    # Store old/new values for changed sections
    old_feature_engineering: str = ""
    new_feature_engineering: str = ""
    old_architecture: str = ""
    new_architecture: str = ""
    old_training_config: str = ""
    new_training_config: str = ""
    old_evaluation: str = ""
    new_evaluation: str = ""

    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return (
            self.feature_engineering_changed
            or self.architecture_changed
            or self.training_config_changed
            or self.evaluation_changed
        )

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

        # Feature Engineering
        if self.feature_engineering_changed:
            lines.extend(
                [
                    "**Feature Engineering** *(Changed)*",
                    new_plan.feature_engineering,
                    "",
                    "<details><summary>Show previous version</summary>",
                    "",
                    self.old_feature_engineering,
                    "",
                    "</details>",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "**Feature Engineering**",
                    new_plan.feature_engineering,
                    "",
                ]
            )

        # Architecture
        if self.architecture_changed:
            lines.extend(
                [
                    "**Model Architecture & Loss** *(Changed)*",
                    new_plan.model_architecture_and_loss,
                    "",
                    "<details><summary>Show previous version</summary>",
                    "",
                    self.old_architecture,
                    "",
                    "</details>",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "**Model Architecture & Loss**",
                    new_plan.model_architecture_and_loss,
                    "",
                ]
            )

        # Training Config
        if self.training_config_changed:
            lines.extend(
                [
                    "**Training Configuration** *(Changed)*",
                    new_plan.training_configuration,
                    "",
                    "<details><summary>Show previous version</summary>",
                    "",
                    self.old_training_config,
                    "",
                    "</details>",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "**Training Configuration**",
                    new_plan.training_configuration,
                    "",
                ]
            )

        # Evaluation
        if self.evaluation_changed:
            lines.extend(
                [
                    "**Evaluation** *(Changed)*",
                    new_plan.evaluation,
                    "",
                    "<details><summary>Show previous version</summary>",
                    "",
                    self.old_evaluation,
                    "",
                    "</details>",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "**Evaluation**",
                    new_plan.evaluation,
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
    if old_plan.feature_engineering != new_plan.feature_engineering:
        diff.feature_engineering_changed = True
        diff.old_feature_engineering = old_plan.feature_engineering
        diff.new_feature_engineering = new_plan.feature_engineering

    if old_plan.model_architecture_and_loss != new_plan.model_architecture_and_loss:
        diff.architecture_changed = True
        diff.old_architecture = old_plan.model_architecture_and_loss
        diff.new_architecture = new_plan.model_architecture_and_loss

    if old_plan.training_configuration != new_plan.training_configuration:
        diff.training_config_changed = True
        diff.old_training_config = old_plan.training_configuration
        diff.new_training_config = new_plan.training_configuration

    if old_plan.evaluation != new_plan.evaluation:
        diff.evaluation_changed = True
        diff.old_evaluation = old_plan.evaluation
        diff.new_evaluation = new_plan.evaluation

    return diff
