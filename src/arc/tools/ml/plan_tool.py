"""ML Plan tool for creating and revising ML plans."""

from __future__ import annotations

import contextlib

import yaml

from arc.core.agents.ml_plan import MLPlanAgent
from arc.tools.base import BaseTool, ToolResult


class MLPlanTool(BaseTool):
    """Tool for creating and revising ML plans with technical decisions."""

    def __init__(
        self,
        services,
        api_key: str | None,
        base_url: str | None,
        model: str | None,
        ui_interface=None,
        agent=None,
    ) -> None:
        self.services = services
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.ui = ui_interface
        self.agent = agent  # Reference to parent agent for auto_accept flag

    async def execute(
        self,
        *,
        name: str | None = None,
        instruction: str | None = None,
        source_tables: str | None = None,
        previous_plan: dict | None = None,
        section_to_update: str | None = None,
        conversation_history: str | None = None,  # noqa: ARG002
        verbose: bool = False,
    ) -> ToolResult:
        # Early validation for common errors (before any section printing)
        if not self.api_key:
            return ToolResult.error_result(
                "API key required for ML planning. "
                "Set ARC_API_KEY or configure an API key before using this tool."
            )

        if not self.services:
            return ToolResult.error_result(
                "ML planning service unavailable. Database services not initialized."
            )

        if not name or not instruction or not source_tables:
            return ToolResult.error_result(
                "Parameters 'name', 'instruction', and 'source_tables' "
                "are required for ML planning."
            )

        # Handle section update mode (different workflow - no UI section needed)
        if section_to_update:
            # Section update mode requires previous_plan and instruction
            if not previous_plan:
                return ToolResult.error_result(
                    "Parameter 'previous_plan' is required when updating a section."
                )
            if not instruction:
                return ToolResult.error_result(
                    "Parameter 'instruction' is required when updating a section."
                )

            # Extract the original section content
            section_content = previous_plan.get(section_to_update)
            if section_content is None:
                return ToolResult.error_result(
                    f"Section '{section_to_update}' not found in previous plan."
                )

            # Create agent and update section
            agent = MLPlanAgent(
                self.services,
                self.api_key,
                self.base_url,
                self.model,
                verbose=verbose,
            )

            try:
                updated_section = await agent.update_section(
                    section_name=section_to_update,
                    original_section=str(section_content),
                    feedback_content=str(instruction),
                )

                # Update the plan with new section
                updated_plan = previous_plan.copy()
                updated_plan[section_to_update] = updated_section

                return ToolResult(
                    success=True,
                    output=f"Section '{section_to_update}' updated successfully.",
                    metadata={
                        "ml_plan": updated_plan,
                        "section_updated": section_to_update,
                        "is_revision": True,
                    },
                )

            except Exception as exc:
                from arc.core.ml_plan import MLPlanError

                if isinstance(exc, MLPlanError):
                    return ToolResult.error_result(str(exc))
                return ToolResult.error_result(
                    f"Unexpected error updating section: {exc}"
                )

        # Full plan generation mode
        # Use context manager for section printing
        with self._section_printer(self.ui, "ML Plan", color="cyan") as printer:
            # Show task description
            if printer:
                printer.print(f"[dim]Task: {instruction}[/dim]")
                printer.print("")  # Empty line after task

            # Helper to show error and return
            def _error_in_section(message: str) -> ToolResult:
                if printer:
                    printer.print("")
                    printer.print(f"✗ {message}")
                return ToolResult(
                    success=False,
                    output=message,
                    metadata={"error_shown": True, "error_message": message},
                )

            # Helper to print progress messages within the section
            def _progress_callback(message: str):
                if printer:
                    printer.print(message)

            agent = MLPlanAgent(
                self.services,
                self.api_key,
                self.base_url,
                self.model,
                progress_callback=_progress_callback if printer else None,
                verbose=verbose,
            )

            try:
                # Import MLPlan for plan management
                from arc.core.ml_plan import MLPlan

                # Check if auto-accept is enabled
                if self.agent and self.agent.ml_plan_auto_accept:
                    # Auto-accept mode - skip workflow
                    pass  # Continue to generate plan but skip confirmation

                # Internal loop for handling user instruction and revision feedback
                current_instruction = instruction

                # Get version from database to avoid conflicts
                latest_plan = self.services.ml_plans.get_latest_plan_by_name(str(name))
                version = latest_plan.version + 1 if latest_plan else 1

                while True:
                    try:
                        # Generate the plan (pass source_tables as
                        # comma-separated string)
                        analysis = await agent.analyze_problem(
                            user_context=str(current_instruction),
                            source_tables=str(source_tables),
                            instruction=current_instruction
                            if current_instruction != instruction
                            else None,
                            stream=False,
                        )

                        # Inject the passed-in name into the analysis result
                        # (LLM doesn't generate it since we already have it)
                        analysis["name"] = str(name)

                        # Show completion message
                        if printer:
                            printer.print("[dim]✓ Plan generated successfully[/dim]")

                        # Determine stage
                        if previous_plan:
                            stage = previous_plan.get("stage", "initial")
                            instruction_lower = str(current_instruction).lower()
                            if (
                                current_instruction != instruction
                                and "training" in instruction_lower
                            ):
                                stage = "post_training"
                            elif (
                                current_instruction != instruction
                                and "evaluation" in instruction_lower
                            ):
                                stage = "post_evaluation"
                            reason = (
                                f"Revised based on instruction: "
                                f"{current_instruction[:100]}..."
                                if current_instruction != instruction
                                else "Plan revision"
                            )
                        else:
                            stage = "initial"
                            reason = None

                        plan = MLPlan.from_analysis(
                            analysis, version=version, stage=stage, reason=reason
                        )

                        # Save plan to database immediately with "draft" status
                        # This allows other tools to reference it even before
                        # user confirms
                        try:
                            from datetime import UTC, datetime

                            from arc.database.models.ml_plan import (
                                MLPlan as MLPlanModel,
                            )
                            from arc.ml.runtime import _slugify_name

                            # Convert plan to dict for storage
                            plan_dict = plan.to_dict()
                            plan_dict["source_tables"] = str(source_tables)

                            # Convert plan to YAML format for better readability
                            plan_yaml = yaml.dump(
                                plan_dict, default_flow_style=False, sort_keys=False
                            )

                            # Create database model - use name for plan ID
                            base_slug = _slugify_name(str(name))
                            plan_id = f"{base_slug}-v{version}"

                            now = datetime.now(UTC)
                            db_plan = MLPlanModel(
                                plan_id=plan_id,
                                name=str(name),
                                version=version,
                                user_context=str(instruction),
                                source_tables=str(source_tables),
                                plan_yaml=plan_yaml,  # Store as YAML string
                                status="draft",  # Initially draft until user confirms
                                created_at=now,
                                updated_at=now,
                            )

                            # Save to database
                            self.services.ml_plans.create_plan(db_plan)

                            # Add plan_id to plan_dict for use in confirmation workflow
                            plan_dict["plan_id"] = plan_id

                        except Exception as e:
                            # Log error but continue - plan still works in memory
                            if printer:
                                printer.print(
                                    f"[dim yellow]⚠ Could not save plan to "
                                    f"database: {e}[/dim yellow]"
                                )
                            plan_id = None  # Track that we don't have a DB plan

                    except Exception as e:
                        # Handle errors during plan generation
                        error_msg = f"Failed to generate ML plan: {str(e)}"
                        if printer:
                            printer.print("")
                            printer.print(f"✗ {error_msg}")
                        return ToolResult(
                            success=False,
                            output="",
                            metadata={"error_shown": True},
                        )

                    # If auto-accept is enabled, skip workflow
                    if self.agent and self.agent.ml_plan_auto_accept:
                        output_message = (
                            f"Plan '{plan_id}' automatically accepted "
                            f"(auto-accept enabled)"
                        )
                        break

                    # Display plan and run confirmation workflow
                    if self.ui:
                        from arc.utils.ml_plan_workflow import (
                            MLPlanConfirmationWorkflow,
                        )

                        try:
                            workflow = MLPlanConfirmationWorkflow(self.ui)
                            result = await workflow.run_workflow(
                                plan, previous_plan is not None
                            )
                            choice = result.get("choice")
                        except Exception as e:
                            # Handle workflow errors
                            error_msg = f"Workflow execution failed: {str(e)}"
                            self.ui._printer.console.print(f"[red]❌ {error_msg}[/red]")
                            return ToolResult.error_result(error_msg)

                        if choice == "accept":
                            # Update status to confirmed in database
                            if plan_id:
                                try:
                                    self.services.ml_plans.update_status(
                                        plan_id, "confirmed"
                                    )
                                except Exception as e:
                                    # Log but don't fail
                                    if printer:
                                        printer.print(
                                            f"[dim yellow]⚠ Could not update plan "
                                            f"status: {e}[/dim yellow]"
                                        )
                            output_message = (
                                f"Plan '{plan_id}' accepted. "
                                f"Ready to proceed with implementation."
                            )
                            break
                        elif choice == "accept_all":
                            # Update status to confirmed in database
                            if plan_id:
                                try:
                                    self.services.ml_plans.update_status(
                                        plan_id, "confirmed"
                                    )
                                except Exception as e:
                                    # Log but don't fail
                                    if printer:
                                        printer.print(
                                            f"[dim yellow]⚠ Could not update plan "
                                            f"status: {e}[/dim yellow]"
                                        )
                            # Enable auto-accept for this session
                            if self.agent:
                                self.agent.ml_plan_auto_accept = True
                            output_message = (
                                f"Plan '{plan_id}' accepted. "
                                f"Auto-accept enabled for this session."
                            )
                            break
                        elif choice == "feedback":
                            # Mark current plan as rejected since user wants to revise
                            if plan_id:
                                with contextlib.suppress(Exception):
                                    self.services.ml_plans.update_status(
                                        plan_id, "rejected"
                                    )
                            # Get instruction and loop to revise
                            current_instruction = result.get("feedback", "")
                            version += 1
                            # Continue loop to generate revised plan
                            continue
                        elif choice == "cancel":
                            # Mark plan as rejected
                            if plan_id:
                                with contextlib.suppress(Exception):
                                    self.services.ml_plans.update_status(
                                        plan_id, "rejected"
                                    )
                            # Print cancellation message inside section
                            if printer:
                                printer.print("")  # Empty line
                                printer.print("[dim]✗ ML plan cancelled by user.[/dim]")
                            # Return to main agent with context message
                            # (Message already displayed in section,
                            # but agent needs context)
                            return ToolResult(
                                success=True,
                                output="✗ ML plan cancelled by user.",
                                metadata={"cancelled": True, "suppress_output": True},
                            )
                    else:
                        # Headless mode - auto-accept
                        # Update status to confirmed since it's auto-accepted
                        if plan_id:
                            with contextlib.suppress(Exception):
                                self.services.ml_plans.update_status(
                                    plan_id, "confirmed"
                                )
                        formatted_result = plan.format_for_display()
                        output_message = (
                            f"Plan '{plan_id}' created successfully.\n\n"
                            f"{formatted_result}"
                        )
                        break

                # Display registration confirmation in the ML Plan section
                if printer and plan_id:
                    printer.print("")  # Empty line before confirmation
                    table_count = len(source_tables.split(","))
                    printer.print(
                        f"[dim]✓ Plan '{plan_id}' saved to database "
                        f"(v{version} • {stage} • {table_count} tables)[/dim]"
                    )

                return ToolResult(
                    success=True,
                    output=output_message,
                    metadata={
                        "ml_plan": plan_dict,
                        "plan_id": plan_id,  # Top-level for easy LLM access
                        "is_revision": previous_plan is not None,
                        "recommended_knowledge_ids": plan.recommended_knowledge_ids,
                    },
                )

            except Exception as exc:
                from arc.core.ml_plan import MLPlanError

                if isinstance(exc, MLPlanError):
                    return _error_in_section(str(exc))
                return _error_in_section(f"Unexpected error during ML planning: {exc}")
