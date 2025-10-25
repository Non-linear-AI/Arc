"""ML Plan generation tool for creating ML workflow plans."""

from __future__ import annotations

from datetime import UTC, datetime

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
        instruction: str | None = None,
        source_tables: str | None = None,
        previous_plan: dict | None = None,
        section_to_update: str | None = None,
        conversation_history: str | None = None,  # noqa: ARG002
        verbose: bool = False,
    ) -> ToolResult:
        if not self.api_key:
            return ToolResult.error_result(
                "API key required for ML planning. "
                "Set ARC_API_KEY or configure an API key before using this tool."
            )

        if not self.services:
            return ToolResult.error_result(
                "ML planning service unavailable. Database services not initialized."
            )

        if not instruction or not source_tables:
            return ToolResult.error_result(
                "Parameters 'instruction' and 'source_tables' "
                "are required for ML planning."
            )

        # Handle section update mode (different workflow)
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
        # Show section title before generation starts
        # Keep the section printer reference to use later for messages
        ml_plan_section_printer = None
        if self.ui:
            self._ml_plan_section = self.ui._printer.section(color="cyan", add_dot=True)
            ml_plan_section_printer = self._ml_plan_section.__enter__()
            ml_plan_section_printer.print("ML Plan")

        # Helper to print progress messages within the section
        def _progress_callback(message: str):
            if ml_plan_section_printer:
                ml_plan_section_printer.print(message)

        agent = MLPlanAgent(
            self.services,
            self.api_key,
            self.base_url,
            self.model,
            progress_callback=_progress_callback if ml_plan_section_printer else None,
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
            latest_plan = self.services.ml_plans.get_latest_plan_for_tables(
                str(source_tables)
            )
            version = latest_plan.version + 1 if latest_plan else 1

            while True:
                try:
                    # Generate the plan (pass source_tables as comma-separated string)
                    analysis = await agent.analyze_problem(
                        user_context=str(current_instruction),
                        source_tables=str(source_tables),
                        instruction=current_instruction
                        if current_instruction != instruction
                        else None,
                        stream=False,
                    )

                    # Show completion message
                    if ml_plan_section_printer:
                        ml_plan_section_printer.print(" Plan generated successfully")

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

                except Exception as e:
                    # Handle errors during plan generation
                    error_msg = f"Failed to generate ML plan: {str(e)}"
                    if ml_plan_section_printer:
                        ml_plan_section_printer.print("")
                        ml_plan_section_printer.print(f" {error_msg}")
                    # Close the section
                    if self.ui and hasattr(self, "_ml_plan_section"):
                        self._ml_plan_section.__exit__(None, None, None)
                    return ToolResult(
                        success=False,
                        output="",
                        metadata={"error_shown": True},
                    )

                # If auto-accept is enabled, skip workflow
                if self.agent and self.agent.ml_plan_auto_accept:
                    output_message = "Plan automatically accepted (auto-accept enabled)"
                    break

                # Display plan and run confirmation workflow
                if self.ui:
                    from arc.utils.ml_plan_workflow import MLPlanConfirmationWorkflow

                    try:
                        workflow = MLPlanConfirmationWorkflow(self.ui)
                        result = await workflow.run_workflow(
                            plan, previous_plan is not None
                        )
                        choice = result.get("choice")
                    except Exception as e:
                        # Handle workflow errors
                        error_msg = f"Workflow execution failed: {str(e)}"
                        self.ui._printer.console.print(f"[red]L {error_msg}[/red]")
                        return ToolResult.error_result(error_msg)

                    if choice == "accept":
                        output_message = (
                            "Plan accepted. Ready to proceed with implementation."
                        )
                        break
                    elif choice == "accept_all":
                        # Enable auto-accept for this session
                        if self.agent:
                            self.agent.ml_plan_auto_accept = True
                        output_message = (
                            "Plan accepted. Auto-accept enabled for this session."
                        )
                        break
                    elif choice == "feedback":
                        # Get instruction and loop to revise
                        current_instruction = result.get("feedback", "")
                        version += 1
                        # Continue loop to generate revised plan
                        continue
                    elif choice == "cancel":
                        # Print cancellation message inside section
                        if ml_plan_section_printer:
                            ml_plan_section_printer.print(
                                "ML plan cancelled. What would you like to do instead?"
                            )
                        # Close the section
                        if hasattr(self, "_ml_plan_section"):
                            self._ml_plan_section.__exit__(None, None, None)
                        # Return to main agent with context message
                        # (Message already displayed in section,
                        # but agent needs context)
                        return ToolResult(
                            success=True,
                            output=(
                                "ML plan cancelled. What would you like to do instead?"
                            ),
                            metadata={"cancelled": True, "suppress_output": True},
                        )
                else:
                    # Headless mode - auto-accept
                    formatted_result = plan.format_for_display()
                    output_message = (
                        "I've created a comprehensive ML workflow plan based on "
                        f"your requirements.\n\n{formatted_result}"
                    )
                    break

            # Save plan to database after acceptance
            try:
                from arc.database.models.ml_plan import MLPlan as MLPlanModel
                from arc.ml.runtime import _slugify_name

                # Convert plan to dict for storage
                plan_dict = plan.to_dict()
                plan_dict["source_tables"] = str(source_tables)

                # Convert plan to YAML format for better readability
                plan_yaml = yaml.dump(
                    plan_dict, default_flow_style=False, sort_keys=False
                )

                # Create database model - use first table for plan ID
                first_table = source_tables.split(",")[0].strip()
                base_slug = _slugify_name(f"{first_table}-plan")
                plan_id = f"{base_slug}-v{version}"

                now = datetime.now(UTC)
                db_plan = MLPlanModel(
                    plan_id=plan_id,
                    version=version,
                    user_context=str(instruction),
                    source_tables=str(source_tables),
                    plan_yaml=plan_yaml,  # Store as YAML string
                    status="approved",  # Plan was accepted by user
                    created_at=now,
                    updated_at=now,
                )

                # Save to database
                self.services.ml_plans.create_plan(db_plan)

                # Add plan_id to metadata for linking
                plan_dict["plan_id"] = plan_id

                # Display registration confirmation in the ML Plan section
                if ml_plan_section_printer:
                    ml_plan_section_printer.print("")  # Empty line before confirmation
                    table_count = len(source_tables.split(","))
                    ml_plan_section_printer.print(
                        f"[dim] Plan '{plan_id}' saved to database "
                        f"(v{version} • {stage} • {table_count} tables)[/dim]"
                    )

            except Exception as e:
                # Log error but don't fail - plan still in memory
                if self.ui:
                    self.ui.show_warning(
                        f" Plan saved to session but not database: {e}"
                    )

            # Close the ML Plan section
            if self.ui and hasattr(self, "_ml_plan_section"):
                self._ml_plan_section.__exit__(None, None, None)

            return ToolResult(
                success=True,
                output=output_message,
                metadata={
                    "ml_plan": plan_dict,
                    "is_revision": previous_plan is not None,
                },
            )

        except Exception as exc:
            from arc.core.ml_plan import MLPlanError

            # Close the ML Plan section before returning error
            if self.ui and hasattr(self, "_ml_plan_section"):
                self._ml_plan_section.__exit__(None, None, None)

            if isinstance(exc, MLPlanError):
                return ToolResult.error_result(str(exc))
            return ToolResult.error_result(
                f"Unexpected error during ML planning: {exc}"
            )
