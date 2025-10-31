"""Data processing tool for generating SQL feature engineering YAML configurations."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING

import yaml

from arc.tools.base import BaseTool, ToolResult
from arc.utils import ConfirmationService
from arc.utils.yaml_workflow import YamlConfirmationWorkflow

if TYPE_CHECKING:
    from arc.core.agents.ml_data.ml_data import (
        MLDataAgent,
    )
    from arc.database.services.container import ServiceContainer
else:
    # Avoid circular imports at runtime
    try:
        from arc.core.agents.ml_data.ml_data import (
            MLDataAgent,
            MLDataError,
        )
    except ImportError:
        MLDataAgent = None
        MLDataError = Exception


class MLDataTool(BaseTool):
    """Tool for generating data processing YAML configurations from natural language.

    This tool has a single purpose: convert natural language descriptions into
    structured SQL data processing pipelines using LLM generation.
    It requires an API key and uses the MLDataAgent.
    """

    def __init__(
        self,
        services: ServiceContainer,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        ui_interface=None,
    ):
        """Initialize MLDataTool.

        Args:
            services: ServiceContainer instance providing database access
            api_key: API key for LLM calls (can be empty for enterprise gateway)
            base_url: Base URL for LLM calls
            model: Model name for LLM calls
            ui_interface: UI interface for interactive confirmation workflow

        Raises:
            RuntimeError: If MLDataAgent cannot be initialized

        Note:
            Empty API keys are allowed to support enterprise gateway environments
            where authentication is handled by the gateway (only base_url needed).
            The validation is handled at the CLI level in _get_ml_tool_config().
        """
        # Allow empty API key for enterprise gateway mode
        # The OpenAI SDK will use the provided base_url for authentication

        if MLDataAgent is None:
            raise RuntimeError(
                "MLDataAgent is not available. "
                "This is likely due to missing dependencies or circular imports."
            )

        self.services = services
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.ui = ui_interface

        # Initialize the generator agent (required)
        try:
            self.generator_agent = MLDataAgent(
                services=services, api_key=api_key, base_url=base_url, model=model
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize MLDataAgent: {str(e)}. "
                "This tool requires a working LLM connection for data processing "
                "generation."
            ) from e

    def _parse_sql_to_array(self, sql_string: str) -> list[str]:
        """Parse SQL string into array of clean SQL statements.

        Splits the SQL by step markers and removes transaction boilerplate.

        Args:
            sql_string: Full SQL string with step markers and transaction

        Returns:
            List of clean SQL statements without step markers
        """
        if not sql_string:
            return []

        import re

        # Remove transaction markers from the entire SQL first
        # This handles cases where COMMIT appears at the end of SQL blocks
        sql_string = sql_string.replace("BEGIN TRANSACTION;", "")
        sql_string = sql_string.replace("COMMIT;", "")

        # Split by step markers (e.g., "-- Step 1/6: ...")
        # Pattern: -- Step X/Y: step_name (type)
        step_pattern = r"-- Step \d+/\d+: [^\n]+\n"

        # Split on step markers but keep the SQL parts
        parts = re.split(step_pattern, sql_string)

        # Clean and collect non-empty SQL statements
        sql_statements = []
        for part in parts:
            # Clean up whitespace
            cleaned = part.strip()

            # Skip empty parts
            if not cleaned:
                continue

            sql_statements.append(cleaned)

        return sql_statements

    def _build_data_result(
        self,
        status: str,
        data_processing_id: str,
        output_tables: list[str] | None = None,
        sql_operations: list[str] | None = None,
    ) -> str:
        """Build structured JSON result for ML data tool.

        Args:
            status: "accepted" or "cancelled"
            data_processing_id: Data processing execution ID
            output_tables: List of created table names
            sql_operations: List of SQL statements (or single string to parse)

        Returns:
            JSON string with structured data processing result
        """
        # Parse SQL if it's a single string
        if sql_operations and isinstance(sql_operations, str):
            sql_operations = self._parse_sql_to_array(sql_operations)

        result = {
            "status": status,
            "data_processing_id": data_processing_id,
            "execution": {
                "output_tables": output_tables or [],
                "sql_operations": sql_operations or [],
            },
        }
        return json.dumps(result)

    async def generate(
        self,
        name: str,
        source_tables: list[str],
        instruction: str | None = None,
        database: str = "user",
        auto_confirm: bool = False,
        plan_id: str | None = None,
        recommended_knowledge_ids: list[str] | None = None,
    ) -> ToolResult:
        """Generate YAML configuration from instruction using LLM.

        Args:
            name: Name for the data processor (will be registered in database)
            source_tables: List of source tables to read from (required to narrow
                scope of data exploration)
            instruction: Detailed instruction for data processing (PRIMARY driver,
                shaped by main agent or provided directly)
            database: Database to use - "system" or "user"
            auto_confirm: Skip interactive confirmation workflow
            plan_id: Optional ML plan ID (e.g., 'pidd-plan-v1') containing
                feature engineering guidance (SECONDARY baseline)

        Note on instruction vs plan_id precedence:
            - instruction: PRIMARY driver - user's immediate, specific data
              processing request
            - plan_id: SECONDARY baseline - loads plan from DB for feature
              engineering guidance
            - When there's a conflict, instruction takes precedence
            - Example: If instruction says "create interaction features" but plan says
              "use raw features only", the processor should create interaction features
              (instruction wins)
            - The LLM agent should use plan as baseline feature engineering guidance
              and augment/override it with specifics from instruction

        Returns:
            ToolResult with operation result
        """
        # Build metadata for section title (do this before validation for clean code)
        metadata_parts = []
        if plan_id:
            metadata_parts.append(plan_id)
        if recommended_knowledge_ids:
            metadata_parts.extend(recommended_knowledge_ids)

        # Use context manager for section printing (automatic cleanup and spacing)
        with self._section_printer(
            self.ui, "ML Data", metadata=metadata_parts
        ) as printer:
            # Helper to show error and return (define early for use in validation)
            def _error_in_section(message: str) -> ToolResult:
                if printer:
                    printer.print("")
                    printer.print(f"✗ {message}")
                return ToolResult(
                    success=False,
                    output=message,
                    metadata={"error_shown": True, "error_message": message},
                )

            # Validate required parameters
            if not name:
                return _error_in_section(
                    "Name is required for data processor registration. "
                    "Provide a name for the data processor."
                )

            if not instruction:
                return _error_in_section(
                    "Instruction is required for YAML generation. "
                    "Provide a detailed instruction for your data processing needs."
                )

            if not source_tables or len(source_tables) == 0:
                return _error_in_section(
                    "source_tables is required to narrow the scope of data "
                    "exploration. Specify which tables to read from "
                    "(e.g., ['users', 'transactions'])."
                )

            # Validate database
            if database not in ["system", "user"]:
                return _error_in_section(
                    f"Invalid database: {database}. Must be 'system' or 'user'."
                )

            # Load plan from database if plan_id is provided
            # If plan is provided, use it as baseline and augment with instruction
            ml_plan_data_plan = None
            if plan_id:
                try:
                    # Load plan from database using service
                    ml_plan = self.services.ml_plans.get_plan_content(plan_id)

                    from arc.core.ml_plan import MLPlan

                    plan = MLPlan.from_dict(ml_plan)
                    # Extract data plan guidance for data processing
                    ml_plan_data_plan = plan.data_plan

                    # Extract stage-specific knowledge IDs from plan
                    if not recommended_knowledge_ids:
                        recommended_knowledge_ids = plan.knowledge.get("data", [])
                except ValueError as e:
                    return _error_in_section(f"Failed to load ML plan '{plan_id}': {e}")
                except Exception as e:
                    return _error_in_section(
                        f"Unexpected error loading ML plan '{plan_id}': {e}"
                    )

            # Build final instruction: use ML plan as baseline context if available
            # Main agent can provide shaped instruction that builds on the plan
            if ml_plan_data_plan:
                # ML plan provides baseline, instruction adds specifics
                enhanced_instruction = (
                    f"{instruction}\n\n"
                    f"ML Plan Data Plan Guidance (use as baseline):\n"
                    f"{ml_plan_data_plan}"
                )
            else:
                enhanced_instruction = instruction

            # Show task description
            if printer:
                printer.print(f"[dim]Task: {instruction}[/dim]")
                printer.print("")  # Empty line after task

            try:
                # Set progress callback for this invocation
                # (agent can be reused with different printers)
                if printer:
                    self.generator_agent.progress_callback = printer.print
                else:
                    self.generator_agent.progress_callback = None

                # Preload stage-specific knowledge from plan
                preloaded_knowledge = None
                if recommended_knowledge_ids:
                    preloaded_knowledge = (
                        self.generator_agent.knowledge_loader.load_multiple(
                            recommended_knowledge_ids
                        )
                    )

                # Generate using LLM (generator_agent is guaranteed to exist)
                (
                    spec,
                    yaml_content,
                    conversation_history,  # Store for interactive editing workflow
                ) = await self.generator_agent.generate_data_processing_yaml(
                    instruction=enhanced_instruction,
                    name=name,
                    source_tables=source_tables,
                    database=database,
                    preloaded_knowledge=preloaded_knowledge,
                )

                # Show completion message
                if printer:
                    printer.print("[dim]✓ Data processor generated successfully[/dim]")

                # Spec is already validated by agent (with retries)
                # No need for redundant validation here

                # Interactive confirmation workflow
                # (unless auto_confirm is True or no UI available)
                # Get UI from singleton if not provided directly
                ui = self.ui
                if ui is None:
                    ui = ConfirmationService.get_instance()._ui

                if not auto_confirm and ui:
                    workflow = YamlConfirmationWorkflow(
                        validator_func=self._create_validator(),
                        editor_func=self._create_editor(),
                        ui_interface=ui,
                        yaml_type_name="data processor",
                        yaml_suffix=".arc-data.yaml",
                    )

                    context_dict = {
                        "name": name,
                        "instruction": str(enhanced_instruction),
                        "source_tables": source_tables,
                        "database": database,
                    }

                    try:
                        proceed, final_yaml = await workflow.run_workflow(
                            yaml_content,
                            context_dict,
                            None,  # No output path - saved to DB only
                            conversation_history,  # Pass history for editing
                        )
                        if not proceed:
                            # Generate placeholder data_processing_id for cancelled state
                            cancelled_id = f"cancelled_{uuid.uuid4().hex[:8]}"
                            output_json = self._build_data_result(
                                status="cancelled",
                                data_processing_id=cancelled_id,
                                output_tables=[],
                                sql_operations=[],
                            )
                            return ToolResult(
                                success=True,
                                output=output_json,
                                metadata={"cancelled": True},
                            )
                        yaml_content = final_yaml
                        # Re-parse spec from edited YAML
                        from arc.graph.features.data_source import DataSourceSpec

                        spec = DataSourceSpec.from_yaml(yaml_content)
                    finally:
                        workflow.cleanup()

                # Register data processor in database
                from arc.ml.runtime import MLRuntime, MLRuntimeError

                runtime = MLRuntime(self.services, artifacts_dir="artifacts")
                try:
                    # Use spec.description if available, otherwise use name
                    description = spec.description if spec.description else name
                    processor = runtime.register_data_processor(
                        name=name, spec=spec, description=description
                    )
                    # Display registration confirmation in the Data Processor section
                    if printer:
                        printer.print("")
                        printer.print(
                            f"[dim]✓ Data processor '{name}' registered to database "
                            f"({processor.id} • {len(spec.steps)} steps)[/dim]"
                        )
                except MLRuntimeError as e:
                    return _error_in_section(
                        f"Failed to register data processor: {str(e)}"
                    )

                # Execute the pipeline automatically after confirmation
                from arc.ml.data_source_executor import (
                    DataSourceExecutionError,
                    execute_data_source_pipeline,
                )

                # Show execution message
                if printer:
                    printer.print("")
                    printer.print("[dim]→ Executing data processing pipeline...[/dim]")

                # Define progress callback for real-time updates
                def progress_callback(message: str, level: str):
                    """Handle progress updates during execution."""
                    if printer:
                        if level == "step":
                            printer.print(f"[dim]  {message}[/dim]")
                        elif level == "warning":
                            printer.print(f"[yellow]  ⚠️ {message}[/yellow]")
                        elif level == "error":
                            printer.print(f"[red]  ❌ {message}[/red]")

                try:
                    execution_result = await execute_data_source_pipeline(
                        spec, database, self.services.db_manager, progress_callback
                    )

                    # Invalidate schema cache since new tables were created
                    self.services.schema.invalidate_cache(database)

                    # Store execution record in database
                    # Use processor.id as data_processing_id for consistency with other ML tools
                    # (e.g., "diabetes-feature-processing-v3" instead of random "data_98f4ac51")
                    data_processing_id = processor.id
                    if execution_result.sql and execution_result.outputs:
                        try:
                            self.services.plan_executions.store_execution(
                                execution_id=data_processing_id,
                                plan_id=plan_id
                                or "standalone",  # Use "standalone" for ad-hoc executions
                                step_type="data_processing",
                                context=execution_result.sql,
                                outputs=execution_result.outputs,
                                status="completed",
                            )
                        except Exception as store_error:
                            # Log but don't fail the operation if storage fails
                            if printer:
                                printer.print(
                                    f"[yellow]⚠️  Failed to store execution record: {store_error}[/yellow]"
                                )

                except DataSourceExecutionError as e:
                    # Generation and registration succeeded, but execution failed
                    if printer:
                        printer.print("")
                        printer.print(
                            f"[yellow]⚠️  Pipeline execution failed: {str(e)}[/yellow]"
                        )
                        printer.print("")
                        printer.print(
                            "[dim]The data processor was successfully generated and "
                            "registered to the database.[/dim]"
                        )
                        printer.print(
                            "[dim]You can review and fix the SQL queries, then re-run "
                            "the processor.[/dim]"
                        )

                    # Return FAILURE with detailed metadata for agent to understand
                    # Generation/registration succeeded, but execution failed
                    return ToolResult(
                        success=False,  # Execution failed - agent needs to know
                        output=f"Data processor '{name}' registered as {processor.id}, "
                        f"but execution failed: {str(e)}",
                        metadata={
                            "processor_id": processor.id,
                            "processor_name": name,
                            "generation_succeeded": True,
                            "registration_succeeded": True,
                            "execution_succeeded": False,
                            "execution_error": str(e),
                            "error_type": "execution_failure",
                        },
                    )

                # Show success summary
                if printer:
                    printer.print("")
                    printer.print(
                        f"[dim]✓ Pipeline executed successfully "
                        f"({', '.join(execution_result.created_tables)} created • "
                        f"{execution_result.execution_time:.2f}s)[/dim]"
                    )

                # Build structured JSON output
                output_json = self._build_data_result(
                    status="accepted",
                    data_processing_id=data_processing_id,
                    output_tables=execution_result.created_tables,
                    sql_operations=execution_result.sql or [],
                )

                # Build metadata
                metadata = {
                    "processor_id": processor.id,
                    "processor_name": name,
                    "generation_succeeded": True,
                    "registration_succeeded": True,
                    "execution_succeeded": True,
                    "created_tables": execution_result.created_tables,
                    "execution_time": execution_result.execution_time,
                    "data_processing_id": data_processing_id,
                }

                # Return success with structured JSON output
                return ToolResult(
                    success=True,
                    output=output_json,
                    metadata=metadata,
                )

            except Exception as e:
                # Preserve validation error details for main LLM
                error_msg = str(e)
                return _error_in_section(
                    f"Failed to generate data processor after 3 attempts: {error_msg}"
                )

    async def execute(self, **kwargs) -> ToolResult:
        """Execute method for BaseTool compatibility."""
        return await self.generate(**kwargs)

    def _create_validator(self):
        """Create validator function for the workflow.

        Returns:
            Validator function that takes YAML string and returns list of errors
        """
        from arc.graph.features.data_source import DataSourceSpec

        def validate(yaml_str: str) -> list[str]:
            """Validate YAML string as DataSourceSpec.

            Args:
                yaml_str: YAML string to validate

            Returns:
                List of validation error messages (empty if valid)
            """
            try:
                # Try to parse and validate as DataSourceSpec
                DataSourceSpec.from_yaml(yaml_str)
                return []  # No errors
            except yaml.YAMLError as e:
                return [f"Invalid YAML: {e}"]
            except ValueError as e:
                return [f"Validation error: {e}"]
            except Exception as e:
                return [f"Unexpected error: {e}"]

        return validate

    def _create_editor(self):
        """Create editor function for AI-assisted editing in the workflow.

        Returns:
            Editor function that takes (yaml, feedback, context, conversation_history)
            and returns tuple of (edited_yaml, updated_history)
        """

        async def edit(
            yaml_content: str,
            feedback: str,
            context: dict[str, str],
            conversation_history: list[dict[str, str]] | None = None,
        ) -> tuple[str | None, list[dict[str, str]] | None]:
            """Edit YAML with AI assistance using conversation history.

            Args:
                yaml_content: Current YAML content
                feedback: User feedback describing desired changes
                context: Context dictionary with generation parameters
                conversation_history: Optional conversation history for
                    continuing conversation

            Returns:
                Tuple of (edited_yaml, updated_conversation_history) or
                (None, None) if failed
            """
            try:
                # Set progress callback for editing invocation
                # Note: progress_callback should already be set from generate(),
                # but we keep it here for robustness
                # (it's the same agent instance, so callback is preserved)

                # Call the generator agent with conversation history
                # feedback becomes the instruction in edit mode
                (
                    _spec,
                    edited_yaml,
                    updated_history,
                ) = await self.generator_agent.generate_data_processing_yaml(
                    instruction=feedback,  # User's change request
                    name=context.get("name", ""),
                    source_tables=context.get("source_tables"),
                    database=context.get("database", "user"),
                    existing_yaml=yaml_content,
                    recommended_knowledge_ids=None,  # Editing uses conversation_history
                    conversation_history=conversation_history,  # Continue conversation
                )
                return edited_yaml, updated_history

            except Exception as e:
                if self.ui:
                    self.ui.show_system_error(f"❌ AI editing failed: {str(e)}")
                return None, None

        return edit
