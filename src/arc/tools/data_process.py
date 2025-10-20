"""Data processing tool for generating SQL feature engineering YAML configurations."""

from __future__ import annotations

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


class MLDataProcessTool(BaseTool):
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
        """Initialize MLDataProcessTool.

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

    async def generate(
        self,
        name: str,
        source_tables: list[str],
        instruction: str | None = None,
        database: str = "user",
        auto_confirm: bool = False,
        ml_plan: dict | None = None,
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
            ml_plan: Optional ML plan dict containing feature engineering guidance
                (SECONDARY baseline, automatically injected by the main agent)

        Note on instruction vs ml_plan precedence:
            - instruction: PRIMARY driver - user's immediate, specific data
              processing request
            - ml_plan: SECONDARY baseline - background feature engineering guidance
            - When there's a conflict, instruction takes precedence
            - Example: If instruction says "create interaction features" but plan says
              "use raw features only", the processor should create interaction features
              (instruction wins)
            - The LLM agent should use ml_plan as baseline feature engineering guidance
              and augment/override it with specifics from instruction

        Returns:
            ToolResult with operation result
        """
        if not name:
            return ToolResult.error_result(
                "Name is required for data processor registration. "
                "Provide a name for the data processor."
            )

        if not instruction:
            return ToolResult.error_result(
                "Instruction is required for YAML generation. "
                "Provide a detailed instruction for your data processing needs."
            )

        if not source_tables or len(source_tables) == 0:
            return ToolResult.error_result(
                "source_tables is required to narrow the scope of data exploration. "
                "Specify which tables to read from (e.g., ['users', 'transactions'])."
            )

        # Validate database
        if database not in ["system", "user"]:
            return ToolResult.error_result(
                f"Invalid database: {database}. Must be 'system' or 'user'."
            )

        # Extract feature engineering guidance from ML plan if provided
        # If ML plan is provided, use it as baseline and augment with instruction
        ml_plan_feature_engineering = None
        if ml_plan:
            from arc.core.ml_plan import MLPlan

            plan = MLPlan.from_dict(ml_plan)
            # Extract feature engineering guidance for data processing
            ml_plan_feature_engineering = plan.feature_engineering

        # Show section title and ML plan guidance if provided
        ml_data_process_section_printer = None
        if self.ui:
            self._ml_data_process_section = self.ui._printer.section(
                color="magenta", add_dot=True
            )
            ml_data_process_section_printer = self._ml_data_process_section.__enter__()
            ml_data_process_section_printer.print("ML Data")

            # Show ML plan feature engineering guidance if provided
            if ml_plan_feature_engineering:
                ml_data_process_section_printer.print(
                    "[dim][cyan]ℹ Using ML plan feature engineering "
                    "guidance[/cyan][/dim]"
                )

            ml_data_process_section_printer.print(
                "[dim]Generating Arc-Graph data processor specification...[/dim]"
            )

        # Build final instruction: use ML plan as baseline context if available
        # Main agent can provide shaped instruction that builds on the plan
        if ml_plan_feature_engineering:
            # ML plan provides baseline, instruction adds specifics
            enhanced_instruction = (
                f"{instruction}\n\n"
                f"ML Plan Feature Engineering Guidance (use as baseline):\n"
                f"{ml_plan_feature_engineering}"
            )
        else:
            enhanced_instruction = instruction

        try:
            # Generate using LLM (generator_agent is guaranteed to exist)
            (
                spec,
                yaml_content,
            ) = await self.generator_agent.generate_data_processing_yaml(
                instruction=enhanced_instruction,
                source_tables=source_tables,
                database=database,
            )

            # Validate the generated spec before confirmation workflow
            try:
                # Parse and validate structure
                from arc.graph.features.data_source import DataSourceSpec

                validation_result = DataSourceSpec.validate_yaml_string(yaml_content)
                if not validation_result.success:
                    return ToolResult.error_result(
                        f"Generated data processor failed validation: "
                        f"{validation_result.error}"
                    )

                # Validate dependencies and execution order
                spec.validate_dependencies()
                _ = spec.get_execution_order()

            except Exception as e:
                return ToolResult.error_result(
                    f"Data processor validation failed: {str(e)}"
                )

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
                    "instruction": str(enhanced_instruction),
                    "source_tables": source_tables,
                    "database": database,
                }

                try:
                    proceed, final_yaml = await workflow.run_workflow(
                        yaml_content,
                        context_dict,
                        None,  # No output path - saved to DB only
                    )
                    if not proceed:
                        # Close the section before returning
                        if self.ui and hasattr(self, "_ml_data_process_section"):
                            self._ml_data_process_section.__exit__(None, None, None)
                        return ToolResult.success_result(
                            "✗ Data processor generation cancelled by user."
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
                processor = runtime.register_data_processor(
                    name=name, spec=spec, description=spec.description
                )
                # Display registration confirmation in the Data Processor section
                if ml_data_process_section_printer:
                    ml_data_process_section_printer.print("")
                    ml_data_process_section_printer.print(
                        f"[dim]✓ Data processor '{name}' registered to database "
                        f"({processor.id} • {len(spec.steps)} steps)[/dim]"
                    )
            except MLRuntimeError as e:
                return ToolResult.error_result(
                    f"Failed to register data processor: {str(e)}"
                )

            # Execute the pipeline automatically after confirmation
            from arc.ml.data_source_executor import (
                DataSourceExecutionError,
                execute_data_source_pipeline,
            )

            # Show execution message
            if ml_data_process_section_printer:
                ml_data_process_section_printer.print("")
                ml_data_process_section_printer.print(
                    "→ Executing data processing pipeline..."
                )

            # Define progress callback for real-time updates
            def progress_callback(message: str, level: str):
                """Handle progress updates during execution."""
                if ml_data_process_section_printer:
                    if level == "step":
                        ml_data_process_section_printer.print(f"[dim]  {message}[/dim]")
                    elif level == "warning":
                        ml_data_process_section_printer.print(
                            f"[yellow]  ⚠️ {message}[/yellow]"
                        )
                    elif level == "error":
                        ml_data_process_section_printer.print(
                            f"[red]  ❌ {message}[/red]"
                        )

            try:
                execution_result = await execute_data_source_pipeline(
                    spec, database, self.services.db_manager, progress_callback
                )

                # Invalidate schema cache since new tables were created
                self.services.schema.invalidate_cache(database)

            except DataSourceExecutionError as e:
                # Generation and registration succeeded, but execution failed
                if ml_data_process_section_printer:
                    ml_data_process_section_printer.print("")
                    ml_data_process_section_printer.print(
                        f"[yellow]⚠️  Pipeline execution failed: {str(e)}[/yellow]"
                    )
                    ml_data_process_section_printer.print("")
                    ml_data_process_section_printer.print(
                        "[dim]The data processor was successfully generated and "
                        "registered to the database.[/dim]"
                    )
                    ml_data_process_section_printer.print(
                        "[dim]You can review and fix the SQL queries, then re-run "
                        "the processor.[/dim]"
                    )
                # Close the section before returning
                if self.ui and hasattr(self, "_ml_data_process_section"):
                    self._ml_data_process_section.__exit__(None, None, None)

                # Return success with warning message
                # Generation/registration succeeded, execution failed
                return ToolResult.success_result(
                    f"Data processor '{name}' registered as {processor.id}, "
                    f"but execution failed: {str(e)}"
                )

            # Show success summary
            if ml_data_process_section_printer:
                ml_data_process_section_printer.print("")
                ml_data_process_section_printer.print(
                    f"[dim]✓ Pipeline executed successfully "
                    f"({', '.join(execution_result.created_tables)} created • "
                    f"{execution_result.execution_time:.2f}s)[/dim]"
                )

            # Close the Data Processor section
            if self.ui and hasattr(self, "_ml_data_process_section"):
                self._ml_data_process_section.__exit__(None, None, None)

            # Build simple output for ToolResult
            lines = [
                f"Data processor '{name}' registered successfully as {processor.id}",
                f"Pipeline executed: "
                f"{', '.join(execution_result.created_tables)} created",
            ]

            return ToolResult.success_result("\n".join(lines))

        except Exception as e:
            # Close the section before returning error
            if self.ui and hasattr(self, "_ml_data_process_section"):
                self._ml_data_process_section.__exit__(None, None, None)
            return ToolResult.error_result(
                f"Failed to generate YAML using LLM: {str(e)}. "
                "Please check your API key and network connection, "
                "or try simplifying your request."
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
            Editor function that takes (yaml, feedback, context) and returns edited YAML
        """

        async def edit(
            yaml_content: str, feedback: str, context: dict[str, str]
        ) -> str | None:
            """Edit YAML with AI assistance.

            Args:
                yaml_content: Current YAML content
                feedback: User feedback describing desired changes
                context: Context dictionary with generation parameters

            Returns:
                Edited YAML string or None if editing failed
            """
            try:
                # Call the generator agent with existing YAML
                # feedback becomes the instruction in edit mode
                (
                    _spec,
                    edited_yaml,
                ) = await self.generator_agent.generate_data_processing_yaml(
                    instruction=feedback,  # User's change request
                    source_tables=context.get("source_tables"),
                    database=context.get("database", "user"),
                    existing_yaml=yaml_content,
                )
                return edited_yaml

            except Exception as e:
                if self.ui:
                    self.ui.show_system_error(f"❌ AI editing failed: {str(e)}")
                return None

        return edit
