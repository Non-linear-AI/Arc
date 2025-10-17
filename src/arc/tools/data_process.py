"""Data processing tool for generating SQL feature engineering YAML configurations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from arc.tools.base import BaseTool, ToolResult
from arc.utils import ConfirmationService
from arc.utils.yaml_workflow import YamlConfirmationWorkflow

if TYPE_CHECKING:
    from arc.core.agents.data_process.data_process import (
        DataProcessorGeneratorAgent,
    )
    from arc.database.services.container import ServiceContainer
else:
    # Avoid circular imports at runtime
    try:
        from arc.core.agents.data_process.data_process import (
            DataProcessorGeneratorAgent,
            DataProcessorGeneratorError,
        )
    except ImportError:
        DataProcessorGeneratorAgent = None
        DataProcessorGeneratorError = Exception


class MLDataProcessTool(BaseTool):
    """Tool for generating data processing YAML configurations from natural language.

    This tool has a single purpose: convert natural language descriptions into
    structured SQL data processing pipelines using LLM generation.
    It requires an API key and uses the DataProcessorGeneratorAgent.
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
            api_key: API key for LLM calls (required)
            base_url: Base URL for LLM calls
            model: Model name for LLM calls
            ui_interface: UI interface for interactive confirmation workflow

        Raises:
            ValueError: If API key is not provided
            RuntimeError: If DataProcessorGeneratorAgent cannot be initialized
        """
        if not api_key:
            raise ValueError(
                "API key is required for MLDataProcessTool. "
                "This tool uses LLM-powered generation and cannot function "
                "without an API key."
            )

        if DataProcessorGeneratorAgent is None:
            raise RuntimeError(
                "DataProcessorGeneratorAgent is not available. "
                "This is likely due to missing dependencies or circular imports."
            )

        self.services = services
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.ui = ui_interface

        # Initialize the generator agent (required)
        try:
            self.generator_agent = DataProcessorGeneratorAgent(
                services=services, api_key=api_key, base_url=base_url, model=model
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize DataProcessorGeneratorAgent: {str(e)}. "
                "This tool requires a working LLM connection for data processing "
                "generation."
            ) from e

    async def generate(
        self,
        name: str,
        context: str,
        target_tables: list[str] | None = None,
        output_path: str | None = None,
        target_db: str = "user",
        auto_confirm: bool = False,
        ml_plan: dict | None = None,
    ) -> ToolResult:
        """Generate YAML configuration from natural language context using LLM.

        Args:
            name: Name for the data processor (will be registered in database)
            context: Natural language description of data processing requirements
            target_tables: List of tables to analyze for generation
            output_path: Path to save generated YAML file (optional, for backup)
            target_db: Target database - "system" or "user"
            auto_confirm: Skip interactive confirmation workflow
            ml_plan: Optional ML plan dict containing feature engineering guidance

        Returns:
            ToolResult with operation result
        """
        if not name:
            return ToolResult.error_result(
                "Name is required for data processor registration. "
                "Provide a name for the data processor."
            )

        if not context:
            return ToolResult.error_result(
                "Context is required for YAML generation. "
                "Provide a natural language description of your data processing needs."
            )

        # Validate target database
        if target_db not in ["system", "user"]:
            return ToolResult.error_result(
                f"Invalid target database: {target_db}. Must be 'system' or 'user'."
            )

        # Extract feature engineering guidance from ML plan if provided
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
                color="cyan", add_dot=True
            )
            ml_data_process_section_printer = self._ml_data_process_section.__enter__()
            ml_data_process_section_printer.print("Data Processor")

            # Show ML plan feature engineering guidance if provided
            if ml_plan_feature_engineering:
                ml_data_process_section_printer.print(
                    "[dim][cyan]ℹ Using ML plan feature engineering "
                    "guidance[/cyan][/dim]"
                )

            ml_data_process_section_printer.print(
                "[dim]Generating Arc-Graph data processor specification...[/dim]"
            )

        # Augment context with ML plan feature engineering guidance if provided
        enhanced_context = context
        if ml_plan_feature_engineering:
            enhanced_context = (
                f"{context}\n\n"
                f"ML Plan Feature Engineering Guidance:\n"
                f"{ml_plan_feature_engineering}"
            )

        try:
            # Generate using LLM (generator_agent is guaranteed to exist)
            (
                spec,
                yaml_content,
            ) = await self.generator_agent.generate_data_processing_yaml(
                context=enhanced_context,
                target_tables=target_tables,
                target_db=target_db,
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
                    "context": str(enhanced_context),
                    "target_tables": target_tables,
                    "target_db": target_db,
                }

                try:
                    proceed, final_yaml = await workflow.run_workflow(
                        yaml_content, context_dict, output_path
                    )
                    if not proceed:
                        # Show cancellation message in the Data Processor section
                        if ml_data_process_section_printer:
                            ml_data_process_section_printer.print(
                                ""
                            )  # Empty line before cancellation
                            ml_data_process_section_printer.print(
                                "✗ Data processor generation cancelled by user"
                            )
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

            # Save YAML to file for backup (optional)
            if output_path:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_text(yaml_content)

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
                    spec, target_db, self.services.db_manager, progress_callback
                )

                # Invalidate schema cache since new tables were created
                self.services.schema.invalidate_cache(target_db)

            except DataSourceExecutionError as e:
                # YAML is saved even if execution fails (for debugging)
                if ml_data_process_section_printer:
                    ml_data_process_section_printer.print("")
                    ml_data_process_section_printer.print(
                        f"[red]❌ Pipeline execution failed: {str(e)}[/red]"
                    )
                    if output_path:
                        ml_data_process_section_printer.print(
                            f"[dim]YAML saved to: {output_path}[/dim]"
                        )
                # Close the section before returning
                if self.ui and hasattr(self, "_ml_data_process_section"):
                    self._ml_data_process_section.__exit__(None, None, None)
                return ToolResult.error_result(f"Pipeline execution failed: {str(e)}")

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

    # Keep execute method for backward compatibility (delegate to generate)
    async def execute(self, **kwargs) -> ToolResult:
        """Execute method for BaseTool compatibility."""
        # Remove action parameter if present and delegate to generate
        kwargs.pop("action", None)  # Remove deprecated action parameter
        kwargs.pop("yaml_content", None)  # Remove validation-related parameter
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
                # and editing instructions
                (
                    _spec,
                    edited_yaml,
                ) = await self.generator_agent.generate_data_processing_yaml(
                    context=context["context"],
                    target_tables=context.get("target_tables"),
                    target_db=context.get("target_db", "user"),
                    existing_yaml=yaml_content,
                    editing_instructions=feedback,
                )
                return edited_yaml

            except Exception as e:
                if self.ui:
                    self.ui.show_system_error(f"❌ AI editing failed: {str(e)}")
                return None

        return edit
