"""Data processing tool for generating SQL feature engineering YAML configurations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from arc.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from arc.core.agents.data_processor_generator.data_processor_generator import (
        DataProcessorGeneratorAgent,
    )
    from arc.database.services.container import ServiceContainer
else:
    # Avoid circular imports at runtime
    try:
        from arc.core.agents.data_processor_generator.data_processor_generator import (
            DataProcessorGeneratorAgent,
            DataProcessorGeneratorError,
        )
    except ImportError:
        DataProcessorGeneratorAgent = None
        DataProcessorGeneratorError = Exception


class DataProcessorGeneratorTool(BaseTool):
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
    ):
        """Initialize DataProcessorGeneratorTool.

        Args:
            services: ServiceContainer instance providing database access
            api_key: API key for LLM calls (required)
            base_url: Base URL for LLM calls
            model: Model name for LLM calls

        Raises:
            ValueError: If API key is not provided
            RuntimeError: If DataProcessorGeneratorAgent cannot be initialized
        """
        if not api_key:
            raise ValueError(
                "API key is required for DataProcessorGeneratorTool. "
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
        context: str,
        target_tables: list[str] | None = None,
        output_path: str | None = None,
        target_db: str = "user",
    ) -> ToolResult:
        """Generate YAML configuration from natural language context using LLM.

        Args:
            context: Natural language description of data processing requirements
            target_tables: List of tables to analyze for generation
            output_path: Path to save generated YAML file
            target_db: Target database - "system" or "user"

        Returns:
            ToolResult with operation result
        """
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

        try:
            # Show progress to build trust
            progress_lines = [
                "🔍 Analyzing database schema and table structure...",
            ]

            # Get schema information first to show what we found
            schema_service = self.services.schema
            schema_info = schema_service.get_schema_info(target_db)

            if target_tables:
                available_tables = [
                    t for t in target_tables if schema_info.table_exists(t)
                ]
                progress_lines.append(
                    f"📊 Found {len(available_tables)} specified tables with "
                    "detailed schema"
                )
                if len(available_tables) != len(target_tables):
                    missing = set(target_tables) - set(available_tables)
                    progress_lines.append(
                        f"⚠️  Note: {len(missing)} tables not found: "
                        f"{', '.join(missing)}"
                    )
            else:
                progress_lines.append(
                    f"📊 Discovered {len(schema_info.tables)} available tables in "
                    f"{target_db} database"
                )

            progress_lines.append("🤖 Generating optimized SQL pipeline with LLM...")

            # Show progress so far (for logging purposes)
            # progress_result = ToolResult.success_result("\n".join(progress_lines))

            # Generate using LLM (generator_agent is guaranteed to exist)
            (
                spec,
                yaml_content,
            ) = await self.generator_agent.generate_data_processing_yaml(
                context=context, target_tables=target_tables, target_db=target_db
            )

            # Save to file if path provided
            if output_path:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_text(yaml_content)

            # Create concise context for header (similar to SQL Query format)
            context_summary = context[:80] + "..." if len(context) > 80 else context

            lines = [
                f"Data Processor Generator: {context_summary}",
                "Generated successfully using LLM",
                f"Steps: {len(spec.steps)}, Outputs: {', '.join(spec.outputs)}",
            ]

            if target_tables:
                lines.append(f"Target tables: {', '.join(target_tables)}")

            if output_path:
                lines.append(f"Saved to: {output_path}")

            if spec.vars:
                lines.append(f"Variables: {', '.join(spec.vars.keys())}")

            lines.append("")
            lines.append(yaml_content)
            lines.append("")
            lines.append(
                "Note: Generated configuration is production-ready and should "
                "not be modified manually."
            )

            return ToolResult.success_result("\n".join(lines))

        except Exception as e:
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
