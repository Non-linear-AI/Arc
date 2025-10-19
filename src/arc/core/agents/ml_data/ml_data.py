"""Arc ML data agent for creating SQL feature configurations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jinja2

from arc.core.client import ArcClient
from arc.graph.features.data_source import DataSourceSpec

if TYPE_CHECKING:
    from arc.database.services.container import ServiceContainer


class MLDataError(Exception):
    """Exception raised when data processor generation fails."""


class MLDataAgent:
    """Agent for generating data processing YAML from natural language."""

    def __init__(
        self,
        services: ServiceContainer,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize MLDataAgent.

        Args:
            services: Service container for database access
            api_key: API key for LLM calls
            base_url: Base URL for LLM API
            model: Model name to use
        """
        self.services = services
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = ArcClient(api_key, self.model, base_url)

    async def generate_data_processing_yaml(
        self,
        instruction: str,
        source_tables: list[str] | None = None,
        database: str = "user",
        existing_yaml: str | None = None,
    ) -> tuple[DataSourceSpec, str]:
        """Generate or edit data processing YAML from instruction.

        Args:
            instruction: Detailed instruction for data processing.
                For generation: shaped by main agent or from ML plan.
                For editing: user feedback on what to change.
            source_tables: List of source tables to read from (optional)
            database: Database to use for schema discovery
            existing_yaml: Existing YAML content to edit (optional).
                If provided, switches to editing mode where instruction
                describes the changes to make.

        Returns:
            Tuple of (DataSourceSpec object, YAML string)

        Raises:
            MLDataError: If generation fails
        """
        try:
            # Get schema information for available tables
            schema_info = await self._get_schema_context(source_tables, database)

            # Render prompt and build messages
            # Mode determined by presence of existing_yaml
            system_prompt = await self._render_system_prompt(
                instruction,
                schema_info,
                source_tables,
                existing_yaml,
            )
            messages = [
                {"role": "system", "content": system_prompt},
            ]

            # Generate YAML using LLM
            yaml_content = await self._generate_yaml_with_llm(messages)

            # Parse YAML and create DataSourceSpec
            try:
                spec = DataSourceSpec.from_yaml(yaml_content)
                return spec, yaml_content
            except ValueError as e:
                # Try to retry with error feedback
                retry_result = await self._retry_generation_with_feedback(
                    messages, yaml_content, str(e)
                )
                if retry_result:
                    return retry_result
                raise MLDataError(f"Generated invalid YAML configuration: {e}") from e

        except Exception as e:
            if isinstance(e, MLDataError):
                raise
            raise MLDataError(
                f"Failed to generate data processing configuration: {e}"
            ) from e

    async def _get_schema_context(
        self, source_tables: list[str] | None, database: str
    ) -> dict:
        """Get schema information with statistics for context.

        Args:
            source_tables: Specific source tables to analyze
            database: Database to use

        Returns:
            Dictionary with schema information and table statistics
        """
        try:
            schema_service = self.services.schema
            schema_info = schema_service.get_schema_info(database)

            context = {
                "database": database,
                "tables": [],
                "total_tables": len(schema_info.tables),
            }

            # If specific tables requested, get detailed info with statistics
            if source_tables:
                for table_name in source_tables:
                    if schema_info.table_exists(table_name):
                        # Get schema columns
                        columns = schema_info.get_columns_for_table(table_name)

                        # Get table statistics using ml_data service
                        table_info = {
                            "name": table_name,
                            "columns": [
                                {
                                    "name": col.column_name,
                                    "type": col.data_type,
                                    "nullable": col.is_nullable,
                                }
                                for col in columns
                            ],
                        }

                        # Try to get dataset info for row counts
                        try:
                            dataset_info = self.services.ml_data.get_dataset_info(
                                table_name
                            )
                            if dataset_info:
                                table_info["row_count"] = dataset_info.row_count
                        except Exception:
                            # If statistics fail, continue without them
                            pass

                        context["tables"].append(table_info)
            else:
                # Get basic info for all tables (limit to 10)
                for table in schema_info.tables[:10]:
                    columns = schema_info.get_columns_for_table(table.name)

                    table_info = {
                        "name": table.name,
                        "columns": [
                            {
                                "name": col.column_name,
                                "type": col.data_type,
                                "nullable": col.is_nullable,
                            }
                            for col in columns[:5]  # Limit to 5 columns
                        ],
                    }

                    # Try to get row count for each table
                    try:
                        dataset_info = self.services.ml_data.get_dataset_info(
                            table.name
                        )
                        if dataset_info:
                            table_info["row_count"] = dataset_info.row_count
                    except Exception:
                        # If statistics fail, continue without them
                        pass

                    context["tables"].append(table_info)

            return context

        except Exception as e:
            # Return empty context if schema discovery fails
            return {
                "database": database,
                "tables": [],
                "total_tables": 0,
                "error": str(e),
            }

    async def _render_system_prompt(
        self,
        instruction: str,
        schema_info: dict,
        source_tables: list[str] | None,
        existing_yaml: str | None = None,
    ) -> str:
        """Render the system prompt template with instruction.

        Args:
            instruction: Detailed instruction (for generation or editing)
            schema_info: Database schema information
            source_tables: Source tables list
            existing_yaml: Existing YAML content to edit (optional)

        Returns:
            Rendered prompt string for system message

        Raises:
            MLDataError: If template cannot be loaded or rendered
        """
        template_path = Path(__file__).parent / "templates" / "prompt.j2"

        try:
            # Create Jinja2 environment
            template_dir = template_path.parent
            env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(template_dir),
                trim_blocks=True,
                lstrip_blocks=True,
            )

            # Load and render template
            template = env.get_template("prompt.j2")
            prompt = template.render(
                user_instruction=instruction,
                schema_info=schema_info,
                source_tables=source_tables or [],
                existing_yaml=existing_yaml,
            )

            return prompt

        except jinja2.TemplateNotFound as e:
            raise MLDataError(
                f"Prompt template not found: {template_path}. "
                "This is a configuration error in the MLDataAgent."
            ) from e
        except jinja2.TemplateError as e:
            raise MLDataError(
                f"Failed to render prompt template: {e}. "
                "Check the template syntax in {template_path}."
            ) from e
        except Exception as e:
            raise MLDataError(f"Unexpected error loading prompt template: {e}") from e

    async def _generate_yaml_with_llm(self, messages: list[dict[str, str]]) -> str:
        """Generate YAML content using LLM.

        Args:
            messages: List of message dictionaries for LLM chat

        Returns:
            Generated YAML content

        Raises:
            MLDataError: If LLM generation fails
        """
        try:
            response = await self.client.chat(messages, tools=[])

            if not response.content:
                raise MLDataError("LLM returned empty response")

            # Extract YAML from response (handle cases where LLM adds markdown)
            yaml_content = response.content.strip()

            # Remove markdown code blocks if present
            if yaml_content.startswith("```yaml"):
                yaml_content = yaml_content[7:]
            elif yaml_content.startswith("```"):
                yaml_content = yaml_content[3:]

            if yaml_content.endswith("```"):
                yaml_content = yaml_content[:-3]

            return yaml_content.strip()

        except Exception as e:
            raise MLDataError(f"LLM generation failed: {e}") from e

    async def _retry_generation_with_feedback(
        self,
        original_messages: list[dict[str, str]],
        failed_yaml: str,
        error_message: str,
    ) -> tuple[DataSourceSpec, str] | None:
        """Retry generation with feedback about the error.

        Args:
            original_messages: The original messages that were sent
            failed_yaml: The YAML that failed to parse
            error_message: The error message from parsing failure

        Returns:
            Tuple of (DataSourceSpec, YAML) if retry succeeds, None otherwise
        """
        try:
            # Add error feedback to the conversation
            retry_messages = original_messages + [
                {"role": "assistant", "content": failed_yaml},
                {
                    "role": "user",
                    "content": (
                        f"Your previous response failed to parse with this error: "
                        f"{error_message}\n\nPlease fix the YAML and try again. "
                        f"Ensure it's valid YAML that matches the schema exactly."
                    ),
                },
            ]

            yaml_content = await self._generate_yaml_with_llm(retry_messages)
            spec = DataSourceSpec.from_yaml(yaml_content)
            return spec, yaml_content

        except Exception:
            # If retry also fails, return None to fall back to original error
            return None
