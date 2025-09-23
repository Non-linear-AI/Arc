"""Data processor generator agent for creating SQL feature engineering configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import jinja2

from ...client import ArcClient
from ....graph.features.data_source import DataSourceSpec

if TYPE_CHECKING:
    from ...database.services.container import ServiceContainer


class DataProcessorGeneratorError(Exception):
    """Exception raised when data processor generation fails."""


class DataProcessorGeneratorAgent:
    """Agent for generating data processing YAML configurations from natural language."""

    def __init__(
        self,
        services: ServiceContainer,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize DataProcessorGeneratorAgent.

        Args:
            services: Service container for database access
            api_key: API key for LLM calls
            base_url: Base URL for LLM API
            model: Model name to use
        """
        self.services = services
        self.api_key = api_key
        self.base_url = base_url
        self.model = model or "gpt-4"
        self.client = ArcClient(api_key, self.model, base_url)

    async def generate_data_processing_yaml(
        self,
        context: str,
        target_tables: list[str] | None = None,
        target_db: str = "user",
    ) -> tuple[DataSourceSpec, str]:
        """Generate data processing YAML from natural language description.

        Args:
            context: Natural language description of data processing requirements
            target_tables: List of tables to analyze (optional)
            target_db: Target database for schema discovery

        Returns:
            Tuple of (DataSourceSpec object, YAML string)

        Raises:
            DataProcessorGeneratorError: If generation fails
        """
        try:
            # Get schema information for available tables
            schema_info = await self._get_schema_context(target_tables, target_db)

            # Load and render the prompt template
            prompt = await self._render_prompt(context, schema_info, target_tables)

            # Generate YAML using LLM
            yaml_content = await self._generate_with_llm(prompt)

            # Parse and validate the generated YAML
            try:
                spec = DataSourceSpec.from_yaml(yaml_content)
                return spec, yaml_content
            except ValueError as e:
                raise DataProcessorGeneratorError(
                    f"Generated invalid YAML configuration: {e}"
                ) from e

        except Exception as e:
            if isinstance(e, DataProcessorGeneratorError):
                raise
            raise DataProcessorGeneratorError(
                f"Failed to generate data processing configuration: {e}"
            ) from e

    async def _get_schema_context(
        self, target_tables: list[str] | None, target_db: str
    ) -> dict:
        """Get schema information for context.

        Args:
            target_tables: Specific tables to analyze
            target_db: Target database

        Returns:
            Dictionary with schema information
        """
        try:
            schema_service = self.services.schema
            schema_info = schema_service.get_schema_info(target_db)

            context = {
                "database": target_db,
                "tables": [],
                "total_tables": len(schema_info.tables),
            }

            # If specific tables requested, get detailed info for those
            if target_tables:
                for table_name in target_tables:
                    if schema_info.table_exists(table_name):
                        columns = schema_info.get_columns_for_table(table_name)
                        context["tables"].append({
                            "name": table_name,
                            "columns": [
                                {
                                    "name": col.column_name,
                                    "type": col.data_type,
                                    "nullable": col.is_nullable,
                                }
                                for col in columns
                            ],
                        })
            else:
                # Get basic info for all tables
                for table in schema_info.tables[:10]:  # Limit to first 10 tables
                    columns = schema_info.get_columns_for_table(table.name)
                    context["tables"].append({
                        "name": table.name,
                        "columns": [
                            {
                                "name": col.column_name,
                                "type": col.data_type,
                                "nullable": col.is_nullable,
                            }
                            for col in columns[:5]  # Limit to first 5 columns
                        ],
                    })

            return context

        except Exception as e:
            # Return empty context if schema discovery fails
            return {
                "database": target_db,
                "tables": [],
                "total_tables": 0,
                "error": str(e),
            }

    async def _render_prompt(
        self, context: str, schema_info: dict, target_tables: list[str] | None
    ) -> str:
        """Render the prompt template with context.

        Args:
            context: User's natural language description
            schema_info: Database schema information
            target_tables: Target tables list

        Returns:
            Rendered prompt string
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
            return template.render(
                user_context=context,
                schema_info=schema_info,
                target_tables=target_tables or [],
            )

        except Exception as e:
            # Fallback to basic prompt if template loading fails
            return f"""Generate a SQL data processing pipeline YAML configuration based on this description:

Context: {context}

Available tables: {', '.join(table['name'] for table in schema_info.get('tables', []))}

Create a YAML configuration with the following structure:
- data_source section
- vars for parameter substitution (optional)
- steps with name, depends_on, and sql
- outputs listing final tables to materialize

Generate YAML only, no explanations."""

    async def _generate_with_llm(self, prompt: str) -> str:
        """Generate YAML content using LLM.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            Generated YAML content

        Raises:
            DataProcessorGeneratorError: If LLM generation fails
        """
        try:
            messages = [
                {"role": "system", "content": "You are an expert SQL data engineer. Generate only valid YAML configurations for data processing pipelines."},
                {"role": "user", "content": prompt}
            ]

            response = await self.client.chat(messages, tools=[])

            if not response.content:
                raise DataProcessorGeneratorError("LLM returned empty response")

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
            raise DataProcessorGeneratorError(
                f"LLM generation failed: {e}"
            ) from e