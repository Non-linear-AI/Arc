"""Data processor generator agent for creating SQL feature configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import jinja2

from arc.core.client import ArcClient
from arc.graph.features.data_source import DataSourceSpec

if TYPE_CHECKING:
    from arc.database.services.container import ServiceContainer


class DataProcessorGeneratorError(Exception):
    """Exception raised when data processor generation fails."""


class DataProcessorGeneratorAgent:
    """Agent for generating data processing YAML from natural language."""

    # System message for data processing generation
    SYSTEM_MESSAGE = (
        "You are an expert SQL data engineer specializing in feature engineering "
        "and data transformation pipelines. Generate COMPLETE, PRODUCTION-READY "
        "JSON configurations for data processing pipelines with concrete SQL queries. "
        "Your generated configuration must be comprehensive and require no further "
        "modifications or enhancements. Follow the exact JSON schema provided in the "
        "prompt. Write specific, concrete SQL without variables or placeholders. "
        "Include all necessary data cleaning, validation, and transformation steps. "
        "Do not include any explanations or markdown formatting - return only the "
        "complete JSON object."
    )

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

            # Render prompt and build messages
            user_prompt = await self._render_prompt(context, schema_info, target_tables)
            messages = [
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": user_prompt},
            ]

            # Generate JSON using LLM
            json_content = await self._generate_json_with_llm(messages)

            # Parse JSON and create DataSourceSpec
            try:
                data = json.loads(json_content)
                spec = DataSourceSpec.from_dict(data)
                yaml_content = spec.to_yaml()
                return spec, yaml_content
            except (json.JSONDecodeError, ValueError) as e:
                # Try to retry with error feedback
                retry_result = await self._retry_generation_with_feedback(
                    messages, json_content, str(e)
                )
                if retry_result:
                    return retry_result
                raise DataProcessorGeneratorError(
                    f"Generated invalid JSON configuration: {e}"
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
                        context["tables"].append(
                            {
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
                        )
            else:
                # Get basic info for all tables
                for table in schema_info.tables[:10]:  # Limit to first 10 tables
                    columns = schema_info.get_columns_for_table(table.name)
                    context["tables"].append(
                        {
                            "name": table.name,
                            "columns": [
                                {
                                    "name": col.column_name,
                                    "type": col.data_type,
                                    "nullable": col.is_nullable,
                                }
                                for col in columns[:5]  # Limit to first 5 columns
                            ],
                        }
                    )

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
            Rendered prompt string for user message

        Raises:
            DataProcessorGeneratorError: If template cannot be loaded or rendered
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
                user_context=context,
                schema_info=schema_info,
                target_tables=target_tables or [],
            )

            return prompt

        except jinja2.TemplateNotFound as e:
            raise DataProcessorGeneratorError(
                f"Prompt template not found: {template_path}. "
                "This is a configuration error in the DataProcessorGeneratorAgent."
            ) from e
        except jinja2.TemplateError as e:
            raise DataProcessorGeneratorError(
                f"Failed to render prompt template: {e}. "
                "Check the template syntax in {template_path}."
            ) from e
        except Exception as e:
            raise DataProcessorGeneratorError(
                f"Unexpected error loading prompt template: {e}"
            ) from e

    async def _generate_json_with_llm(self, messages: list[dict[str, str]]) -> str:
        """Generate JSON content using LLM.

        Args:
            messages: List of message dictionaries for LLM chat

        Returns:
            Generated JSON content

        Raises:
            DataProcessorGeneratorError: If LLM generation fails
        """
        try:
            response = await self.client.chat(messages, tools=[])

            if not response.content:
                raise DataProcessorGeneratorError("LLM returned empty response")

            # Extract JSON from response (handle cases where LLM adds markdown)
            json_content = response.content.strip()

            # Remove markdown code blocks if present
            if json_content.startswith("```json"):
                json_content = json_content[7:]
            elif json_content.startswith("```"):
                json_content = json_content[3:]

            if json_content.endswith("```"):
                json_content = json_content[:-3]

            return json_content.strip()

        except Exception as e:
            raise DataProcessorGeneratorError(f"LLM generation failed: {e}") from e

    async def _retry_generation_with_feedback(
        self,
        original_messages: list[dict[str, str]],
        failed_json: str,
        error_message: str,
    ) -> tuple[DataSourceSpec, str] | None:
        """Retry generation with feedback about the error.

        Args:
            original_messages: The original messages that were sent
            failed_json: The JSON that failed to parse
            error_message: The error message from parsing failure

        Returns:
            Tuple of (DataSourceSpec, YAML) if retry succeeds, None otherwise
        """
        try:
            # Add error feedback to the conversation
            retry_messages = original_messages + [
                {"role": "assistant", "content": failed_json},
                {
                    "role": "user",
                    "content": (
                        f"Your previous response failed to parse with this error: "
                        f"{error_message}\n\nPlease fix the JSON and try again. "
                        f"Ensure it's valid JSON that matches the schema exactly."
                    ),
                },
            ]

            json_content = await self._generate_json_with_llm(retry_messages)
            data = json.loads(json_content)
            spec = DataSourceSpec.from_dict(data)
            yaml_content = spec.to_yaml()
            return spec, yaml_content

        except Exception:
            # If retry also fails, return None to fall back to original error
            return None
