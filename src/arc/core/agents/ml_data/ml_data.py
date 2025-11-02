"""Arc ML data agent for creating SQL feature configurations."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from arc.core.agents.shared.base_agent import AgentError, BaseAgent
from arc.graph.features.data_source import DataSourceSpec

if TYPE_CHECKING:
    from arc.database.services.container import ServiceContainer


class MLDataError(AgentError):
    """Exception raised when data processor generation fails."""


class MLDataAgent(BaseAgent):
    """Agent for generating data processing YAML from natural language."""

    def __init__(
        self,
        services: ServiceContainer,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize MLDataAgent.

        Args:
            services: Service container for database access
            api_key: API key for LLM calls
            base_url: Base URL for LLM API
            model: Model name to use
            progress_callback: Optional callback to report progress/tool usage
        """
        super().__init__(services, api_key, base_url, model)
        self.progress_callback = progress_callback

    def get_template_directory(self) -> Path:
        """Get the template directory for data processing generation.

        Returns:
            Path to the data processing template directory
        """
        return Path(__file__).parent / "templates"

    def get_allowed_phases(self) -> list[str]:
        """Get the phases this agent is allowed to access.

        ML Data agent only accesses data phase for feature engineering guidance.

        Returns:
            List containing ["data"]
        """
        return ["data"]

    async def generate_data_processing_yaml(
        self,
        instruction: str,
        name: str,
        data_source_type: str,
        data_sources: list[str],
        database: str = "user",
        existing_yaml: str | None = None,
        knowledge_references: list[str] | None = None,
        preloaded_knowledge: list[dict[str, str]] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> tuple[DataSourceSpec, str, list[dict[str, str]]]:
        """Generate or edit data processing YAML from instruction.

        Args:
            instruction: Detailed instruction for data processing.
                For generation: shaped by main agent or from ML plan.
                For editing: user feedback on what to change.
            name: Name for the data processor (provided by user/tool).
                Used directly in YAML instead of asking LLM to generate it.
            data_source_type: Type of data sources - "file" or "table"
            data_sources: List of data sources (file paths/URLs or table names)
            database: Database to use for schema discovery
            existing_yaml: Existing YAML content to edit (optional).
                If provided, switches to editing mode where instruction
                describes the changes to make.
            knowledge_references: Optional list of knowledge IDs referenced
                by this request
            preloaded_knowledge: Optional list of preloaded knowledge docs
                (deprecated)
            conversation_history: Optional conversation history for editing
                workflow

        Returns:
            Tuple of (DataSourceSpec object, YAML string, conversation_history)

        Raises:
            MLDataError: If generation fails
        """
        # Route to appropriate generation path
        if conversation_history is None:
            # Fresh generation - build full context
            return await self._generate_fresh(
                instruction=instruction,
                name=name,
                data_source_type=data_source_type,
                data_sources=data_sources,
                database=database,
                existing_yaml=existing_yaml,
                knowledge_references=knowledge_references,
                preloaded_knowledge=preloaded_knowledge,
            )
        else:
            # Continue conversation - just append feedback
            return await self._continue_conversation(
                feedback=instruction,
                name=name,
                conversation_history=conversation_history,
            )

    async def _generate_fresh(
        self,
        instruction: str,
        name: str,
        data_source_type: str,
        data_sources: list[str],
        database: str = "user",
        existing_yaml: str | None = None,
        knowledge_references: list[str] | None = None,
        preloaded_knowledge: list[dict[str, str]] | None = None,  # noqa: ARG002
    ) -> tuple[DataSourceSpec, str, list[dict[str, str]]]:
        """Fresh generation with full context building.

        This path is used for initial generation or when starting a
        new conversation. It builds the complete system message with
        schema discovery and knowledge loading.
        """
        try:
            # Get schema information for available tables
            # Skip row counts by default - agent can query if needed
            # Only fetch schema for table sources, not file sources
            schema_info = await self._get_schema_context(
                data_source_type, data_sources, database, include_row_counts=False
            )

            # Build system message with all context
            system_message = self._render_template(
                "prompt.j2",
                {
                    "processor_name": name,
                    "user_instruction": instruction,
                    "schema_info": schema_info,
                    "data_source_type": data_source_type,
                    "data_sources": data_sources,
                    "existing_yaml": existing_yaml,
                },
            )

            # User message guides tool usage and mentions knowledge references
            if existing_yaml:
                user_message = (
                    f"Edit the existing data processing specification with "
                    f"these changes: {instruction}."
                )
            else:
                user_message = "Generate the data processing specification."

            # Add knowledge references hint if provided
            if knowledge_references:
                user_message += (
                    f"\n\nThis request references the following knowledge: "
                    f"{', '.join(knowledge_references)}. "
                    f"Use list_available_knowledge and read_knowledge_content "
                    f"to review these references or discover additional "
                    f"knowledge as needed."
                )
            else:
                user_message += (
                    "\n\nUse list_available_knowledge and read_knowledge_content "
                    "if you need data processing guidance."
                )

            # Get ML tools from BaseAgent
            tools = self._get_ml_tools()

            # Generate with multi-turn tool support
            spec, yaml_content, conversation_history = await self._generate_with_tools(
                system_message=system_message,
                user_message=user_message,
                tools=tools,
                tool_executor=self._execute_ml_tool,
                validator_func=self._validate_data_processing_comprehensive,
                validation_context={"schema_info": schema_info, "processor_name": name},
                max_iterations=3,
                conversation_history=None,  # Fresh start
                progress_callback=self.progress_callback,
            )

            # Inject name into yaml_content if not already present
            # (LLM doesn't generate it, we add it programmatically)
            if not yaml_content.strip().startswith("name:"):
                yaml_content = f"name: {name}\n\n{yaml_content}"

            return spec, yaml_content, conversation_history

        except AgentError as e:
            raise MLDataError(str(e)) from e
        except Exception as e:
            raise MLDataError(
                f"Failed to generate data processing configuration: {e}"
            ) from e

    async def _continue_conversation(
        self,
        feedback: str,
        name: str,
        conversation_history: list[dict[str, str]],
    ) -> tuple[DataSourceSpec, str, list[dict[str, str]]]:
        """Continue existing conversation with user feedback.

        This path is used during interactive editing when conversation history exists.
        It simply appends the user's feedback to the existing conversation without
        rebuilding the system message.
        """
        # Get ML tools from BaseAgent
        tools = self._get_ml_tools()

        # Continue conversation with feedback
        try:
            spec, yaml_content, updated_history = await self._generate_with_tools(
                system_message="",  # Not used - already in conversation_history
                user_message=feedback,
                tools=tools,
                tool_executor=self._execute_ml_tool,
                validator_func=self._validate_data_processing_comprehensive,
                validation_context={
                    "schema_info": None,  # Already in conversation history
                    "processor_name": name,
                },
                max_iterations=3,
                conversation_history=conversation_history,
                progress_callback=self.progress_callback,
            )

            # Inject name into yaml_content if not already present
            # (LLM doesn't generate it, we add it programmatically)
            if not yaml_content.strip().startswith("name:"):
                yaml_content = f"name: {name}\n\n{yaml_content}"

            return spec, yaml_content, updated_history

        except AgentError as e:
            raise MLDataError(str(e)) from e
        except Exception as e:
            raise MLDataError(
                f"Failed to generate data processing configuration: {e}"
            ) from e

    async def _get_schema_context(
        self,
        data_source_type: str,
        data_sources: list[str],
        database: str,
        include_row_counts: bool = True,
    ) -> dict:
        """Get schema information with statistics for context.

        Args:
            data_source_type: Type of data sources - "file" or "table"
            data_sources: List of data sources (file paths/URLs or table names)
            database: Database to use
            include_row_counts: Whether to fetch row counts (default: True).
                Set to False to skip potentially slow COUNT(*) queries.

        Returns:
            Dictionary with schema information and table statistics
        """
        # For file sources, return minimal context (no tables to analyze)
        if data_source_type == "file":
            return {
                "database": database,
                "tables": [],
                "total_tables": 0,
                "data_source_type": "file",
            }

        # For table sources, fetch schema information
        try:
            schema_service = self.services.schema
            schema_info = schema_service.get_schema_info(database)

            context = {
                "database": database,
                "tables": [],
                "total_tables": len(schema_info.tables),
                "data_source_type": "table",
            }

            # If specific tables requested, get detailed info with statistics
            if data_sources:
                for table_name in data_sources:
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

                        # Try to get dataset info for row counts (if enabled)
                        if include_row_counts:
                            try:
                                dataset_info = self.services.ml_data.get_dataset_info(
                                    table_name, include_row_count=True
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

                    # Try to get row count for each table (if enabled)
                    if include_row_counts:
                        try:
                            dataset_info = self.services.ml_data.get_dataset_info(
                                table.name, include_row_count=True
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

    async def _validate_data_processing_comprehensive(
        self,
        yaml_content: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Comprehensive validation of generated data processing spec.

        Args:
            yaml_content: Generated YAML string
            context: Generation context for validation (includes schema_info
                with database, and processor_name to inject)

        Returns:
            Dictionary with validation results:
            {"valid": bool, "object": DataSourceSpec, "error": str}
        """
        try:
            # Parse YAML syntax
            self._validate_yaml_syntax(yaml_content)

            # Inject processor name into YAML before parsing
            # LLM generates description, steps, outputs - we add the name field
            processor_name = context.get("processor_name")
            if processor_name and not yaml_content.strip().startswith("name:"):
                # Add name field at the beginning of YAML
                yaml_content = f"name: {processor_name}\n\n{yaml_content}"

            # Try to parse into DataSourceSpec
            try:
                spec = DataSourceSpec.from_yaml(yaml_content)
            except ValueError as e:
                return {
                    "valid": False,
                    "error": f"Failed to parse into DataSourceSpec: {str(e)}",
                }

            # Validate dependencies (catch circular dependencies in retry loop)
            try:
                spec.validate_dependencies()
            except ValueError as e:
                return {
                    "valid": False,
                    "error": f"Dependency validation failed: {str(e)}",
                }

            # Validate execution order (catch other structural issues)
            try:
                _ = spec.get_execution_order()
            except ValueError as e:
                return {
                    "valid": False,
                    "error": f"Execution order validation failed: {str(e)}",
                }

            # DuckDB dry-run validation (catch runtime errors like table not found)
            # Extract database from context
            schema_info = context.get("schema_info", {})
            database = schema_info.get("database", "user")

            from arc.ml.data_source_executor import dry_run_data_source_pipeline

            success, error = await dry_run_data_source_pipeline(
                spec, database, self.services.db_manager
            )

            if not success:
                return {
                    "valid": False,
                    "error": f"Dry-run validation failed: {error}",
                }

            # Return the modified yaml_content with name injected
            return {
                "valid": True,
                "object": spec,
                "error": None,
                "yaml_content": yaml_content,
            }

        except AgentError as e:
            # AgentError messages are already well-formatted, don't wrap them
            return {
                "valid": False,
                "error": str(e),
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation exception: {str(e)}",
            }
