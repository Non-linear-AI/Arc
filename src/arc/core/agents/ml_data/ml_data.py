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

    async def generate_data_processing_yaml(
        self,
        instruction: str,
        source_tables: list[str] | None = None,
        database: str = "user",
        existing_yaml: str | None = None,
        recommended_knowledge_ids: list[str] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> tuple[DataSourceSpec, str, list[dict[str, str]]]:
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
            recommended_knowledge_ids: Optional list of knowledge IDs
                recommended by ML Plan
            conversation_history: Optional conversation history for editing workflow

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
                source_tables=source_tables,
                database=database,
                existing_yaml=existing_yaml,
                recommended_knowledge_ids=recommended_knowledge_ids,
            )
        else:
            # Continue conversation - just append feedback
            return await self._continue_conversation(
                feedback=instruction,
                conversation_history=conversation_history,
            )

    async def _generate_fresh(
        self,
        instruction: str,
        source_tables: list[str] | None = None,
        database: str = "user",
        existing_yaml: str | None = None,
        recommended_knowledge_ids: list[str] | None = None,
    ) -> tuple[DataSourceSpec, str, list[dict[str, str]]]:
        """Fresh generation with full context building.

        This path is used for initial generation or when starting a
        new conversation. It builds the complete system message with
        schema discovery and knowledge loading.
        """
        try:
            # Get schema information for available tables
            # Skip row counts by default - agent can query if needed
            schema_info = await self._get_schema_context(
                source_tables, database, include_row_counts=False
            )

            # Pre-load recommended knowledge content (handle missing gracefully)
            recommended_knowledge = ""
            loaded_knowledge_ids = []
            if recommended_knowledge_ids:
                for knowledge_id in recommended_knowledge_ids:
                    content = self.knowledge_loader.load_knowledge(knowledge_id, "data")
                    if content:
                        # Successfully loaded - add to system context
                        recommended_knowledge += (
                            f"\n\n# Data Processing Knowledge: {knowledge_id}"
                            f"\n\n{content}"
                        )
                        loaded_knowledge_ids.append(knowledge_id)
                    # If missing, silently skip (already logged at debug level)

            # Build system message with all context
            system_message = self._render_template(
                "prompt.j2",
                {
                    "user_instruction": instruction,
                    "schema_info": schema_info,
                    "source_tables": source_tables or [],
                    "existing_yaml": existing_yaml,
                    "recommended_knowledge": recommended_knowledge,
                },
            )

            # User message guides tool usage and lists pre-loaded knowledge
            if existing_yaml:
                user_message = (
                    f"Edit the existing data processing specification with "
                    f"these changes: {instruction}."
                )
            else:
                user_message = "Generate the data processing specification."

            # Tell agent which knowledge IDs are already provided
            if loaded_knowledge_ids:
                user_message += (
                    f"\n\nPre-loaded knowledge (already in system message): "
                    f"{', '.join(loaded_knowledge_ids)}. "
                    f"Do NOT reload these. Only use knowledge tools for "
                    f"additional guidance if needed."
                )
            else:
                user_message += (
                    "\n\nNo knowledge was pre-loaded. Use list_available_knowledge "
                    "and read_knowledge_content if you need data processing guidance."
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
                validation_context={"schema_info": schema_info},
                max_iterations=3,
                conversation_history=None,  # Fresh start
                progress_callback=self.progress_callback,
            )

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
                    "schema_info": None
                },  # Already in conversation history
                max_iterations=3,
                conversation_history=conversation_history,
                progress_callback=self.progress_callback,
            )

            return spec, yaml_content, updated_history

        except AgentError as e:
            raise MLDataError(str(e)) from e
        except Exception as e:
            raise MLDataError(
                f"Failed to generate data processing configuration: {e}"
            ) from e

    async def _get_schema_context(
        self,
        source_tables: list[str] | None,
        database: str,
        include_row_counts: bool = True,
    ) -> dict:
        """Get schema information with statistics for context.

        Args:
            source_tables: Specific source tables to analyze
            database: Database to use
            include_row_counts: Whether to fetch row counts (default: True).
                Set to False to skip potentially slow COUNT(*) queries.

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
            context: Generation context for validation (includes schema_info with database)

        Returns:
            Dictionary with validation results:
            {"valid": bool, "object": DataSourceSpec, "error": str}
        """
        try:
            # Parse YAML syntax
            self._validate_yaml_syntax(yaml_content)

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

            return {"valid": True, "object": spec, "error": None}

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
