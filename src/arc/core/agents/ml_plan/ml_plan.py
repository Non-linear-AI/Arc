"""ML plan analysis agent for comprehensive ML workflow planning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from arc.core.agents.shared.base_agent import AgentError, BaseAgent
from arc.database.services import ServiceContainer


class MLPlanError(AgentError):
    """Raised when ML plan analysis fails."""


class MLPlanAgent(BaseAgent):
    """Conversational agent for ML problem analysis and comprehensive planning."""

    def __init__(
        self,
        services: ServiceContainer,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
    ):
        """Initialize ML plan agent.

        Args:
            services: Service container for database access
            api_key: API key for LLM interactions
            base_url: Optional base URL
            model: Optional model name
        """
        super().__init__(services, api_key, base_url, model)

    def get_template_directory(self) -> Path:
        """Get the template directory for ML planning.

        Returns:
            Path to the ml_plan template directory
        """
        return Path(__file__).parent / "templates"

    def get_template_name(self) -> str:
        """Get the name of the template file.

        Returns:
            Template filename relative to the template directory
        """
        return "base.j2"

    async def analyze_problem(
        self,
        user_context: str,
        source_tables: str,
        conversation_history: list[dict] | None = None,
        feedback: str | None = None,
        stream: bool = False,
    ):
        """Analyze ML problem and provide comprehensive plan.

        Args:
            user_context: User description of the problem
            source_tables: Comma-separated source table names for data exploration
            conversation_history: Full conversation history for LLM to analyze
                (required)
            feedback: Optional user feedback for refining analysis
            stream: Whether to stream the output (default: False)

        Returns:
            If stream=False: Dictionary containing plan fields
            If stream=True: Async generator yielding chunks

        Raises:
            MLPlanError: If analysis fails or conversation_history is missing
        """
        # Require conversation_history for comprehensive analysis
        if conversation_history is None:
            raise MLPlanError(
                "conversation_history is required for ML planning. "
                "Pass the full conversation history to enable context-aware planning."
            )

        # Get data profiles for all source tables
        table_list = [t.strip() for t in source_tables.split(",")]
        data_profiles = await self._get_multiple_data_profiles(table_list)

        # Build analysis context with full conversation history
        context = {
            "user_context": user_context,
            "data_profiles": data_profiles,
            "source_tables": source_tables,
            "conversation_history": conversation_history,
            "feedback": feedback,
        }

        # Generate analysis using LLM
        try:
            if stream:
                # Return streaming generator
                return self._analyze_problem_stream(context)
            else:
                # Non-streaming: return complete result
                analysis_result = await self._generate_analysis_with_validation(context)
                return analysis_result

        except Exception as e:
            raise MLPlanError(f"Failed to analyze problem: {str(e)}") from e

    async def update_section(
        self,
        section_name: str,
        original_section: str,
        feedback_content: str,
    ) -> str:
        """Update a specific section of an ML plan based on feedback.

        Args:
            section_name: Name of section to update (e.g., "model_architecture_and_loss")
            original_section: The original section content
            feedback_content: Feedback or implementation details to incorporate

        Returns:
            Updated section text (plain text, not YAML-wrapped)

        Raises:
            MLPlanError: If update fails
        """
        # Build update context
        context = {
            "mode": "update_section",
            "section_to_update": section_name,
            "original_section": original_section,
            "feedback_content": feedback_content,
        }

        # Generate updated section using LLM
        try:
            prompt = self._render_template(self.get_template_name(), context)
            messages = [{"role": "user", "content": prompt}]
            response = await self.arc_client.chat(messages, tools=None)

            # Extract the updated section from response
            updated_section = response.strip()

            # Remove any markdown code fences if present
            if updated_section.startswith("```yaml"):
                updated_section = updated_section[7:]
            if updated_section.startswith("```"):
                updated_section = updated_section[3:]
            if updated_section.endswith("```"):
                updated_section = updated_section[:-3]

            return updated_section.strip()

        except Exception as e:
            raise MLPlanError(f"Failed to update section '{section_name}': {str(e)}") from e

    async def _analyze_problem_stream(self, context):
        """Stream the analysis with final result.

        Args:
            context: Analysis context

        Yields:
            Streaming chunks and final analysis result
        """
        analysis_result = None
        async for chunk in self._generate_analysis_with_streaming(context):
            if isinstance(chunk, dict) and chunk.get("type") == "analysis_complete":
                analysis_result = chunk["analysis"]
            else:
                yield chunk

        # Yield the final analysis
        if analysis_result:
            yield {"type": "analysis_complete", "analysis": analysis_result}

    async def _generate_analysis_with_validation(
        self, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate analysis with validation.

        Args:
            context: Analysis context

        Returns:
            Validated analysis result dictionary
        """
        # Call LLM with context (it will render template internally)
        response = await self._generate_with_llm(context)

        # Parse and validate response
        analysis_result = self._parse_analysis_response(response)

        return analysis_result

    async def _generate_analysis_with_streaming(self, context: dict[str, Any]):
        """Generate analysis with streaming output.

        Args:
            context: Analysis context

        Yields:
            Streaming chunks of the analysis
        """

        prompt = self._render_template(self.get_template_name(), context)

        # Build a minimal message list for one-shot generations
        messages = [{"role": "user", "content": prompt}]

        # Stream the response
        full_response = ""
        async for chunk in self.arc_client.chat_stream(messages, tools=None):
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    full_response += delta.content
                    yield delta.content

        # After streaming completes, parse and return the analysis
        if full_response:
            analysis_result = self._parse_analysis_response(full_response)
            yield {"type": "analysis_complete", "analysis": analysis_result}

    def _parse_analysis_response(self, response: str) -> dict[str, Any]:
        """Parse LLM analysis response into structured format.

        Args:
            response: Raw LLM response

        Returns:
            Structured analysis dictionary
        """
        import yaml

        # Try to extract YAML from response
        try:
            # Look for YAML block in markdown code fence (fallback)
            if "```yaml" in response:
                start = response.find("```yaml") + 7
                end = response.find("```", start)
                yaml_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                yaml_str = response[start:end].strip()
            else:
                # Response should be clean YAML
                yaml_str = response.strip()

            result = yaml.safe_load(yaml_str)

            # Validate required fields
            required_fields = [
                "summary",
                "feature_engineering",
                "model_architecture_and_loss",
                "training_configuration",
                "evaluation",
            ]
            missing = [f for f in required_fields if f not in result]
            if missing:
                raise ValueError(f"Missing required fields: {missing}")

            return result

        except yaml.YAMLError as e:
            # Show first and last 250 chars for better debugging context
            if len(response) > 500:
                response_preview = (
                    f"{response[:250]}...\n\n[truncated]\n\n...{response[-250:]}"
                )
            else:
                response_preview = response

            raise MLPlanError(
                f"Failed to parse YAML from LLM response: {str(e)}\n"
                f"Response preview:\n{response_preview}\n"
                f"Ensure the LLM returns valid YAML without markdown code fences."
            ) from e
        except ValueError as e:
            # Missing required fields
            raise MLPlanError(
                f"Invalid ML plan structure: {str(e)}\n"
                f"The plan must include all required sections: "
                f"summary, feature_engineering, model_architecture_and_loss, "
                f"training_configuration, and evaluation."
            ) from e

    async def _get_multiple_data_profiles(
        self, table_names: list[str]
    ) -> dict[str, Any]:
        """Get data profiles for multiple tables.

        Args:
            table_names: List of table names

        Returns:
            Dictionary mapping table names to their profiles
        """
        profiles = {}
        for table_name in table_names:
            profile = await self._get_single_table_profile(table_name)
            profiles[table_name] = profile
        return profiles

    async def _get_single_table_profile(self, table_name: str) -> dict[str, Any]:
        """Get data profile for a single table.

        Args:
            table_name: Database table name

        Returns:
            Data profile for the table
        """
        try:
            # Use ML data service for dataset information
            dataset_info = self.services.ml_data.get_dataset_info(table_name)

            if dataset_info is None:
                return {
                    "error": f"Table {table_name} not found or invalid",
                    "summary": f"Table {table_name} not found",
                    "feature_count": 0,
                    "feature_types": {},
                }

            # Get all columns from the table
            all_columns = dataset_info.columns

            # Count feature types and gather detailed statistics
            feature_types = {}
            column_details = []

            for col in all_columns:
                col_type = col.get("type", "unknown")
                feature_types[col_type] = feature_types.get(col_type, 0) + 1

                # Build detailed column info
                col_detail = {
                    "name": col.get("name"),
                    "type": col_type,
                }

                # Add null percentage if available
                if "null_count" in col and "total_count" in col:
                    null_pct = (col["null_count"] / col["total_count"]) * 100
                    col_detail["null_pct"] = f"{null_pct:.1f}%"
                elif "null_pct" in col:
                    col_detail["null_pct"] = f"{col['null_pct']:.1f}%"

                # Add cardinality for categorical/string columns
                if "unique_count" in col:
                    col_detail["cardinality"] = col["unique_count"]

                # Add min/max for numerical columns
                if col_type in ("INTEGER", "DOUBLE", "FLOAT", "BIGINT"):
                    if "min" in col:
                        col_detail["min"] = col["min"]
                    if "max" in col:
                        col_detail["max"] = col["max"]
                    if "mean" in col:
                        col_detail["mean"] = f"{col['mean']:.2f}"

                # Add sample values if available
                if "sample_values" in col and col["sample_values"]:
                    # Limit to first 3 sample values
                    samples = col["sample_values"][:3]
                    col_detail["samples"] = samples

                column_details.append(col_detail)

            # Build enhanced summary with statistics
            total_cols = len(all_columns)
            numeric_cols = sum(
                count for dtype, count in feature_types.items()
                if dtype in ("INTEGER", "DOUBLE", "FLOAT", "BIGINT")
            )
            categorical_cols = sum(
                count for dtype, count in feature_types.items()
                if dtype in ("VARCHAR", "STRING", "TEXT")
            )

            summary_parts = [
                f"Table '{table_name}' with {total_cols} columns",
                f"({numeric_cols} numerical, {categorical_cols} categorical)",
            ]

            # Base profile structure with enhanced information
            profile = {
                "table_name": dataset_info.name,
                "feature_columns": all_columns,
                "column_details": column_details,
                "total_columns": total_cols,
                "feature_count": total_cols,
                "feature_types": feature_types,
                "summary": " ".join(summary_parts),
            }

            return profile

        except Exception as e:
            return {
                "error": f"Failed to analyze table {table_name}: {str(e)}",
                "summary": f"Error analyzing table {table_name}",
                "feature_count": 0,
                "feature_types": {},
            }

    async def _analyze_target_column(
        self, table_name: str, target_column: str
    ) -> dict[str, Any]:
        """Analyze target column to provide information for problem analysis.

        Args:
            table_name: Database table name
            target_column: Target column to analyze

        Returns:
            Dictionary with target column information
        """
        try:
            # Get target column basic info
            dataset_info = self.services.ml_data.get_dataset_info(table_name)
            if not dataset_info:
                return {"error": f"Table {table_name} not found"}

            # Find target column in dataset
            target_col_info = None
            for col in dataset_info.columns:
                if col["name"] == target_column:
                    target_col_info = col
                    break

            if not target_col_info:
                return {"error": f"Target column '{target_column}' not found"}

            # Basic target info
            target_type = target_col_info.get("type", "unknown")

            # Get distribution information if available
            distribution = "Distribution information not available"
            if "unique_count" in target_col_info:
                unique_count = target_col_info["unique_count"]
                distribution = f"{unique_count} unique values"

            return {
                "name": target_column,
                "type": target_type,
                "distribution": distribution,
            }

        except Exception as e:
            return {"error": f"Failed to analyze target column: {str(e)}"}
