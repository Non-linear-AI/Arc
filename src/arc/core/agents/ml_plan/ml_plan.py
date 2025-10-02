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
        self.conversation_memory: list[dict[str, Any]] = []

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
        table_name: str,
        target_column: str | None = None,
        conversation_history: list[dict] | None = None,
        feedback: str | None = None,
        stream: bool = False,
    ):
        """Analyze ML problem and provide comprehensive plan.

        Args:
            user_context: User description of the problem
            table_name: Database table name for data exploration
            target_column: Target column for supervised learning
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

        # Get data profile for analysis
        data_profile = await self._get_unified_data_profile(table_name, target_column)

        # Build analysis context with full conversation history
        context = {
            "user_context": user_context,
            "data_profile": data_profile,
            "target_column": target_column,
            "conversation_history": conversation_history,
            "feedback": feedback,
            "available_components": ["mlp", "dcn", "transformer", "mmoe"],
            "component_descriptions": {
                "mlp": "Multi-Layer Perceptron for tabular data, feature processing, "
                "classification/regression",
                "transformer": "Attention-based architecture for sequential data, "
                "complex patterns, temporal modeling",
                "dcn": "Deep & Cross Network for feature interactions, CTR prediction, "
                "recommendation systems",
                "mmoe": "Multi-gate Mixture of Experts for multi-task learning, "
                "shared representations",
            },
        }

        # Generate analysis using LLM
        try:
            if stream:
                # Return streaming generator
                return self._analyze_problem_stream(context, user_context, feedback)
            else:
                # Non-streaming: return complete result
                analysis_result = await self._generate_analysis_with_validation(context)

                # Update conversation memory
                self.conversation_memory.append(
                    {
                        "user_context": user_context,
                        "feedback": feedback,
                        "selected_components": analysis_result.get(
                            "selected_components"
                        ),
                        "summary": analysis_result.get("summary"),
                    }
                )

                return analysis_result

        except Exception as e:
            raise MLPlanError(f"Failed to analyze problem: {str(e)}") from e

    async def _analyze_problem_stream(self, context, user_context, feedback):
        """Stream the analysis with final result."""
        analysis_result = None
        async for chunk in self._generate_analysis_with_streaming(context):
            if isinstance(chunk, dict) and chunk.get("type") == "analysis_complete":
                analysis_result = chunk["analysis"]
            else:
                yield chunk

        # Update conversation memory after streaming completes
        if analysis_result:
            self.conversation_memory.append(
                {
                    "user_context": user_context,
                    "feedback": feedback,
                    "selected_components": analysis_result.get("selected_components"),
                    "summary": analysis_result.get("summary"),
                }
            )
            # Yield the final analysis
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

        except (yaml.YAMLError, ValueError) as e:
            raise MLPlanError(
                f"Failed to parse analysis response: {str(e)}\nResponse: {response}"
            ) from e

    async def _get_unified_data_profile(
        self, table_name: str, target_column: str | None = None
    ) -> dict[str, Any]:
        """Get unified data profile with target-aware analysis for LLM context.

        Args:
            table_name: Database table name
            target_column: Optional target column for task-aware analysis

        Returns:
            Unified data profile with target and feature information
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

            # Build exclude set (only target column)
            exclude_set = set()
            if target_column:
                exclude_set.add(target_column)

            # Separate target and feature columns
            all_columns = dataset_info.columns
            feature_columns = [
                col for col in all_columns if col["name"] not in exclude_set
            ]

            # Count feature types
            feature_types = {}
            for col in feature_columns:
                col_type = col.get("type", "unknown")
                feature_types[col_type] = feature_types.get(col_type, 0) + 1

            # Base profile structure
            profile = {
                "table_name": dataset_info.name,
                "feature_columns": feature_columns,
                "total_columns": len(all_columns),
                "feature_count": len(feature_columns),
                "feature_types": feature_types,
                "summary": f"Table '{table_name}' with {len(feature_columns)} features",
            }

            # Add target analysis if target column specified
            if target_column:
                target_analysis = await self._analyze_target_column(
                    table_name, target_column
                )
                profile["target_info"] = target_analysis
                if "error" not in target_analysis:
                    profile["summary"] += (
                        f" and target '{target_column}' "
                        f"({target_analysis.get('type', 'unknown')} type)"
                    )

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
