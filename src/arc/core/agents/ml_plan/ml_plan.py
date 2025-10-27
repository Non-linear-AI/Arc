"""ML plan analysis agent for comprehensive ML workflow planning."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

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
        progress_callback: Any | None = None,
        verbose: bool = False,
    ):
        """Initialize ML plan agent.

        Args:
            services: Service container for database access
            api_key: API key for LLM interactions
            base_url: Optional base URL
            model: Optional model name
            progress_callback: Optional callback for reporting progress/errors
            verbose: If True, show detailed tool results. If False, show
                only tool calls.
        """
        super().__init__(services, api_key, base_url, model)
        self.progress_callback = progress_callback
        # Note: knowledge_loader is now initialized in BaseAgent
        self.verbose = verbose

    def get_template_directory(self) -> Path:
        """Get the template directory for ML planning.

        Returns:
            Path to the ml_plan template directory
        """
        return Path(__file__).parent / "templates"

    async def analyze_problem(
        self,
        user_context: str,
        source_tables: str,
        instruction: str | None = None,
        stream: bool = False,
    ):
        """Analyze ML problem and provide comprehensive plan.

        Args:
            user_context: User description of the problem
            source_tables: Comma-separated source table names for data exploration
            instruction: Optional instruction for initial generation or refinement
            stream: Whether to stream the output (default: False)

        Returns:
            If stream=False: Dictionary containing plan fields
            If stream=True: Async generator yielding chunks

        Raises:
            MLPlanError: If analysis fails
        """

        # Build analysis context
        context = {
            "user_context": user_context,
            "source_tables": source_tables,
            "instruction": instruction,
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
            section_name: Name of section to update
                (e.g., "model_architecture_and_loss")
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
            raise MLPlanError(
                f"Failed to update section '{section_name}': {str(e)}"
            ) from e

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
        self, context: dict[str, Any], max_iterations: int = 3
    ) -> dict[str, Any]:
        """Generate analysis with validation and retry on invalid YAML.

        This now uses conversational tool-based generation, allowing the LLM
        to explore knowledge documents during plan creation.

        Args:
            context: Analysis context
            max_iterations: Maximum number of validation attempts

        Returns:
            Validated analysis result dictionary
        """
        # Render system message and user message from template
        # Use the template to generate clear instructions for the LLM
        system_message = self._render_template(self.get_template_name(), context)
        user_message = (
            "Please analyze this ML problem and generate a comprehensive plan. "
            "Use the knowledge exploration tools if you need to understand specific "
            "architectural patterns."
        )

        # Get all tools from BaseAgent (knowledge + database)
        tools = self._get_ml_tools()

        # Define validator function
        def validator(content: str, ctx: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG001
            try:
                result = self._parse_analysis_response(content)
                return {"valid": True, "object": result, "error": None}
            except MLPlanError as e:
                error_msg = str(e)
                # Report validation error to user
                if self.progress_callback:
                    msg = f"  ✗ Validation error: {error_msg[:200]}..."
                    self.progress_callback(msg)
                return {"valid": False, "object": None, "error": error_msg}

        # Use tool-based generation with validation
        try:
            analysis_result, _raw_content, _history = await self._generate_with_tools(
                system_message=system_message,
                user_message=user_message,
                tools=tools,
                tool_executor=self._execute_tool,
                validator_func=validator,
                validation_context=context,
                max_iterations=max_iterations,
            )

            return analysis_result

        except AgentError as e:
            # Report the full error to user before re-raising
            if self.progress_callback:
                self.progress_callback(f"  ✗ Generation failed: {str(e)[:300]}")
            raise MLPlanError(str(e)) from e

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
                # Response should be clean YAML, but may have preamble text
                # Find where YAML actually starts by looking for first required field
                yaml_str = response.strip()

                # If response starts with explanatory text before YAML,
                # find where the YAML begins (first required field at line start)
                required_fields = [
                    "feature_engineering:",
                    "model_architecture_and_loss:",
                    "training_configuration:",
                    "evaluation:",
                ]

                # Find earliest required field occurrence at line start
                earliest_match = None
                for field in required_fields:
                    pattern = rf"^{re.escape(field)}"
                    match = re.search(pattern, yaml_str, re.MULTILINE)
                    if match and (
                        earliest_match is None or match.start() < earliest_match
                    ):
                        earliest_match = match.start()

                # If we found a field, extract from there
                if earliest_match is not None and earliest_match > 0:
                    yaml_str = yaml_str[earliest_match:].strip()

            result = yaml.safe_load(yaml_str)

            # Validate required fields
            required_fields = [
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
                f"feature_engineering, model_architecture_and_loss, "
                f"training_configuration, and evaluation."
            ) from e

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

    async def _execute_tool(self, tool_name: str, arguments: str) -> str:
        """Execute ML Plan tools with progress reporting.

        Wraps BaseAgent's _execute_ml_tool() with progress callbacks.

        Args:
            tool_name: Name of the tool to execute
            arguments: JSON string of tool arguments

        Returns:
            Tool execution result as string
        """
        args = json.loads(arguments)

        # Report tool call
        if self.progress_callback:
            self._report_tool_call(tool_name, args)

        # Execute using BaseAgent's implementation
        result = await self._execute_ml_tool(tool_name, arguments)

        # Report result (only in verbose mode)
        if self.verbose and self.progress_callback and result:
            self._report_tool_result(tool_name, result, args)

        return result

    def _report_tool_call(self, tool_name: str, args: dict):
        """Report tool call with readable description."""
        if tool_name == "list_available_knowledge":
            self.progress_callback("[dim]▸ Listing available knowledges[/dim]")

        elif tool_name == "read_knowledge_content":
            knowledge_id = args.get("knowledge_id", "")
            self.progress_callback(f"[dim]▸ Reading knowledge: {knowledge_id}[/dim]")

        elif tool_name == "database_query":
            query = args.get("query", "")
            # Show condensed query if too long
            query_display = query[:97] + "..." if len(query) > 100 else query
            self.progress_callback(f"[dim]▸ Query: {query_display}[/dim]")

    def _report_tool_result(self, tool_name: str, result: str, args: dict):  # noqa: ARG002
        """Display tool result in readable format."""
        if tool_name == "database_query":
            # DatabaseQueryTool already formats nicely, just indent it
            self.progress_callback("    Result:")
            for line in result.split("\n"):
                if line.strip():
                    self.progress_callback(f"    {line}")

        elif tool_name == "list_available_knowledge":
            # Show available patterns
            self.progress_callback("    Available:")
            for line in result.split("\n"):
                if line.strip() and line.startswith("-"):
                    self.progress_callback(f"    {line}")

        elif tool_name == "read_knowledge_content":
            # Show preview of knowledge content
            lines = [line for line in result.split("\n") if line.strip()]
            self.progress_callback("    Preview:")
            for line in lines[:8]:
                # Truncate long lines
                display_line = line if len(line) <= 100 else line[:97] + "..."
                self.progress_callback(f"    {display_line}")
            if len(lines) > 8:
                self.progress_callback(f"    ... ({len(lines) - 8} more lines)")

        # Add blank line for readability
        self.progress_callback("")

    # Note: Handler methods (_handle_list_knowledge, _handle_read_knowledge,
    # _handle_database_query) are now inherited from BaseAgent
