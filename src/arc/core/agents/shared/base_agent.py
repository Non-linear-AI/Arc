"""Base agent class for shared LLM interaction functionality.

This class now handles both Jinja template rendering and direct
network requests to the LLM (via ArcClient). It no longer relies
on an external ArcAgent to pass prompts as user messages.
"""

from __future__ import annotations

import abc
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jinja2
import yaml

from arc.core.client import ArcClient
from arc.database.services import ServiceContainer

logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Base exception for agent errors."""


class BaseAgent(abc.ABC):
    """Base class for Arc AI agents with shared LLM interaction functionality."""

    def __init__(
        self,
        services: ServiceContainer,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
    ):
        """Initialize the base agent.

        Args:
            services: Service container for database access
            api_key: API key for Arc/OpenAI
            base_url: Optional base URL for the API
            model: Optional model name
        """
        self.services = services
        # Initialize an API client dedicated to this agent
        self.arc_client = ArcClient(api_key=api_key, model=model, base_url=base_url)
        # Initialize knowledge loader for all agents
        from arc.core.agents.shared.knowledge_loader import KnowledgeLoader

        self.knowledge_loader = KnowledgeLoader()

    @abc.abstractmethod
    def get_template_directory(self) -> Path:
        """Get the template directory for this agent.

        Returns:
            Path to the template directory
        """

    def get_template_name(self) -> str:
        """Get the name of the template file.

        Returns:
            Template filename (always prompt.j2)
        """
        return "prompt.j2"

    def _load_template(self, template_name: str) -> jinja2.Template:
        """Load a Jinja2 template.

        Args:
            template_name: Name of the template file

        Returns:
            Loaded Jinja2 template

        Raises:
            AgentError: If template loading fails
        """
        try:
            template_dir = self.get_template_directory()
            env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(template_dir),
                trim_blocks=True,
                lstrip_blocks=True,
            )
            return env.get_template(template_name)
        except Exception as e:
            raise AgentError(f"Failed to load template {template_name}: {e}") from e

    def _render_template(self, template_name: str, context: dict[str, Any]) -> str:
        """Render a template with the given context.

        Args:
            template_name: Name of the template file
            context: Context dictionary for template rendering

        Returns:
            Rendered template string

        Raises:
            AgentError: If template rendering fails
        """
        try:
            template = self._load_template(template_name)
            return template.render(**context)
        except Exception as e:
            raise AgentError(f"Failed to render template {template_name}: {e}") from e

    async def _generate_with_llm(
        self,
        context: dict[str, Any],
        max_retries: int = 3,
        timeout: float = 90.0,
        progress_callback: Callable[[str], None] | None = None,
    ) -> str:
        """Generate content using LLM with error handling and retries.

        Args:
            context: Context for generation
            max_retries: Maximum number of retry attempts
            timeout: Timeout in seconds for LLM calls (default: 90 seconds)
            progress_callback: Optional callback to report progress/errors
                during retries

        Returns:
            Generated content string

        Raises:
            AgentError: If generation fails after all retries
        """
        import asyncio

        prompt = self._render_template(self.get_template_name(), context)

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # Build a minimal message list for one-shot generations
                messages = [{"role": "user", "content": prompt}]

                # Call the API with a timeout
                response_msg = await asyncio.wait_for(
                    self.arc_client.chat(messages, tools=None), timeout=timeout
                )

                content = (response_msg.content or "").strip()
                if not content:
                    raise AgentError("No content received from LLM")

                return self._clean_llm_response(content)

            except TimeoutError:
                last_error = f"LLM request timed out after {timeout} seconds"
                if attempt < max_retries:
                    # Report retry to UI if callback provided
                    if progress_callback:
                        msg = (
                            f"[dim]✗ Attempt {attempt + 1}/{max_retries + 1} "
                            f"failed: {last_error}. Retrying...[/dim]"
                        )
                        progress_callback(msg)
                    continue

            except Exception as e:
                last_error = f"LLM generation failed: {str(e)}"
                if attempt < max_retries:
                    # Report retry to UI if callback provided
                    if progress_callback:
                        msg = (
                            f"[dim]✗ Attempt {attempt + 1}/{max_retries + 1} "
                            f"failed: {last_error}. Retrying...[/dim]"
                        )
                        progress_callback(msg)
                    continue

        raise AgentError(
            f"LLM generation failed after {max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )

    def _clean_llm_response(self, response: str) -> str:
        """Clean up LLM response by removing markdown code blocks and preamble text.

        Handles multiple cases:
        1. Code fences at start/end of response
        2. Preamble text before code fence
        3. Explanatory text after code fence
        4. Preamble text without code fences (detect YAML start)

        Args:
            response: Raw LLM response

        Returns:
            Cleaned response content (YAML only, no markdown fences)
        """
        content = response.strip()

        # Case 1: Try to extract YAML from markdown code block (```yaml ... ```)
        # This handles preamble text before the code fence
        if "```yaml" in content:
            start = content.find("```yaml") + 7  # Skip "```yaml"
            end = content.find("```", start)
            if end != -1:
                # Found matching closing fence
                return content[start:end].strip()
            else:
                # No closing fence, take everything after opening
                return content[start:].strip()

        # Case 2: Try to extract from generic code block (``` ... ```)
        if "```" in content:
            start = content.find("```") + 3  # Skip "```"
            end = content.find("```", start)
            if end != -1:
                # Found matching closing fence
                return content[start:end].strip()
            else:
                # No closing fence, take everything after opening
                return content[start:].strip()

        # Case 3: No code fences, but detect preamble before YAML
        # Look for common YAML starting patterns
        lines = content.split("\n")
        yaml_start_idx = None

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # Check if line looks like a valid YAML key-value pair
            # Must start with alphanumeric or underscore (valid YAML key)
            # and have a colon not preceded by markdown formatting
            if ":" in stripped:
                # Split on first colon
                parts = stripped.split(":", 1)
                if len(parts) == 2:
                    potential_key = parts[0].strip()
                    # Valid YAML key: alphanumeric/underscore, no special chars
                    # Reject markdown patterns like "**text**:", "1.", etc.
                    if (
                        potential_key
                        and not potential_key[0].isdigit()  # Not a numbered list
                        and "**" not in potential_key  # Not markdown bold
                        and "`" not in potential_key  # Not markdown code
                        and "." not in potential_key  # Not a numbered list
                        and potential_key.replace("_", "").isalnum()  # Valid identifier
                    ):
                        yaml_start_idx = idx
                        break

            # Check for YAML list item: "- item" (but not markdown list "- **text**")
            if stripped.startswith("- "):
                # Make sure it's not a markdown list with formatting
                list_content = stripped[2:].strip()
                if (
                    list_content
                    and "**" not in list_content
                    and "`" not in list_content
                ):
                    yaml_start_idx = idx
                    break

        if yaml_start_idx is not None and yaml_start_idx > 0:
            # Found preamble, remove it
            return "\n".join(lines[yaml_start_idx:]).strip()

        # Case 4: No code fences, no preamble detected, return as-is
        return content

    def _validate_yaml_syntax(self, yaml_content: str) -> dict[str, Any]:
        """Validate YAML syntax and parse into dictionary.

        Args:
            yaml_content: YAML string to validate

        Returns:
            Parsed YAML dictionary

        Raises:
            AgentError: If YAML is invalid
        """
        try:
            parsed = yaml.safe_load(yaml_content)
            if not isinstance(parsed, dict):
                raise AgentError("Generated content is not a valid YAML object")
            return parsed
        except yaml.YAMLError as e:
            raise AgentError(f"Invalid YAML syntax: {e}") from e

    def _save_to_file(self, content: str, output_path: str) -> None:
        """Save content to file.

        Args:
            content: Content to save
            output_path: Path to save the file

        Raises:
            AgentError: If file saving fails
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(content, encoding="utf-8")
        except Exception as e:
            raise AgentError(f"Failed to save file {output_path}: {e}") from e

    async def _generate_with_validation_loop(
        self,
        context: dict[str, Any],
        validator_func: Callable[[str, dict[str, Any]], dict[str, Any]],
        max_iterations: int = 3,
    ) -> tuple[Any, str]:
        """Generate content with validation and iterative improvement.

        Args:
            context: Initial context for generation
            validator_func: Function to validate generated content
            max_iterations: Maximum number of generation attempts

        Returns:
            Tuple of (validated_object, raw_content)

        Raises:
            AgentError: If generation fails after max iterations
        """
        last_error = None

        for attempt in range(max_iterations):
            try:
                # Add previous error to context for fixing
                if last_error and attempt > 0:
                    context["previous_error"] = last_error
                    context["attempt_number"] = attempt + 1

                # Generate content using LLM
                raw_content = await self._generate_with_llm(context)

                # Validate the generated content
                validation_result = validator_func(raw_content, context)

                if validation_result["valid"]:
                    return validation_result["object"], raw_content
                else:
                    last_error = validation_result["error"]
                    if attempt == max_iterations - 1:
                        raise AgentError(
                            f"Failed to generate valid content after "
                            f"{max_iterations} attempts. Final error: {last_error}"
                        )
                    else:
                        # Log the error and inform that we're retrying
                        logger.warning(
                            f"Validation failed on attempt "
                            f"{attempt + 1}/{max_iterations}: {last_error}. Retrying..."
                        )

            except Exception as e:
                last_error = f"Generation error: {str(e)}"
                if attempt == max_iterations - 1:
                    raise AgentError(f"Content generation failed: {e}") from e
                else:
                    # Log the error and inform that we're retrying
                    logger.warning(
                        f"Generation failed on attempt {attempt + 1}/{max_iterations}: "
                        f"{last_error}. Retrying..."
                    )

        raise AgentError(f"Content generation failed after {max_iterations} attempts")

    async def _generate_with_tools(
        self,
        system_message: str,
        user_message: str,
        tools: list[Any] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        validator_func: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        validation_context: dict[str, Any] | None = None,
        max_iterations: int = 3,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> tuple[Any, str, list[dict[str, str]]]:
        """Generate content with tool support and validation.

        Args:
            system_message: System message with persistent context
            user_message: User message with specific task
            tools: Optional list of ArcTool objects
            tool_executor: Optional function to execute tools
            validator_func: Optional function to validate generated content
            validation_context: Optional context for validation
            max_iterations: Maximum number of generation attempts
            conversation_history: Optional conversation history for editing

        Returns:
            Tuple of (validated_object, raw_content, conversation_history)

        Raises:
            AgentError: If generation fails after max iterations
        """
        import asyncio

        # Initialize conversation history
        if conversation_history is None:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        else:
            # Use existing history and append new user message
            messages = conversation_history.copy()
            messages.append({"role": "user", "content": user_message})

        last_error = None

        for attempt in range(max_iterations):
            try:
                # Inner loop for tool call conversation
                # (separate from validation retries)
                max_tool_rounds = 10  # Allow up to 10 rounds of tool calls
                for _tool_round in range(max_tool_rounds):
                    # Call LLM with tool support
                    response_msg = await asyncio.wait_for(
                        self.arc_client.chat(messages, tools=tools), timeout=90.0
                    )

                    # Handle tool calls
                    if response_msg.tool_calls:
                        # Add assistant message with tool calls
                        messages.append(
                            {
                                "role": "assistant",
                                "content": response_msg.content,
                                "tool_calls": [
                                    {
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments,
                                        },
                                    }
                                    for tc in response_msg.tool_calls
                                ],
                            }
                        )

                        # Execute each tool call
                        for tool_call in response_msg.tool_calls:
                            if tool_executor:
                                result = await tool_executor(
                                    tool_call.function.name,
                                    tool_call.function.arguments,
                                )

                                # Add tool result to messages
                                messages.append(
                                    {
                                        "role": "tool",
                                        "content": str(result),
                                        "tool_call_id": tool_call.id,
                                    }
                                )

                        # Continue to get next response (stay in tool round loop)
                        continue
                    else:
                        # Got final output, break out of tool round loop
                        break

                # Got final output
                content = (response_msg.content or "").strip()
                if not content:
                    raise AgentError("No content received from LLM")

                raw_content = self._clean_llm_response(content)

                # Add assistant response to history
                messages.append({"role": "assistant", "content": raw_content})

                # Validate if validator provided
                if validator_func and validation_context:
                    validation_result = validator_func(raw_content, validation_context)

                    if validation_result["valid"]:
                        return validation_result["object"], raw_content, messages
                    else:
                        last_error = validation_result["error"]
                        if attempt < max_iterations - 1:
                            # Add error feedback for retry
                            error_msg = (
                                f"The generated YAML has validation errors:\n"
                                f"{last_error}\n\n"
                                f"Please fix these errors and generate a "
                                f"corrected version."
                            )
                            messages.append({"role": "user", "content": error_msg})
                            logger.warning(
                                f"Validation failed on attempt "
                                f"{attempt + 1}/{max_iterations}: "
                                f"{last_error}. Retrying..."
                            )
                            continue
                        else:
                            raise AgentError(
                                f"Failed to generate valid content after "
                                f"{max_iterations} attempts. "
                                f"Final error: {last_error}"
                            )
                else:
                    # No validation, return as-is
                    return None, raw_content, messages

            except TimeoutError as e:
                last_error = "LLM request timed out after 90 seconds"
                if attempt < max_iterations - 1:
                    logger.warning(f"Timeout on attempt {attempt + 1}, retrying...")
                    continue
                else:
                    raise AgentError(last_error) from e

            except Exception as e:
                last_error = f"Generation error: {str(e)}"
                if attempt < max_iterations - 1:
                    logger.warning(
                        f"Generation failed on attempt "
                        f"{attempt + 1}/{max_iterations}: "
                        f"{last_error}. Retrying..."
                    )
                    continue
                else:
                    raise AgentError(f"Content generation failed: {e}") from e

        raise AgentError(f"Content generation failed after {max_iterations} attempts")

    # ========== ML Tools Infrastructure (Shared across all ML agents) ==========

    def _get_ml_tools(self) -> list[Any]:
        """Get standard ML tools available to all agents.

        Returns:
            List of ArcTool objects for ML tasks (database query + knowledge tools)
        """

        return [
            self._create_database_query_tool(),
            self._create_list_knowledge_tool(),
            self._create_read_knowledge_tool(),
        ]

    def _create_database_query_tool(self) -> Any:
        """Create database query tool for data exploration.

        Returns:
            ArcTool for executing read-only SQL queries
        """
        from arc.tools.tools import ArcTool

        return ArcTool(
            name="database_query",
            description=(
                "Execute read-only SQL queries on the user database to analyze data. "
                "Use this to verify data characteristics, check column existence, "
                "analyze distributions, validate assumptions, or gather statistics. "
                "Only SELECT, DESCRIBE, and SHOW queries are allowed. "
                "Results are limited to 10 rows for brevity."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "SQL query to execute (SELECT only). "
                            "Example: 'SELECT target_column, COUNT(*) FROM table "
                            "GROUP BY target_column'"
                        ),
                    },
                },
                "required": ["query"],
            },
        )

    def _create_list_knowledge_tool(self) -> Any:
        """Create list knowledge tool for browsing available architectures.

        Returns:
            ArcTool for listing available knowledge documents
        """
        from arc.tools.tools import ArcTool

        return ArcTool(
            name="list_available_knowledge",
            description=(
                "List all available architecture and pattern knowledge documents. "
                "Returns metadata including knowledge ID, name, and description. "
                "Use this to explore what architectural patterns, training strategies, "
                "evaluation techniques, or data processing patterns are available."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )

    def _create_read_knowledge_tool(self) -> Any:
        """Create read knowledge tool for loading specific guidance.

        Returns:
            ArcTool for reading knowledge document content
        """
        from arc.tools.tools import ArcTool

        return ArcTool(
            name="read_knowledge_content",
            description=(
                "Read the full content of a specific knowledge document. "
                "Provides detailed architectural guidance, patterns, or strategies. "
                "Use this after identifying relevant knowledge with "
                "list_available_knowledge, or when you need specific guidance "
                "not provided in recommended knowledge."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "knowledge_id": {
                        "type": "string",
                        "description": (
                            "The ID of the knowledge document to read "
                            "(e.g., 'dcn', 'mlp', 'transformer', "
                            "'feature-interaction', 'sequence-generation')"
                        ),
                    },
                    "domain": {
                        "type": "string",
                        "description": (
                            "The domain of knowledge to read (default: 'model'). "
                            "Options: 'model' (architectures), "
                            "'train' (training strategies), 'evaluate' (metrics), "
                            "'data' (feature engineering patterns)"
                        ),
                        "enum": ["model", "train", "evaluate", "data"],
                        "default": "model",
                    },
                },
                "required": ["knowledge_id"],
            },
        )

    async def _execute_ml_tool(self, tool_name: str, arguments: str) -> str:
        """Execute ML tools with shared implementation.

        Args:
            tool_name: Name of the tool to execute
            arguments: JSON string of tool arguments

        Returns:
            Tool execution result as string
        """
        # Parse arguments with error handling
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON arguments for {tool_name}: {str(e)}"

        # Execute the tool
        if tool_name == "database_query":
            return await self._handle_database_query(args.get("query"))
        elif tool_name == "list_available_knowledge":
            return self._handle_list_knowledge()
        elif tool_name == "read_knowledge_content":
            return self._handle_read_knowledge(
                args.get("knowledge_id"), args.get("domain", "model")
            )
        else:
            return f"Unknown tool: {tool_name}"

    def _handle_list_knowledge(self) -> str:
        """Handle list_available_knowledge tool call.

        Returns:
            Formatted list of available knowledge with metadata
        """
        metadata_map = self.knowledge_loader.scan_metadata()

        if not metadata_map:
            return "No knowledge documents available."

        lines = ["Available Architecture and Pattern Knowledge:\n"]
        for knowledge_id, metadata in metadata_map.items():
            lines.append(f"- **{knowledge_id}**: {metadata.name}")
            lines.append(f"  Description: {metadata.description}")
            if metadata.keywords:
                lines.append(f"  Keywords: {', '.join(metadata.keywords)}")
            lines.append("")

        return "\n".join(lines)

    def _handle_read_knowledge(self, knowledge_id: str, domain: str = "model") -> str:
        """Handle read_knowledge_content tool call.

        Args:
            knowledge_id: Knowledge document ID
            domain: Knowledge domain (model, train, evaluate, data)

        Returns:
            Knowledge content or error message
        """
        if not knowledge_id:
            return "Error: knowledge_id parameter is required"

        # Validate domain parameter
        valid_domains = ["model", "train", "evaluate", "data"]
        if domain not in valid_domains:
            return (
                f"Error: Invalid domain '{domain}'. "
                f"Must be one of: {', '.join(valid_domains)}"
            )

        content = self.knowledge_loader.load_knowledge(knowledge_id, domain)

        if content:
            # Add header for context
            metadata_map = self.knowledge_loader.scan_metadata()
            metadata = metadata_map.get(knowledge_id)
            header = f"# Knowledge: {knowledge_id}"
            if metadata:
                header += f" - {metadata.name}"
            return f"{header}\n\n{content}"
        else:
            return (
                f"Error: Knowledge '{knowledge_id}' not found in domain '{domain}'. "
                f"Use list_available_knowledge to see available options."
            )

    async def _handle_database_query(self, query: str) -> str:
        """Handle database_query tool call.

        Args:
            query: SQL query to execute

        Returns:
            Query result or error message
        """
        if not query:
            return "Error: query parameter is required"

        # Validate query is read-only
        query_upper = query.strip().upper()
        allowed_statements = ["SELECT", "DESCRIBE", "SHOW", "EXPLAIN", "WITH"]

        # Check if query starts with an allowed statement
        is_allowed = any(query_upper.startswith(stmt) for stmt in allowed_statements)

        if not is_allowed:
            return (
                f"Error: Only read-only queries are allowed "
                f"(SELECT, DESCRIBE, SHOW, EXPLAIN, WITH). "
                f"Query starts with: "
                f"{query_upper.split()[0] if query_upper else 'empty'}"
            )

        # Check for multiple statements (SQL injection protection)
        # Allow single trailing semicolon, but not multiple statements
        semicolon_count = query.count(";")
        if semicolon_count > 1 or (
            semicolon_count == 1 and not query.rstrip().endswith(";")
        ):
            return (
                "Error: Multiple SQL statements not allowed for security. "
                "Please execute one query at a time."
            )

        try:
            # Use the existing DatabaseQueryTool with timeout
            import asyncio

            from arc.tools.database_query import DatabaseQueryTool

            db_tool = DatabaseQueryTool(self.services)
            result = await asyncio.wait_for(
                db_tool.execute(
                    query=query,
                    target_db="user",  # Query user database
                    validate_schema=True,
                ),
                timeout=30.0,  # 30 second timeout for queries
            )

            if result.success:
                # Return the formatted output (already brief from the tool)
                return result.output
            else:
                return f"Query Error: {result.output}"

        except TimeoutError:
            return (
                "Query Timeout: Query took longer than 30 seconds to execute. "
                "Try simplifying your query or adding LIMIT clause."
            )
        except Exception as e:
            return f"Query Execution Error: {str(e)}"
