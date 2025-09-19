"""Base agent class for shared LLM interaction functionality."""

from __future__ import annotations

import abc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jinja2
import yaml

from ....database.services import ServiceContainer
from ...agent import ArcAgent


class AgentError(Exception):
    """Base exception for agent errors."""


class BaseAgent(abc.ABC):
    """Base class for Arc AI agents with shared LLM interaction functionality."""

    def __init__(self, services: ServiceContainer, agent: ArcAgent):
        """Initialize the base agent.

        Args:
            services: Service container for database access
            agent: Arc agent for LLM interactions
        """
        self.services = services
        self.agent = agent

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
        self, context: dict[str, Any], max_retries: int = 3, timeout: float = 30.0
    ) -> str:
        """Generate content using LLM with error handling and retries.

        Args:
            context: Context for generation
            max_retries: Maximum number of retry attempts
            timeout: Timeout in seconds for LLM calls

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
                # Use process_user_message with timeout
                chat_entries = await asyncio.wait_for(
                    self.agent.process_user_message(prompt), timeout=timeout
                )

                # Extract the response content
                for entry in chat_entries:
                    if entry.type == "assistant" and entry.content:
                        return self._clean_llm_response(entry.content)

                raise AgentError("No valid response received from LLM")

            except TimeoutError:
                last_error = f"LLM request timed out after {timeout} seconds"
                if attempt < max_retries:
                    continue

            except Exception as e:
                last_error = f"LLM generation failed: {str(e)}"
                if attempt < max_retries:
                    continue

        raise AgentError(
            f"LLM generation failed after {max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )

    def _clean_llm_response(self, response: str) -> str:
        """Clean up LLM response by removing markdown code blocks.

        Args:
            response: Raw LLM response

        Returns:
            Cleaned response content
        """
        content = response.strip()

        # Remove markdown code blocks if present
        if content.startswith("```yaml"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]

        if content.endswith("```"):
            content = content[:-3]

        return content.strip()

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

            except Exception as e:
                last_error = f"Generation error: {str(e)}"
                if attempt == max_iterations - 1:
                    raise AgentError(f"Content generation failed: {e}") from e

        raise AgentError(f"Content generation failed after {max_iterations} attempts")
