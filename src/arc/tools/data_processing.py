"""Data processing tool for generating SQL feature engineering YAML configurations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..graph.features.data_source import DataSourceSpec
from .base import BaseTool, ToolResult

if TYPE_CHECKING:
    from ..database.services.container import ServiceContainer


class DataProcessingTool(BaseTool):
    """Tool for generating data processing YAML configurations from natural language."""

    def __init__(self, services: ServiceContainer, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        """Initialize DataProcessingTool.

        Args:
            services: ServiceContainer instance providing database access
            api_key: API key for LLM calls (for future generator agent)
            base_url: Base URL for LLM calls (for future generator agent)
            model: Model name for LLM calls (for future generator agent)
        """
        self.services = services
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    async def execute(
        self,
        action: str = "generate",
        context: str | None = None,
        target_tables: list[str] | None = None,
        output_path: str | None = None,
        target_db: str = "user",
        yaml_content: str | None = None,
    ) -> ToolResult:
        """Execute data processing operation.

        Args:
            action: Type of action - "generate" (create YAML from context) or "validate" (check YAML)
            context: Natural language description of data processing requirements
            target_tables: List of tables to analyze for generation
            output_path: Path to save generated YAML file
            target_db: Target database - "system" or "user"
            yaml_content: Raw YAML content for validation

        Returns:
            ToolResult with operation result
        """
        try:
            if action == "generate":
                return await self._generate_yaml(context, target_tables, output_path, target_db)
            elif action == "validate":
                return await self._validate_yaml(yaml_content, output_path)
            else:
                return ToolResult.error_result(
                    f"Invalid action: {action}. Supported actions: 'generate', 'validate'"
                )

        except Exception as e:
            return ToolResult.error_result(f"Data processing tool error: {str(e)}")

    async def _generate_yaml(
        self,
        context: str | None,
        target_tables: list[str] | None,
        output_path: str | None,
        target_db: str,
    ) -> ToolResult:
        """Generate YAML configuration from natural language context.

        For now, this creates a template YAML. In the future, this will use
        a data processor generator agent with LLM capabilities.
        """
        if not context:
            return ToolResult.error_result(
                "Context is required for YAML generation. "
                "Provide a natural language description of your data processing needs."
            )

        # Validate target database
        if target_db not in ["system", "user"]:
            return ToolResult.error_result(
                f"Invalid target database: {target_db}. Must be 'system' or 'user'."
            )

        try:
            # For prototype: create a template YAML based on context
            # In future: this will use DataProcessorGeneratorAgent
            template_yaml = self._create_template_yaml(context, target_tables or [])

            # Save to file if path provided
            if output_path:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_text(template_yaml)

            lines = [
                "Data processing YAML generated successfully!",
                f"Context: {context}",
            ]

            if target_tables:
                lines.append(f"Target tables: {', '.join(target_tables)}")

            if output_path:
                lines.append(f"Saved to: {output_path}")

            lines.append("\nGenerated YAML:")
            lines.append("=" * 50)
            lines.append(template_yaml)
            lines.append("=" * 50)

            return ToolResult.success_result("\n".join(lines))

        except Exception as e:
            return ToolResult.error_result(f"Failed to generate YAML: {str(e)}")

    async def _validate_yaml(self, yaml_content: str | None, output_path: str | None) -> ToolResult:
        """Validate YAML configuration."""
        if not yaml_content and not output_path:
            return ToolResult.error_result(
                "Either yaml_content or output_path must be provided for validation"
            )

        try:
            if output_path:
                # Load from file
                yaml_file = Path(output_path)
                if not yaml_file.exists():
                    return ToolResult.error_result(f"YAML file not found: {output_path}")
                spec = DataSourceSpec.from_yaml_file(output_path)
            else:
                # Parse from content
                spec = DataSourceSpec.from_yaml(yaml_content)

            # Validate dependencies
            spec.validate_dependencies()

            # Get execution order
            ordered_steps = spec.get_execution_order()

            lines = [
                "✅ YAML validation successful!",
                f"Steps: {len(spec.steps)}",
                f"Outputs: {len(spec.outputs)}",
                f"Variables: {len(spec.vars) if spec.vars else 0}",
                "",
                "Execution order:",
            ]

            for i, step in enumerate(ordered_steps, 1):
                lines.append(f"  {i}. {step.name}")

            lines.append(f"\nOutput tables: {', '.join(spec.outputs)}")

            return ToolResult.success_result("\n".join(lines))

        except ValueError as e:
            return ToolResult.error_result(f"YAML validation failed: {str(e)}")
        except Exception as e:
            return ToolResult.error_result(f"Unexpected validation error: {str(e)}")

    def _create_template_yaml(self, context: str, target_tables: list[str]) -> str:
        """Create a template YAML based on context and target tables.

        This is a simple template generator for prototyping.
        In the future, this will be replaced by an LLM-powered generator agent.
        """
        # Simple heuristics based on context
        context_lower = context.lower()

        # Detect common patterns
        is_aggregation = any(word in context_lower for word in
                           ["aggregate", "sum", "count", "average", "group", "daily", "monthly"])
        is_cleaning = any(word in context_lower for word in
                         ["clean", "filter", "remove", "null", "invalid"])
        is_joining = any(word in context_lower for word in
                        ["join", "combine", "merge", "with"])

        # Use first target table or default
        base_table = target_tables[0] if target_tables else "source_table"

        steps = []
        step_counter = 1

        # Add cleaning step if needed
        if is_cleaning:
            steps.append({
                "name": f"clean_{base_table}",
                "depends_on": [base_table],
                "sql": f"""SELECT *
    FROM {base_table}
    WHERE column IS NOT NULL
    AND status = 'valid'"""
            })
            base_table = f"clean_{base_table}"
            step_counter += 1

        # Add aggregation step if needed
        if is_aggregation:
            agg_name = "aggregated_data"
            steps.append({
                "name": agg_name,
                "depends_on": [base_table],
                "sql": f"""SELECT
        date_column,
        group_column,
        COUNT(*) as record_count,
        SUM(amount_column) as total_amount,
        AVG(amount_column) as avg_amount
    FROM {base_table}
    GROUP BY date_column, group_column"""
            })
            final_output = agg_name
        else:
            # Simple transformation
            final_name = "processed_data"
            steps.append({
                "name": final_name,
                "depends_on": [base_table],
                "sql": f"""SELECT
        *,
        CURRENT_DATE as processed_date
    FROM {base_table}"""
            })
            final_output = final_name

        # Create the data source spec
        data_source = {
            "vars": {
                "processing_date": "2025-01-01",
                "threshold_value": "100"
            },
            "steps": steps,
            "outputs": [final_output]
        }

        # Convert to YAML
        import yaml
        return yaml.dump({"data_source": data_source}, default_flow_style=False)