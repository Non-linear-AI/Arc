"""Data source specification for SQL-based feature engineering."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple

try:
    import yaml
except ImportError as e:
    raise RuntimeError(
        "PyYAML is required for Arc-Graph. "
        "Install with 'uv add pyyaml' or 'pip install pyyaml'."
    ) from e

try:
    from sqlfluff.core import FluffConfig, Linter
except ImportError as e:
    raise RuntimeError(
        "sqlfluff is required for Arc-Graph. "
        "Install with 'uv add sqlfluff' or 'pip install sqlfluff'."
    ) from e


class ValidationResult(NamedTuple):
    """Result of YAML validation operation."""

    success: bool
    spec: DataSourceSpec | None
    steps_count: int
    outputs_count: int
    variables_count: int
    execution_order: list[str]
    outputs: list[str]
    error: str | None


@dataclass
class DataSourceStep:
    """Configuration for a single data processing step."""

    name: str
    depends_on: list[str]
    sql: str
    type: str = "table"  # Default to 'table' for backward compatibility

    def __post_init__(self):
        """Validate step configuration after initialization."""
        if not self.name or not self.name.strip():
            raise ValueError("Step name cannot be empty")

        if not self.sql or not self.sql.strip():
            raise ValueError(f"SQL query cannot be empty for step '{self.name}'")

        # Ensure depends_on is a list
        if not isinstance(self.depends_on, list):
            raise ValueError(f"depends_on must be a list for step '{self.name}'")

        # Validate type field
        valid_types = ["table", "view", "execute"]
        if self.type not in valid_types:
            raise ValueError(
                f"Invalid type '{self.type}' for step '{self.name}'. "
                f"Must be one of: {', '.join(valid_types)}"
            )


@dataclass
class DataSourceSpec:
    """Complete data source specification for SQL feature engineering."""

    name: str
    description: str
    steps: list[DataSourceStep]
    outputs: list[str]
    vars: dict[str, str] | None = None

    def __post_init__(self):
        """Validate spec configuration after initialization."""
        if not self.name or not self.name.strip():
            raise ValueError("Name is required and cannot be empty")

        if not self.description or not self.description.strip():
            raise ValueError("Description is required and cannot be empty")

        if not self.steps:
            raise ValueError("At least one step is required")

        if not self.outputs:
            raise ValueError("At least one output must be specified")

        # Validate that all outputs exist as steps
        step_names = {step.name for step in self.steps}
        for output in self.outputs:
            if output not in step_names:
                raise ValueError(f"Output '{output}' not found in steps")

        # Validate that execute-type steps are not in outputs
        for step in self.steps:
            if step.type == "execute" and step.name in self.outputs:
                raise ValueError(
                    f"Step '{step.name}' has type='execute' but is listed in outputs. "
                    "Execute-only steps don't produce tables and cannot be outputs."
                )

        # Validate that all dependencies exist
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in step_names and not self._is_table_reference(dep):
                    raise ValueError(
                        f"Dependency '{dep}' in step '{step.name}' not found in steps "
                        f"(assuming it's an existing table)"
                    )

    def _is_table_reference(self, _name: str) -> bool:
        """Check if name looks like a table reference (not a step name)."""
        # For now, assume names not in steps are table references
        # Could be enhanced with actual table validation
        return True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataSourceSpec:
        """Create DataSourceSpec from dictionary.

        Args:
            data: Dictionary containing data source specification

        Returns:
            DataSourceSpec: Parsed and validated data source specification

        Raises:
            ValueError: If data doesn't contain valid spec
        """
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")

        # Parse name (required)
        if "name" not in data:
            raise ValueError("Data must contain 'name' field")
        name = data["name"]
        if not isinstance(name, str):
            raise ValueError("name must be a string")

        # Parse description (required)
        if "description" not in data:
            raise ValueError("Data must contain 'description' field")
        description = data["description"]
        if not isinstance(description, str):
            raise ValueError("description must be a string")

        # Parse vars (optional)
        vars_dict = data.get("vars")
        if vars_dict is not None and not isinstance(vars_dict, dict):
            raise ValueError("vars must be a mapping")

        # Parse steps (required)
        if "steps" not in data:
            raise ValueError("Data must contain 'steps'")

        steps_data = data["steps"]
        if not isinstance(steps_data, list):
            raise ValueError("steps must be a list")

        steps = []
        for i, step_data in enumerate(steps_data):
            if not isinstance(step_data, dict):
                raise ValueError(f"Step {i} must be a mapping")

            required_fields = ["name", "depends_on", "sql"]
            for field in required_fields:
                if field not in step_data:
                    raise ValueError(f"Step {i} must have '{field}' field")

            steps.append(
                DataSourceStep(
                    name=step_data["name"],
                    depends_on=step_data["depends_on"],
                    sql=step_data["sql"],
                    type=step_data.get("type", "table"),  # Default to 'table'
                )
            )

        # Parse outputs (required)
        if "outputs" not in data:
            raise ValueError("Data must contain 'outputs'")

        outputs = data["outputs"]
        if not isinstance(outputs, list):
            raise ValueError("outputs must be a list")

        return cls(
            name=name,
            description=description,
            steps=steps,
            outputs=outputs,
            vars=vars_dict,
        )

    @classmethod
    def from_yaml(cls, yaml_str: str) -> DataSourceSpec:
        """Parse DataSourceSpec from YAML string.

        Args:
            yaml_str: YAML string containing data source specification

        Returns:
            DataSourceSpec: Parsed and validated data source specification

        Raises:
            ValueError: If YAML is invalid or doesn't contain valid spec
        """
        try:
            data = yaml.safe_load(yaml_str)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("Top-level YAML must be a mapping")

        # Parse name (required)
        if "name" not in data:
            raise ValueError("YAML must contain 'name' field")
        name = data["name"]
        if not isinstance(name, str):
            raise ValueError("name must be a string")

        # Parse description (required)
        if "description" not in data:
            raise ValueError("YAML must contain 'description' field")
        description = data["description"]
        if not isinstance(description, str):
            raise ValueError("description must be a string")

        # Parse vars (optional)
        vars_dict = data.get("vars")
        if vars_dict is not None and not isinstance(vars_dict, dict):
            raise ValueError("vars must be a mapping")

        # Parse steps (required)
        if "steps" not in data:
            raise ValueError("YAML must contain 'steps'")

        steps_data = data["steps"]
        if not isinstance(steps_data, list):
            raise ValueError("steps must be a list")

        steps = []
        for i, step_data in enumerate(steps_data):
            if not isinstance(step_data, dict):
                raise ValueError(f"Step {i} must be a mapping")

            if "name" not in step_data:
                raise ValueError(f"Step {i} must have 'name'")

            if "depends_on" not in step_data:
                raise ValueError(f"Step {i} must have 'depends_on'")

            if "sql" not in step_data:
                raise ValueError(f"Step {i} must have 'sql'")

            steps.append(
                DataSourceStep(
                    name=step_data["name"],
                    depends_on=step_data["depends_on"],
                    sql=step_data["sql"],
                    type=step_data.get("type", "table"),  # Default to 'table'
                )
            )

        # Parse outputs (required)
        if "outputs" not in data:
            raise ValueError("YAML must contain 'outputs'")

        outputs = data["outputs"]
        if not isinstance(outputs, list):
            raise ValueError("outputs must be a list")

        return cls(
            name=name,
            description=description,
            steps=steps,
            outputs=outputs,
            vars=vars_dict,
        )

    @classmethod
    def from_yaml_file(cls, path: str) -> DataSourceSpec:
        """Parse DataSourceSpec from YAML file.

        Args:
            path: Path to YAML file containing data source specification

        Returns:
            DataSourceSpec: Parsed and validated data source specification

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid or doesn't contain valid spec
        """
        try:
            with open(path, encoding="utf-8") as f:
                return cls.from_yaml(f.read())
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data source file not found: {path}") from e

    def to_dict(self) -> dict[str, Any]:
        """Convert DataSourceSpec to dictionary.

        Returns:
            Dictionary representation of the data source specification
        """
        # Build dictionary manually to exclude None values and improve formatting
        result_dict = {
            "name": self.name,
            "description": self.description,
            "steps": [asdict(step) for step in self.steps],
            "outputs": self.outputs,
        }

        # Only include vars if it's not None
        if self.vars is not None:
            result_dict["vars"] = self.vars

        return result_dict

    def to_json(self) -> str:
        """Convert DataSourceSpec to JSON string.

        Returns:
            JSON string representation of the data source specification
        """
        return json.dumps(self.to_dict(), indent=2)

    def _format_sql(self, sql: str) -> str:
        """Format SQL string using sqlfluff library.

        Args:
            sql: SQL string to format

        Returns:
            Formatted SQL string
        """
        if not sql.strip():
            return sql

        # First, unescape any escaped newlines and other sequences
        # This handles SQL that comes pre-escaped from LLM JSON responses
        unescaped_sql = (
            sql.replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace('\\"', '"')
            .replace("\\'", "'")
        )

        # Use sqlfluff for professional SQL formatting with readable multi-line output
        try:
            config = FluffConfig(
                overrides={
                    "dialect": "duckdb",  # DuckDB dialect for Arc's database
                    "core": {
                        "max_line_length": 80,
                    },
                    "layout": {
                        "type": {
                            "comma": {
                                "line_position": "trailing",
                                "spacing_before": "touch",
                            },
                        },
                    },
                }
            )
            linter = Linter(config=config)
            result = linter.lint_string(unescaped_sql, fix=True)
            formatted_sql = result.tree.raw if result.tree else unescaped_sql
        except Exception:
            # If formatting fails, return the unescaped SQL
            formatted_sql = unescaped_sql

        return formatted_sql.strip()

    def _add_yaml_spacing(self, yaml_content: str) -> str:
        """Add blank lines between YAML sections and steps for better readability.
        Also fix indentation for list items.

        Args:
            yaml_content: Original YAML content string

        Returns:
            YAML content with added spacing and proper indentation
        """
        lines = yaml_content.splitlines()
        result_lines = []
        in_steps_section = False
        in_outputs_section = False

        for i, line in enumerate(lines):
            # Track which section we're in
            if line.strip() == "steps:":
                in_steps_section = True
                in_outputs_section = False
            elif line.strip() == "outputs:":
                in_steps_section = False
                in_outputs_section = True
            elif line.strip() == "vars:":
                in_steps_section = False
                in_outputs_section = False
            elif line and not line[0].isspace() and not line.startswith("-"):
                # Root level key (not a list item), reset sections
                if not line.startswith(("name:", "description:")):
                    in_steps_section = False
                    in_outputs_section = False

            # Fix indentation for list items at wrong level
            processed_line = line
            if (
                line.startswith("- ")
                and not line.startswith("  ")
                and (in_steps_section or in_outputs_section)
            ):
                # List item with no indentation, should be indented
                processed_line = "  " + line

            result_lines.append(processed_line)

            # Add blank line after 'name:' field (when value is on same line)
            if line.startswith("name: ") and not line.startswith("  "):
                result_lines.append("")

            # Add blank line before 'steps:', 'outputs:' and 'vars:' sections
            if line.strip() in ["steps:", "outputs:", "vars:"] and i > 0:
                # Insert blank line before this section
                result_lines.insert(-1, "")

            # Add blank line between steps (before each '- name:' except the first one)
            elif line.strip().startswith("- name:") and i > 0 and in_steps_section:
                # Check if this is not the first step by looking for previous steps
                has_previous_step = False
                for j in range(i - 1, -1, -1):
                    if lines[j].strip().startswith("- name:"):
                        has_previous_step = True
                        break
                    elif lines[j].strip() == "steps:":
                        break

                if has_previous_step:
                    # Insert blank line before this step
                    result_lines.insert(-1, "")

        return "\n".join(result_lines)

    def to_yaml(self) -> str:
        """Convert DataSourceSpec to YAML string with proper formatting.

        Returns:
            YAML string representation of the data source specification
        """
        lines = []

        # Name and description
        lines.append(f"name: {self.name}")
        lines.append("")
        lines.append(f"description: {self.description}")
        lines.append("")

        # Variables (optional)
        if self.vars:
            lines.append("vars:")
            for key, value in self.vars.items():
                # Escape value if it contains special characters
                if isinstance(value, str) and (":" in value or "#" in value):
                    value = f'"{value}"'
                lines.append(f"  {key}: {value}")
            lines.append("")

        # Steps
        lines.append("steps:")
        for i, step in enumerate(self.steps):
            if i > 0:
                lines.append("")  # Blank line between steps

            lines.append(f"  - name: {step.name}")

            # type field
            lines.append(f"    type: {step.type}")

            # depends_on list
            lines.append("    depends_on:")
            if step.depends_on:
                for dep in step.depends_on:
                    lines.append(f"      - {dep}")
            else:
                lines.append("      []")

            # SQL - format and indent properly
            formatted_sql = self._format_sql(step.sql)
            if "\n" in formatted_sql:
                # Multi-line SQL - use literal block style
                lines.append("    sql: |-")
                for sql_line in formatted_sql.split("\n"):
                    if sql_line.strip():  # Skip empty lines
                        lines.append(f"      {sql_line}")
            else:
                # Single line SQL
                lines.append(f"    sql: {formatted_sql}")

        # Outputs
        lines.append("")
        lines.append("outputs:")
        for output in self.outputs:
            lines.append(f"  - {output}")

        return "\n".join(lines)

    def to_yaml_file(self, path: str) -> None:
        """Save DataSourceSpec to YAML file.

        Args:
            path: Path to save the YAML file
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_yaml())

    def substitute_vars(self, sql: str) -> str:
        """Substitute variables in SQL query.

        Args:
            sql: SQL query string that may contain ${var} placeholders

        Returns:
            SQL string with variables substituted

        Raises:
            ValueError: If a variable is referenced but not defined
        """
        if not self.vars:
            return sql

        # Find all ${var} patterns
        pattern = r"\$\{([^}]+)\}"
        matches = re.findall(pattern, sql)

        # Check that all referenced variables are defined
        for var_name in matches:
            if var_name not in self.vars:
                raise ValueError(
                    f"Variable '${{{var_name}}}' not defined in vars section"
                )

        # Substitute variables
        result = sql
        for var_name, var_value in self.vars.items():
            result = result.replace(f"${{{var_name}}}", str(var_value))

        return result

    def get_step_names(self) -> list[str]:
        """Get list of all step names.

        Returns:
            List of step names in order
        """
        return [step.name for step in self.steps]

    def get_dependencies(self) -> dict[str, list[str]]:
        """Get dependency mapping for all steps.

        Returns:
            Dictionary mapping step names to their dependencies
        """
        return {step.name: step.depends_on.copy() for step in self.steps}

    def validate_dependencies(self) -> None:
        """Validate that dependency graph has no cycles.

        Raises:
            ValueError: If circular dependencies are detected
        """
        # Use DFS to detect cycles
        visited = set()
        rec_stack = set()

        def has_cycle(step_name: str) -> bool:
            visited.add(step_name)
            rec_stack.add(step_name)

            step = next((s for s in self.steps if s.name == step_name), None)
            if step is None:
                return False

            for dep in step.depends_on:
                # Skip table references (not step names)
                if dep not in {s.name for s in self.steps}:
                    continue

                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(step_name)
            return False

        for step in self.steps:
            if step.name not in visited and has_cycle(step.name):
                raise ValueError("Circular dependency detected in steps")

    def get_execution_order(self) -> list[DataSourceStep]:
        """Get steps in topological order for execution.

        Returns:
            List of steps ordered by dependencies (dependencies first)

        Raises:
            ValueError: If circular dependencies are detected
        """
        # First validate no cycles
        self.validate_dependencies()

        # Build mapping of step names to steps
        step_map = {step.name: step for step in self.steps}
        step_names = set(step_map.keys())

        in_degree = dict.fromkeys(step_names, 0)

        for step in self.steps:
            for dep in step.depends_on:
                # Only count dependencies that are other steps (not external tables)
                if dep in step_names:
                    in_degree[step.name] += 1

        # Initialize queue with steps that have no dependencies
        queue = [name for name in step_names if in_degree[name] == 0]
        result = []

        while queue:
            # Take a step with no remaining dependencies
            current = queue.pop(0)
            result.append(step_map[current])

            # For each step that depends on the current step
            for step in self.steps:
                if current in step.depends_on:
                    in_degree[step.name] -= 1
                    if in_degree[step.name] == 0:
                        queue.append(step.name)

        # Check if all steps were processed (no cycles)
        if len(result) != len(self.steps):
            raise ValueError("Circular dependency detected in steps")

        return result

    @classmethod
    def validate_yaml_string(cls, yaml_str: str) -> ValidationResult:
        """Validate YAML string and return detailed results.

        Args:
            yaml_str: YAML string to validate

        Returns:
            ValidationResult with success status and details
        """
        try:
            spec = cls.from_yaml(yaml_str)
            spec.validate_dependencies()
            ordered_steps = spec.get_execution_order()

            return ValidationResult(
                success=True,
                spec=spec,
                steps_count=len(spec.steps),
                outputs_count=len(spec.outputs),
                variables_count=len(spec.vars) if spec.vars else 0,
                execution_order=[step.name for step in ordered_steps],
                outputs=spec.outputs,
                error=None,
            )
        except ValueError as e:
            return ValidationResult(
                success=False,
                spec=None,
                steps_count=0,
                outputs_count=0,
                variables_count=0,
                execution_order=[],
                outputs=[],
                error=f"YAML validation failed: {str(e)}",
            )
        except Exception as e:
            return ValidationResult(
                success=False,
                spec=None,
                steps_count=0,
                outputs_count=0,
                variables_count=0,
                execution_order=[],
                outputs=[],
                error=f"Unexpected validation error: {str(e)}",
            )

    @classmethod
    def validate_yaml_file(cls, file_path: str) -> ValidationResult:
        """Validate YAML file and return detailed results.

        Args:
            file_path: Path to YAML file to validate

        Returns:
            ValidationResult with success status and details
        """
        from pathlib import Path

        try:
            yaml_file = Path(file_path)
            if not yaml_file.exists():
                return ValidationResult(
                    success=False,
                    spec=None,
                    steps_count=0,
                    outputs_count=0,
                    variables_count=0,
                    execution_order=[],
                    outputs=[],
                    error=f"YAML file not found: {file_path}",
                )

            spec = cls.from_yaml_file(file_path)
            spec.validate_dependencies()
            ordered_steps = spec.get_execution_order()

            return ValidationResult(
                success=True,
                spec=spec,
                steps_count=len(spec.steps),
                outputs_count=len(spec.outputs),
                variables_count=len(spec.vars) if spec.vars else 0,
                execution_order=[step.name for step in ordered_steps],
                outputs=spec.outputs,
                error=None,
            )
        except ValueError as e:
            return ValidationResult(
                success=False,
                spec=None,
                steps_count=0,
                outputs_count=0,
                variables_count=0,
                execution_order=[],
                outputs=[],
                error=f"YAML validation failed: {str(e)}",
            )
        except Exception as e:
            return ValidationResult(
                success=False,
                spec=None,
                steps_count=0,
                outputs_count=0,
                variables_count=0,
                execution_order=[],
                outputs=[],
                error=f"Unexpected validation error: {str(e)}",
            )

    @classmethod
    def get_json_schema(cls) -> dict[str, Any]:
        """Get JSON schema for DataSourceSpec for LLM prompts.

        Returns:
            JSON schema dictionary that can be used in LLM prompts
        """
        return {
            "type": "object",
            "required": ["name", "description", "steps", "outputs"],
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the data processing pipeline",
                },
                "description": {
                    "type": "string",
                    "description": "Description of what this pipeline does",
                },
                "vars": {
                    "type": "object",
                    "description": (
                        "Optional variables for SQL substitution (${var} syntax)"
                    ),
                },
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "depends_on", "sql", "type"],
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": ("Unique name for this processing step"),
                            },
                            "type": {
                                "type": "string",
                                "enum": ["table", "view", "execute"],
                                "description": (
                                    "Step type: 'table' for final output tables "
                                    "(SELECT), 'view' for intermediate results "
                                    "(SELECT), 'execute' for DDL/DML (DROP, etc.)"
                                ),
                            },
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "List of table names or step names this "
                                    "step depends on"
                                ),
                            },
                            "sql": {
                                "type": "string",
                                "description": (
                                    "Concrete SQL query for this transformation step"
                                ),
                            },
                        },
                    },
                    "description": (
                        "List of data processing steps to execute in dependency order"
                    ),
                },
                "outputs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of step names that should be materialized "
                        "as final output tables"
                    ),
                },
            },
        }
