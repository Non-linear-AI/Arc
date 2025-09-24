"""Data source specification for SQL-based feature engineering."""

from __future__ import annotations

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
    import json
except ImportError as e:
    raise RuntimeError("JSON support is required") from e


class ValidationResult(NamedTuple):
    """Result of YAML validation operation."""
    success: bool
    spec: "DataSourceSpec | None"
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

    def __post_init__(self):
        """Validate step configuration after initialization."""
        if not self.name or not self.name.strip():
            raise ValueError("Step name cannot be empty")

        if not self.sql or not self.sql.strip():
            raise ValueError(f"SQL query cannot be empty for step '{self.name}'")

        # Ensure depends_on is a list
        if not isinstance(self.depends_on, list):
            raise ValueError(f"depends_on must be a list for step '{self.name}'")


@dataclass
class DataSourceSpec:
    """Complete data source specification for SQL feature engineering."""

    steps: list[DataSourceStep]
    outputs: list[str]
    vars: dict[str, str] | None = None

    def __post_init__(self):
        """Validate spec configuration after initialization."""
        if not self.steps:
            raise ValueError("At least one step is required")

        if not self.outputs:
            raise ValueError("At least one output must be specified")

        # Validate that all outputs exist as steps
        step_names = {step.name for step in self.steps}
        for output in self.outputs:
            if output not in step_names:
                raise ValueError(f"Output '{output}' not found in steps")

        # Validate that all dependencies exist
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in step_names and not self._is_table_reference(dep):
                    raise ValueError(
                        f"Dependency '{dep}' in step '{step.name}' not found in steps "
                        f"(assuming it's an existing table)"
                    )

    def _is_table_reference(self, name: str) -> bool:
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

        if "data_source" not in data:
            raise ValueError("Data must contain 'data_source' section")

        ds_data = data["data_source"]

        if not isinstance(ds_data, dict):
            raise ValueError("data_source must be a mapping")

        # Parse vars (optional)
        vars_dict = ds_data.get("vars")
        if vars_dict is not None and not isinstance(vars_dict, dict):
            raise ValueError("vars must be a mapping")

        # Parse steps (required)
        if "steps" not in ds_data:
            raise ValueError("data_source must contain 'steps'")

        steps_data = ds_data["steps"]
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

            steps.append(DataSourceStep(
                name=step_data["name"],
                depends_on=step_data["depends_on"],
                sql=step_data["sql"]
            ))

        # Parse outputs (required)
        if "outputs" not in ds_data:
            raise ValueError("data_source must contain 'outputs'")

        outputs = ds_data["outputs"]
        if not isinstance(outputs, list):
            raise ValueError("outputs must be a list")

        return cls(
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

        if "data_source" not in data:
            raise ValueError("YAML must contain 'data_source' section")

        ds_data = data["data_source"]

        if not isinstance(ds_data, dict):
            raise ValueError("data_source must be a mapping")

        # Parse vars (optional)
        vars_dict = ds_data.get("vars")
        if vars_dict is not None and not isinstance(vars_dict, dict):
            raise ValueError("vars must be a mapping")

        # Parse steps (required)
        if "steps" not in ds_data:
            raise ValueError("data_source must contain 'steps'")

        steps_data = ds_data["steps"]
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

            steps.append(DataSourceStep(
                name=step_data["name"],
                depends_on=step_data["depends_on"],
                sql=step_data["sql"]
            ))

        # Parse outputs (required)
        if "outputs" not in ds_data:
            raise ValueError("data_source must contain 'outputs'")

        outputs = ds_data["outputs"]
        if not isinstance(outputs, list):
            raise ValueError("outputs must be a list")

        return cls(
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
        except FileNotFoundError:
            raise FileNotFoundError(f"Data source file not found: {path}")

    def to_dict(self) -> dict[str, Any]:
        """Convert DataSourceSpec to dictionary.

        Returns:
            Dictionary representation of the data source specification
        """
        return {"data_source": asdict(self)}

    def to_json(self) -> str:
        """Convert DataSourceSpec to JSON string.

        Returns:
            JSON string representation of the data source specification
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_yaml(self) -> str:
        """Convert DataSourceSpec to YAML string.

        Returns:
            YAML string representation of the data source specification
        """
        return yaml.dump(self.to_dict(), default_flow_style=False)

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
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, sql)

        # Check that all referenced variables are defined
        for var_name in matches:
            if var_name not in self.vars:
                raise ValueError(f"Variable '${{{var_name}}}' not defined in vars section")

        # Substitute variables
        result = sql
        for var_name, var_value in self.vars.items():
            result = result.replace(f"${{{var_name}}}", var_value)

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
            if step.name not in visited:
                if has_cycle(step.name):
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

        in_degree = {name: 0 for name in step_names}

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
            current_step = step_map[current]
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
    def validate_yaml_string(cls, yaml_str: str) -> "ValidationResult":
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
                error=None
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
                error=f"YAML validation failed: {str(e)}"
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
                error=f"Unexpected validation error: {str(e)}"
            )

    @classmethod
    def validate_yaml_file(cls, file_path: str) -> "ValidationResult":
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
                    error=f"YAML file not found: {file_path}"
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
                error=None
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
                error=f"YAML validation failed: {str(e)}"
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
                error=f"Unexpected validation error: {str(e)}"
            )

    @classmethod
    def get_json_schema(cls) -> dict[str, Any]:
        """Get JSON schema for DataSourceSpec for LLM prompts.

        Returns:
            JSON schema dictionary that can be used in LLM prompts
        """
        return {
            "type": "object",
            "required": ["data_source"],
            "properties": {
                "data_source": {
                    "type": "object",
                    "required": ["steps", "outputs"],
                    "properties": {
                        "vars": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                            "description": "Optional variables for SQL substitution using ${var_name} syntax"
                        },
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["name", "depends_on", "sql"],
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "Unique name for this processing step"
                                    },
                                    "depends_on": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of table names or step names this step depends on"
                                    },
                                    "sql": {
                                        "type": "string",
                                        "description": "SQL query for this transformation step"
                                    }
                                }
                            },
                            "description": "List of data processing steps to execute in dependency order"
                        },
                        "outputs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of step names that should be materialized as final output tables"
                        }
                    }
                }
            }
        }