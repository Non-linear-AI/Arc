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

            steps.append(
                DataSourceStep(
                    name=step_data["name"],
                    depends_on=step_data["depends_on"],
                    sql=step_data["sql"],
                )
            )

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

            steps.append(
                DataSourceStep(
                    name=step_data["name"],
                    depends_on=step_data["depends_on"],
                    sql=step_data["sql"],
                )
            )

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
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data source file not found: {path}") from e

    def to_dict(self) -> dict[str, Any]:
        """Convert DataSourceSpec to dictionary.

        Returns:
            Dictionary representation of the data source specification
        """
        # Build dictionary manually to exclude None values and improve formatting
        data_source_dict = {
            "steps": [asdict(step) for step in self.steps],
            "outputs": self.outputs,
        }

        # Only include vars if it's not None
        if self.vars is not None:
            data_source_dict["vars"] = self.vars

        return {"data_source": data_source_dict}

    def to_json(self) -> str:
        """Convert DataSourceSpec to JSON string.

        Returns:
            JSON string representation of the data source specification
        """
        return json.dumps(self.to_dict(), indent=2)

    def _format_sql(self, sql: str, format_sql: bool = True) -> str:
        """Format SQL string with proper indentation and alignment.

        Args:
            sql: SQL string to format
            format_sql: Whether to apply SQL formatting

        Returns:
            Formatted SQL string
        """
        if not format_sql or not sql.strip():
            return sql

        # First, unescape any escaped newlines and other sequences
        # This handles SQL that comes pre-escaped from LLM JSON responses
        unescaped_sql = (
            sql.replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace('\\"', '"')
            .replace("\\'", "'")
        )

        # Simple but effective SQL formatting
        # Normalize whitespace first (but preserve intentional newlines now)
        sql = " ".join(unescaped_sql.split())

        # Keywords that should start on new lines (order matters - longer first)
        keywords = [
            "LEFT JOIN",
            "RIGHT JOIN",
            "INNER JOIN",
            "FULL JOIN",
            "OUTER JOIN",
            "GROUP BY",
            "ORDER BY",
            "SELECT",
            "FROM",
            "WHERE",
            "HAVING",
            "LIMIT",
            "UNION",
            "JOIN",
            "INSERT",
            "UPDATE",
            "DELETE",
            "WITH",
        ]

        # Start formatting
        result = sql

        # Replace keywords with newline + keyword (case-insensitive)
        for keyword in keywords:
            # Use word boundaries to avoid matching parts of other words
            import re

            pattern = r"\b" + re.escape(keyword) + r"\b"
            replacement = "\n" + keyword
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # Clean up and format lines
        lines = [line.strip() for line in result.split("\n") if line.strip()]
        formatted_lines = []

        for line in lines:
            line_upper = line.upper()

            # Handle SELECT specially - format columns
            if line_upper.startswith("SELECT "):
                select_part = line[7:]  # Remove 'SELECT '
                if "," in select_part:
                    columns = [col.strip() for col in select_part.split(",")]
                    formatted_lines.append("SELECT")
                    for i, col in enumerate(columns):
                        suffix = "," if i < len(columns) - 1 else ""
                        formatted_lines.append(f"  {col}{suffix}")
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)

        return "\n".join(formatted_lines)

    def to_yaml(self, format_sql: bool = True, add_comments: bool = False) -> str:
        """Convert DataSourceSpec to YAML string with enhanced formatting.

        Args:
            format_sql: Whether to format SQL statements for better readability
            add_comments: Whether to add section comments

        Returns:
            YAML string representation of the data source specification
        """

        class SQLFormattingDumper(yaml.SafeDumper):
            """Custom YAML dumper that formats SQL strings as literal blocks."""

            def represent_str(self, data):
                # For now, use default representation - we'll post-process SQL blocks
                return self.represent_scalar("tag:yaml.org,2002:str", data)

        # Add the custom string representer
        SQLFormattingDumper.add_representer(str, SQLFormattingDumper.represent_str)

        # Create a copy of the data with formatted SQL if requested
        data_dict = self.to_dict()
        if format_sql:
            for step_data in data_dict["data_source"]["steps"]:
                if "sql" in step_data:
                    step_data["sql"] = self._format_sql(step_data["sql"], format_sql)

        # Generate base YAML
        yaml_content = yaml.dump(
            data_dict,
            Dumper=SQLFormattingDumper,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120,  # Wider lines for better SQL formatting
        )

        # Enhanced post-processing for better structure
        lines = yaml_content.split("\n")
        result_lines = []
        current_section = None

        for i, line in enumerate(lines):
            # Add section comments if requested
            if add_comments:
                if line.strip() == "data_source:":
                    result_lines.append("# Arc Data Processing Configuration")
                    result_lines.append(line)
                    continue
                elif line.strip().startswith("vars:"):
                    result_lines.append("")
                    result_lines.append("  # Variable definitions")
                    result_lines.append(line)
                    continue
                elif line.strip() == "steps:":
                    result_lines.append("")
                    result_lines.append("  # Processing steps")
                    result_lines.append(line)
                    continue
                elif line.strip().startswith("outputs:"):
                    result_lines.append("")
                    result_lines.append("  # Output tables")
                    result_lines.append(line)
                    continue

            result_lines.append(line)

            # Track current section for spacing
            if "data_source:" in line:
                current_section = "data_source"
            elif line.strip().startswith("vars:"):
                current_section = "vars"
            elif line.strip() == "steps:":
                current_section = "steps"
            elif line.strip().startswith("outputs:"):
                current_section = "outputs"

            # Add spacing between major sections
            if (
                current_section
                and not add_comments
                and (
                    line.strip().startswith("vars:")
                    or line.strip() == "steps:"
                    or line.strip().startswith("outputs:")
                )
                and result_lines
                and result_lines[-2].strip()
            ):
                result_lines.insert(-1, "")

            # Add spacing between steps
            if (
                current_section == "steps"
                and line.strip()
                and not line.startswith("  - name:")
                and not line.startswith("    depends_on:")
                and not line.startswith("    sql:")
                and not line.startswith("      ")  # SQL content lines
                and i < len(lines) - 1
                and lines[i + 1].startswith("  - name:")
            ):
                result_lines.append("")

        # Clean up excessive blank lines
        final_lines = []
        prev_blank = False
        for line in result_lines:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue  # Skip multiple consecutive blank lines
            final_lines.append(line)
            prev_blank = is_blank

        # Post-process to convert SQL quoted strings to literal blocks
        final_yaml = "\n".join(final_lines)
        if format_sql:
            final_yaml = self._post_process_sql_blocks(final_yaml)

        return final_yaml

    def _post_process_sql_blocks(self, yaml_content: str) -> str:
        """Post-process YAML to convert SQL quoted strings to literal blocks.

        Args:
            yaml_content: YAML string content

        Returns:
            YAML with SQL strings as literal blocks
        """
        import re

        lines = yaml_content.split("\n")
        processed_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Look for SQL field start: "    sql: "..."
            sql_start_match = re.match(r'^(\s+)sql:\s*"(.*)$', line)
            if sql_start_match:
                indent = sql_start_match.group(1)
                sql_content = sql_start_match.group(2)

                # Collect the entire multi-line quoted string
                if not sql_content.endswith('"'):
                    # Multi-line quoted string - collect all lines until closing quote
                    i += 1
                    while i < len(lines) and not lines[i].rstrip().endswith('"'):
                        sql_content += lines[i]
                        i += 1
                    if i < len(lines):
                        # Add the final line with closing quote
                        final_line = lines[i].rstrip()
                        if final_line.endswith('"'):
                            sql_content += final_line[:-1]  # Remove closing quote

                # Remove trailing quote from single-line case
                if sql_content.endswith('"'):
                    sql_content = sql_content[:-1]

                # Check if this looks like SQL and has newlines (escaped or real)
                sql_keywords = ["SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "DELETE"]
                if any(kw in sql_content.upper() for kw in sql_keywords) and (
                    "\\n" in sql_content or "\n" in sql_content
                ):
                    # Unescape the SQL content and clean up continuation chars
                    unescaped_sql = (
                        sql_content.replace("\\n", "\n")
                        .replace("\\t", "\t")
                        .replace('\\"', '"')
                        .replace("\\'", "'")
                        .replace("\\ ", "")  # Remove line continuation chars
                    )

                    # Create literal block
                    processed_lines.append(f"{indent}sql: |-")
                    for sql_line in unescaped_sql.split("\n"):
                        processed_lines.append(f"{indent}  {sql_line}")
                else:
                    # Not SQL, keep original format
                    processed_lines.append(f'{indent}sql: "{sql_content}"')
            else:
                processed_lines.append(line)

            i += 1

        return "\n".join(processed_lines)

    def to_yaml_file(
        self, path: str, format_sql: bool = True, add_comments: bool = False
    ) -> None:
        """Save DataSourceSpec to YAML file.

        Args:
            path: Path to save the YAML file
            format_sql: Whether to format SQL statements for better readability
            add_comments: Whether to add section comments
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_yaml(format_sql=format_sql, add_comments=add_comments))

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
            "required": ["data_source"],
            "properties": {
                "data_source": {
                    "type": "object",
                    "required": ["steps", "outputs"],
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["name", "depends_on", "sql"],
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": (
                                            "Unique name for this processing step"
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
                                            "Concrete SQL query for this "
                                            "transformation step"
                                        ),
                                    },
                                },
                            },
                            "description": (
                                "List of data processing steps to execute in "
                                "dependency order"
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
            },
        }
