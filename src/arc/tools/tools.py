"""Tool definitions and management."""

from pathlib import Path

import yaml

from arc.core.client import ArcTool


def validate_tools_yaml(tools_config: dict) -> None:
    """Validate tools.yaml structure.

    Args:
        tools_config: Loaded YAML configuration

    Raises:
        ValueError: If YAML structure is invalid
    """
    if not isinstance(tools_config, dict):
        raise ValueError("Tools configuration must be a dictionary")

    if "tools" not in tools_config:
        raise ValueError("Tools configuration must have a 'tools' key")

    if not isinstance(tools_config["tools"], list):
        raise ValueError("'tools' must be a list")

    seen_names = set()
    for i, tool_def in enumerate(tools_config["tools"]):
        if not isinstance(tool_def, dict):
            raise ValueError(f"Tool at index {i} must be a dictionary")

        # Check required fields
        required_fields = ["name", "description", "parameters"]
        for field in required_fields:
            if field not in tool_def:
                raise ValueError(f"Tool at index {i} missing required field: {field}")

        # Check for duplicate names
        name = tool_def["name"]
        if name in seen_names:
            raise ValueError(f"Duplicate tool name: {name}")
        seen_names.add(name)

        # Validate parameters structure
        params = tool_def["parameters"]
        if not isinstance(params, dict):
            raise ValueError(f"Tool '{name}' parameters must be a dictionary")

        if "type" not in params:
            raise ValueError(f"Tool '{name}' parameters must have a 'type' field")

        if params["type"] != "object":
            param_type = params["type"]
            raise ValueError(
                f"Tool '{name}' parameters type must be 'object', got '{param_type}'"
            )


def get_base_tools() -> list[ArcTool]:
    """Get the base set of tools available to the agent.

    Loads tool definitions from YAML configuration file with validation.

    Returns:
        List of ArcTool instances

    Raises:
        ValueError: If tools.yaml is invalid
        FileNotFoundError: If tools.yaml not found
    """
    # Load tools from YAML file
    tools_yaml_path = Path(__file__).parent.parent / "templates" / "tools.yaml"

    if not tools_yaml_path.exists():
        raise FileNotFoundError(f"Tools configuration not found: {tools_yaml_path}")

    with open(tools_yaml_path) as f:
        tools_config = yaml.safe_load(f)

    # Validate YAML structure
    validate_tools_yaml(tools_config)

    # Convert YAML to ArcTool instances
    tools = []
    for tool_def in tools_config["tools"]:
        tools.append(
            ArcTool(
                name=tool_def["name"],
                description=tool_def["description"],
                parameters=tool_def["parameters"],
            )
        )

    return tools


def get_tool_names() -> list[str]:
    """Get list of all tool names defined in tools.yaml.

    Returns:
        List of tool names

    Raises:
        ValueError: If tools.yaml is invalid
        FileNotFoundError: If tools.yaml not found
    """
    tools = get_base_tools()
    return [tool.name for tool in tools]
