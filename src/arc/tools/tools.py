"""Tool definitions and management."""

from pathlib import Path

import yaml

from arc.core.client import ArcTool


def get_base_tools() -> list[ArcTool]:
    """Get the base set of tools available to the agent.

    Loads tool definitions from YAML configuration file.
    """
    # Load tools from YAML file
    tools_yaml_path = Path(__file__).parent.parent / "templates" / "tools.yaml"

    with open(tools_yaml_path) as f:
        tools_config = yaml.safe_load(f)

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
