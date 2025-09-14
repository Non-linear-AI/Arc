"""Tool definitions and management."""

from ..core.client import ArcTool


def get_base_tools() -> list[ArcTool]:
    """Get the base set of tools available to the agent."""
    return [
        ArcTool(
            name="view_file",
            description="View contents of a file or list directory contents",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to file or directory to view",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": ("Starting line number for partial file view (optional)"),  # noqa
                    },
                    "end_line": {
                        "type": "integer",
                        "description": ("Ending line number for partial file view (optional)"),  # noqa
                    },
                },
                "required": ["path"],
            },
        ),
        ArcTool(
            name="create_file",
            description="Create a new file with specified content",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path where the file should be created",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
        ),
        ArcTool(
            name="str_replace_editor",
            description=("Replace specific text in a file. Use this for single line edits only"),  # noqa
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit",
                    },
                    "old_str": {
                        "type": "string",
                        "description": (
                            "Text to replace (must match exactly, or will use fuzzy matching for multi-line strings)"  # noqa: E501
                        ),
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Text to replace with",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": (
                            "Replace all occurrences (default: false, "
                            "only replaces first occurrence)"
                        ),  # noqa
                    },
                },
                "required": ["path", "old_str", "new_str"],
            },
        ),
        ArcTool(
            name="bash",
            description="Execute a bash command",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute",
                    },
                },
                "required": ["command"],
            },
        ),
        ArcTool(
            name="search",
            description="Unified search tool for finding text content or files",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for or file name/path pattern",
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["text", "files", "both"],
                        "description": (
                            "Type of search: 'text' for content search, 'files' for "
                            "file names, 'both' for both (default: 'both')"
                        ),
                    },
                    "include_pattern": {
                        "type": "string",
                        "description": ("Glob pattern for files to include (e.g. '*.ts', '*.js')"),  # noqa
                    },
                    "exclude_pattern": {
                        "type": "string",
                        "description": (
                            "Glob pattern for files to exclude (e.g. '*.log', 'node_modules')"
                        ),  # noqa
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": ("Whether search should be case sensitive (default: false)"),  # noqa
                    },
                    "whole_word": {
                        "type": "boolean",
                        "description": ("Whether to match whole words only (default: false)"),  # noqa
                    },
                    "regex": {
                        "type": "boolean",
                        "description": ("Whether query is a regex pattern (default: false)"),  # noqa
                    },
                    "max_results": {
                        "type": "integer",
                        "description": ("Maximum number of results to return (default: 50)"),  # noqa
                    },
                    "file_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File types to search (e.g. ['js', 'ts', 'py'])",
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": ("Whether to include hidden files (default: false)"),  # noqa
                    },
                },
                "required": ["query"],
            },
        ),
        ArcTool(
            name="create_todo_list",
            description="Create a new todo list for planning and tracking tasks",
            parameters={
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "Array of todo items to create",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": ("Unique identifier for the todo item"),  # noqa
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Description of the todo item",
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "description": (
                                        "Current status of the todo item (default: pending)"
                                    ),  # noqa
                                },
                            },
                            "required": ["id", "content", "status"],
                        },
                    },
                },
                "required": ["todos"],
            },
        ),
        ArcTool(
            name="update_todo_list",
            description="Update existing todos in the todo list by ID",
            parameters={
                "type": "object",
                "properties": {
                    "updates": {
                        "type": "array",
                        "description": "Array of todo updates",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "ID of the todo item to update",
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "description": "New status for the todo item",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "New content for the todo item",
                                },
                                "priority": {
                                    "type": "string",
                                    "enum": ["high", "medium", "low"],
                                    "description": "New priority for the todo item",
                                },
                            },
                            "required": ["id"],
                        },
                    },
                },
                "required": ["updates"],
            },
        ),
    ]
