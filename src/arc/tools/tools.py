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
                        "description": (
                            "Starting line number for partial file view (optional)"
                        ),  # noqa
                    },
                    "end_line": {
                        "type": "integer",
                        "description": (
                            "Ending line number for partial file view (optional)"
                        ),  # noqa
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
            description=(
                "Replace specific text in a file. Use this for single line edits only"
            ),  # noqa
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
                        "description": (
                            "Glob pattern for files to include (e.g. '*.ts', '*.js')"
                        ),  # noqa
                    },
                    "exclude_pattern": {
                        "type": "string",
                        "description": (
                            "Glob pattern for files to exclude "
                            "(e.g. '*.log', 'node_modules')"
                        ),  # noqa
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": (
                            "Whether search should be case sensitive (default: false)"
                        ),  # noqa
                    },
                    "whole_word": {
                        "type": "boolean",
                        "description": (
                            "Whether to match whole words only (default: false)"
                        ),  # noqa
                    },
                    "regex": {
                        "type": "boolean",
                        "description": (
                            "Whether query is a regex pattern (default: false)"
                        ),  # noqa
                    },
                    "max_results": {
                        "type": "integer",
                        "description": (
                            "Maximum number of results to return (default: 50)"
                        ),  # noqa
                    },
                    "file_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File types to search (e.g. ['js', 'ts', 'py'])",
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": (
                            "Whether to include hidden files (default: false)"
                        ),  # noqa
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
                                    "description": (
                                        "Unique identifier for the todo item"
                                    ),  # noqa
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Description of the todo item",
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "description": (
                                        "Current status of the todo item "
                                        "(default: pending)"
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
        ArcTool(
            name="database_query",
            description="Execute SQL queries against system or user databases",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute",
                    },
                    "target_db": {
                        "type": "string",
                        "enum": ["system", "user"],
                        "description": (
                            "Target database: 'system' for read-only queries "
                            "(default), 'user' for full SQL access"
                        ),
                    },
                    "validate_schema": {
                        "type": "boolean",
                        "description": (
                            "Whether to validate query against database schema "
                            "(default: true)"
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        ArcTool(
            name="schema_discovery",
            description="Discover and explore database schema information",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list_tables", "describe_table", "show_schema"],
                        "description": (
                            "Schema discovery action: 'list_tables' shows all tables, "
                            "'describe_table' shows detailed table structure, "
                            "'show_schema' shows complete database overview"
                        ),
                    },
                    "target_db": {
                        "type": "string",
                        "enum": ["system", "user"],
                        "description": (
                            "Target database: 'system' for Arc metadata, "
                            "'user' for training data (default: system)"
                        ),
                    },
                    "table_name": {
                        "type": "string",
                        "description": (
                            "Specific table name (required for 'describe_table' action)"
                        ),
                    },
                },
                "required": ["action"],
            },
        ),
        ArcTool(
            name="ml_create_model",
            description="Register a new Arc-Graph model from a YAML schema file",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Human-friendly name for the model",
                    },
                    "schema_path": {
                        "type": "string",
                        "description": "Path to the Arc-Graph YAML schema file",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional description stored with the model metadata",
                    },
                    "model_type": {
                        "type": "string",
                        "description": (
                            "Optional type identifier to store with the model. Defaults to 'ml.arc_graph'"
                        ),
                    },
                },
                "required": ["name", "schema_path"],
            },
        ),
        ArcTool(
            name="ml_train",
            description="Launch an Arc-Graph training job on a dataset",
            parameters={
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Name of the registered model to train",
                    },
                    "train_table": {
                        "type": "string",
                        "description": "User database table or dataset name with training data",
                    },
                    "target_column": {
                        "type": "string",
                        "description": "Target column for supervised learning (optional)",
                    },
                    "validation_table": {
                        "type": "string",
                        "description": "Optional validation table or dataset name",
                    },
                    "validation_split": {
                        "type": "number",
                        "description": (
                            "Optional validation split (0-1). Overrides graph configuration if provided"
                        ),
                    },
                    "epochs": {
                        "type": "integer",
                        "description": "Override training epochs",
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Override training batch size",
                    },
                    "learning_rate": {
                        "type": "number",
                        "description": "Override learning rate",
                    },
                    "checkpoint_dir": {
                        "type": "string",
                        "description": "Directory path for checkpoints (optional)",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional description for the training job",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags to attach to the training job",
                    },
                },
                "required": ["model_name", "train_table"],
            },
        ),
        ArcTool(
            name="ml_predict",
            description="Run inference with a trained Arc-Graph model and save results to a table",
            parameters={
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Name of the registered model to use for prediction",
                    },
                    "table_name": {
                        "type": "string",
                        "description": "User database table or dataset containing features",
                    },
                    "output_table": {
                        "type": "string",
                        "description": "Destination table name for prediction results",
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Batch size for prediction (default: 32)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Limit the number of rows to process",
                    },
                    "device": {
                        "type": "string",
                        "description": "Torch device for inference (default: cpu)",
                    },
                },
                "required": ["model_name", "table_name", "output_table"],
            },
        ),
    ]
