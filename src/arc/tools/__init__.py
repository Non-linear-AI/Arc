"""Tool implementations for Arc CLI."""

from arc.tools.base import ToolResult
from arc.tools.bash import BashTool
from arc.tools.data_process import DataProcessorGeneratorTool
from arc.tools.database_query import DatabaseQueryTool
from arc.tools.file_editor import FileEditorTool
from arc.tools.ml import (
    MLEvaluateTool,
    MLModelTool,
    MLPlanTool,
    MLTrainTool,
)
from arc.tools.registry import ToolRegistry
from arc.tools.schema_discovery import SchemaDiscoveryTool
from arc.tools.search import SearchTool
from arc.tools.todo import (
    CreateTodoListTool,
    TodoManager,
    TodoTool,
    UpdateTodoListTool,
)

__all__ = [
    "ToolResult",
    "ToolRegistry",
    "BashTool",
    "DataProcessorGeneratorTool",
    "DatabaseQueryTool",
    "FileEditorTool",
    "SchemaDiscoveryTool",
    "SearchTool",
    "TodoTool",
    "CreateTodoListTool",
    "UpdateTodoListTool",
    "TodoManager",
    "MLModelTool",
    "MLPlanTool",
    "MLTrainTool",
    "MLEvaluateTool",
]
