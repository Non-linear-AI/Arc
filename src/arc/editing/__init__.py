"""Multi-strategy editing system for Arc CLI."""

from .base import EditInstruction, EditResult, EditStrategy
from .diff_editor import DiffEditor
from .editor_manager import EditorManager
from .search_replace import SearchReplaceEditor
from .whole_file import WholeFileEditor

__all__ = [
    "EditStrategy",
    "EditResult",
    "EditInstruction",
    "SearchReplaceEditor",
    "WholeFileEditor",
    "DiffEditor",
    "EditorManager",
]
