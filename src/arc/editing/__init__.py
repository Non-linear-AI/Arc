"""Multi-strategy editing system for Arc CLI."""

from arc.editing.base import EditInstruction, EditResult, EditStrategy
from arc.editing.diff_editor import DiffEditor
from arc.editing.editor_manager import EditorManager
from arc.editing.search_replace import SearchReplaceEditor
from arc.editing.whole_file import WholeFileEditor

__all__ = [
    "EditStrategy",
    "EditResult",
    "EditInstruction",
    "SearchReplaceEditor",
    "WholeFileEditor",
    "DiffEditor",
    "EditorManager",
]
