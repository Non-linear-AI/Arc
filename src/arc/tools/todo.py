"""TODO list management tool."""

from typing import Any

from .base import BaseTool, ToolResult


class TodoItem:
    """Represents a single TODO item."""

    def __init__(
        self, id: str, content: str, status: str = "pending", priority: str = "medium"
    ):
        self.id = id
        self.content = content
        self.status = status  # pending, in_progress, completed
        self.priority = priority  # high, medium, low

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "status": self.status,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TodoItem":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            status=data.get("status", "pending"),
            priority=data.get("priority", "medium"),
        )


class TodoTool(BaseTool):
    """Tool for managing TODO lists."""

    def __init__(self):
        self.todos: list[TodoItem] = []

    async def execute(self, action: str, **kwargs) -> ToolResult:
        """Execute TODO operation."""
        if action == "start":
            return await self.start_todo(kwargs["todo_id"])
        elif action == "update":
            return await self.update_todos(kwargs["todos"])
        else:
            return ToolResult.error_result(f"Unknown TODO action: {action}")

    async def start_todo(self, todo_id: str) -> ToolResult:
        """Start working on a specific todo item."""
        try:
            # Find the TODO item
            todo = None
            for t in self.todos:
                if t.id == todo_id:
                    todo = t
                    break

            if not todo:
                return ToolResult.error_result(f"Todo item '{todo_id}' not found")

            # Set status to in_progress
            todo.status = "in_progress"

            formatted = self._format_todo_list()
            return ToolResult.success_result(f"Started todo:\n{formatted}")

        except Exception as e:
            return ToolResult.error_result(f"Failed to start todo: {str(e)}")

    async def update_todos(self, todos: list[dict[str, Any]]) -> ToolResult:
        """Update the entire todo list."""
        try:
            # Clear existing todos
            self.todos.clear()

            # Add new todos with auto-generated IDs
            for i, todo_data in enumerate(todos):
                todo_id = f"todo_{i+1}"
                todo = TodoItem(
                    id=todo_id,
                    content=todo_data["content"],
                    status=todo_data["status"]
                )
                self.todos.append(todo)

            formatted = self._format_todo_list()
            return ToolResult.success_result(f"Todo list updated:\n{formatted}")

        except Exception as e:
            return ToolResult.error_result(f"Failed to update todos: {str(e)}")

    def _format_todo_list(self) -> str:
        """Format TODO list with progress bar style."""
        if not self.todos:
            return "No todos created yet"

        # Calculate progress
        completed = sum(1 for todo in self.todos if todo.status == "completed")
        total = len(self.todos)
        
        # Create progress bar (10 blocks)
        progress_ratio = completed / total if total > 0 else 0
        filled_blocks = int(progress_ratio * 10)
        progress_bar = "█" * filled_blocks + "░" * (10 - filled_blocks)
        
        # Header with progress
        lines = [f"📋 Update plan [{progress_bar}] {completed}/{total}"]
        
        # Add todo items
        for todo in self.todos:
            if todo.status == "completed":
                marker = "●"
                line_text = f"  └ {marker} [strike]{todo.content}[/strike]"
            elif todo.status == "in_progress":
                marker = "◐"
                line_text = f"  └ {marker} {todo.content}"
            else:
                marker = "○"  # empty
                line_text = f"  └ {marker} {todo.content}"
            
            lines.append(line_text)

        return "\n".join(lines)

    def get_todo_summary(self) -> dict[str, int]:
        """Get summary of TODO statuses."""
        summary = {"pending": 0, "in_progress": 0, "completed": 0}
        for todo in self.todos:
            summary[todo.status] += 1
        return summary
