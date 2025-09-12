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
        if action == "create":
            return await self.create_todo_list(kwargs["todos"])
        elif action == "update":
            return await self.update_todo_list(kwargs["updates"])
        else:
            return ToolResult.error_result(f"Unknown TODO action: {action}")

    async def create_todo_list(self, todos: list[dict[str, Any]]) -> ToolResult:
        """Create a new TODO list."""
        try:
            # Clear existing todos
            self.todos.clear()

            # Add new todos
            for todo_data in todos:
                todo = TodoItem.from_dict(todo_data)
                self.todos.append(todo)

            formatted = self._format_todo_list()
            return ToolResult.success_result(f"TODO list created:\n{formatted}")

        except Exception as e:
            return ToolResult.error_result(f"Failed to create TODO list: {str(e)}")

    async def update_todo_list(self, updates: list[dict[str, Any]]) -> ToolResult:
        """Update existing TODO items."""
        try:
            updated_count = 0

            for update in updates:
                todo_id = update["id"]

                # Find the TODO item
                todo = None
                for t in self.todos:
                    if t.id == todo_id:
                        todo = t
                        break

                if not todo:
                    continue

                # Apply updates
                if "status" in update:
                    todo.status = update["status"]
                if "content" in update:
                    todo.content = update["content"]
                if "priority" in update:
                    todo.priority = update["priority"]

                updated_count += 1

            formatted = self._format_todo_list()
            return ToolResult.success_result(
                f"Updated {updated_count} TODO item(s):\n{formatted}"
            )

        except Exception as e:
            return ToolResult.error_result(f"Failed to update TODO list: {str(e)}")

    def _format_todo_list(self) -> str:
        """Format TODO list for display (simple, arc-cli style)."""
        if not self.todos:
            return "No todos created yet"

        lines: list[str] = []

        for idx, todo in enumerate(self.todos):
            if todo.status == "completed":
                marker = "●"  # filled
                # Use rich markup strike for clarity
                line_text = f"{marker} [strike]{todo.content}[/strike]"
            elif todo.status == "in_progress":
                marker = "◐"  # half-filled
                line_text = f"{marker} {todo.content}"
            else:
                marker = "○"  # empty
                line_text = f"{marker} {todo.content}"

            # First line no indent; subsequent lines indented by two spaces
            indent = "" if idx == 0 else "  "
            line = f"{indent}{line_text}"
            lines.append(line)

        return "\n".join(lines)

    def get_todo_summary(self) -> dict[str, int]:
        """Get summary of TODO statuses."""
        summary = {"pending": 0, "in_progress": 0, "completed": 0}
        for todo in self.todos:
            summary[todo.status] += 1
        return summary
