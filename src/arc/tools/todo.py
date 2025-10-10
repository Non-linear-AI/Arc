"""TODO list management tool."""

from typing import Any

from arc.tools.base import BaseTool, ToolResult


class TodoItem:
    """Represents a single TODO item."""

    def __init__(self, id: str, content: str, status: str = "pending"):
        self.id = id
        self.content = content
        self.status = status  # pending, completed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TodoItem":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            status=data.get("status", "pending"),
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
        elif action == "view":
            return await self.view_todo_list()
        else:
            return ToolResult.error_result(f"Unknown TODO action: {action}")

    async def create_todo_list(self, todos: list[dict[str, Any]]) -> ToolResult:
        """Create a new TODO list with items."""
        try:
            # Validate and create todos
            new_todos = []
            for todo_data in todos:
                # Auto-generate ID if not provided
                todo_id = todo_data.get("id", f"todo_{len(new_todos) + 1}")

                todo = TodoItem(
                    id=todo_id,
                    content=todo_data["content"],
                    status=todo_data.get("status", "pending"),
                )
                new_todos.append(todo)

            # Replace existing todos
            self.todos = new_todos
            formatted = self._format_todo_list()
            return ToolResult.success_result(f"Todo list created:\n{formatted}")

        except Exception as e:
            return ToolResult.error_result(f"Failed to create todo list: {str(e)}")

    async def update_todo_list(self, updates: list[dict[str, Any]]) -> ToolResult:
        """Update existing TODO items by ID."""
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
                    return ToolResult.error_result(f"Todo item '{todo_id}' not found")

                # Apply updates
                if "status" in update:
                    todo.status = update["status"]
                if "content" in update:
                    todo.content = update["content"]

                updated_count += 1

            formatted = self._format_todo_list()
            return ToolResult.success_result(
                f"Updated {updated_count} TODO item(s):\n{formatted}"
            )

        except Exception as e:
            return ToolResult.error_result(f"Failed to update todo list: {str(e)}")

    async def view_todo_list(self) -> ToolResult:
        """View the current TODO list."""
        formatted = self._format_todo_list()
        return ToolResult.success_result(formatted)

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

        # Header with progress - simpler title
        lines = [f"📋 [{progress_bar}] {completed}/{total}"]

        # Add todo items with IDs shown for reference
        for todo in self.todos:
            if todo.status == "completed":
                marker = "●"
                line_text = f"  └ {marker} [strike]{todo.content}[/strike]"
            else:
                marker = "○"
                line_text = f"  └ {marker} {todo.content}"

            lines.append(line_text)

        return "\n".join(lines)

    def get_todo_summary(self) -> dict[str, int]:
        """Get summary of TODO statuses."""
        summary = {"pending": 0, "completed": 0}
        for todo in self.todos:
            summary[todo.status] += 1
        return summary
