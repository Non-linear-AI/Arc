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


class TodoManager:
    """Centralized state manager for TODO lists.

    Shared across CreateTodoListTool and UpdateTodoListTool to maintain
    a single source of truth for the todo list state.
    """

    def __init__(self):
        self.todos: list[TodoItem] = []

    def create_todos(self, todos_data: list[dict[str, Any]]) -> str:
        """Create a new TODO list, replacing any existing todos.

        Args:
            todos_data: List of todo dictionaries with 'content' and optional 'status'

        Returns:
            Formatted string representation of the new todo list
        """
        new_todos = []
        for todo_data in todos_data:
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
        return self.format_todo_list()

    def update_todos(self, updates: list[dict[str, Any]]) -> tuple[int, str]:
        """Update existing TODO items by ID.

        Args:
            updates: List of update dictionaries with 'id' and fields to update

        Returns:
            Tuple of (updated_count, formatted_todo_list)

        Raises:
            ValueError: If a todo ID is not found
        """
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
                raise ValueError(f"Todo item '{todo_id}' not found")

            # Apply updates
            if "status" in update:
                todo.status = update["status"]
            if "content" in update:
                todo.content = update["content"]

            updated_count += 1

        return updated_count, self.format_todo_list()

    def format_todo_list(self) -> str:
        """Format TODO list with minimal style.

        Returns:
            Formatted string with progress count and todo items
        """
        if not self.todos:
            return "No todos created yet"

        # Calculate progress
        completed = sum(1 for todo in self.todos if todo.status == "completed")
        in_progress = sum(1 for todo in self.todos if todo.status == "in_progress")
        total = len(self.todos)

        # Header with just progress count
        lines = [f"{completed}/{total}"]

        # Add todo items with status symbols
        for todo in self.todos:
            if todo.status == "completed":
                marker = "✓"
                line_text = f"{marker} {todo.content}"
            elif todo.status == "in_progress":
                marker = "→"
                line_text = f"{marker} {todo.content}"
            else:
                marker = "○"
                line_text = f"{marker} {todo.content}"

            lines.append(line_text)

        return "\n".join(lines)

    def get_todo_summary(self) -> dict[str, int]:
        """Get summary of TODO statuses.

        Returns:
            Dictionary with counts for each status
        """
        summary = {"pending": 0, "completed": 0}
        for todo in self.todos:
            summary[todo.status] += 1
        return summary


class CreateTodoListTool(BaseTool):
    """Tool for creating a new TODO list."""

    def __init__(self, todo_manager: TodoManager):
        """Initialize with a shared TodoManager instance.

        Args:
            todo_manager: Shared TodoManager for state management
        """
        super().__init__()
        self.todo_manager = todo_manager

    async def execute(self, todos: list[dict[str, Any]]) -> ToolResult:
        """Create a new TODO list with items.

        Args:
            todos: List of todo dictionaries with 'content' and optional 'status'

        Returns:
            ToolResult with formatted todo list or error
        """
        try:
            formatted = self.todo_manager.create_todos(todos)
            return ToolResult.success_result(f"Todo list created:\n{formatted}")
        except Exception as e:
            return ToolResult.error_result(f"Failed to create todo list: {str(e)}")


class UpdateTodoListTool(BaseTool):
    """Tool for updating existing TODO items."""

    def __init__(self, todo_manager: TodoManager):
        """Initialize with a shared TodoManager instance.

        Args:
            todo_manager: Shared TodoManager for state management
        """
        super().__init__()
        self.todo_manager = todo_manager

    async def execute(self, updates: list[dict[str, Any]]) -> ToolResult:
        """Update existing TODO items by ID.

        Args:
            updates: List of update dictionaries with 'id' and fields to update

        Returns:
            ToolResult with formatted todo list or error
        """
        try:
            updated_count, formatted = self.todo_manager.update_todos(updates)
            return ToolResult.success_result(
                f"Updated {updated_count} TODO item(s):\n{formatted}"
            )
        except ValueError as e:
            return ToolResult.error_result(str(e))
        except Exception as e:
            return ToolResult.error_result(f"Failed to update todo list: {str(e)}")
