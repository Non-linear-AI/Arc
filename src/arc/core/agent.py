"""Arc AI Agent implementation."""

import json
import os
from datetime import datetime
from textwrap import dedent

from openai.types.chat import ChatCompletionMessageParam

from ..tools import BashTool, FileEditorTool, SearchTool, TodoTool, ToolResult
from ..tools.tools import get_base_tools
from ..utils import TokenCounter
from .client import ArcClient, ArcToolCall
from .config import SettingsManager


class ChatEntry:
    """Represents a single entry in the chat history."""

    def __init__(
        self,
        type: str,
        content: str,
        timestamp: datetime | None = None,
        tool_calls: list[ArcToolCall] | None = None,
        tool_call: ArcToolCall | None = None,
        tool_result: ToolResult | None = None,
        is_streaming: bool = False,
    ):
        self.type = type  # "user", "assistant", "tool_result", "tool_call"
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.tool_calls = tool_calls or []
        self.tool_call = tool_call
        self.tool_result = tool_result
        self.is_streaming = is_streaming


class StreamingChunk:
    """Represents a chunk of streaming response."""

    def __init__(
        self,
        type: str,
        content: str | None = None,
        tool_calls: list[ArcToolCall] | None = None,
        tool_call: ArcToolCall | None = None,
        tool_result: ToolResult | None = None,
        token_count: int | None = None,
    ):
        self.type = (
            type  # "content", "tool_calls", "tool_result", "done", "token_count"
        )
        self.content = content
        self.tool_calls = tool_calls or []
        self.tool_call = tool_call
        self.tool_result = tool_result
        self.token_count = token_count


class ArcAgent:
    """Main AI agent for Arc CLI."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        max_tool_rounds: int = 400,
    ):
        # Initialize settings and model
        self.settings_manager = SettingsManager()
        saved_model = self.settings_manager.get_current_model()
        model_to_use = model or saved_model or "gpt-4"

        self.max_tool_rounds = max_tool_rounds
        self.arc_client = ArcClient(api_key, model_to_use, base_url)

        # Initialize tools
        self.file_editor = FileEditorTool()
        self.bash_tool = BashTool()
        self.search_tool = SearchTool()
        self.todo_tool = TodoTool()

        # Initialize chat history
        self.chat_history: list[ChatEntry] = []
        self.messages: list[ChatCompletionMessageParam] = []
        self.token_counter = TokenCounter(model_to_use)

        # Load custom instructions and initialize system message
        self._initialize_system_message()

    def _initialize_system_message(self) -> None:
        """Initialize the system message with instructions."""
        custom_instructions = self._load_custom_instructions()
        custom_instructions_section = (
            f"\n\nCUSTOM INSTRUCTIONS:\n{custom_instructions}\n\n"
            "The above custom instructions should be followed alongside "
            "the standard instructions below."
            if custom_instructions
            else ""
        )

        system_content = dedent(
            f"""
            You are Arc CLI, an AI assistant that EXCLUSIVELY helps with file editing, coding tasks, and system operations. You do NOT provide information about general life advice, or non-technical topics.{custom_instructions_section}

            SCOPE RESTRICTIONS:
            - ONLY answer questions related to programming, file editing, system operations, databases, data analysis, and software development
            - For non-technical questions (cooking, general knowledge, etc.), politely redirect to technical topics
            - Always stay within your role as a coding and system operations assistant

            You have access to these tools:
            - view_file: View file contents or directory listings
            - create_file: Create new files with content (ONLY use this for files that don't exist yet)
            - str_replace_editor: Replace text in existing files (ALWAYS use this to edit or update existing files)
            - bash: Execute bash commands (use for searching, file discovery, navigation, and system operations)
            - search: Unified search tool for finding text content or files (similar to Cursor's search functionality)
            - create_todo_list: Create a visual todo list for planning and tracking tasks
            - update_todo_list: Update existing todos in your todo list
            - show_todo_list: Display current todos with summary and focus
            - start_todo: Start a specific todo (or first pending)
            - complete_todo: Complete a specific todo (or current)
            - advance_todo: Complete current and start the next pending

            IMPORTANT TOOL USAGE RULES:
            - NEVER use create_file on files that already exist - this will overwrite them completely
            - ALWAYS use str_replace_editor to modify existing files, even for small changes
            - Before editing a file, use view_file to see its current contents
            - Use create_file ONLY when creating entirely new files that don't exist

            SEARCHING AND EXPLORATION:
            - Use search for fast, powerful text search across files or finding files by name (unified search tool)
            - Examples: search for text content like "import.*react", search for files like "component.tsx"
            - Use bash with commands like 'find', 'grep', 'rg', 'ls' for complex file operations and navigation
            - view_file is best for reading specific files you already know exist

            When a user asks you to edit, update, modify, or change an existing file:
            1. First use view_file to see the current contents
            2. Then use str_replace_editor to make the specific changes
            3. Never use create_file for existing files

            When a user asks you to create a new file that doesn't exist:
            1. Use create_file with the full content

            TASK PLANNING WITH TODO LISTS:
            - For complex requests with multiple steps, ALWAYS create a todo list first to plan your approach
            - Use create_todo_list to break down tasks into manageable items with priorities
            - Use show_todo_list to keep the plan visible
            - Mark tasks as 'in_progress' when you start working (use start_todo) — only one at a time
            - Mark tasks as 'completed' immediately when finished (use complete_todo)
            - Prefer advance_todo to complete the current and automatically start the next
            - Use update_todo_list for bulk edits (renames, priorities)
            - Todo lists provide visual feedback with colors: ✅ Green (completed), 🔄 Cyan (in progress), ⏳ Yellow (pending)
            - Always create todos with priorities: 'high' (🔴), 'medium' (🟡), 'low' (🟢)

            Be helpful, direct, and efficient. Always explain what you're doing and show the results.

            IMPORTANT RESPONSE GUIDELINES:
            - After using tools, do NOT respond with pleasantries like "Thanks for..." or "Great!"
            - Only provide necessary explanations or next steps if relevant to the task
            - Keep responses concise and focused on the actual work being done
            - If a tool execution completes the user's request, you can remain silent or give a brief confirmation

            Current working directory: {os.getcwd()}
            """
        )

        self.messages.append({"role": "system", "content": system_content})

    def _load_custom_instructions(self) -> str | None:
        """Load custom instructions from settings."""
        # This would typically load from a config file
        # For now, return None as we don't have the settings infrastructure
        return None

    async def process_user_message_stream(self, message: str):
        """Process a user message with streaming response."""
        # Add user message
        user_entry = ChatEntry(type="user", content=message)
        self.chat_history.append(user_entry)
        self.messages.append({"role": "user", "content": message})

        yield StreamingChunk(type="user_message", content=message)

        try:
            tools = get_base_tools()
            tool_rounds = 0

            while tool_rounds < self.max_tool_rounds:
                # Stream one assistant turn
                current_content = ""
                current_tool_calls: list = []
                streaming_tool_calls: dict[int, dict] = {}

                async for chunk in self.arc_client.chat_stream(self.messages, tools):
                    if chunk.choices and len(chunk.choices) > 0:
                        choice = chunk.choices[0]

                        # Stream content
                        if choice.delta and choice.delta.content:
                            current_content += choice.delta.content
                            yield StreamingChunk(type="content", content=choice.delta.content)

                        # Accumulate tool call deltas
                        if choice.delta and choice.delta.tool_calls:
                            for tc_delta in choice.delta.tool_calls:
                                idx = getattr(tc_delta, "index", 0)
                                if idx not in streaming_tool_calls:
                                    streaming_tool_calls[idx] = {
                                        "id": tc_delta.id or "",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                if tc_delta.id:
                                    streaming_tool_calls[idx]["id"] = tc_delta.id
                                if tc_delta.function:
                                    if tc_delta.function.name:
                                        streaming_tool_calls[idx]["function"]["name"] += tc_delta.function.name
                                    if tc_delta.function.arguments:
                                        streaming_tool_calls[idx]["function"]["arguments"] += tc_delta.function.arguments

                # Convert accumulated tool calls
                for _, tc_data in streaming_tool_calls.items():
                    if tc_data["function"]["name"]:
                        class MockFunction:
                            def __init__(self, name, arguments):
                                self.name = name
                                self.arguments = arguments

                        class MockToolCall:
                            def __init__(self, id, func):
                                self.id = id
                                self.function = func

                        mock_tc = MockToolCall(
                            tc_data["id"],
                            MockFunction(
                                tc_data["function"]["name"],
                                tc_data["function"]["arguments"],
                            ),
                        )
                        current_tool_calls.append(mock_tc)
                        yield StreamingChunk(
                            type="tool_calls",
                            tool_call=ArcToolCall.from_openai_tool_call(mock_tc),
                        )

                # If no tool calls, finalize and exit
                if not current_tool_calls:
                    final_entry = ChatEntry(type="assistant", content=current_content)
                    self.chat_history.append(final_entry)
                    self.messages.append({"role": "assistant", "content": current_content})
                    yield StreamingChunk(type="done")
                    break

                # Otherwise, record assistant message with tool calls
                assistant_entry = ChatEntry(
                    type="assistant",
                    content=current_content or "Using tools to help you...",
                    tool_calls=[ArcToolCall.from_openai_tool_call(tc) for tc in current_tool_calls],
                )
                self.chat_history.append(assistant_entry)
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": current_content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in current_tool_calls
                        ],
                    }
                )

                # Execute tools and append results
                for tool_call in current_tool_calls:
                    arc_tool_call = ArcToolCall.from_openai_tool_call(tool_call)
                    result = await self._execute_tool_call(arc_tool_call)
                    yield StreamingChunk(type="tool_result", tool_call=arc_tool_call, tool_result=result)
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result.output or result.error or "Tool completed",
                        }
                    )

                tool_rounds += 1

        except Exception as e:
            error_entry = ChatEntry(
                type="assistant", content=f"Sorry, I encountered an error: {str(e)}"
            )
            self.chat_history.append(error_entry)
            yield StreamingChunk(type="error", content=str(e))

    async def process_user_message(self, message: str) -> list[ChatEntry]:
        """Process a user message and return new chat entries."""
        # Add user message
        user_entry = ChatEntry(type="user", content=message)
        self.chat_history.append(user_entry)
        self.messages.append({"role": "user", "content": message})

        new_entries = [user_entry]
        tool_rounds = 0

        try:
            tools = get_base_tools()
            current_response = await self.arc_client.chat(
                self.messages,
                tools,
            )

            # Agent loop - continue until no more tool calls or max rounds reached
            while tool_rounds < self.max_tool_rounds:
                if not current_response:
                    break

                # Handle tool calls
                if current_response.tool_calls:
                    tool_rounds += 1

                    # Add assistant message with tool calls
                    assistant_entry = ChatEntry(
                        type="assistant",
                        content=current_response.content
                        or "Using tools to help you...",
                        tool_calls=[
                            ArcToolCall.from_openai_tool_call(tc)
                            for tc in current_response.tool_calls
                        ],
                    )
                    self.chat_history.append(assistant_entry)
                    new_entries.append(assistant_entry)

                    # Add assistant message to conversation
                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": current_response.content or "",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in current_response.tool_calls
                            ],
                        }
                    )

                    # Execute tool calls
                    for openai_tool_call in current_response.tool_calls:
                        tool_call = ArcToolCall.from_openai_tool_call(openai_tool_call)

                        # Create tool call entry
                        tool_call_entry = ChatEntry(
                            type="tool_call",
                            content="Executing...",
                            tool_call=tool_call,
                        )
                        self.chat_history.append(tool_call_entry)
                        new_entries.append(tool_call_entry)

                        # Execute the tool
                        result = await self._execute_tool(tool_call)

                        # Update the tool call entry to tool result
                        tool_call_entry.type = "tool_result"
                        tool_call_entry.content = (
                            result.output
                            if result.success
                            else result.error or "Error occurred"
                        )
                        tool_call_entry.tool_result = result

                        # Add tool result to messages
                        self.messages.append(
                            {
                                "role": "tool",
                                "content": result.output
                                if result.success
                                else result.error or "Error",
                                "tool_call_id": tool_call.id,
                            }
                        )

                    # Get next response
                    current_response = await self.arc_client.chat(
                        self.messages,
                        tools,
                    )
                else:
                    # No more tool calls, add final response
                    final_entry = ChatEntry(
                        type="assistant",
                        content=current_response.content
                        or "I understand, but I don't have a specific response.",
                    )
                    self.chat_history.append(final_entry)
                    new_entries.append(final_entry)
                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": current_response.content or "",
                        }
                    )
                    break

            if tool_rounds >= self.max_tool_rounds:
                warning_entry = ChatEntry(
                    type="assistant",
                    content="Maximum tool execution rounds reached. Stopping to prevent infinite loops.",
                )
                self.chat_history.append(warning_entry)
                new_entries.append(warning_entry)

            return new_entries

        except Exception as e:
            error_entry = ChatEntry(
                type="assistant",
                content=f"Sorry, I encountered an error: {str(e)}",
            )
            self.chat_history.append(error_entry)
            return [user_entry, error_entry]

    async def _execute_tool(self, tool_call: ArcToolCall) -> ToolResult:
        """Execute a tool call."""
        try:
            args = json.loads(tool_call.arguments)

            if tool_call.name == "view_file":
                return await self.file_editor.execute(
                    action="view",
                    path=args["path"],
                    start_line=args.get("start_line"),
                    end_line=args.get("end_line"),
                )
            elif tool_call.name == "create_file":
                return await self.file_editor.execute(
                    action="create",
                    path=args["path"],
                    content=args["content"],
                )
            elif tool_call.name == "str_replace_editor":
                return await self.file_editor.execute(
                    action="str_replace",
                    path=args["path"],
                    old_str=args["old_str"],
                    new_str=args["new_str"],
                    replace_all=args.get("replace_all", False),
                )
            elif tool_call.name == "bash":
                return await self.bash_tool.execute(command=args["command"])
            elif tool_call.name == "search":
                return await self.search_tool.execute(
                    query=args["query"],
                    search_type=args.get("search_type", "both"),
                    include_pattern=args.get("include_pattern"),
                    exclude_pattern=args.get("exclude_pattern"),
                    case_sensitive=args.get("case_sensitive", False),
                    whole_word=args.get("whole_word", False),
                    regex=args.get("regex", False),
                    max_results=args.get("max_results", 50),
                    file_types=args.get("file_types"),
                    include_hidden=args.get("include_hidden", False),
                )
            elif tool_call.name == "create_todo_list":
                return await self.todo_tool.execute(
                    action="create", todos=args["todos"]
                )
            elif tool_call.name == "update_todo_list":
                return await self.todo_tool.execute(
                    action="update", updates=args["updates"]
                )
            else:
                return ToolResult.error_result(f"Unknown tool: {tool_call.name}")

        except Exception as e:
            return ToolResult.error_result(f"Tool execution error: {str(e)}")

    async def _execute_tool_call(self, tool_call: ArcToolCall) -> ToolResult:
        """Execute a tool call (alias for _execute_tool)."""
        return await self._execute_tool(tool_call)

    def get_chat_history(self) -> list[ChatEntry]:
        """Get the complete chat history."""
        return self.chat_history.copy()

    def get_current_directory(self) -> str:
        """Get the current working directory."""
        return self.bash_tool.get_current_directory()

    async def execute_bash_command(self, command: str) -> ToolResult:
        """Execute a bash command directly."""
        return await self.bash_tool.execute(command=command)

    def get_current_model(self) -> str:
        """Get the current model."""
        return self.arc_client.get_current_model()

    def set_model(self, model: str) -> None:
        """Set the current model."""
        self.arc_client.set_model(model)
        self.token_counter = TokenCounter(model)
