"""Arc AI Agent implementation."""

import logging
import os
from datetime import datetime
from pathlib import Path

import jinja2
from openai.types.chat import ChatCompletionMessageParam

from arc.core.client import ArcClient, ArcToolCall
from arc.core.config import SettingsManager
from arc.tools.base import ToolResult
from arc.tools.bash import BashTool
from arc.tools.database_query import DatabaseQueryTool
from arc.tools.file_editor import CreateFileTool, EditFileTool, ViewFileTool
from arc.tools.knowledge import ReadKnowledgeTool
from arc.tools.ml import MLEvaluateTool, MLModelTool, MLPlanTool
from arc.tools.ml_data import MLDataTool
from arc.tools.schema_discovery import SchemaDiscoveryTool
from arc.tools.search import SearchTool
from arc.tools.todo import CreateTodoListTool, TodoManager, UpdateTodoListTool
from arc.utils import TokenCounter


class StreamingToolCallFunction:
    """Adapter for streaming tool call function data."""

    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class StreamingToolCall:
    """Adapter for streaming tool call data."""

    def __init__(self, id: str, function: StreamingToolCallFunction):
        self.id = id
        self.function = function


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
        services,
        base_url: str,
        model: str,
        max_tool_rounds: int = 50,
        ui_interface=None,
    ):
        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize settings manager (kept for compatibility)
        self.settings_manager = SettingsManager()

        self.max_tool_rounds = max_tool_rounds
        self.arc_client = ArcClient(api_key, model, base_url)
        self.api_key = api_key
        self.base_url = base_url
        self.ui_interface = ui_interface

        # Track the model currently configured for the client so tool calls
        # stay consistent
        self.current_model_name = model
        self.logger.info(f"ArcAgent initialized with model: {self.current_model_name}")

        # ML Plan management
        self.ml_plan_auto_accept: bool = False  # Session-scoped auto-accept flag

        # Initialize tool registry
        from arc.tools.registry import ToolRegistry

        self.tool_registry = ToolRegistry(default_timeout=300)

        # Initialize and register tools
        self.view_file_tool = ViewFileTool()
        self.create_file_tool = CreateFileTool()
        self.edit_file_tool = EditFileTool()
        self.bash_tool = BashTool()
        self.search_tool = SearchTool()

        # Initialize todo manager and tools with shared state
        self.todo_manager = TodoManager()
        self.create_todo_tool = CreateTodoListTool(self.todo_manager)
        self.update_todo_tool = UpdateTodoListTool(self.todo_manager)

        # Initialize knowledge tool
        self.read_knowledge_tool = ReadKnowledgeTool()

        # Register basic tools
        self.tool_registry.register("view_file", self.view_file_tool)
        self.tool_registry.register("create_file", self.create_file_tool)
        self.tool_registry.register("edit_file", self.edit_file_tool)
        self.tool_registry.register("bash", self.bash_tool)
        self.tool_registry.register("search", self.search_tool)
        self.tool_registry.register("create_todo_list", self.create_todo_tool)
        self.tool_registry.register("update_todo_list", self.update_todo_tool)
        self.tool_registry.register("read_knowledge", self.read_knowledge_tool)

        # Initialize TensorBoard manager
        try:
            from arc.ml import TensorBoardManager

            self.tensorboard_manager = TensorBoardManager()
            self.logger.debug("TensorBoard manager initialized successfully")
        except ImportError:
            self.logger.debug("TensorBoard not available (import failed)")
            self.tensorboard_manager = None
        except Exception as e:
            self.logger.warning(
                f"TensorBoard manager initialization failed: {e}", exc_info=True
            )
            self.tensorboard_manager = None

        # Initialize and register database/ML tools
        self.database_query_tool = DatabaseQueryTool(services)
        self.schema_discovery_tool = SchemaDiscoveryTool(services)
        self.ml_plan_tool = MLPlanTool(
            services,
            self.api_key,
            self.base_url,
            model,
            self.ui_interface,
            agent=self,  # Pass agent reference for auto_accept flag
        )
        self.ml_model_tool = MLModelTool(
            services,
            services.ml_runtime,
            self.api_key,
            self.base_url,
            model,
            self.ui_interface,
            self.tensorboard_manager,
        )
        self.ml_evaluate_tool = MLEvaluateTool(
            services,
            services.ml_runtime,
            self.ui_interface,
            self.tensorboard_manager,
        )
        self.ml_data_tool = MLDataTool(
            services,
            self.api_key,
            self.base_url,
            model,
            self.ui_interface,
        )

        # Register database and ML tools
        self.tool_registry.register("database_query", self.database_query_tool)
        self.tool_registry.register("schema_discovery", self.schema_discovery_tool)
        self.tool_registry.register("ml_plan", self.ml_plan_tool)
        self.tool_registry.register("ml_model", self.ml_model_tool)
        self.tool_registry.register("ml_evaluate", self.ml_evaluate_tool)
        self.tool_registry.register("ml_data", self.ml_data_tool)

        # Validate tool registry matches tools.yaml
        self._validate_tool_registry()
        # Initialize chat history
        self.chat_history: list[ChatEntry] = []
        self.messages: list[ChatCompletionMessageParam] = []
        self.token_counter = TokenCounter(model)

        # Load custom instructions and initialize system message
        self._initialize_system_message()

    def _validate_tool_registry(self) -> None:
        """Validate that tool registry matches tools.yaml definitions.

        Logs warnings if there are mismatches but doesn't fail initialization.
        """
        try:
            from arc.tools.tools import get_tool_names

            yaml_tool_names = get_tool_names()
            validation = self.tool_registry.validate_against_yaml_tools(yaml_tool_names)

            if not validation["valid"]:
                error_msg = validation["error"]
                self.logger.warning(f"Tool registry validation failed: {error_msg}")
                if validation["missing_in_registry"]:
                    self.logger.warning(
                        f"  Tools defined in YAML but not registered: "
                        f"{', '.join(validation['missing_in_registry'])}"
                    )
                if validation["extra_in_registry"]:
                    self.logger.warning(
                        f"  Tools registered but not in YAML: "
                        f"{', '.join(validation['extra_in_registry'])}"
                    )
            else:
                self.logger.info(
                    f"Tool registry validated: {len(yaml_tool_names)} tools registered"
                )
        except Exception as e:
            self.logger.warning(f"Could not validate tool registry: {e}")

    def _initialize_system_message(self) -> None:
        """Initialize the system message with instructions."""

        # Generate system schema
        system_schema = None
        try:
            services = self.database_query_tool.services
            if hasattr(services, "schema"):
                system_schema = services.schema.generate_system_schema_prompt()
        except Exception as e:
            self.logger.warning(f"Failed to generate system schema: {e}")
            # Continue without system schema if generation fails

        # Load system prompt from Jinja2 template
        template_path = Path(__file__).parent.parent / "templates" / "system_prompt.j2"

        try:
            # Create Jinja2 environment
            template_dir = template_path.parent
            env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(template_dir),
                trim_blocks=True,
                lstrip_blocks=True,
            )

            # Load and render template
            template = env.get_template("system_prompt.j2")
            system_content = template.render(
                current_directory=os.getcwd(),
                system_schema=system_schema,
            )
        except Exception as e:
            # Raise exception with error details instead of using fallback
            raise RuntimeError(
                f"Failed to load system prompt template: {str(e)}"
            ) from e

        self.messages.append({"role": "system", "content": system_content})

    async def process_user_message_stream(self, message: str):
        """Process a user message with streaming response."""
        # Add user message
        user_entry = ChatEntry(type="user", content=message)
        self.chat_history.append(user_entry)
        self.messages.append({"role": "user", "content": message})

        yield StreamingChunk(type="user_message", content=message)

        try:
            from arc.tools.tools import get_base_tools

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
                            yield StreamingChunk(
                                type="content", content=choice.delta.content
                            )

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
                                        streaming_tool_calls[idx]["function"][
                                            "name"
                                        ] += tc_delta.function.name
                                    if tc_delta.function.arguments:
                                        streaming_tool_calls[idx]["function"][
                                            "arguments"
                                        ] += tc_delta.function.arguments

                # Convert accumulated tool calls
                for _, tc_data in streaming_tool_calls.items():
                    if tc_data["function"]["name"]:
                        streaming_tc = StreamingToolCall(
                            id=tc_data["id"],
                            function=StreamingToolCallFunction(
                                name=tc_data["function"]["name"],
                                arguments=tc_data["function"]["arguments"],
                            ),
                        )
                        current_tool_calls.append(streaming_tc)
                        yield StreamingChunk(
                            type="tool_calls",
                            tool_call=ArcToolCall.from_openai_tool_call(streaming_tc),
                        )

                # If no tool calls, finalize and exit
                if not current_tool_calls:
                    final_entry = ChatEntry(type="assistant", content=current_content)
                    self.chat_history.append(final_entry)
                    self.messages.append(
                        {"role": "assistant", "content": current_content}
                    )
                    yield StreamingChunk(type="done")
                    break

                # Otherwise, record assistant message with tool calls
                assistant_entry = ChatEntry(
                    type="assistant",
                    content=current_content or "Using tools to help you...",
                    tool_calls=[
                        ArcToolCall.from_openai_tool_call(tc)
                        for tc in current_tool_calls
                    ],
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
                    yield StreamingChunk(
                        type="tool_result", tool_call=arc_tool_call, tool_result=result
                    )
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result.output
                            or result.error
                            or "Tool completed",
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
            from arc.tools.tools import get_base_tools

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
                    content=(
                        "Maximum tool execution rounds reached. "
                        "Stopping to prevent infinite loops."
                    ),
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

    def _has_recent_data_exploration(self, lookback_messages: int = 10) -> bool:
        """Check if recent chat history contains successful data exploration tool calls.

        Args:
            lookback_messages: Number of recent messages to check

        Returns:
            True if recent successful database_query or schema_discovery calls exist
        """
        # Look at recent chat entries
        recent_entries = (
            self.chat_history[-lookback_messages:] if self.chat_history else []
        )

        for entry in recent_entries:
            # Only consider entries with tool results (completed tool calls)
            # and check if the tool call succeeded
            if (
                entry.tool_result
                and entry.tool_call
                and entry.tool_result.success
                and entry.tool_call.name in ("database_query", "schema_discovery")
            ):
                return True

        return False

    async def _execute_tool(self, tool_call: ArcToolCall) -> ToolResult:
        """Execute a tool call using the tool registry."""
        # Use tool registry for execution
        # ML tools should not have timeouts since they wait for user input
        ml_tools = {"ml_plan", "ml_model", "ml_evaluate", "ml_data"}
        if tool_call.name in ml_tools:
            # No timeout for ML tools (they wait for user interaction)
            return await self.tool_registry.execute(tool_call, timeout=None)
        else:
            # Use default timeout for other tools
            return await self.tool_registry.execute(tool_call)

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
        self.current_model_name = model
        if getattr(self, "ml_plan_tool", None):
            self.ml_plan_tool.model = model
        if getattr(self, "ml_model_tool", None):
            self.ml_model_tool.model = model

    def cleanup(self) -> None:
        """Clean up resources including TensorBoard processes."""
        if self.tensorboard_manager:
            try:
                count = self.tensorboard_manager.stop_all()
                if count > 0:
                    self.logger.info(f"Stopped {count} TensorBoard process(es)")
            except Exception as e:
                self.logger.warning(
                    f"Failed to stop TensorBoard processes: {e}", exc_info=True
                )
