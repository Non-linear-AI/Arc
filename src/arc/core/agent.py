"""Arc AI Agent implementation."""

import json
import os
from datetime import datetime
from pathlib import Path

import jinja2
from openai.types.chat import ChatCompletionMessageParam

from arc.core.client import ArcClient, ArcToolCall
from arc.core.config import SettingsManager
from arc.tools import (
    BashTool,
    DatabaseQueryTool,
    DataProcessorGeneratorTool,
    FileEditorTool,
    MLModelGeneratorTool,
    MLPlanTool,
    MLPredictorGeneratorTool,
    MLPredictTool,
    MLTrainTool,
    SchemaDiscoveryTool,
    SearchTool,
    TodoTool,
    ToolResult,
)
from arc.utils import TokenCounter


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
        services=None,
        ui_interface=None,
    ):
        # Initialize settings and model
        self.settings_manager = SettingsManager()
        saved_model = self.settings_manager.get_current_model()
        model_to_use = model or saved_model or "gpt-4"

        self.max_tool_rounds = max_tool_rounds
        self.arc_client = ArcClient(api_key, model_to_use, base_url)
        self.api_key = api_key
        self.base_url = self.arc_client.base_url
        self.ui_interface = ui_interface

        # Track the model currently configured for the client so tool calls
        # stay consistent
        self.current_model_name = self.arc_client.get_current_model()

        # ML Plan management - stores current plan across workflow
        self.current_ml_plan: dict | None = None
        self.ml_plan_auto_accept: bool = False  # Session-scoped auto-accept flag
        # Track timestamp for filtering conversation history in revisions
        self.last_ml_plan_timestamp: datetime | None = None

        # Initialize tools
        self.file_editor = FileEditorTool()
        self.bash_tool = BashTool()
        self.search_tool = SearchTool()
        self.todo_tool = TodoTool()

        # Initialize TensorBoard manager
        try:
            from arc.ml import TensorBoardManager

            self.tensorboard_manager = TensorBoardManager()
            print(f"DEBUG: TensorBoardManager created: {self.tensorboard_manager}")
        except Exception as e:
            # If TensorBoard manager fails to initialize, log but continue
            print(f"Warning: TensorBoard manager initialization failed: {e}")
            self.tensorboard_manager = None
        self.database_query_tool = DatabaseQueryTool(services) if services else None
        self.schema_discovery_tool = SchemaDiscoveryTool(services) if services else None
        self.ml_predict_tool = MLPredictTool(services.ml_runtime) if services else None
        self.ml_plan_tool = (
            MLPlanTool(
                services,
                self.api_key,
                self.base_url,
                self.current_model_name,
                self.ui_interface,
                agent=self,  # Pass agent reference for auto_accept flag
            )
            if services
            else None
        )
        self.ml_model_generator_tool = (
            MLModelGeneratorTool(
                services,
                self.api_key,
                self.base_url,
                self.current_model_name,
                self.ui_interface,
            )
            if services
            else None
        )
        self.ml_train_tool = (
            MLTrainTool(
                services,
                services.ml_runtime,
                self.api_key,
                self.base_url,
                self.current_model_name,
                self.ui_interface,
                self.tensorboard_manager,
            )
            if services
            else None
        )
        self.ml_predictor_generator_tool = (
            MLPredictorGeneratorTool(
                services,
                self.api_key,
                self.base_url,
                self.current_model_name,
                self.ui_interface,
            )
            if services
            else None
        )
        self.data_processor_generator_tool = (
            DataProcessorGeneratorTool(
                services,
                self.api_key,
                self.base_url,
                self.current_model_name,
            )
            if services
            else None
        )
        # Initialize chat history
        self.chat_history: list[ChatEntry] = []
        self.messages: list[ChatCompletionMessageParam] = []
        self.token_counter = TokenCounter(model_to_use)

        # Load custom instructions and initialize system message
        self._initialize_system_message()

    def _initialize_system_message(self) -> None:
        """Initialize the system message with instructions."""

        # Generate system schema if services are available
        system_schema = None
        if hasattr(self, "database_query_tool") and self.database_query_tool:
            try:
                # Access schema service through the database query tool's services
                services = self.database_query_tool.services
                if services and hasattr(services, "schema"):
                    system_schema = services.schema.generate_system_schema_prompt()
            except Exception:
                # Continue without system schema if generation fails
                pass

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
            elif tool_call.name == "database_query":
                if self.database_query_tool:
                    return await self.database_query_tool.execute(
                        query=args["query"],
                        target_db=args.get("target_db", "system"),
                        validate_schema=args.get("validate_schema", True),
                    )
                else:
                    return ToolResult.error_result(
                        "Database query tool not available. "
                        "Database services not initialized."
                    )
            elif tool_call.name == "schema_discovery":
                if self.schema_discovery_tool:
                    return await self.schema_discovery_tool.execute(
                        action=args["action"],
                        target_db=args.get("target_db", "system"),
                        table_name=args.get("table_name"),
                    )
                else:
                    return ToolResult.error_result(
                        "Schema discovery tool not available. "
                        "Database services not initialized."
                    )
            elif tool_call.name == "ml_predict":
                if self.ml_predict_tool:
                    return await self.ml_predict_tool.execute(
                        model_name=args.get("model_name"),
                        table_name=args.get("table_name"),
                        output_table=args.get("output_table"),
                        batch_size=args.get("batch_size"),
                        limit=args.get("limit"),
                        device=args.get("device"),
                    )
                return ToolResult.error_result(
                    "ML predict tool not available. Database services not initialized."
                )
            elif tool_call.name == "ml_plan":
                if self.ml_plan_tool:
                    # Prepare conversation history for the planner
                    # For initial: all history. For revisions: messages since last plan
                    conversation_history = self._prepare_conversation_for_ml_plan(
                        from_timestamp=self.last_ml_plan_timestamp
                        if self.current_ml_plan
                        else None
                    )

                    # Execute with current plan for revisions
                    result = await self.ml_plan_tool.execute(
                        user_context=args.get("user_context"),
                        data_table=args.get("data_table"),
                        target_column=args.get("target_column"),
                        conversation_history=conversation_history,
                        feedback=args.get("feedback"),
                        previous_plan=self.current_ml_plan,
                    )

                    # Store the new plan and track timestamp if successful
                    if (
                        result.success
                        and result.metadata
                        and "ml_plan" in result.metadata
                    ):
                        self.current_ml_plan = result.metadata["ml_plan"]
                        # Track current timestamp for next revision
                        self.last_ml_plan_timestamp = datetime.now()

                    return result
                return ToolResult.error_result(
                    "ML plan tool not available. Database services not initialized."
                )
            elif tool_call.name == "ml_model_generator":
                if self.ml_model_generator_tool:
                    return await self.ml_model_generator_tool.execute(
                        name=args.get("name"),
                        context=args.get("context"),
                        data_table=args.get("data_table"),
                        target_column=args.get("target_column"),
                        category=args.get("category"),
                        ml_plan=self.current_ml_plan,  # Pass current plan
                    )
                return ToolResult.error_result(
                    "ML model generator tool not available. "
                    "Database services not initialized."
                )
            elif tool_call.name == "ml_train":
                if self.ml_train_tool:
                    return await self.ml_train_tool.execute(
                        name=args.get("name"),
                        context=args.get("context"),
                        model_id=args.get("model_id"),
                        train_table=args.get("train_table"),
                    )
                return ToolResult.error_result(
                    "ML train tool not available. Database services not initialized."
                )
            elif tool_call.name == "ml_predictor_generator":
                if self.ml_predictor_generator_tool:
                    return await self.ml_predictor_generator_tool.execute(
                        context=args.get("context"),
                        model_spec_path=args.get("model_spec_path"),
                        trainer_spec_path=args.get("trainer_spec_path"),
                        output_path=args.get("output_path"),
                    )
                return ToolResult.error_result(
                    "ML predictor generator tool not available. "
                    "Database services not initialized."
                )
            elif tool_call.name == "data_processor_generator":
                if self.data_processor_generator_tool:
                    return await self.data_processor_generator_tool.execute(
                        action=args.get("action", "generate"),
                        context=args.get("context"),
                        target_tables=args.get("target_tables"),
                        output_path=args.get("output_path"),
                        target_db=args.get("target_db", "user"),
                        yaml_content=args.get("yaml_content"),
                    )
                return ToolResult.error_result(
                    "Data processor generator tool not available. "
                    "Database services not initialized."
                )
            else:
                return ToolResult.error_result(f"Unknown tool: {tool_call.name}")

        except Exception as e:
            return ToolResult.error_result(f"Tool execution error: {str(e)}")

    async def _execute_tool_call(self, tool_call: ArcToolCall) -> ToolResult:
        """Execute a tool call (alias for _execute_tool)."""
        return await self._execute_tool(tool_call)

    def _prepare_conversation_for_ml_plan(
        self, from_timestamp: datetime | None = None
    ) -> list[dict]:
        """Prepare conversation history for ML plan tool.

        Converts chat history to a simplified format suitable for ML planning.
        For revisions, only includes messages after the last ML plan timestamp
        to avoid context pollution. For initial plans, includes all history.

        Args:
            from_timestamp: Timestamp to filter messages from (for revisions).
                           If None, includes all conversation history (initial plan).

        Returns:
            List of conversation messages
        """
        conversation = []

        for entry in self.chat_history:
            # For revisions, only include messages after the last plan timestamp
            if from_timestamp and entry.timestamp <= from_timestamp:
                continue

            if entry.type == "user":
                conversation.append({"role": "user", "content": entry.content})
            elif entry.type == "assistant" and entry.content:
                conversation.append({"role": "assistant", "content": entry.content})

        return conversation

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
        if getattr(self, "ml_model_generator_tool", None):
            self.ml_model_generator_tool.model = model
        if getattr(self, "ml_predictor_generator_tool", None):
            self.ml_predictor_generator_tool.model = model
