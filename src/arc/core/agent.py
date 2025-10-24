"""Arc AI Agent implementation."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import jinja2
from openai.types.chat import ChatCompletionMessageParam

from arc.core.client import ArcClient, ArcToolCall
from arc.core.config import SettingsManager
from arc.tools import (
    BashTool,
    CreateFileTool,
    CreateTodoListTool,
    DatabaseQueryTool,
    EditFileTool,
    MLDataProcessTool,
    MLEvaluateTool,
    MLModelTool,
    MLPlanTool,
    MLTrainTool,
    ReadKnowledgeTool,
    SchemaDiscoveryTool,
    SearchTool,
    TodoManager,
    ToolResult,
    UpdateTodoListTool,
    ViewFileTool,
)
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

        # ML Plan management - stores current plan across workflow
        self.current_ml_plan: dict | None = None
        self.ml_plan_auto_accept: bool = False  # Session-scoped auto-accept flag
        # Track timestamp for filtering conversation history in revisions
        self.last_ml_plan_timestamp: datetime | None = None

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
            self.api_key,
            self.base_url,
            model,
            self.ui_interface,
        )
        self.ml_train_tool = MLTrainTool(
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
            self.api_key,
            self.base_url,
            model,
            self.ui_interface,
            self.tensorboard_manager,
        )
        self.data_process_tool = MLDataProcessTool(
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
        self.tool_registry.register("ml_train", self.ml_train_tool)
        self.tool_registry.register("ml_evaluate", self.ml_evaluate_tool)
        self.tool_registry.register("data_process", self.data_process_tool)

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

    async def _execute_tool(self, tool_call: ArcToolCall) -> ToolResult:
        """Execute a tool call using the tool registry.

        Handles special preprocessing for tools that need agent context:
        - ml_plan: Injects previous_plan for revisions
        - ml_model, ml_train, ml_evaluate, data_process: Inject current_ml_plan
        """
        # Special handling for ml_plan: inject previous plan for revisions
        if tool_call.name == "ml_plan":
            try:
                args = json.loads(tool_call.arguments)
                # Inject previous plan if it exists (for revisions)
                args["previous_plan"] = self.current_ml_plan
                # Recreate tool call with modified arguments
                tool_call = ArcToolCall(
                    id=tool_call.id,
                    name=tool_call.name,
                    arguments=json.dumps(args),
                )
            except Exception as e:
                return ToolResult.error_result(
                    f"Error preparing ml_plan context: {str(e)}"
                )

        # Special handling for ml_model: inject current plan
        if tool_call.name == "ml_model":
            try:
                args = json.loads(tool_call.arguments)
                # Only inject ml_plan if it exists and is valid
                if self.current_ml_plan is not None:
                    # Validate ml_plan structure before injecting
                    if not isinstance(self.current_ml_plan, dict):
                        return ToolResult.error_result(
                            "Internal error: ml_plan is not a dictionary. "
                            "Cannot inject invalid plan into ml_model."
                        )
                    # Basic validation: check for required fields
                    if "model_architecture_and_loss" not in self.current_ml_plan:
                        self.logger.warning(
                            "ml_plan missing 'model_architecture_and_loss' section"
                        )
                args["ml_plan"] = self.current_ml_plan
                tool_call = ArcToolCall(
                    id=tool_call.id,
                    name=tool_call.name,
                    arguments=json.dumps(args),
                )
            except Exception as e:
                return ToolResult.error_result(
                    f"Error preparing ml_model context: {str(e)}"
                )

        # Special handling for ml_train: inject current plan
        if tool_call.name == "ml_train":
            try:
                args = json.loads(tool_call.arguments)
                # Only inject ml_plan if it exists and is valid
                if self.current_ml_plan is not None:
                    # Validate ml_plan structure before injecting
                    if not isinstance(self.current_ml_plan, dict):
                        return ToolResult.error_result(
                            "Internal error: ml_plan is not a dictionary. "
                            "Cannot inject invalid plan into ml_train."
                        )
                    # Basic validation: check for required fields
                    if "training_configuration" not in self.current_ml_plan:
                        self.logger.warning(
                            "ml_plan missing 'training_configuration' section"
                        )
                args["ml_plan"] = self.current_ml_plan
                tool_call = ArcToolCall(
                    id=tool_call.id,
                    name=tool_call.name,
                    arguments=json.dumps(args),
                )
            except Exception as e:
                return ToolResult.error_result(
                    f"Error preparing ml_train context: {str(e)}"
                )

        # Special handling for ml_evaluate: inject current plan
        if tool_call.name == "ml_evaluate":
            try:
                args = json.loads(tool_call.arguments)
                # Only inject ml_plan if it exists and is valid
                if self.current_ml_plan is not None:
                    # Validate ml_plan structure before injecting
                    if not isinstance(self.current_ml_plan, dict):
                        return ToolResult.error_result(
                            "Internal error: ml_plan is not a dictionary. "
                            "Cannot inject invalid plan into ml_evaluate."
                        )
                    # Basic validation: check for required fields
                    if "evaluation" not in self.current_ml_plan:
                        self.logger.warning("ml_plan missing 'evaluation' section")
                args["ml_plan"] = self.current_ml_plan
                tool_call = ArcToolCall(
                    id=tool_call.id,
                    name=tool_call.name,
                    arguments=json.dumps(args),
                )
            except Exception as e:
                return ToolResult.error_result(
                    f"Error preparing ml_evaluate context: {str(e)}"
                )

        # Special handling for data_process: inject current plan
        if tool_call.name == "data_process":
            try:
                args = json.loads(tool_call.arguments)
                # Only inject ml_plan if it exists and is valid
                if self.current_ml_plan is not None:
                    # Validate ml_plan structure before injecting
                    if not isinstance(self.current_ml_plan, dict):
                        return ToolResult.error_result(
                            "Internal error: ml_plan is not a dictionary. "
                            "Cannot inject invalid plan into data_process."
                        )
                    # Basic validation: check for required fields
                    if "feature_engineering" not in self.current_ml_plan:
                        self.logger.warning(
                            "ml_plan missing 'feature_engineering' section"
                        )
                args["ml_plan"] = self.current_ml_plan
                tool_call = ArcToolCall(
                    id=tool_call.id,
                    name=tool_call.name,
                    arguments=json.dumps(args),
                )
            except Exception as e:
                return ToolResult.error_result(
                    f"Error preparing data_process context: {str(e)}"
                )

        # Use tool registry for execution
        result = await self.tool_registry.execute(tool_call)

        # Post-processing for ml_plan: store plan state
        if (
            tool_call.name == "ml_plan"
            and result.success
            and result.metadata
            and "ml_plan" in result.metadata
        ):
            # Validate ml_plan before storing
            plan_data = result.metadata["ml_plan"]
            if not isinstance(plan_data, dict):
                self.logger.error(
                    f"ml_plan tool returned invalid plan: not a dictionary "
                    f"(type: {type(plan_data).__name__})"
                )
                # Don't store invalid plan
                return ToolResult.error_result(
                    "Internal error: ml_plan tool returned invalid plan data. "
                    "Expected dictionary, got {type(plan_data).__name__}."
                )

            # Validate required fields
            required_fields = [
                "summary",
                "feature_engineering",
                "model_architecture_and_loss",
                "training_configuration",
                "evaluation",
            ]
            missing_fields = [f for f in required_fields if f not in plan_data]
            if missing_fields:
                self.logger.warning(
                    f"ml_plan missing required fields: {', '.join(missing_fields)}"
                )
                # Store anyway but log warning (plan might be partial/revision)

            self.current_ml_plan = plan_data
            self.last_ml_plan_timestamp = datetime.now()

        # Tools now return factual data in metadata
        # (plan_comparison, plan_training_config, plan_evaluation)
        # The agent's reasoning will analyze this data and decide if plan
        # revision is warranted

        return result

    async def _execute_tool_call(self, tool_call: ArcToolCall) -> ToolResult:
        """Execute a tool call (alias for _execute_tool)."""
        return await self._execute_tool(tool_call)

    def _prepare_conversation_for_ml_plan(
        self, from_timestamp: datetime | None = None, max_turns: int = 10
    ) -> str:
        """Prepare conversation history for ML plan tool.

        Converts chat history to a simplified format suitable for ML planning.
        Returns recent conversation turns to provide context without overwhelming
        the LLM with too much history.

        Args:
            from_timestamp: Timestamp to filter messages from (for revisions).
                           If None, includes all conversation history.
            max_turns: Maximum number of conversation turns (user+assistant pairs)
                      to include. Default is 10 turns (20 messages).

        Returns:
            Formatted conversation history string
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

        # Limit to recent turns (user + assistant = 1 turn, so max_turns * 2 messages)
        if max_turns and len(conversation) > max_turns * 2:
            conversation = conversation[-(max_turns * 2) :]

        # Format as readable conversation
        if not conversation:
            return ""

        formatted = []
        for msg in conversation:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted.append(f"{role}: {content}")

        return "\n\n".join(formatted)

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
        if getattr(self, "ml_train_tool", None):
            self.ml_train_tool.model = model
        if getattr(self, "ml_evaluate_tool", None):
            self.ml_evaluate_tool.model = model

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
