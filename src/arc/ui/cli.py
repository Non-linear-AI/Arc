"""Command-line interface for Arc CLI."""

import asyncio
import json
import os
import sys
import time

import click
from dotenv import load_dotenv

from ..core import ArcAgent, SettingsManager
from ..database import DatabaseError, DatabaseManager, QueryValidationError
from ..database.services import ServiceContainer
from ..error_handling import error_handler
from ..utils import ConfirmationService, performance_manager
from .console import InteractiveInterface

# Load environment variables
load_dotenv()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Arc CLI - A conversational AI agent for file editing and system operations."""
    pass


@cli.command()
@click.option("-d", "--directory", default=None, help="Set working directory")
@click.option(
    "-k", "--api-key", default=None, help="Arc API key (or set ARC_API_KEY env var)"
)
@click.option(
    "-u",
    "--base-url",
    default=None,
    help="Arc API base URL (or set ARC_BASE_URL env var)",
)
@click.option(
    "-m", "--model", default=None, help="AI model to use (e.g., gpt-4, claude-3-sonnet)"
)
@click.option(
    "-p",
    "--prompt",
    default=None,
    help="Process a single prompt and exit (headless mode)",
)
@click.option(
    "--max-tool-rounds", default=400, help="Maximum number of tool execution rounds"
)
def chat(
    directory: str | None,
    api_key: str | None,
    base_url: str | None,
    model: str | None,
    prompt: str | None,
    max_tool_rounds: int,
):
    """Start an interactive chat session with Arc CLI."""
    ui = InteractiveInterface()

    # Change directory if specified
    if directory:
        try:
            os.chdir(directory)
        except OSError as e:
            ui.show_system_error(f"Error changing directory to {directory}: {e}")
            sys.exit(1)

    # Get configuration
    settings_manager = SettingsManager()

    api_key = api_key or settings_manager.get_api_key()
    if not api_key:
        ui.show_system_error(
            "API key required. Set ARC_API_KEY environment variable, "
            "use --api-key flag, or save to ~/.arc/user-settings.json"
        )
        sys.exit(1)

    base_url = base_url or settings_manager.get_base_url()
    model = model or settings_manager.get_current_model()

    # Save command line settings if provided
    if api_key and click.get_current_context().params.get("api_key"):
        settings_manager.update_user_setting("apiKey", api_key)
        ui.show_system_success("API key saved to ~/.arc/user-settings.json")

    if base_url and click.get_current_context().params.get("base_url"):
        settings_manager.update_user_setting("baseURL", base_url)
        ui.show_system_success("Base URL saved to ~/.arc/user-settings.json")

    # Initialize database services (shared by both modes)
    system_db_path = settings_manager.get_system_database_path()
    user_db_path = settings_manager.get_user_database_path()
    db_manager = DatabaseManager(system_db_path, user_db_path)
    services = ServiceContainer(db_manager)

    # Run the appropriate mode
    if prompt:
        asyncio.run(
            run_headless_mode(
                prompt, api_key, base_url, model, max_tool_rounds, services
            )
        )
    else:
        asyncio.run(
            run_interactive_mode(api_key, base_url, model, max_tool_rounds, services)
        )


async def handle_sql_command(
    query_service, ui, user_input: str, current_database: str
) -> str:
    """Handle SQL command execution using InteractiveQueryService.

    Args:
        query_service: InteractiveQueryService instance
        ui: InteractiveInterface instance
        user_input: Raw user input starting with /sql
        current_database: Current database context ("system" or "user")

    Returns:
        Updated current_database after processing the command
    """
    # Parse the command: /sql use [system|user] OR /sql <query>
    parts = user_input.split(" ", 2)

    if len(parts) < 2:
        ui.show_system_error(
            "SQL command requires arguments. "
            "Usage: /sql use [system|user] OR /sql <query>"
        )
        return current_database

    # Check if this is a database switch command
    if len(parts) >= 3 and parts[1].lower() == "use":
        db_name = parts[2].lower()
        if db_name in ["system", "user"]:
            ui.show_system_success(f"Switched to {db_name} database")
            return db_name
        else:
            ui.show_system_error("Invalid database. Use 'system' or 'user'")
            return current_database

    # Otherwise, treat as a query to execute against current database
    query = " ".join(parts[1:]).strip()

    if not query:
        ui.show_system_error("❌ Empty SQL query provided.")
        return current_database

    try:
        # Execute the query using the current database context
        result = query_service.execute_query(query, current_database)

        # Display results using the UI formatter
        execution_time = getattr(result, "query_execution_time", result.execution_time)
        ui.show_sql_result(result, current_database, query, execution_time)

    except QueryValidationError as e:
        ui.show_system_error(f"Query Error: {str(e)}")
    except DatabaseError as e:
        ui.show_system_error(f"Database Error: {str(e)}")
    except Exception as e:
        ui.show_system_error(f"Unexpected error executing SQL: {str(e)}")

    return current_database


async def run_headless_mode(
    prompt: str,
    api_key: str,
    base_url: str | None,
    model: str | None,
    max_tool_rounds: int,
    _: ServiceContainer,
):
    """Run in headless mode - process prompt and exit."""
    try:
        agent = ArcAgent(api_key, base_url, model, max_tool_rounds)

        # Configure confirmation service for headless mode
        confirmation_service = ConfirmationService()
        confirmation_service.set_session_flag("allOperations", True)

        # Process the user message
        chat_entries = await agent.process_user_message(prompt)

        # Output each message as JSON (OpenAI compatible format)
        for entry in chat_entries:
            if entry.type == "user":
                print(json.dumps({"role": "user", "content": entry.content}))
            elif entry.type == "assistant":
                message = {"role": "assistant", "content": entry.content}
                if entry.tool_calls:
                    message["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                        for tc in entry.tool_calls
                    ]
                print(json.dumps(message))
            elif entry.type == "tool_result" and entry.tool_call:
                print(
                    json.dumps(
                        {
                            "role": "tool",
                            "tool_call_id": entry.tool_call.id,
                            "content": entry.content,
                        }
                    )
                )

    except Exception as e:
        # Output error in OpenAI compatible format
        print(
            json.dumps(
                {
                    "role": "assistant",
                    "content": f"Error: {str(e)}",
                }
            )
        )
        sys.exit(1)


async def run_interactive_mode(
    api_key: str,
    base_url: str | None,
    model: str | None,
    max_tool_rounds: int,
    services: ServiceContainer,
):
    """Run in interactive mode with enhanced UX."""
    try:
        agent = ArcAgent(api_key, base_url, model, max_tool_rounds)
        ui = InteractiveInterface()

        # Database context for SQL commands - defaults to system database
        current_database = "system"

        # Show enhanced welcome screen
        ui.show_welcome(agent.get_current_model(), agent.get_current_directory())

        while True:
            try:
                # Get user input with styled prompt
                user_input = ui.get_user_input()

                # Display the user message in chat history with different coloring
                ui.show_user_message(user_input)

                # Handle system commands (only with / prefix)
                if user_input.startswith("/"):
                    cmd = user_input[
                        1:
                    ].lower()  # Remove the / prefix and convert to lowercase

                    if cmd in ["exit", "quit", "bye"]:
                        ui.show_goodbye()
                        break
                    elif cmd == "help":
                        ui.show_commands()
                        continue
                    elif cmd == "stats":
                        # Show editing statistics if available
                        if hasattr(agent.file_editor, "editor_manager"):
                            stats = (
                                agent.file_editor.editor_manager.get_strategy_stats()
                            )
                            ui.show_edit_summary(stats)
                        else:
                            ui.show_info("📊 No editing statistics available yet.")
                        continue
                    elif cmd == "tree":
                        ui.show_file_tree(agent.get_current_directory())
                        continue
                    elif cmd == "clear":
                        ui.clear_screen()
                        continue
                    elif cmd == "config":
                        # Show current configuration with fresh settings
                        settings_manager = SettingsManager()
                        current_api_key = settings_manager.get_api_key()
                        current_base_url = settings_manager.get_base_url()
                        current_model = settings_manager.get_current_model()

                        config_text = (
                            f"API Key: {'*' * 8 if current_api_key else 'Not set'}\n"
                            f"Base URL: {current_base_url}\n"
                            f"Model: {current_model or 'Not set'}\n"
                            f"Max Tool Rounds: {max_tool_rounds}"
                        )
                        ui.show_config_panel(config_text)
                        continue
                    elif cmd == "performance":
                        # Show performance metrics
                        metrics = performance_manager.get_metrics()
                        error_stats = error_handler.get_error_stats()
                        ui.show_performance_metrics(metrics, error_stats)
                        continue
                    elif cmd.startswith("sql"):
                        # Handle SQL queries and update current database context
                        current_database = await handle_sql_command(
                            services.query, ui, user_input, current_database
                        )
                        continue
                    else:
                        ui.show_system_error(f"Unknown system command: /{cmd}")
                        continue

                # Handle special exit commands without prefix (for convenience)
                elif user_input.lower() in ["exit", "quit", "bye"]:
                    ui.show_goodbye()
                    break

                if not user_input:
                    continue

                # Process streaming response with clean context management
                start_time = time.time()

                with ui.stream_response(start_time) as handler:
                    async for chunk in agent.process_user_message_stream(user_input):
                        handler.handle_chunk(chunk)

            except KeyboardInterrupt:
                ui.show_goodbye()
                break
            except EOFError:
                ui.show_goodbye()
                break
            except Exception as e:
                ui.show_system_error(str(e))

    except Exception as e:
        ui = InteractiveInterface()
        ui.show_system_error(f"Error initializing Arc CLI: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
