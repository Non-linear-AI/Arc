"""Command-line interface for Arc CLI."""

import asyncio
import json
import os
import sys
import time

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from ..core import ArcAgent, SettingsManager
from ..database import DatabaseError, DatabaseManager, QueryValidationError
from ..database.services import ServiceContainer
from ..error_handling import error_handler
from ..utils import ConfirmationService, performance_manager
from .console import InteractiveInterface

# Load environment variables
load_dotenv()

console = Console()


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
    # Change directory if specified
    if directory:
        try:
            os.chdir(directory)
        except OSError as e:
            console.print(
                f"❌ Error changing directory to {directory}: {e}", style="red"
            )
            sys.exit(1)

    # Get configuration
    settings_manager = SettingsManager()

    api_key = api_key or settings_manager.get_api_key()
    if not api_key:
        console.print(
            "❌ Error: API key required. Set ARC_API_KEY environment variable, "
            "use --api-key flag, or save to ~/.arc/user-settings.json",
            style="red",
        )
        sys.exit(1)

    base_url = base_url or settings_manager.get_base_url()
    model = model or settings_manager.get_current_model()

    # Save command line settings if provided
    if api_key and click.get_current_context().params.get("api_key"):
        settings_manager.update_user_setting("apiKey", api_key)
        console.print("✅ API key saved to ~/.arc/user-settings.json")

    if base_url and click.get_current_context().params.get("base_url"):
        settings_manager.update_user_setting("baseURL", base_url)
        console.print("✅ Base URL saved to ~/.arc/user-settings.json")

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


async def handle_sql_command(query_service, ui, user_input: str) -> None:
    """Handle SQL command execution using InteractiveQueryService."""
    # Parse the command: /sql [system|user] <query>
    parts = user_input.split(" ", 2)

    if len(parts) < 2:
        console.print(
            "❌ SQL command requires a query. Usage: /sql [system|user] <query>"
        )
        return

    # Determine target database and query
    if len(parts) >= 3 and parts[1].lower() in ["system", "user"]:
        target_db = parts[1].lower()
        query = parts[2].strip()
    else:
        target_db = "system"  # Default to system database
        query = " ".join(parts[1:]).strip()

    if not query:
        console.print("❌ Empty SQL query provided.")
        return

    try:
        # Execute the query using the service - now returns QueryResult directly
        result = query_service.execute_query(query, target_db)

        # Display results using the UI formatter
        # TimedQueryResult has execution_time in the execution_time property
        execution_time = getattr(result, "query_execution_time", result.execution_time)
        ui.show_sql_result(result, target_db, query, execution_time)

    except QueryValidationError as e:
        console.print(f"❌ Query Error: {str(e)}", style="red")
    except DatabaseError as e:
        console.print(f"❌ Database Error: {str(e)}", style="red")
    except Exception as e:
        console.print(f"❌ Unexpected error executing SQL: {str(e)}", style="red")


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

        # Show enhanced welcome screen
        ui.show_welcome(agent.get_current_model(), agent.get_current_directory())

        while True:
            try:
                # Get user input with styled prompt
                user_input = console.input("\n[bold green]>[/bold green] ").strip()

                # Display the user message in chat history with different coloring
                ui.show_user_message(user_input)

                # Handle system commands (only with / prefix)
                if user_input.startswith("/"):
                    cmd = user_input[
                        1:
                    ].lower()  # Remove the / prefix and convert to lowercase

                    if cmd in ["exit", "quit", "bye"]:
                        console.print("👋 Goodbye!", style="cyan")
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
                            console.print("📊 No editing statistics available yet.")
                        continue
                    elif cmd == "tree":
                        ui.show_file_tree(agent.get_current_directory())
                        continue
                    elif cmd == "clear":
                        console.clear()
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
                        console.print(
                            Panel(
                                config_text,
                                title="⚙️ Current Configuration",
                                border_style="blue",
                            )
                        )
                        continue
                    elif cmd == "performance":
                        # Show performance metrics
                        metrics = performance_manager.get_metrics()
                        error_stats = error_handler.get_error_stats()
                        ui.show_performance_metrics(metrics, error_stats)
                        continue
                    elif cmd.startswith("sql"):
                        # Handle SQL queries
                        await handle_sql_command(services.query, ui, user_input)
                        continue
                    else:
                        console.print(f"❌ Unknown system command: /{cmd}")
                        console.print("Type /help to see available commands")
                        continue

                # Handle special exit commands without prefix (for convenience)
                elif user_input.lower() in ["exit", "quit", "bye"]:
                    console.print("👋 Goodbye!", style="cyan")
                    break

                if not user_input:
                    continue

                # Process streaming response with proper spacing
                start_time = time.time()
                current_content = ""

                async for chunk in agent.process_user_message_stream(user_input):
                    if chunk.type == "content" and chunk.content:
                        # Accumulate assistant thoughts to render as a step later
                        current_content += chunk.content

                    elif chunk.type == "tool_calls" and chunk.tool_call:
                        # Flush any pending assistant thoughts before tool execution
                        if current_content.strip():
                            ui.show_assistant_step(current_content)
                            current_content = ""
                        args = {}
                        if chunk.tool_call and chunk.tool_call.arguments:
                            try:
                                args = json.loads(chunk.tool_call.arguments)
                            except json.JSONDecodeError:
                                args = {"raw_arguments": chunk.tool_call.arguments}

                        tool_name = (
                            chunk.tool_call.name if chunk.tool_call else "Unknown Tool"
                        )
                        ui.show_tool_execution(tool_name, args)

                    elif (
                        chunk.type == "tool_result"
                        and chunk.tool_result
                        and chunk.tool_call
                    ):
                        # Show tool result
                        tool_time = time.time() - start_time
                        ui.show_tool_result(
                            chunk.tool_call.name, chunk.tool_result, tool_time
                        )

                    elif chunk.type == "error":
                        console.print(f"\n❌ Error: {chunk.content}")

                    elif chunk.type == "done":
                        # Flush any remaining assistant thoughts at end
                        if current_content.strip():
                            ui.show_assistant_step(current_content)
                            current_content = ""
                        console.print()  # blank line after completion
                        break

            except KeyboardInterrupt:
                console.print("\n👋 Goodbye!", style="cyan")
                break
            except EOFError:
                console.print("\n👋 Goodbye!", style="cyan")
                break
            except Exception as e:
                console.print(f"\n❌ Error: {str(e)}", style="red")

    except Exception as e:
        console.print(f"❌ Error initializing Arc CLI: {str(e)}", style="red")
        sys.exit(1)


if __name__ == "__main__":
    cli()
