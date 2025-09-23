"""Command-line interface for Arc CLI."""

import asyncio
import json
import os
import shlex
import sys
import time
from contextlib import suppress
from pathlib import Path

import click
from dotenv import load_dotenv

from ..core import ArcAgent, SettingsManager
from ..core.agents import ModelGeneratorAgent, TrainerGeneratorAgent
from ..core.agents.model_generator.model_generator import ModelGeneratorError
from ..core.agents.predictor_generator import PredictorGeneratorAgent
from ..core.agents.trainer_generator.trainer_generator import TrainerGeneratorError
from ..database import DatabaseError, DatabaseManager, QueryValidationError
from ..database.services import ServiceContainer
from ..ml.runtime import MLRuntime, MLRuntimeError
from ..utils import ConfirmationService
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
    services = ServiceContainer(db_manager, artifacts_dir="artifacts")

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
        ui.show_sql_result(result, current_database, execution_time)

    except QueryValidationError as e:
        ui.show_system_error(f"Query Error: {str(e)}")
    except DatabaseError as e:
        ui.show_system_error(f"Database Error: {str(e)}")
    except Exception as e:
        ui.show_system_error(f"Unexpected error executing SQL: {str(e)}")

    return current_database


class CommandError(Exception):
    """Raised when ML command parsing or validation fails."""


def _parse_options(args: list[str], spec: dict[str, bool]) -> dict[str, str | bool]:
    options: dict[str, str | bool] = {}
    idx = 0
    while idx < len(args):
        token = args[idx]
        if not token.startswith("--"):
            raise CommandError(f"Unexpected argument '{token}'")
        key = token[2:]
        if key not in spec:
            raise CommandError(f"Unknown option '--{key}'")
        expects_value = spec[key]
        if not expects_value:
            options[key] = True
            idx += 1
            continue
        idx += 1
        if idx >= len(args):
            raise CommandError(f"Option '--{key}' requires a value")
        options[key] = args[idx]
        idx += 1
    return options


async def handle_ml_command(
    user_input: str, ui: InteractiveInterface, runtime: "MLRuntime"
) -> None:
    try:
        tokens = shlex.split(user_input)
    except ValueError as e:
        ui.show_system_error(f"Failed to parse command: {e}")
        return

    if not tokens or tokens[0] != "/ml":
        ui.show_system_error("Invalid ML command. Use /ml <subcommand> ...")
        return

    if len(tokens) < 2:
        ui.show_system_error(
            "Usage: /ml <create-model|train|predict|jobs|generate-model|"
            "generate-trainer|generate-predictor> ..."
        )
        return

    subcommand = tokens[1]
    args = tokens[2:]

    try:
        if subcommand == "create-model":
            _ml_create_model(args, ui, runtime)
        elif subcommand == "train":
            _ml_train(args, ui, runtime)
        elif subcommand == "predict":
            _ml_predict(args, ui, runtime)
        elif subcommand == "jobs":
            _ml_jobs(args, ui, runtime)
        elif subcommand == "generate-model":
            await _ml_generate_model(args, ui, runtime)
        elif subcommand == "generate-trainer":
            await _ml_generate_trainer(args, ui, runtime)
        elif subcommand == "generate-predictor":
            await _ml_generate_predictor(args, ui, runtime)
        else:
            raise CommandError(f"Unknown ML command: {subcommand}")
    except CommandError as e:
        ui.show_system_error(str(e))
    except Exception as e:
        ui.show_system_error(f"ML command failed: {e}")


def _ml_create_model(
    args: list[str], ui: InteractiveInterface, runtime: "MLRuntime"
) -> None:
    options = _parse_options(
        args,
        {
            "name": True,
            "schema": True,
        },
    )

    name = options.get("name")
    schema_path = options.get("schema")

    if not name or not schema_path:
        raise CommandError("/ml create-model requires --name and --schema")
    schema_path_obj = Path(str(schema_path))

    try:
        model = runtime.create_model(
            name=str(name),
            schema_path=schema_path_obj,
        )
    except MLRuntimeError as exc:
        raise CommandError(str(exc)) from exc

    ui.show_system_success(
        f"Model '{model.name}' registered (version {model.version}, id={model.id})."
    )


def _ml_train(args: list[str], ui: InteractiveInterface, runtime: "MLRuntime") -> None:
    options = _parse_options(
        args,
        {
            "model": True,
            "data": True,
            "target": True,
        },
    )

    model_name = options.get("model")
    train_table = options.get("data")
    target_column = options.get("target")

    if not model_name or not train_table:
        raise CommandError("/ml train requires --model and --data")

    try:
        job_id = runtime.train_model(
            model_name=str(model_name),
            train_table=str(train_table),
            target_column=str(target_column) if target_column else None,
        )
    except MLRuntimeError as exc:
        raise CommandError(str(exc)) from exc

    ui.show_system_success(
        "Training job submitted. Use /ml jobs status <job_id> to monitor."
    )
    ui.show_info(f"Job ID: {job_id}")


def _ml_predict(
    args: list[str], ui: InteractiveInterface, runtime: "MLRuntime"
) -> None:
    options = _parse_options(
        args,
        {
            "model": True,
            "data": True,
            "output": True,
        },
    )

    model_name = options.get("model")
    table_name = options.get("data")

    if not model_name or not table_name:
        raise CommandError("/ml predict requires --model and --data")

    # Parse optional output table parameter
    output_table = options.get("output")

    try:
        summary = runtime.predict(
            model_name=str(model_name),
            table_name=str(table_name),
            output_table=output_table,
        )
    except MLRuntimeError as exc:
        raise CommandError(str(exc)) from exc

    outputs_display = ", ".join(summary.outputs) if summary.outputs else "None"
    ui.show_system_success(
        f"Generated {summary.total_predictions} predictions "
        f"with outputs: {outputs_display}"
    )

    if summary.saved_table:
        ui.show_system_success(f"Predictions saved to table '{summary.saved_table}'")


def _ml_jobs(args: list[str], ui: InteractiveInterface, runtime: MLRuntime) -> None:
    if not args:
        raise CommandError("Usage: /ml jobs <list|status <job_id>>")

    sub = args[0]
    if sub == "list":
        jobs = runtime.job_service.list_jobs(limit=20)
        rows = []
        for job in jobs:
            job_type = job.type.value if hasattr(job.type, "value") else str(job.type)
            status = (
                job.status.value if hasattr(job.status, "value") else str(job.status)
            )
            rows.append(
                [
                    job.job_id,
                    job_type,
                    status,
                    job.message or "",
                    job.updated_at.isoformat()
                    if hasattr(job.updated_at, "isoformat")
                    else str(job.updated_at),
                ]
            )
        ui.show_table(
            title="ML Jobs",
            columns=["Job ID", "Type", "Status", "Message", "Updated"],
            rows=rows,
        )
    elif sub == "status":
        if len(args) < 2:
            raise CommandError("Usage: /ml jobs status <job_id>")
        job_id = args[1]
        job = runtime.job_service.get_job_by_id(job_id)
        if job is None:
            raise CommandError(f"Job '{job_id}' not found")
        job_type = job.type.value if hasattr(job.type, "value") else str(job.type)
        status = job.status.value if hasattr(job.status, "value") else str(job.status)
        rows = [
            ["Job ID", job.job_id],
            ["Type", job_type],
            ["Status", status],
            ["Message", job.message or ""],
            [
                "Created",
                job.created_at.isoformat()
                if hasattr(job.created_at, "isoformat")
                else str(job.created_at),
            ],
            [
                "Updated",
                job.updated_at.isoformat()
                if hasattr(job.updated_at, "isoformat")
                else str(job.updated_at),
            ],
        ]
        ui.show_key_values("Job Status", rows)
    else:
        raise CommandError(f"Unknown jobs subcommand: {sub}")


async def _ml_generate_model(
    args: list[str], ui: InteractiveInterface, runtime: "MLRuntime"
) -> None:
    """Handle model specification generation command."""
    options = _parse_options(
        args,
        {
            "name": True,
            "context": True,
            "data-table": True,
            "output": True,
        },
    )

    name = options.get("name")
    context = options.get("context")
    data_table = options.get("data-table")
    output_path = options.get("output")

    if not name or not context or not data_table:
        raise CommandError(
            "/ml generate-model requires --name, --context, and --data-table"
        )

    # Default output path if not specified
    if not output_path:
        output_path = f"{name}_model.yaml"

    try:
        ui.show_info(f"🤖 Generating model specification for '{name}'...")
        ui.show_info(f"📊 Analyzing data table '{data_table}'...")

        # Get the agent from the runtime or create one for model generation
        services = runtime.services
        settings_manager = SettingsManager()
        api_key = settings_manager.get_api_key()
        base_url = settings_manager.get_base_url()
        model = settings_manager.get_current_model()

        if not api_key:
            raise CommandError("API key required for model generation")

        # Create model generator agent (no ArcAgent dependency)
        model_generator = ModelGeneratorAgent(services, api_key, base_url, model)

        # Generate the model specification
        model_spec, model_yaml = await model_generator.generate_model(
            name=str(name),
            user_context=str(context),
            table_name=str(data_table),
        )

        # Save to file
        Path(output_path).write_text(model_yaml)

        ui.show_system_success("✅ Model specification generated successfully!")
        ui.show_info(f"📄 Saved to: {output_path}")
        ui.show_info(f"🏗️ Model: {name}")

        # Show brief summary of generated model
        input_count = len(model_spec.inputs)
        node_count = len(model_spec.graph)
        output_count = len(model_spec.outputs)

        ui.show_info(f"📥 Inputs: {input_count}")
        ui.show_info(f"🧠 Model nodes: {node_count}")
        ui.show_info(f"📤 Outputs: {output_count}")

        # Suggest next steps
        ui.show_info("\n💡 Next steps:")
        ui.show_info(
            f"   /ml generate-trainer --name {name}_trainer --context "
            f"'training config' --model-spec {output_path}"
        )

    except ModelGeneratorError as exc:
        raise CommandError(f"Model generation failed: {exc}") from exc
    except Exception as exc:
        raise CommandError(f"Unexpected error during model generation: {exc}") from exc


async def _ml_generate_trainer(
    args: list[str], ui: InteractiveInterface, runtime: "MLRuntime"
) -> None:
    """Handle trainer specification generation command."""
    options = _parse_options(
        args,
        {
            "name": True,
            "context": True,
            "model-spec": True,
            "output": True,
        },
    )

    name = options.get("name")
    context = options.get("context")
    model_spec_path = options.get("model-spec")
    output_path = options.get("output")

    if not name or not context or not model_spec_path:
        raise CommandError(
            "/ml generate-trainer requires --name, --context, and --model-spec"
        )

    # Default output path if not specified
    if not output_path:
        output_path = f"{name}_trainer.yaml"

    try:
        ui.show_info(f"🤖 Generating trainer specification for '{name}'...")
        ui.show_info(f"📋 Using model specification: {model_spec_path}")

        # Check that model specification file exists
        model_spec_file = Path(str(model_spec_path))
        if not model_spec_file.exists():
            raise CommandError(f"Model specification file not found: {model_spec_path}")

        # Get the agent from the runtime or create one for trainer generation
        services = runtime.services
        settings_manager = SettingsManager()
        api_key = settings_manager.get_api_key()
        base_url = settings_manager.get_base_url()
        model = settings_manager.get_current_model()

        if not api_key:
            raise CommandError("API key required for trainer generation")

        # Create trainer generator agent (no ArcAgent dependency)
        trainer_generator = TrainerGeneratorAgent(services, api_key, base_url, model)

        # Generate the trainer specification
        trainer_spec, trainer_yaml = await trainer_generator.generate_trainer(
            name=str(name),
            user_context=str(context),
            model_spec_path=str(model_spec_path),
        )

        # Save to file
        Path(output_path).write_text(trainer_yaml)

        ui.show_system_success("✅ Trainer specification generated successfully!")
        ui.show_info(f"📄 Saved to: {output_path}")
        ui.show_info(f"🏋️ Trainer: {name}")

        # Show brief summary of generated trainer
        ui.show_info(f"🎯 Loss function: {trainer_spec.loss.type}")
        ui.show_info(f"⚡ Optimizer: {trainer_spec.optimizer.type}")
        if hasattr(trainer_spec, "epochs"):
            ui.show_info(f"🔄 Epochs: {trainer_spec.epochs}")
        if hasattr(trainer_spec, "batch_size"):
            ui.show_info(f"📦 Batch size: {trainer_spec.batch_size}")

        # Suggest next steps
        ui.show_info("\n💡 Next steps:")
        ui.show_info(
            f"   /ml create-model --name {name.replace('_trainer', '')} "
            f"--schema <combined_schema>"
        )
        ui.show_info(
            f"   /ml train --model {name.replace('_trainer', '')} --data <table_name>"
        )

    except TrainerGeneratorError as exc:
        raise CommandError(f"Trainer generation failed: {exc}") from exc
    except Exception as exc:
        raise CommandError(
            f"Unexpected error during trainer generation: {exc}"
        ) from exc


async def _ml_generate_predictor(
    args: list[str], ui: InteractiveInterface, runtime: "MLRuntime"
) -> None:
    """Handle predictor specification generation command."""
    options = _parse_options(
        args,
        {
            "model-spec": True,
            "context": True,
            "trainer-spec": True,
            "output": True,
        },
    )

    model_spec_path = options.get("model-spec")
    context = options.get("context")
    trainer_spec_path = options.get("trainer-spec")
    output_path = options.get("output")

    if not model_spec_path or not context:
        raise CommandError("/ml generate-predictor requires --model-spec and --context")

    # Check that model spec file exists
    model_spec_file = Path(str(model_spec_path))
    if not model_spec_file.exists():
        raise CommandError(f"Model specification file not found: {model_spec_path}")

    # Check trainer spec file if provided
    if trainer_spec_path:
        trainer_spec_file = Path(str(trainer_spec_path))
        if not trainer_spec_file.exists():
            raise CommandError(
                f"Trainer specification file not found: {trainer_spec_path}"
            )

    # Default output path if not specified
    if not output_path:
        output_path = "predictor.yaml"

    try:
        ui.show_info("🤖 Generating predictor specification...")
        ui.show_info(f"📋 Using model specification: {model_spec_path}")
        if trainer_spec_path:
            ui.show_info(f"🏋️ Using trainer specification: {trainer_spec_path}")

        # Get the agent configuration
        services = runtime.services
        settings_manager = SettingsManager()
        api_key = settings_manager.get_api_key()
        base_url = settings_manager.get_base_url()
        model = settings_manager.get_current_model()

        if not api_key:
            raise CommandError("API key required for predictor generation")

        # Create predictor generator agent (no ArcAgent dependency)
        predictor_generator = PredictorGeneratorAgent(
            services, api_key, base_url, model
        )

        # Generate the predictor specification
        predictor_yaml = await predictor_generator.generate_predictor(
            user_context=context,
            model_spec_path=str(model_spec_path),
            trainer_spec_path=str(trainer_spec_path) if trainer_spec_path else None,
        )

        # Save to file
        Path(output_path).write_text(predictor_yaml)

        ui.show_system_success("✅ Predictor specification generated successfully!")
        ui.show_info(f"📄 Saved to: {output_path}")

        # Show YAML preview
        ui.show_info("\n📋 Generated YAML:")
        # Show first few lines of the YAML
        yaml_lines = predictor_yaml.strip().split("\n")
        for line in yaml_lines[:10]:  # Show first 10 lines
            ui.show_info(f"   {line}")
        if len(yaml_lines) > 10:
            ui.show_info("   ...")

        # Suggest next steps
        ui.show_info("\n💡 Next steps:")
        ui.show_info("   Use this predictor specification for model inference")

    except Exception as exc:
        raise CommandError(f"Predictor generation failed: {exc}") from exc


async def run_headless_mode(
    prompt: str,
    api_key: str,
    base_url: str | None,
    model: str | None,
    max_tool_rounds: int,
    services: ServiceContainer,
):
    """Run in headless mode - process prompt and exit."""
    try:
        agent = ArcAgent(api_key, base_url, model, max_tool_rounds, services)

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
        agent = ArcAgent(api_key, base_url, model, max_tool_rounds, services)
        ui = InteractiveInterface()

        # Database context for SQL commands - defaults to system database
        current_database = "system"

        # Show enhanced welcome screen
        ui.show_welcome(agent.get_current_model(), agent.get_current_directory())

        while True:
            try:
                # Get user input with styled prompt
                user_input = await ui.get_user_input_async()

                # Display the user message in chat history with different coloring
                ui.show_user_message(user_input)

                # Handle system commands (only with / prefix)
                if user_input.startswith("/"):
                    if user_input.startswith("/ml"):
                        await handle_ml_command(user_input, ui, services.ml_runtime)
                        continue

                    cmd = user_input[
                        1:
                    ].lower()  # Remove the / prefix and convert to lowercase

                    if cmd == "exit":
                        ui.show_goodbye()
                        break
                    elif cmd == "help":
                        ui.show_commands()
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
                    elif cmd.startswith("sql"):
                        # Handle SQL queries and update current database context
                        current_database = await handle_sql_command(
                            services.query, ui, user_input, current_database
                        )
                        continue
                    else:
                        ui.show_system_error(f"Unknown system command: /{cmd}")
                        continue

                # No special exit without slash; only /exit is supported

                if not user_input:
                    continue

                # Process streaming response with clean context management
                start_time = time.time()

                with ui.escape_watcher() as esc:
                    interrupted = False
                    with ui.stream_response(start_time) as handler:
                        agen = agent.process_user_message_stream(user_input).__aiter__()
                        esc_task = asyncio.create_task(esc.event.wait())
                        try:
                            while True:
                                next_task = asyncio.create_task(agen.__anext__())
                                done, _ = await asyncio.wait(
                                    {esc_task, next_task},
                                    return_when=asyncio.FIRST_COMPLETED,
                                )

                                if esc_task in done:
                                    interrupted = True
                                    next_task.cancel()
                                    with suppress(Exception):
                                        await agen.aclose()
                                    break

                                # Otherwise, next_task completed
                                chunk = next_task.result()
                                handler.handle_chunk(chunk)
                        except StopAsyncIteration:
                            pass
                        finally:
                            if not esc_task.done():
                                esc_task.cancel()

                if interrupted:
                    ui.show_info("⏹ Interrupted.")

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
    finally:
        with suppress(Exception):
            services.shutdown()


if __name__ == "__main__":
    cli()


@cli.command("generate-model")
@click.option("-n", "--name", required=True, help="Model name")
@click.option("-c", "--context", required=True, help="Model description and context")
@click.option("-t", "--data-table", required=True, help="Database table name for data")
@click.option("-o", "--output", help="Output file path (default: {name}_model.yaml)")
@click.option("-k", "--api-key", help="Arc API key (or set ARC_API_KEY env var)")
@click.option("-u", "--base-url", help="Arc API base URL")
@click.option("-m", "--model", "ai_model", help="AI model to use")
def generate_model(
    name: str,
    context: str,
    data_table: str,
    output: str | None,
    api_key: str | None,
    base_url: str | None,
    ai_model: str | None,
):
    """Generate a model specification using Arc model generator agent."""
    ui = InteractiveInterface()

    # Get configuration
    settings_manager = SettingsManager()
    api_key = api_key or settings_manager.get_api_key()
    if not api_key:
        ui.show_system_error(
            "API key required. Set ARC_API_KEY environment variable or use --api-key"
        )
        sys.exit(1)

    base_url = base_url or settings_manager.get_base_url()
    ai_model = ai_model or settings_manager.get_current_model()

    # Initialize services
    system_db_path = settings_manager.get_system_database_path()
    user_db_path = settings_manager.get_user_database_path()
    db_manager = DatabaseManager(system_db_path, user_db_path)
    services = ServiceContainer(db_manager)
    runtime = MLRuntime(services)

    # Run model generation
    asyncio.run(
        _ml_generate_model_cli(
            name, context, data_table, output, api_key, base_url, ai_model, ui, runtime
        )
    )


@cli.command("generate-trainer")
@click.option("-n", "--name", required=True, help="Trainer name")
@click.option(
    "-c", "--context", required=True, help="Training context and requirements"
)
@click.option(
    "-s", "--model-spec", required=True, help="Path to model specification file"
)
@click.option("-o", "--output", help="Output file path (default: {name}_trainer.yaml)")
@click.option("-k", "--api-key", help="Arc API key (or set ARC_API_KEY env var)")
@click.option("-u", "--base-url", help="Arc API base URL")
@click.option("-m", "--model", "ai_model", help="AI model to use")
def generate_trainer(
    name: str,
    context: str,
    model_spec: str,
    output: str | None,
    api_key: str | None,
    base_url: str | None,
    ai_model: str | None,
):
    """Generate a trainer specification using Arc trainer generator agent."""
    ui = InteractiveInterface()

    # Get configuration
    settings_manager = SettingsManager()
    api_key = api_key or settings_manager.get_api_key()
    if not api_key:
        ui.show_system_error(
            "API key required. Set ARC_API_KEY environment variable or use --api-key"
        )
        sys.exit(1)

    base_url = base_url or settings_manager.get_base_url()
    ai_model = ai_model or settings_manager.get_current_model()

    # Initialize services
    system_db_path = settings_manager.get_system_database_path()
    user_db_path = settings_manager.get_user_database_path()
    db_manager = DatabaseManager(system_db_path, user_db_path)
    services = ServiceContainer(db_manager)
    runtime = MLRuntime(services)

    # Run trainer generation
    asyncio.run(
        _ml_generate_trainer_cli(
            name, context, model_spec, output, api_key, base_url, ai_model, ui, runtime
        )
    )


@cli.command("generate-predictor")
@click.option(
    "-c", "--context", required=True, help="Prediction requirements and use case"
)
@click.option(
    "-s", "--model-spec", required=True, help="Path to model specification file"
)
@click.option("-t", "--trainer-spec", help="Path to trainer specification file")
@click.option("-o", "--output", help="Output file path (default: predictor.yaml)")
@click.option("-k", "--api-key", help="Arc API key (or set ARC_API_KEY env var)")
@click.option("-u", "--base-url", help="Arc API base URL")
@click.option("-m", "--model", "ai_model", help="AI model to use")
def generate_predictor(
    context: str,
    model_spec: str,
    trainer_spec: str | None,
    output: str | None,
    api_key: str | None,
    base_url: str | None,
    ai_model: str | None,
):
    """Generate a predictor specification using Arc predictor generator agent."""
    ui = InteractiveInterface()

    # Get configuration
    settings_manager = SettingsManager()
    api_key = api_key or settings_manager.get_api_key()
    if not api_key:
        ui.show_system_error(
            "API key required. Set ARC_API_KEY environment variable or use --api-key"
        )
        sys.exit(1)

    base_url = base_url or settings_manager.get_base_url()
    ai_model = ai_model or settings_manager.get_current_model()

    # Initialize services
    system_db_path = settings_manager.get_system_database_path()
    user_db_path = settings_manager.get_user_database_path()
    db_manager = DatabaseManager(system_db_path, user_db_path)
    services = ServiceContainer(db_manager)
    runtime = MLRuntime(services)

    # Run predictor generation
    asyncio.run(
        _ml_generate_predictor_cli(
            context,
            model_spec,
            trainer_spec,
            output,
            api_key,
            base_url,
            ai_model,
            ui,
            runtime,
        )
    )


async def _ml_generate_model_cli(
    name: str,
    context: str,
    data_table: str,
    output_path: str | None,
    _api_key: str,
    _base_url: str | None,
    _ai_model: str | None,
    ui: InteractiveInterface,
    runtime: MLRuntime,
) -> None:
    """CLI wrapper for model generation."""
    try:
        args = ["--name", name, "--context", context, "--data-table", data_table]
        if output_path:
            args.extend(["--output", output_path])

        await _ml_generate_model(args, ui, runtime)

    except Exception as e:
        ui.show_system_error(f"Model generation failed: {e}")
        sys.exit(1)


async def _ml_generate_trainer_cli(
    name: str,
    context: str,
    model_spec: str,
    output_path: str | None,
    _api_key: str,
    _base_url: str | None,
    _ai_model: str | None,
    ui: InteractiveInterface,
    runtime: MLRuntime,
) -> None:
    """CLI wrapper for trainer generation."""
    try:
        args = ["--name", name, "--context", context, "--model-spec", model_spec]
        if output_path:
            args.extend(["--output", output_path])

        await _ml_generate_trainer(args, ui, runtime)

    except Exception as e:
        ui.show_system_error(f"Trainer generation failed: {e}")
        sys.exit(1)


async def _ml_generate_predictor_cli(
    context: str,
    model_spec_path: str,
    trainer_spec_path: str | None,
    output_path: str | None,
    _api_key: str,
    _base_url: str | None,
    _ai_model: str | None,
    ui: InteractiveInterface,
    runtime: MLRuntime,
) -> None:
    """CLI wrapper for predictor generation."""
    try:
        args = ["--model-spec", model_spec_path, "--context", context]

        if trainer_spec_path:
            args.extend(["--trainer-spec", trainer_spec_path])
        if output_path:
            args.extend(["--output", output_path])

        await _ml_generate_predictor(args, ui, runtime)

    except Exception as e:
        ui.show_system_error(f"Predictor generation failed: {e}")
        sys.exit(1)
