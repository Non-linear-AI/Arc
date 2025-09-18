"""Command-line interface for Arc CLI."""

import asyncio
import json
import os
import re
import shlex
import sys
import time
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import torch
from dotenv import load_dotenv

from ..core import ArcAgent, SettingsManager
from ..database import DatabaseError, DatabaseManager, QueryValidationError
from ..database.models.model import Model
from ..database.services import ServiceContainer
from ..error_handling import error_handler
from ..graph.spec import ArcGraph
from ..ml.artifacts import ModelArtifactManager
from ..ml.predictor import ArcPredictor
from ..ml.training_service import TrainingJobConfig, TrainingService
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


class CommandError(Exception):
    """Raised when ML command parsing or validation fails."""


class MLRuntime:
    """Runtime utilities for ML CLI commands."""

    def __init__(self, services: ServiceContainer, artifacts_dir: Path | None = None):
        self.services = services
        self.model_service = services.models
        self.job_service = services.jobs
        self.ml_data_service = services.ml_data
        self.artifacts_root = Path(artifacts_dir or "artifacts")
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self.training_service = TrainingService(
            self.job_service, artifacts_dir=self.artifacts_root
        )

    def shutdown(self) -> None:
        self.training_service.shutdown()


def _slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "model"


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
    user_input: str, ui: InteractiveInterface, runtime: MLRuntime
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
        ui.show_system_error("Usage: /ml <create-model|train|predict|jobs> ...")
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
        else:
            raise CommandError(f"Unknown ML command: {subcommand}")
    except CommandError as e:
        ui.show_system_error(str(e))
    except Exception as e:
        ui.show_system_error(f"ML command failed: {e}")


def _ml_create_model(
    args: list[str], ui: InteractiveInterface, runtime: MLRuntime
) -> None:
    options = _parse_options(
        args,
        {
            "name": True,
            "schema": True,
            "description": True,
            "type": True,
        },
    )

    name = options.get("name")
    schema_path = options.get("schema")

    if not name or not schema_path:
        raise CommandError("/ml create-model requires --name and --schema")

    schema_file = Path(str(schema_path)).expanduser()
    if not schema_file.exists():
        raise CommandError(f"Schema file not found: {schema_file}")

    try:
        schema_text = schema_file.read_text(encoding="utf-8")
    except OSError as e:
        raise CommandError(f"Failed to read schema file: {e}") from e

    try:
        arc_graph = ArcGraph.from_yaml(schema_text)
    except Exception as e:
        raise CommandError(f"Invalid Arc-Graph schema: {e}") from e

    # Ensure graph is ready for training
    arc_graph.to_training_config()

    model_service = runtime.model_service
    latest = model_service.get_latest_model_by_name(str(name))
    version = 1 if latest is None else latest.version + 1

    model_id = f"{_slugify_name(str(name))}-v{version}"
    model_type = str(options.get("type") or "ml.arc_graph")
    description = str(options.get("description") or arc_graph.description or "")

    now = datetime.now(UTC)
    arc_graph_payload = json.dumps(asdict(arc_graph), default=str)

    model = Model(
        id=model_id,
        type=model_type,
        name=str(name),
        version=version,
        description=description,
        base_model_id=None,
        spec=schema_text,
        arc_graph=arc_graph_payload,
        created_at=now,
        updated_at=now,
    )

    try:
        model_service.create_model(model)
    except DatabaseError as e:
        raise CommandError(f"Failed to register model: {e}") from e

    ui.show_system_success(
        f"Model '{name}' registered (version {version}, id={model_id})."
    )


def _ml_train(args: list[str], ui: InteractiveInterface, runtime: MLRuntime) -> None:
    options = _parse_options(
        args,
        {
            "model": True,
            "data": True,
            "target": True,
            "validation-table": True,
            "validation-split": True,
            "epochs": True,
            "batch-size": True,
            "learning-rate": True,
            "checkpoint-dir": True,
            "description": True,
            "tags": True,
        },
    )

    model_name = options.get("model")
    train_table = options.get("data")

    if not model_name or not train_table:
        raise CommandError("/ml train requires --model and --data")

    model_record = runtime.model_service.get_latest_model_by_name(str(model_name))
    if model_record is None:
        raise CommandError(
            f"Model '{model_name}' not found. Use /ml create-model first."
        )

    if not model_record.arc_graph:
        raise CommandError(
            "Stored model is missing arc graph specification; cannot train."
        )

    try:
        arc_graph_dict = json.loads(model_record.arc_graph)
        arc_graph = ArcGraph.from_dict(arc_graph_dict)
    except Exception as e:
        raise CommandError(f"Failed to load Arc-Graph from stored spec: {e}") from e

    feature_columns = arc_graph.features.feature_columns
    if not feature_columns:
        raise CommandError(
            "Arc-Graph must define features.feature_columns before training."
        )

    if not runtime.ml_data_service.dataset_exists(str(train_table)):
        raise CommandError(f"Training table '{train_table}' does not exist in user DB")

    target_column = options.get("target")
    if not target_column:
        targets = arc_graph.features.target_columns or []
        if not targets:
            raise CommandError(
                "Unable to determine target column. Provide --target explicitly."
            )
        target_column = targets[0]

    columns_to_check = list(feature_columns)
    if target_column not in columns_to_check:
        columns_to_check.append(target_column)

    missing = [
        col
        for col, exists in runtime.ml_data_service.validate_columns(
            str(train_table), columns_to_check
        ).items()
        if not exists
    ]
    if missing:
        raise CommandError(
            "Training table is missing required column(s): " + ", ".join(missing)
        )

    overrides: dict[str, Any] = {}
    if "epochs" in options:
        try:
            overrides["epochs"] = int(options["epochs"])
        except ValueError as e:
            raise CommandError("Option --epochs must be an integer") from e
    if "batch-size" in options:
        try:
            overrides["batch_size"] = int(options["batch-size"])
        except ValueError as e:
            raise CommandError("Option --batch-size must be an integer") from e
    if "learning-rate" in options:
        try:
            overrides["learning_rate"] = float(options["learning-rate"])
        except ValueError as e:
            raise CommandError("Option --learning-rate must be a number") from e
    if "validation-split" in options:
        try:
            overrides["validation_split"] = float(options["validation-split"])
        except ValueError as e:
            raise CommandError("Option --validation-split must be a number") from e

    training_config = arc_graph.to_training_config(overrides or None)

    validation_table = options.get("validation-table")
    validation_split = overrides.get(
        "validation_split", training_config.validation_split
    )

    if validation_table and not runtime.ml_data_service.dataset_exists(
        str(validation_table)
    ):
        raise CommandError(
            f"Validation table '{validation_table}' does not exist in user DB"
        )

    checkpoint_dir = options.get("checkpoint-dir")
    checkpoint_path = None
    if checkpoint_dir:
        checkpoint_path = str(Path(str(checkpoint_dir)).expanduser())

    tags_value = options.get("tags")
    tags = None
    if tags_value:
        tags = [tag.strip() for tag in str(tags_value).split(",") if tag.strip()]
        if not tags:
            tags = None

    job_config = TrainingJobConfig(
        model_id=model_record.id,
        model_name=model_record.name,
        arc_graph=arc_graph,
        train_table=str(train_table),
        target_column=str(target_column),
        feature_columns=list(feature_columns),
        validation_table=str(validation_table) if validation_table else None,
        validation_split=validation_split,
        training_config=training_config,
        artifacts_dir=str(runtime.artifacts_root),
        checkpoint_dir=checkpoint_path,
        description=str(options.get("description") or "Training job"),
        tags=tags,
    )

    job_id = runtime.training_service.submit_training_job(job_config)
    ui.show_system_success(
        "Training job submitted. Use /ml jobs status <job_id> to monitor."
    )
    ui.show_info(f"Job ID: {job_id}")


def _ml_predict(args: list[str], ui: InteractiveInterface, runtime: MLRuntime) -> None:
    options = _parse_options(
        args,
        {
            "model": True,
            "data": True,
            "batch-size": True,
            "limit": True,
            "output": True,
            "device": True,
        },
    )

    model_name = options.get("model")
    table_name = options.get("data")

    if not model_name or not table_name:
        raise CommandError("/ml predict requires --model and --data")

    # Validate table exists
    if not runtime.ml_data_service.dataset_exists(str(table_name)):
        raise CommandError(f"Table '{table_name}' does not exist")

    # Load predictor
    predictor = _load_predictor(str(model_name), runtime, options.get("device"))

    # Parse options
    batch_size = 32
    if "batch-size" in options:
        try:
            batch_size = int(options["batch-size"])
        except ValueError as e:
            raise CommandError("Option --batch-size must be an integer") from e

    limit = None
    if "limit" in options:
        try:
            limit = int(options["limit"])
        except ValueError as e:
            raise CommandError("Option --limit must be an integer") from e

    # Run prediction
    try:
        predictions = predictor.predict_from_table(
            ml_data_service=runtime.ml_data_service,
            table_name=str(table_name),
            batch_size=batch_size,
            limit=limit,
        )

        # Show summary
        total_predictions = list(predictions.values())[0].shape[0]
        output_names = list(predictions.keys())

        ui.show_system_success(
            f"Generated {total_predictions} predictions with outputs: "
            f"{', '.join(output_names)}"
        )

        # Optionally save to table
        output_table = options.get("output")
        if output_table:
            _save_predictions_to_table(
                predictions,
                str(output_table),
                runtime,
                predictor,
                str(table_name),
                limit,
            )
            ui.show_system_success(f"Predictions saved to table '{output_table}'")

    except Exception as e:
        raise CommandError(f"Prediction failed: {e}") from e


def _load_predictor(
    model_name: str, runtime: MLRuntime, device: str | None = None
) -> ArcPredictor:
    """Load predictor from model name."""
    # Get model record
    model_record = runtime.model_service.get_latest_model_by_name(model_name)
    if model_record is None:
        raise CommandError(f"Model '{model_name}' not found")

    # Load from artifacts
    artifact_manager = ModelArtifactManager(runtime.artifacts_root)
    try:
        return ArcPredictor.load_from_artifact(
            artifact_manager=artifact_manager,
            model_id=model_record.id,
            device=device or "cpu",
        )
    except Exception as e:
        raise CommandError(f"Failed to load predictor: {e}") from e


def _save_predictions_to_table(
    predictions: dict[str, torch.Tensor],
    table_name: str,
    runtime: MLRuntime,
    predictor: "ArcPredictor",
    source_table: str,
    limit: int | None = None,
) -> None:
    """Save predictions to a database table with original features."""
    try:
        # Get feature columns from Arc-Graph
        feature_columns = predictor.arc_graph.features.feature_columns
        if not feature_columns:
            raise CommandError("No feature columns found in Arc-Graph specification")

        # Use ml_data_service to save predictions with streaming support
        runtime.ml_data_service.save_prediction_results(
            source_table=source_table,
            output_table=table_name,
            feature_columns=feature_columns,
            predictions=predictions,
            batch_size=1000,  # Process in batches of 1000 rows
            limit=limit,
        )

    except Exception as e:
        raise CommandError(f"Failed to save predictions to table: {e}") from e


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

        ml_runtime = MLRuntime(services)

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
                    if user_input.startswith("/ml"):
                        await handle_ml_command(user_input, ui, ml_runtime)
                        continue

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
    finally:
        with suppress(Exception):
            if "ml_runtime" in locals():
                ml_runtime.shutdown()


if __name__ == "__main__":
    cli()
