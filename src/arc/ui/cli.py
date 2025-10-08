"""Command-line interface for Arc CLI."""

import asyncio
import json
import os
import shlex
import sys
import time
from pathlib import Path

import click
from dotenv import load_dotenv

from arc.core import ArcAgent, SettingsManager
from arc.core.agents.predictor_generator import PredictorGeneratorAgent
from arc.database import DatabaseError, DatabaseManager, QueryValidationError
from arc.database.services import ServiceContainer
from arc.graph.features.data_source import DataSourceSpec
from arc.ml.runtime import MLRuntime, MLRuntimeError
from arc.ui.console import InteractiveInterface
from arc.utils import ConfirmationService
from arc.utils.report import (
    build_issue_url,
    compose_issue_body,
    open_in_browser,
)

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
        # Headless mode requires an API key
        if not api_key:
            ui.show_system_error(
                "API key required for headless mode. Set ARC_API_KEY, "
                "use --api-key, or run /config in interactive mode."
            )
            sys.exit(1)
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
    user_input: str,
    ui: InteractiveInterface,
    runtime: "MLRuntime",
    agent: "ArcAgent | None" = None,
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
            "Usage: /ml <plan|revise-plan|create-model|train|predict|jobs|"
            "generate-model|generate-trainer|generate-predictor|data-processing> ..."
        )
        return

    subcommand = tokens[1]
    args = tokens[2:]

    try:
        if subcommand == "plan":
            await _ml_plan(args, ui, agent)
        elif subcommand == "revise-plan":
            await _ml_revise_plan(args, ui, agent)
        elif subcommand == "create-model":
            _ml_create_model(args, ui, runtime)
        elif subcommand == "create-trainer":
            _ml_create_trainer(args, ui, runtime)
        elif subcommand == "train":
            await _ml_train(args, ui, runtime)
        elif subcommand == "predict":
            _ml_predict(args, ui, runtime)
        elif subcommand == "jobs":
            _ml_jobs(args, ui, runtime)
        elif subcommand == "generate-model":
            await _ml_generate_model(args, ui, runtime, agent)
        elif subcommand == "generate-trainer":
            await _ml_generate_trainer(args, ui, runtime)
        elif subcommand == "generate-predictor":
            await _ml_generate_predictor(args, ui, runtime)
        elif subcommand == "data-processing":
            await _ml_data_processing(args, ui, runtime)
        else:
            raise CommandError(f"Unknown ML command: {subcommand}")
    except CommandError as e:
        ui.show_system_error(str(e))
    except Exception as e:
        ui.show_system_error(f"ML command failed: {e}")


async def _ml_plan(
    args: list[str], ui: InteractiveInterface, agent: "ArcAgent | None"
) -> None:
    """Create an ML workflow plan."""
    if not agent:
        raise CommandError("Agent not available for ML planning")

    if not agent.ml_plan_tool:
        raise CommandError(
            "ML plan tool not available. Database services not initialized."
        )

    options = _parse_options(
        args,
        {
            "context": True,
            "data-source": True,
        },
    )

    user_context = options.get("context")
    source_tables = options.get("data-source")

    if not user_context or not source_tables:
        raise CommandError("/ml plan requires --context and --data-source")

    ui.show_info("🤖 Analyzing problem and creating ML workflow plan...")

    # Prepare conversation history
    conversation_history = agent._prepare_conversation_for_ml_plan()

    # Execute ML plan tool
    result = await agent.ml_plan_tool.execute(
        user_context=str(user_context),
        source_tables=str(source_tables),
        conversation_history=conversation_history,
        feedback=None,
        previous_plan=agent.current_ml_plan,
    )

    if result.success:
        # Store the new plan
        if result.metadata and "ml_plan" in result.metadata:
            agent.current_ml_plan = result.metadata["ml_plan"]

        # Display assistant's question
        if result.output:
            ui.show_info(result.output)
    else:
        raise CommandError(f"Failed to create ML plan: {result.error}")


async def _ml_revise_plan(
    args: list[str], ui: InteractiveInterface, agent: "ArcAgent | None"
) -> None:
    """Revise the current ML workflow plan based on feedback."""
    if not agent:
        raise CommandError("Agent not available for ML planning")

    if not agent.ml_plan_tool:
        raise CommandError(
            "ML plan tool not available. Database services not initialized."
        )

    if not agent.current_ml_plan:
        raise CommandError(
            "No current ML plan to revise. Create a plan first with /ml plan"
        )

    options = _parse_options(
        args,
        {
            "feedback": True,
        },
    )

    feedback = options.get("feedback")

    if not feedback:
        raise CommandError("/ml revise-plan requires --feedback")

    ui.show_info("🤖 Revising ML plan based on feedback...")

    # Get context from current plan
    from arc.core.ml_plan import MLPlan

    current_plan_obj = MLPlan.from_dict(agent.current_ml_plan)
    user_context = current_plan_obj.summary
    data_table = agent.current_ml_plan.get("data_table", "")
    target_column = agent.current_ml_plan.get("target_column", "")

    # If data_table/target_column not in plan, we need to ask for them
    if not data_table or not target_column:
        raise CommandError(
            "Current plan missing data table or target column information. "
            "Please create a new plan with /ml plan"
        )

    # Prepare conversation history
    conversation_history = agent._prepare_conversation_for_ml_plan()

    # Execute ML plan tool with feedback
    result = await agent.ml_plan_tool.execute(
        user_context=user_context,
        data_table=data_table,
        target_column=target_column,
        conversation_history=conversation_history,
        feedback=str(feedback),
        previous_plan=agent.current_ml_plan,
    )

    if result.success:
        # Store the revised plan
        if result.metadata and "ml_plan" in result.metadata:
            agent.current_ml_plan = result.metadata["ml_plan"]

        # Display assistant's question
        if result.output:
            ui.show_info(result.output)
    else:
        raise CommandError(f"Failed to revise ML plan: {result.error}")


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


def _ml_create_trainer(
    args: list[str], ui: InteractiveInterface, runtime: "MLRuntime"
) -> None:
    options = _parse_options(
        args,
        {
            "name": True,
            "schema": True,
            "model-id": True,
        },
    )

    name = options.get("name")
    schema_path = options.get("schema")
    model_id = options.get("model-id")

    if not name or not schema_path or not model_id:
        raise CommandError(
            "/ml create-trainer requires --name, --schema, and --model-id"
        )

    schema_path_obj = Path(str(schema_path))

    try:
        trainer = runtime.create_trainer(
            name=str(name),
            schema_path=schema_path_obj,
            model_id=str(model_id),
        )
    except MLRuntimeError as exc:
        raise CommandError(str(exc)) from exc

    ui.show_system_success(
        f"Trainer '{trainer.name}' registered "
        f"(version {trainer.version}, id={trainer.id})."
    )
    ui.show_info(f"Linked to model: {trainer.model_id}")


async def _ml_train(
    args: list[str], ui: InteractiveInterface, runtime: "MLRuntime"
) -> None:
    """Handle training command with trainer generation.

    Always generates a trainer YAML file and launches training.
    Supports two modes:
    1. With plan: Uses plan's training instructions
    2. With context: Uses user-provided training context
    """
    options = _parse_options(
        args,
        {
            "name": True,
            "model-id": True,
            "data": True,
            "context": True,
            "plan-id": True,
        },
    )

    name = options.get("name")
    model_id = options.get("model-id")
    train_table = options.get("data")
    context = options.get("context")
    plan_id = options.get("plan-id")

    # Validate required parameters
    if not name:
        raise CommandError("/ml train requires --name")

    if not model_id:
        raise CommandError("/ml train requires --model-id")

    if not train_table:
        raise CommandError("/ml train requires --data")

    # Must provide either context or plan-id (mutually exclusive)
    if not context and not plan_id:
        raise CommandError("/ml train requires either --context or --plan-id")

    if context and plan_id:
        raise CommandError(
            "/ml train cannot use both --context and --plan-id (choose one)"
        )

    # If using plan, extract training instructions and embed into context
    training_context = context
    if plan_id:
        try:
            # Get the plan from database
            db_plan = runtime.services.ml_plan.get_plan_by_id(str(plan_id))
            if not db_plan:
                raise CommandError(f"Plan '{plan_id}' not found")

            # Parse plan YAML to extract training instructions
            import yaml

            plan_data = yaml.safe_load(db_plan.plan_yaml)

            # Extract training-related information from plan
            training_instructions = []

            if "training" in plan_data:
                training_section = plan_data["training"]
                training_instructions.append("Training Strategy from Plan:")
                training_instructions.append(
                    yaml.dump(training_section, default_flow_style=False)
                )

            if "rationale" in plan_data:
                training_instructions.append("\nPlan Rationale:")
                training_instructions.append(plan_data["rationale"])

            if training_instructions:
                training_context = "\n".join(training_instructions)
            else:
                # Fallback: use entire plan as context
                training_context = (
                    f"Follow the training strategy from plan '{plan_id}':\n"
                    f"{db_plan.plan_yaml}"
                )

            ui.show_info(f"📊 Using ML plan: {plan_id}")
        except Exception as e:
            raise CommandError(f"Failed to load plan '{plan_id}': {e}") from e

    try:
        # Use the MLTrainerTool which includes confirmation workflow
        from arc.tools.ml import MLTrainerTool

        # Get settings for tool initialization
        settings_manager = SettingsManager()
        api_key = settings_manager.get_api_key()
        base_url = settings_manager.get_base_url()
        model = settings_manager.get_current_model()

        if not api_key:
            raise CommandError("API key required for trainer generation")

        # Create the tool with proper dependencies
        tool = MLTrainerTool(runtime.services, runtime, api_key, base_url, model, ui)

        # Execute the tool with confirmation workflow
        # This will generate trainer, confirm, register, and launch training
        result = await tool.execute(
            name=name,
            context=training_context,
            model_id=model_id,
            train_table=train_table,
            train_immediately=True,  # Always train after generation
        )

        if not result.success:
            raise CommandError(f"Training failed: {result.error}")

        # Success message already shown by tool

    except Exception as exc:
        raise CommandError(f"Unexpected error during training: {exc}") from exc


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

        # If this is a training job, show training metrics and TensorBoard info
        if job_type == "training" and runtime.services.training_tracking:
            tracking_service = runtime.services.training_tracking

            # Get training run by job_id
            training_run = tracking_service.get_run_by_job_id(job_id)

            if training_run:
                rows.append(["", ""])  # Separator
                rows.append(["Training Run ID", training_run.run_id])
                if training_run.model_id:
                    rows.append(["Model", training_run.model_id])
                if training_run.trainer_id:
                    rows.append(["Trainer", training_run.trainer_id])

                # Show TensorBoard info
                if training_run.tensorboard_enabled:
                    rows.append(["TensorBoard", "Enabled"])
                    if training_run.tensorboard_log_dir:
                        rows.append(["  Log Directory", training_run.tensorboard_log_dir])
                        rows.append(["  Command", f"tensorboard --logdir {training_run.tensorboard_log_dir}"])

                # Get latest metrics
                metrics = tracking_service.get_metrics(training_run.run_id, limit=10)

                if metrics:
                    rows.append(["", ""])  # Separator
                    rows.append(["Recent Metrics", f"(latest {len(metrics)})"])

                    # Group by metric name for better display
                    from collections import defaultdict
                    metric_groups = defaultdict(list)
                    for metric in metrics:
                        key = f"{metric.metric_name} ({metric.metric_type.value})"
                        metric_groups[key].append(metric)

                    for metric_name, metric_list in metric_groups.items():
                        latest = metric_list[0]  # Most recent
                        rows.append([
                            f"  {metric_name}",
                            f"{latest.value:.6f} (epoch {latest.epoch}, step {latest.step})"
                        ])

        ui.show_key_values("Job Status", rows)
    else:
        raise CommandError(f"Unknown jobs subcommand: {sub}")


async def _ml_generate_model(
    args: list[str],
    ui: InteractiveInterface,
    runtime: "MLRuntime",
    agent: "ArcAgent | None" = None,  # noqa: ARG001
) -> None:
    """Handle model specification generation command."""
    options = _parse_options(
        args,
        {
            "name": True,
            "context": True,
            "data-table": True,
            "target-column": True,  # Target column for task-aware generation
            "plan-id": True,  # ML plan ID to use for guidance
        },
    )

    name = options.get("name")
    context = options.get("context")
    data_table = options.get("data-table")
    target_column = options.get("target-column")
    plan_id = options.get("plan-id")

    # If plan-id is provided, fetch the plan from database
    ml_plan = None
    if plan_id:
        try:
            # Fetch plan from database
            db_plan = runtime.services.ml_plans.get_plan_by_id(str(plan_id))
            if not db_plan:
                raise CommandError(f"Plan '{plan_id}' not found in database")

            # Parse YAML to dict for the tool
            import yaml

            ml_plan = yaml.safe_load(db_plan.plan_yaml)
            ml_plan["plan_id"] = db_plan.plan_id  # Ensure plan_id is in the dict

            ui.show_info(f"📊 Using ML plan: {plan_id}")
        except Exception as e:
            raise CommandError(f"Failed to load plan '{plan_id}': {e}") from e

    # Validate required parameters
    if not name:
        raise CommandError("/ml generate-model requires --name")

    if not data_table:
        raise CommandError("/ml generate-model requires --data-table")

    # context is optional when using a plan
    if not ml_plan and not context:
        raise CommandError(
            "/ml generate-model requires --context when not using --plan-id"
        )

    try:
        # Use the MLModelGeneratorTool which includes confirmation workflow
        from arc.tools.ml import MLModelGeneratorTool

        # Get settings for tool initialization
        settings_manager = SettingsManager()
        api_key = settings_manager.get_api_key()
        base_url = settings_manager.get_base_url()
        model = settings_manager.get_current_model()

        if not api_key:
            raise CommandError("API key required for model generation")

        # Create the tool with proper dependencies
        tool = MLModelGeneratorTool(runtime.services, api_key, base_url, model, ui)

        # Execute the tool with confirmation workflow
        result = await tool.execute(
            name=name,
            context=context,
            data_table=data_table,
            target_column=target_column,
            ml_plan=ml_plan,  # Pass ML plan if available
        )

        if not result.success:
            raise CommandError(f"Model generation failed: {result.error}")

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
            "model": True,
        },
    )

    name = options.get("name")
    context = options.get("context")
    model_name = options.get("model")

    if not name or not context or not model_name:
        raise CommandError(
            "/ml generate-trainer requires --name, --context, and --model"
        )

    try:
        # Use the MLTrainerGeneratorTool which includes confirmation workflow
        from arc.tools.ml import MLTrainerGeneratorTool

        # Get settings for tool initialization
        settings_manager = SettingsManager()
        api_key = settings_manager.get_api_key()
        base_url = settings_manager.get_base_url()
        model = settings_manager.get_current_model()

        if not api_key:
            raise CommandError("API key required for trainer generation")

        # Create the tool with proper dependencies
        tool = MLTrainerGeneratorTool(runtime.services, api_key, base_url, model, ui)

        # Execute the tool with confirmation workflow (auto-registers to DB)
        result = await tool.execute(
            name=name,
            context=context,
            model_name=model_name,
        )

        if result.success:
            # Suggest next steps
            ui.show_info("\n💡 Trainer registered successfully")
            ui.show_info(
                f"   To train: /ml train --name <trainer_name> --model {model_name} "
                f"--context <description> --data <table_name>"
            )
        else:
            ui.show_system_error(result.message)

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


async def _ml_data_processing(
    args: list[str], ui: InteractiveInterface, runtime: "MLRuntime"
) -> None:
    """Handle data processing pipeline execution command."""
    options = _parse_options(
        args,
        {
            "yaml": True,
            "target-db": True,
        },
    )

    yaml_path = options.get("yaml")
    target_db = options.get("target-db", "user")

    if not yaml_path:
        raise CommandError(
            "/ml data-processing requires --yaml <path> to specify the pipeline file"
        )

    # Validate target database
    if target_db not in ["system", "user"]:
        raise CommandError(
            "Invalid target database. Use --target-db system or --target-db user"
        )

    yaml_file = Path(str(yaml_path))
    if not yaml_file.exists():
        raise CommandError(f"Data processing file not found: {yaml_path}")

    try:
        ui.show_info(f"Data Processing: {yaml_path}")

        # Parse the YAML specification
        spec = DataSourceSpec.from_yaml_file(str(yaml_file))

        ui.show_info(f"Pipeline loaded: {len(spec.steps)} steps")

        # Execute the pipeline using the centralized executor
        from arc.ml.data_source_executor import execute_data_source_pipeline

        # Define progress callback for CLI usage
        def cli_progress(message: str, level: str):
            """Handle progress updates for CLI."""
            if level == "success":
                ui.show_system_success(message)
            elif level == "warning":
                ui.show_warning(message)
            elif level == "error":
                ui.show_system_error(message)
            elif level == "step":
                ui.show_info(f"  {message}")
            else:  # "info"
                ui.show_info(message)

        await execute_data_source_pipeline(
            spec, str(target_db), runtime.services.db_manager, cli_progress
        )

    except ValueError as parse_error:
        raise CommandError(
            f"Invalid YAML specification: {str(parse_error)}"
        ) from parse_error
    except Exception as exc:
        raise CommandError(f"Data processing failed: {exc}") from exc


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
        agent = ArcAgent(api_key, base_url, model, max_tool_rounds, services, None)

        # Configure confirmation service for headless mode (singleton)
        confirmation_service = ConfirmationService.get_instance()
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
    api_key: str | None,
    base_url: str | None,
    model: str | None,
    max_tool_rounds: int,
    services: ServiceContainer,
):
    """Run in interactive mode with enhanced UX."""
    try:
        ui = InteractiveInterface()
        settings_manager = SettingsManager()

        # Explicitly prompt to configure if no settings file exists
        if not settings_manager.settings_file.exists():
            ui.show_warning("No configuration found at ~/.arc/user-settings.json")
            resp = (
                (
                    await ui.get_user_input_async(
                        "Configure API key, Base URL, and Model now? (Y/n): "
                    )
                )
                .strip()
                .lower()
            )
            if not resp or resp.startswith("y"):
                # Collect values; allow skip
                new_api = await ui.get_user_input_async(
                    "API key (leave blank to skip): "
                )
                if new_api.strip():
                    settings_manager.update_user_setting("apiKey", new_api.strip())
                    api_key = new_api.strip()
                    ui.show_system_success("API key saved to ~/.arc/user-settings.json")
                new_url = await ui.get_user_input_async(
                    f"Base URL [{base_url or ''}]: "
                )
                if new_url.strip():
                    settings_manager.update_user_setting("baseURL", new_url.strip())
                    base_url = new_url.strip()
                    ui.show_system_success(
                        "Base URL saved to ~/.arc/user-settings.json"
                    )
                new_model = await ui.get_user_input_async(f"Model [{model or ''}]: ")
                if new_model.strip():
                    settings_manager.update_user_setting("model", new_model.strip())
                    model = new_model.strip()
                    ui.show_system_success("Model saved to ~/.arc/user-settings.json")

        # Initialize agent only if API key is available
        agent: ArcAgent | None = None
        if api_key:
            agent = ArcAgent(api_key, base_url, model, max_tool_rounds, services, ui)
        from contextlib import suppress

        with suppress(Exception):
            ConfirmationService.get_instance().set_ui(ui)

        # Database context for SQL commands - defaults to system database
        current_database = "system"

        # Show enhanced welcome screen
        # Provide welcome even if agent is not yet initialized
        current_model_name = (
            agent.get_current_model() if agent else (model or "Not set")
        )
        current_dir = agent.get_current_directory() if agent else os.getcwd()
        ui.show_welcome(current_model_name, current_dir)

        while True:
            try:
                # Get user input with styled prompt
                user_input = await ui.get_user_input_async()

                # Display the user message in chat history with different coloring
                ui.show_user_message(user_input)

                # Handle system commands (only with / prefix)
                if user_input.startswith("/"):
                    if user_input.startswith("/ml"):
                        await handle_ml_command(
                            user_input, ui, services.ml_runtime, agent
                        )
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
                        # Show and optionally edit configuration
                        settings_manager = SettingsManager()
                        current_api_key = settings_manager.get_api_key()
                        current_base_url = settings_manager.get_base_url()
                        current_model = settings_manager.get_current_model()

                        config_text = (
                            f"API Key: {'*' * 8 if current_api_key else 'Not set'}\n"
                            f"Base URL: {current_base_url or 'Not set'}\n"
                            f"Model: {current_model or 'Not set'}\n"
                            f"Max Tool Rounds: {max_tool_rounds}"
                        )
                        ui.show_config_panel(config_text)

                        # Offer inline editing of baseURL, model, and apiKey
                        edit_resp = (
                            (
                                await ui.get_user_input_async(
                                    "Edit configuration values now? (y/N): "
                                )
                            )
                            .strip()
                            .lower()
                        )
                        if edit_resp.startswith("y"):
                            # Note: environment variables override settings at runtime.
                            # Editing here updates ~/.arc/user-settings.json.
                            new_api = await ui.get_user_input_async(
                                "API key (leave blank to keep current): "
                            )
                            if new_api.strip():
                                settings_manager.update_user_setting(
                                    "apiKey", new_api.strip()
                                )
                                ui.show_system_success(
                                    "API key saved to ~/.arc/user-settings.json"
                                )

                            new_url = await ui.get_user_input_async(
                                f"Base URL [{current_base_url or ''}]: "
                            )
                            if new_url.strip():
                                settings_manager.update_user_setting(
                                    "baseURL", new_url.strip()
                                )
                                ui.show_system_success(
                                    "Base URL saved to ~/.arc/user-settings.json"
                                )

                            new_model = await ui.get_user_input_async(
                                f"Model [{current_model or ''}]: "
                            )
                            if new_model.strip():
                                settings_manager.update_user_setting(
                                    "model", new_model.strip()
                                )
                                ui.show_system_success(
                                    "Model saved to ~/.arc/user-settings.json"
                                )

                            # Refresh and show the updated configuration
                            updated_api = settings_manager.get_api_key()
                            updated_base = settings_manager.get_base_url()
                            updated_model = settings_manager.get_current_model()
                            updated_text = (
                                f"API Key: {'*' * 8 if updated_api else 'Not set'}\n"
                                f"Base URL: {updated_base or 'Not set'}\n"
                                f"Model: {updated_model or 'Not set'}\n"
                                f"Max Tool Rounds: {max_tool_rounds}"
                            )
                            ui.show_config_panel(updated_text)

                            # Initialize agent if missing and API key set
                            if agent is None and updated_api:
                                try:
                                    nonlocal_agent = ArcAgent(
                                        updated_api,
                                        updated_base,
                                        updated_model,
                                        max_tool_rounds,
                                        services,
                                        ui,
                                    )
                                    agent = nonlocal_agent
                                    ui.show_system_success(
                                        "AI chat is now enabled with the configured "
                                        "settings."
                                    )
                                except Exception as init_exc:
                                    ui.show_system_error(
                                        f"Failed to initialize agent: {init_exc}"
                                    )
                        continue
                    elif cmd.startswith("sql"):
                        # Handle SQL queries and update current database context
                        current_database = await handle_sql_command(
                            services.query, ui, user_input, current_database
                        )
                        continue
                    elif cmd.startswith("report"):
                        # Simple interactive GitHub issue reporter
                        # Optional initial title after /report
                        initial = user_input[7:].strip() if len(user_input) > 7 else ""
                        ui.show_info(
                            "This will open a prefilled GitHub issue in your browser."
                        )
                        title_prompt = "Issue title (optional): "
                        title = await ui.get_user_input_async(title_prompt)
                        if not title and initial:
                            title = initial

                        desc = await ui.get_user_input_async(
                            "Brief description (one line): "
                        )

                        try:
                            model_name = agent.get_current_model()
                        except Exception:
                            model_name = None
                        body = compose_issue_body(desc, model=model_name)
                        issue_url = build_issue_url(title, body)

                        confirm = (
                            (
                                await ui.get_user_input_async(
                                    "Open browser to create the issue? (Y/n): "
                                )
                            )
                            .strip()
                            .lower()
                        )
                        yes = (not confirm) or confirm.startswith("y")
                        if yes:
                            opened = open_in_browser(issue_url)
                            if opened:
                                ui.show_system_success(
                                    "Opened browser to GitHub issues page."
                                )
                            else:
                                ui.show_warning(
                                    "Could not open browser. Please use the URL below."
                                )
                        else:
                            ui.show_info("Okay, not opening the browser.")

                        ui.show_info("You can also open this URL to file the issue:")
                        ui.show_info(issue_url)
                        continue
                    else:
                        ui.show_system_error(f"Unknown system command: /{cmd}")
                        continue

                # No special exit without slash; only /exit is supported

                if not user_input:
                    continue

                # If no agent and user typed free text, require configuration
                if not user_input.startswith("/") and agent is None:
                    ui.show_system_error(
                        "API key not configured. Use /config to set apiKey, "
                        "baseURL, and model."
                    )
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
