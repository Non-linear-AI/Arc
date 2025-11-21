# Architecture Overview

This document provides an overview of Arc's architecture to help contributors understand the codebase.

## High-Level Architecture

Arc is built on a modular architecture with these main components:

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI Interface                         │
│                 (src/arc/ui/cli.py)                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      Arc Agent                               │
│          (src/arc/core/agent.py)                            │
│   Orchestrates AI interaction loop and tool execution       │
└──────┬──────────────────┬──────────────────┬────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────┐   ┌──────────────┐   ┌─────────────┐
│   Tools     │   │  Specialized │   │  Database   │
│  Registry   │   │  ML Agents   │   │   Layer     │
└─────────────┘   └──────────────┘   └─────────────┘
```

## Core Components

### 1. Agent System (`src/arc/core/`)

**Main Agent** (`agent.py`):
- `ArcAgent` class orchestrates the AI interaction loop
- Manages tool execution with configurable rounds (default: 50)
- Maintains chat history with `ChatEntry` objects
- Handles streaming responses for real-time UI updates

**Specialized ML Agents** (`agents/`):
- `MLPlanAgent`: Generates ML workflow plans
- `MLModelAgent`: Creates and trains models
- `MLDataAgent`: Handles data processing and feature engineering
- `MLEvaluateAgent`: Evaluates model performance

Each agent has:
- System prompt template (Jinja2)
- Available tools specific to its role
- Example workflows for reference

### 2. Tool System (`src/arc/tools/`)

**Tool Registry**:
- Manages tool registration and execution
- Configurable timeouts (ML tools: no timeout for user interaction)
- Tool input/output validation

**Tool Categories**:

**Core Tools**:
- `view_file`: Read file contents
- `create_file`: Create new files
- `edit_file`: Edit existing files
- `bash`: Execute shell commands
- `search`: Search codebase
- `database_query`: Execute SQL queries
- `schema_discovery`: Discover database schema

**ML Tools** (delegate to specialized agents):
- `ml_plan`: Generate ML plan
- `ml_model`: Train model
- `ml_evaluate`: Evaluate model
- `ml_data`: Process data

**Todo Tools**:
- `create_todo_list`: Create task list
- `update_todo_list`: Update tasks
- Shared `TodoManager` state across tools

**Knowledge Tools**:
- `read_knowledge`: Access builtin and user knowledge from `~/.arc/knowledge/`

### 3. Database Layer (`src/arc/database/`)

**Dual Database System**:
- **System DB** (`~/.arc/system.db`): Arc metadata (models, plans, evaluations)
- **User DB** (`~/.arc/user.db`): User data tables

**Service Container Pattern**:
- `MLPlanService`: Manage ML plans
- `ModelService`: Manage models
- `TrainingTrackingService`: Track training runs
- `EvaluationService`: Store evaluation results
- `DataProcessorService`: Manage data processors

**Models**:
- `MLPlan`: ML workflow plan
- `Model`: Trained model metadata
- `Training`: Training run information
- `Evaluation`: Evaluation results
- `DataProcessor`: Data processing pipeline

**Database Technology**:
- DuckDB for both system and user databases
- Supports S3, Snowflake, and local data sources
- SQL-based interface for querying

### 4. ML Workflow (`src/arc/ml/`)

**Runtime** (`runtime.py`):
- `MLRuntime` orchestrates training, prediction, evaluation
- Manages model lifecycle

**Builder** (`builder.py`):
- Builds PyTorch models from Arc-Graph specifications
- Validates layer connections
- Handles multi-input models

**Training** (`training.py`):
- Training loop with epoch management
- Metrics tracking (loss, accuracy, etc.)
- Checkpoint saving
- TensorBoard integration

**Evaluator** (`evaluator.py`):
- Model evaluation with custom metrics
- Classification: accuracy, precision, recall, F1, AUC
- Regression: MSE, RMSE, MAE, R²

**Data Management** (`data.py`):
- Dataset loading from DuckDB tables
- Feature processing
- Train/val/test splits
- PyTorch DataLoader creation

**TensorBoard** (`tensorboard.py`):
- Automatic TensorBoard launch
- Training metrics visualization
- Model architecture graphs

**Processors**:
- Builtin data processors (normalization, encoding, etc.)
- Plugin-based extensibility

### 5. Arc-Graph & Arc-Pipeline (`src/arc/graph/`)

**Arc-Graph**:
- Declarative YAML for ML model architecture
- Components: inputs, graph, outputs, trainer, evaluator
- Validators ensure spec correctness
- Layer types: `torch.nn.*` and `torch.*` format

**Arc-Pipeline**:
- Declarative YAML for feature engineering
- Data loading, transformation, splitting
- Reproducible workflows

**Validation**:
- Schema validation
- Type checking
- Layer compatibility verification

### 6. CLI Interface (`src/arc/ui/`)

**CLI** (`cli.py`):
- Click-based command-line interface
- Interactive and non-interactive modes
- Command routing

**Commands**:
- `/ml`: ML operations (plan, model, evaluate, data, predict, jobs)
- `/sql`: SQL queries
- `/config`: Configuration management
- `/report`: Generate reports
- `/help`: Help information
- `/clear`: Clear context

**InteractiveInterface**:
- UI rendering with Rich library
- Streaming responses
- Escape key interruption support
- Command history

### 7. Plugin System (`src/arc/plugins/`)

Extensibility through plugins:
- Custom data processors
- Custom layer types
- Custom metrics

### 8. Templates (`src/arc/templates/`)

Jinja2 templates for:
- System prompts for agents
- Agent instructions
- Example workflows

### 9. Resources (`src/arc/resources/`)

Built-in knowledge files:
- `knowledge/data_loading.md`: Data loading patterns
- `knowledge/ml_data_preparation.md`: ML data prep
- `knowledge/mlp.md`, `dcn.md`: Model architectures

### 10. Utilities (`src/arc/utils/`)

Helper functions and utilities:
- File operations
- String manipulation
- Configuration management

## Data Flow

### ML Workflow Stages

```
1. User Message
   ↓
2. Arc Agent (LLM + Tools)
   ↓
3. Tool Execution
   │
   ├→ Plan Generation (MLPlanAgent)
   │  └→ Arc-Graph + Arc-Pipeline specs → System DB
   │
   ├→ Data Processing (MLDataAgent)
   │  └→ Execute Arc-Pipeline → Processed tables
   │
   ├→ Model Training (MLModelAgent)
   │  └→ Build PyTorch model → Train → Save model + metadata
   │
   ├→ Evaluation (MLEvaluateAgent)
   │  └→ Run metrics → Store results
   │
   └→ Prediction
      └→ Load model → Inference → Save predictions
   ↓
4. Results back to User
```

### Agent Loop

```
User Message
    ↓
LLM generates response + tool calls
    ↓
Execute tools
    ↓
Feed results back to LLM
    ↓
Repeat until task complete (max 50 rounds)
    ↓
Final response to user
```

### Configuration Management

Settings loaded from:
1. Environment variables (`ARC_API_KEY`, `ARC_BASE_URL`, `ARC_MODEL`)
2. `~/.arc/user-settings.json`

Precedence: Environment variables > Settings file

## Key Design Patterns

### 1. Agent-Based Architecture
- Specialized agents for different ML phases
- Each agent has focused responsibilities
- Agents use tools to accomplish tasks

### 2. Tool Registry Pattern
- Centralized tool management
- Timeout controls per tool
- Input/output validation

### 3. Service Container
- Dependency injection for database services
- Clean separation of concerns
- Testable components

### 4. Declarative ML
- Arc-Graph and Arc-Pipeline YAML specs
- Separate intent from implementation
- Reproducible and portable

### 5. Streaming Responses
- Async generators for real-time UI updates
- Cancellation support (escape key)
- Progressive output

### 6. Dual Database System
- System DB: Arc metadata
- User DB: User data
- Clean separation of concerns

## Extension Points

Contributors can extend Arc at these points:

### 1. Add New Tools
- Implement tool function in `src/arc/tools/`
- Register in tool registry
- Add to appropriate agent's tool list

### 2. Add New Agents
- Create agent directory in `src/arc/core/agents/`
- Implement system prompt template
- Register in agent factory

### 3. Add Built-in Knowledge
- Create markdown file in `src/arc/resources/knowledge/`
- Update `metadata.yaml`
- Test knowledge loading

### 4. Add Layer Types
- Add layer definition in `src/arc/graph/`
- Update validators
- Add tests

### 5. Add Data Processors
- Implement processor in `src/arc/ml/processors/`
- Register in processor registry
- Add tests

### 6. Add Metrics
- Implement metric in `src/arc/ml/evaluator.py`
- Add to evaluator registry
- Add tests

## Testing Architecture

### Test Organization

```
tests/
├── test_core/      # Agent tests
├── test_tools/     # Tool tests
├── test_ml/        # ML runtime tests
├── test_graph/     # Validator tests
├── test_database/  # Database tests
└── test_ui/        # CLI tests
```

### Test Coverage

- Unit tests for individual components
- Integration tests for workflows
- End-to-end tests for complete ML workflows

## Performance Considerations

### 1. Database
- DuckDB: Fast analytical queries
- Columnar storage for efficiency
- S3/Snowflake integrations

### 2. ML Training
- PyTorch: GPU acceleration (when available)
- Batch processing
- Checkpoint saving for resumption

### 3. Tool Execution
- Configurable timeouts
- Parallel tool execution (where safe)
- Caching for expensive operations

## Security Considerations

### 1. API Keys
- Stored in `~/.arc/user-settings.json` (not in code)
- Environment variable support
- Never logged or exposed

### 2. Database
- Local SQLite databases
- User data isolated from system data
- No remote access by default

### 3. Tool Execution
- Bash commands: sanitized inputs
- File operations: path validation
- SQL queries: parameterized queries

## Debugging Tips

### 1. Enable Debug Logging
```bash
ARC_LOG_LEVEL=DEBUG uv run arc chat
```

### 2. Inspect Database
```bash
# System DB
sqlite3 ~/.arc/system.db
# or use DuckDB CLI
duckdb ~/.arc/system.db
```

### 3. View Agent Prompts
Check `src/arc/templates/` for system prompts

### 4. Trace Tool Execution
Add logging in tools to see execution flow

## Next Steps

- **[Development Setup](development-setup.md)** - Set up dev environment and run tests
- **[Contributing Guidelines](../../CONTRIBUTING.md)** - Contribution workflow
