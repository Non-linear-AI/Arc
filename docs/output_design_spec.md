# Arc Output Design Specification

## Design Philosophy

**Goal**: Ultra-clean, minimal design with clear visual hierarchy using shape characters to differentiate tool types while maintaining a streaming workflow feel.

**Principles**:
- Streaming workflow with narrative markers (`▸`) for context
- Shape characters for tool categorization
- Minimal visual clutter
- Clear information hierarchy through indentation
- Consistent spacing and separators

---

## Visual System

### Narrative Flow

Each workflow step begins with a narrative marker explaining what's happening:

```
▸ [what I'm doing now]
```

This creates a conversation-like flow where the user follows the agent's thought process.

### Tool Output Structure

Tool outputs follow immediately after narrative markers:

```
▸ Narrative explaining the action

  ◆ Tool Name • context
  [tool-specific content]

  summary line
```

**Elements**:
- **Shape**: Category identifier (2 chars wide including space)
- **Tool Name**: Human-readable action name
- **Context** (optional): Metadata in `• context` format
- **Content**: Tool-specific output (indented 2 spaces)
- **Summary**: Result count or status (indented 2 spaces, dim)

---

## Tool Categories & Shapes

### Database Operations: `◆`

**Tools**:
- `database_query` → "SQL • db_name"
- `schema_discovery` → "Schema • db_name" or "Schema • table_name"

**Format**:
```
◆ SQL • user
  SELECT split, outcome, COUNT(*) as count
  FROM diabetes_training_data
  GROUP BY split, outcome

  split        outcome   count
  training     0         210
  training     1         104
  validation   0         52
  validation   1         26

  4 rows
```

**Failed query**:
```
◆ SQL • user
  SELECT * FROM nonexistent_table

  × Table 'nonexistent_table' not found
```

**Schema operations**:
```
◆ Schema • pidd

  column_name          data_type    nullable
  pregnancies          INTEGER      YES
  glucose              INTEGER      YES
  outcome              INTEGER      YES

  8 columns
```

### ML Operations: `●`

**Tools**:
- `ml_model` → "ML Train • model_name"
- `ml_evaluate` → "ML Evaluate • model_name"
- Custom workflow outputs

**Training workflow**:
```
● ML Train • pidd-diabetes-predictor

  → listing knowledge
  → reading: mlp
  → querying data
  ✓ model generated

  name: pidd-diabetes-predictor
  data_table: diabetes_training_data
  inputs:
    features:
      dtype: float32
      ...

  Choose action:
  1. Accept and train
  2. Iterate with feedback
  3. Edit manually
  4. Cancel

  ✓ registered • v1 • 1 inputs • 10 nodes • 2 outputs
  → training started • job 1d605a3a

  Monitor: /ml jobs status 1d605a3a
  TensorBoard: http://localhost:6007
```

**Training complete**:
```
● ML Result • pidd-diabetes-predictor-v1

  Time 7.6s • Epochs 31 • Best epoch 21

  Accuracy   52.1%
  Precision  43.6%
  Recall     33.3%
  F1         37.8%
  AUC        0.50

  Train loss  0.3801
  Val loss    0.4642
```

**Evaluation**:
```
● ML Evaluate • pidd-diabetes-predictor-v1

  → loading model
  → processing test data
  ✓ evaluation complete

  Test Metrics:
  Accuracy   54.2%
  Precision  46.1%
  Recall     35.7%
  F1         40.2%

  Confusion Matrix:
  [detailed matrix]
```

### Data Processing: `◇`

**Tools**:
- `ml_data` → "Data Pipeline • name"

**Format**:
```
◇ Data Pipeline • diabetes-preprocessing

  → generating pipeline
  ✓ pipeline generated

  steps:
    - filter_zeros: Remove zero-valued records
    - standardize: StandardScaler normalization
    - train_val_split: 80/20 stratified split

  Choose action:
  1. Accept and execute
  2. Iterate with feedback
  3. Edit manually
  4. Cancel

  ✓ executed

  Output tables:
  • diabetes_training_data (392 rows)
```

### Planning & Progress: `◇`

**Tools**:
- `create_todo_list` → "Plan • progress"
- `update_todo_list` → "Plan • progress"

**Format**:
```
◇ Plan • 4/5
  ✓ Explore pidd table schema
  ✓ Analyze outcome distribution
  ✓ Review ML knowledge
  ✓ Create feature pipeline
  → Train ML model
```

**Progress indicators**:
- `✓` Completed
- `→` In progress
- `○` Pending

### File Operations: `■`

**Tools**:
- `view_file` → "Read • path:lines"
- `create_file` → "Create • path"
- `edit_file` → "Edit • path:line"

**Read**:
```
■ Read • src/arc/ml/trainer.py:150-180

  [file content with line numbers]
```

**Write**:
```
■ Create • src/arc/new_feature.py

  ✓ created • 125 lines
```

**Edit**:
```
■ Edit • src/arc/ml/trainer.py:150

  - old_line = "previous code"
  + new_line = "updated code"

  ✓ 1 change
```

### Search Operations: `◎`

**Tools**:
- `search` → "Search • query"

**Code search**:
```
◎ Search • "class.*Agent"

  ✓ 5 matches • 3 files

  src/arc/core/agents/base_agent.py:45
  src/arc/core/agents/model_generator.py:12
  src/arc/core/agents/trainer_generator.py:18
```

**File search**:
```
◎ Find • "**/*agent*.py"

  ✓ 12 files

  src/arc/core/agents/base_agent.py
  src/arc/core/agents/model_generator/agent.py
  ...
```

### Knowledge Operations: `◐`

**Tools**:
- `list_available_knowledge` → "Knowledge Catalog"
- `read_knowledge` → "Knowledge • id"

**List**:
```
◐ Knowledge Catalog

  ✓ 15 documents

  Architectures:
  • mlp - Multi-layer Perceptron
  • dcn - Deep & Cross Network
  • transformer - Attention-based models

  Patterns:
  • feature-interaction - Feature crossing techniques
  • regularization - Overfitting prevention
```

**Read**:
```
◐ Knowledge • mlp

  Multi-layer Perceptron (MLP)

  Architecture:
  - Fully connected layers
  - Non-linear activations (ReLU, tanh)
  - Dropout for regularization

  Use cases:
  - Tabular data classification
  - Regression tasks
  ...
```

### System Operations: `▶`

**Tools**:
- `bash` → "Run • command"

**Success**:
```
▶ Run • uv run pytest tests/

  ✓ 156 passed • 3.2s
```

**Failure**:
```
▶ Run • npm install

  × command failed
  Error: package.json not found
```

**Background**:
```
▶ Run • tensorboard --logdir runs/

  → running in background • PID 82326

  URL: http://localhost:6006
```

---

## Status Indicators

### Universal Symbols

- `✓` Success / Completed
- `→` In progress / Active
- `×` Failed / Error
- `○` Pending / Not started
- `•` Bullet / Separator

### Progress Context

Within tool outputs, use symbols to show workflow steps:

```
● ML Train • model-name

  → listing knowledge     ← active step
  → reading: mlp          ← active step
  ✓ model generated       ← completed
  → training started      ← active step
```

---

## Message Types

### Info Messages

```
ℹ Monitor training progress:
  • Status: /ml jobs status 1d605a3a
  • Logs: /ml jobs logs 1d605a3a
```

### Success Messages

```
✅ Model successfully deployed
   Endpoint: https://api.example.com/predict
```

### Warning Messages

```
⚠️  High memory usage
   Current: 8.2 GB / 16 GB (51%)
   Consider: Reduce batch size
```

### Error Messages

```
❌ Training failed
   Error: CUDA out of memory

   Suggestion: Reduce batch_size from 32 to 16
```

---

## Tables

### Simple Tables (no borders)

For query results and structured data:

```
column_name   data_type    nullable
pregnancies   INTEGER      YES
glucose       INTEGER      YES
outcome       INTEGER      YES
```

### Metrics Tables

For ML results:

```
Accuracy   52.1%
Precision  43.6%
Recall     33.3%
F1         37.8%
AUC        0.50
```

### Status Tables

For job listings:

```
job_id      type          status      created
abc123      train_model   completed   2025-11-02 06:46
def456      evaluate      running     2025-11-02 06:52
```

---

## Spacing & Hierarchy

### Indentation Levels

1. **Narrative marker**: No indentation
2. **Tool header**: 2 spaces (shape + space + name)
3. **Tool content**: 4 spaces (2 from header + 2 content indent)
4. **Nested content**: 6+ spaces

Example:
```
▸ Let me check the data                    ← 0 spaces (narrative)

  ◆ SQL • user                             ← 2 spaces (tool header)
    SELECT * FROM pidd LIMIT 5             ← 4 spaces (content)

    pregnancies  glucose  outcome          ← 4 spaces (table)
    6            148      1                ← 4 spaces

    5 rows                                 ← 4 spaces (summary)
```

### Vertical Spacing

- Blank line between narrative and tool output
- No blank line between tool header and content
- Blank line between tool content and summary
- Blank line after each tool section

---

## Implementation Plan

### Phase 1: Core Infrastructure

**File**: `src/arc/ui/printer.py`

1. Add shape character mapping function
2. Update `section()` context manager to accept shape parameter
3. Modify dot prefix logic to use shapes instead of `⏺`

**File**: `src/arc/ui/console.py`

4. Update `_action_label()` to use new naming convention
5. Remove `_get_dot_color()` - colors no longer needed
6. Update `show_tool_result()` to use shapes

### Phase 2: Tool-Specific Updates

**Database Tools** (`src/arc/tools/database_query.py`, `schema_discovery.py`):
- Update metadata to include shape: `◆`
- Simplify table rendering (no Rich boxes)
- Update summary format

**ML Tools** (`src/arc/tools/ml_data.py`, ML tool modules):
- Update metadata to include shape: `●` or `◇`
- Update progress indicators in workflow
- Simplify result formatting

**File Tools** (`src/arc/tools/file_editor.py`):
- Update metadata to include shape: `■`
- Simplify diff output

**Search Tools** (`src/arc/tools/search.py`):
- Update metadata to include shape: `◎`
- Update result listing format

**Knowledge Tools** (`src/arc/tools/knowledge.py`):
- Update metadata to include shape: `◐`
- Update catalog and content formatting

**System Tools** (`src/arc/tools/bash.py`):
- Update metadata to include shape: `▶`
- Update command output formatting

### Phase 3: Narrative Markers

**File**: `src/arc/ui/console.py`

Add new method:
```python
def show_narrative(self, content: str):
    """Show narrative step marker."""
    with self._printer.section(shape="▸", color="cyan") as p:
        p.print(content)
```

Update agent streaming to emit narrative steps.

### Phase 4: Testing & Refinement

1. Test all tools with new output format
2. Verify spacing and alignment
3. Test color vs no-color terminals
4. Validate accessibility

---

## Migration Strategy

### Backward Compatibility

During migration:
1. Keep old `_get_dot_color()` function
2. Add new shape parameter as optional
3. Gradually migrate tools one by one
4. Remove old color-based system once complete

### Testing Checklist

- [ ] Database queries (SELECT, error cases)
- [ ] Schema discovery (list tables, describe table)
- [ ] ML training workflow (full pipeline)
- [ ] ML evaluation
- [ ] Data processing pipeline
- [ ] File operations (read, write, edit)
- [ ] Search (code, files)
- [ ] Knowledge operations
- [ ] Bash commands (success, failure, background)
- [ ] Planning/todos (create, update)
- [ ] Messages (info, warning, error, success)
- [ ] Tables (simple, metrics, status)

---

## Open Questions

1. **Color support**: Should shapes also have colors for enhanced terminals?
   - Proposal: Shape + color hybrid (shape works standalone, color enhances)

2. **Shape alternatives**: Are current shapes universally supported?
   - Test on: macOS, Linux, Windows Terminal, WSL

3. **Accessibility**: How do screen readers handle shapes?
   - May need alt-text in metadata

4. **Customization**: Should users be able to customize shapes?
   - Could add to config file for power users
