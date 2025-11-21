# Arc-Knowledge

> **Arc-Knowledge** is Arc's curated collection of ML best practices, patterns, and examples. It guides the AI in generating optimal Arc-Graph and Arc-Pipeline specifications. You can extend it with your own project-specific knowledge.

## 1. Overview

Arc-Knowledge is what makes Arc "AI-native". When you describe what you want in natural language, Arc's AI consults this knowledge to generate production-ready specifications.

### What is Arc-Knowledge?

Arc-Knowledge is a collection of **markdown documents** containing:
- **Data patterns** - How to load and process different data formats
- **Feature engineering** - ML-specific transformation techniques
- **Model architectures** - Proven neural network patterns (DCN, MLP, etc.)
- **Best practices** - Optimization tips and performance patterns

### How Does It Work?

```
Your Request → Arc AI reads relevant knowledge → Generates Arc-Graph/Arc-Pipeline
              ↓
        Built-in Knowledge + Your Custom Knowledge (~/.arc/knowledge/)
```

Arc's AI:
1. **Understands your request** - Parses your natural language description
2. **Selects relevant knowledge** - Chooses applicable patterns from the knowledge base
3. **Applies best practices** - Uses proven patterns instead of generic solutions
4. **Generates specifications** - Creates Arc-Graph and Arc-Pipeline YAML

## 2. How It Works in Code

### 2.1 Knowledge Loading System

Arc loads knowledge from two locations with automatic discovery:

```
Builtin:  src/arc/resources/knowledge/    # Bundled with package
User:     ~/.arc/knowledge/                # Your custom knowledge
```

**Priority:** User knowledge overrides builtin knowledge when the same knowledge ID exists.

### 2.2 File Structure

Knowledge consists of two components:

**1. Metadata File (metadata.yaml)**

A single `metadata.yaml` file at the base path defines all knowledge documents:

```yaml
# src/arc/resources/knowledge/metadata.yaml
mlp:
  name: "Multi-Layer Perceptron"
  description: "Feedforward neural network for tabular data prediction"
  phases: ["model"]

dcn:
  name: "Deep & Cross Network"
  description: "Combines explicit feature crossing with deep representation learning"
  phases: ["model"]

data_loading:
  name: "Data Loading with DuckDB"
  description: "Generic data loading patterns for CSV, Parquet, JSON files"
  phases: ["data"]

ml_data_preparation:
  name: "ML Data Preparation"
  description: "ML-specific data preparation: splits, normalization, encoding"
  phases: ["data"]
```

**Metadata Fields:**
- `name` (required) - Display name for the knowledge document
- `description` (required) - What patterns/techniques it covers
- `phases` (required) - Which ML phases it applies to: `["data", "model", "trainer", "evaluator"]`

**2. Knowledge Files ({knowledge_id}.md)**

Markdown files named by their knowledge ID:

```
src/arc/resources/knowledge/
  ├── metadata.yaml              # Single metadata file
  ├── mlp.md                     # Knowledge ID: mlp
  ├── dcn.md                     # Knowledge ID: dcn
  ├── data_loading.md            # Knowledge ID: data_loading
  └── ml_data_preparation.md     # Knowledge ID: ml_data_preparation
```

### 2.3 Knowledge Discovery

Arc's AI agents use tools to discover and read knowledge:

```python
# 1. List available knowledge
list_available_knowledge()  # Returns categorized list by phase

# 2. Read specific knowledge
read_knowledge(knowledge_id="dcn")  # Returns markdown content
```

**Caching:** Knowledge content is cached after first read to avoid repeated file I/O.

## 3. Built-in Knowledge

Arc includes curated knowledge for common ML scenarios:

### Data Processing (phases: ["data"])
- **data_loading** - CSV, Parquet, JSON patterns with DuckDB
- **ml_data_preparation** - Train/val splits, normalization, encoding

### Model Architectures (phases: ["model"])
- **mlp** - Multi-layer perceptron for tabular data
- **dcn** - Deep & Cross Network for feature interactions

Each includes:
- When to use it
- Complete working examples (Arc-Graph and Arc-Pipeline)
- Best practices and hyperparameter guidance

## 4. Adding Custom Knowledge

### 4.1 Quick Start

Create knowledge files in `~/.arc/knowledge/`:

```bash
mkdir -p ~/.arc/knowledge
```

**Step 1: Create metadata.yaml**

```yaml
# ~/.arc/knowledge/metadata.yaml
financial_features:
  name: "Financial Feature Engineering"
  description: "Stock market and trading-specific feature patterns"
  phases: ["data"]

company_recommendation:
  name: "Company Recommendation Architecture"
  description: "Our standard two-tower recommendation model"
  phases: ["model"]
```

**Step 2: Create knowledge files**

```bash
# Knowledge files must match the IDs in metadata.yaml
touch ~/.arc/knowledge/financial_features.md
touch ~/.arc/knowledge/company_recommendation.md
```

**Step 3: Write knowledge content**

```markdown
# ~/.arc/knowledge/financial_features.md

# Financial Feature Engineering

## Price Movement Features

When working with stock data, calculate price changes:

```yaml
steps:
  - name: price_features
    type: view
    depends_on: [stock_prices]
    sql: |
      SELECT
        ticker,
        date,
        close_price,
        close_price - LAG(close_price, 1) OVER (
          PARTITION BY ticker ORDER BY date
        ) as price_change_1d
      FROM "stock_prices"
```

## When to Use

- Working with financial time-series data
- Predicting stock movements
- Building trading signals
```

### 4.2 Knowledge File Template

```markdown
# [Pattern Name]

## Overview

Brief description of what this pattern does and why it's useful.

## Example

```yaml
# Complete working example (Arc-Pipeline or Arc-Graph)
```

## When to Use

- Scenario 1
- Scenario 2

## Common Mistakes

❌ **Don't do this:**
```sql
-- Anti-pattern
```

✅ **Instead do this:**
```sql
-- Correct pattern
```
```

### 4.3 Best Practices

**Complete Examples**
```markdown
# ✅ Good: Complete, working example
```yaml
steps:
  - name: user_features
    type: view
    depends_on: [raw_users]
    sql: |
      SELECT user_id, COUNT(*) as total_purchases
      FROM "raw_users"
      GROUP BY user_id
```

# ❌ Bad: Incomplete snippet
```yaml
SELECT user_id, COUNT(*) ...
```
```

**Explain the "Why"**
```markdown
# ✅ Good: Explains rationale
## Hash-based Splitting

We use hash-based splitting instead of random sampling because:
- Ensures same user always in same split (prevents data leakage)
- Reproducible across runs

# ❌ Bad: No context
Use `hash(user_id) % 10` for splits.
```

## 5. Knowledge Priority and Overrides

When multiple knowledge sources exist:

**Priority Order:**
1. `~/.arc/knowledge/` (your custom knowledge) - **Highest priority**
2. `src/arc/resources/knowledge/` (built-in knowledge) - Fallback

**Override Behavior:**

If you create `~/.arc/knowledge/mlp.md` with a `mlp` entry in your `metadata.yaml`, it completely replaces the built-in `mlp` knowledge. Arc will use your version instead.

**Use cases:**
- Company-specific model architectures
- Domain-specific feature engineering patterns
- Team conventions that differ from built-in best practices

## 6. Phases and Knowledge Selection

Arc filters knowledge by ML phase to provide relevant context:

| Phase | Purpose | Example Knowledge |
|-------|---------|-------------------|
| `data` | Data loading and feature engineering | data_loading, ml_data_preparation |
| `model` | Model architecture design | mlp, dcn |
| `trainer` | Training configuration | (future) |
| `evaluator` | Model evaluation | (future) |

When Arc's data agent runs, it only sees knowledge with `phases: ["data"]`. When the model agent runs, it sees `phases: ["model"]`.

**Multi-phase knowledge:**
```yaml
hybrid_approach:
  name: "End-to-End Recommendation Pipeline"
  description: "Complete pipeline from data prep to model"
  phases: ["data", "model"]  # Visible to both agents
```

## 7. Troubleshooting

### Knowledge not loading?

**Check:**
1. File is in `~/.arc/knowledge/` directory
2. `metadata.yaml` exists at `~/.arc/knowledge/metadata.yaml`
3. Knowledge ID in metadata matches filename (e.g., `mlp.md` → `mlp:`)
4. File has `.md` extension
5. Metadata YAML is valid

**Debug:**
```bash
# Check if metadata.yaml is valid
cat ~/.arc/knowledge/metadata.yaml

# Verify knowledge files exist
ls ~/.arc/knowledge/*.md
```

### How do I know what knowledge is available?

Within Arc chat, use:
```
Use the list_available_knowledge tool
```

This will show all discovered knowledge documents categorized by phase.

### Can I see what knowledge Arc is using?

Currently, Arc doesn't log which knowledge it consulted. To verify:
1. Check if generated Arc-Graph/Arc-Pipeline specs match your custom patterns
2. Look for patterns/naming conventions from your knowledge

## 8. Version Control

Keep your custom knowledge in Git:

```bash
# Add custom knowledge to your project
git add ~/.arc/knowledge/
git commit -m "Add domain-specific knowledge"

# Or create project-specific knowledge
# (future feature: .arc/knowledge/ in project directory)
```

**Team Collaboration:**
```
~/.arc/knowledge/
  ├── README.md                 # Overview of your team's knowledge
  ├── metadata.yaml             # All knowledge metadata
  ├── data-standards.md         # Team conventions
  └── production-patterns.md    # Proven patterns
```

## 9. Related Documentation

- **[Arc-Graph Specification](arc-graph.md)** - Model architecture YAML schema
- **[Arc-Pipeline Specification](arc-pipeline.md)** - Feature engineering YAML schema
- **[Built-in Data Patterns](../src/arc/resources/knowledge/data_loading.md)** - Data loading knowledge
- **[Built-in ML Preparation](../src/arc/resources/knowledge/ml_data_preparation.md)** - Feature engineering knowledge
