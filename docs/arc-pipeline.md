# Arc-Pipeline Specification

> **Arc-Pipeline** is Arc's declarative YAML schema for defining data processing workflows. It's a general-purpose ETL tool powered by DuckDB, enhanced with ML-specific knowledge for feature engineering and model preparation.

## 1. Overview

**Arc-Pipeline** allows you to define data processing workflows declaratively in YAML. While it can handle any ETL task, Arc's built-in ML knowledge makes it especially powerful for feature engineering and preparing data for machine learning.

### What is Arc-Pipeline?

Arc-Pipeline is a specification that describes:
- **Data Loading** - Import data from CSV, Parquet, JSON, S3, Snowflake
- **Feature Engineering** - Transform raw data into ML-ready features
- **Data Processing** - Train/val/test splits, normalization, encoding
- **Dependency Management** - Define execution order through a DAG

### General ETL with ML Expertise

Arc-Pipeline is a general-purpose ETL tool powered by DuckDB's SQL engine. **What makes it special is built-in ML knowledge** for feature engineering.

**You can use Arc-Pipeline for:**
- General data transformations and ETL workflows
- Business reporting and analytics
- Data warehousing and migrations
- **ML-specific feature engineering** (where Arc's knowledge shines)

**Arc's ML Knowledge provides:**
- Train/validation/test splitting patterns
- Feature normalization and scaling techniques
- Categorical encoding strategies
- Temporal feature extraction
- Data leakage prevention

When you ask Arc to "prepare data for ML training", it uses its knowledge to generate pipelines with ML best practices. For general ETL tasks, it works like any declarative SQL-based pipeline tool.

### Arc-Pipeline Benefits

- **Declarative** - Define what transformations you need, not how to implement them
- **Reproducible** - Same pipeline produces same features every time
- **Versionable** - Track feature engineering logic in Git
- **ML-aware** - Built-in knowledge for ML-specific operations when needed
- **SQL-based** - Leverage DuckDB's powerful SQL engine
- **Flexible** - Use for both general ETL and specialized ML feature engineering

## 2. Basic Structure

An Arc-Pipeline specification consists of:

```yaml
# Arc-Pipeline specification
steps:
  - name: <step_name>
    type: <table|view|execute>
    depends_on: [<parent_steps>]
    sql: |
      <SQL query or command>

outputs: [<final_table_names>]
```

### Step Types

| Type | Purpose | SQL Content | Created Object |
|------|---------|-------------|----------------|
| `table` | Create materialized table | SELECT query only | Persistent table |
| `view` | Create virtual view | SELECT query only | Virtual view |
| `execute` | Run arbitrary SQL | Any DDL/DML | None (side effects) |

**Important:** For `type: table` and `type: view`, the SQL must be a SELECT query only. The CREATE TABLE/VIEW wrapper is added automatically.

## 3. Complete Example: Feature Engineering

```yaml
# Arc-Pipeline: MovieLens feature engineering
steps:
  # 1. Load raw data
  - name: load_ratings
    type: table
    depends_on: []
    sql: |
      SELECT * FROM read_csv('ratings.csv',
        header=true,
        columns={
          'userId': 'INTEGER',
          'movieId': 'INTEGER',
          'rating': 'FLOAT',
          'timestamp': 'INTEGER'
        }
      )

  - name: load_movies
    type: table
    depends_on: []
    sql: |
      SELECT * FROM read_csv('movies.csv',
        header=true,
        columns={
          'movieId': 'INTEGER',
          'title': 'VARCHAR',
          'genres': 'VARCHAR'
        }
      )

  # 2. Feature engineering
  - name: user_features
    type: view
    depends_on: [load_ratings]
    sql: |
      SELECT
        userId,
        COUNT(*) as num_ratings,
        AVG(rating) as avg_rating,
        STDDEV(rating) as rating_std,
        MIN(timestamp) as first_rating_time,
        MAX(timestamp) as last_rating_time
      FROM "load_ratings"
      GROUP BY userId

  - name: movie_features
    type: view
    depends_on: [load_ratings, load_movies]
    sql: |
      SELECT
        m.movieId,
        m.title,
        m.genres,
        COUNT(r.rating) as num_ratings,
        AVG(r.rating) as avg_rating,
        STDDEV(r.rating) as rating_std
      FROM "load_movies" m
      LEFT JOIN "load_ratings" r ON m.movieId = r.movieId
      GROUP BY m.movieId, m.title, m.genres

  # 3. Create training dataset
  - name: training_data
    type: table
    depends_on: [load_ratings, user_features, movie_features]
    sql: |
      SELECT
        r.userId,
        r.movieId,
        r.rating as target,
        u.num_ratings as user_num_ratings,
        u.avg_rating as user_avg_rating,
        u.rating_std as user_rating_std,
        m.num_ratings as movie_num_ratings,
        m.avg_rating as movie_avg_rating,
        m.rating_std as movie_rating_std
      FROM "load_ratings" r
      JOIN "user_features" u ON r.userId = u.userId
      JOIN "movie_features" m ON r.movieId = m.movieId

  # 4. Train/validation split
  - name: train_set
    type: view
    depends_on: [training_data]
    sql: |
      SELECT * FROM "training_data"
      WHERE hash(userId || movieId) % 10 < 8  -- 80% for training

  - name: val_set
    type: view
    depends_on: [training_data]
    sql: |
      SELECT * FROM "training_data"
      WHERE hash(userId || movieId) % 10 >= 8  -- 20% for validation

outputs: [train_set, val_set, training_data]
```

## 4. Common Patterns

### 4.1 Train/Validation/Test Split

```yaml
steps:
  - name: train_set
    type: view
    depends_on: [prepared_data]
    sql: |
      SELECT * FROM "prepared_data"
      WHERE hash(id) % 10 < 7  -- 70% train

  - name: val_set
    type: view
    depends_on: [prepared_data]
    sql: |
      SELECT * FROM "prepared_data"
      WHERE hash(id) % 10 >= 7 AND hash(id) % 10 < 9  -- 20% val

  - name: test_set
    type: view
    depends_on: [prepared_data]
    sql: |
      SELECT * FROM "prepared_data"
      WHERE hash(id) % 10 >= 9  -- 10% test
```

### 4.2 Feature Normalization

```yaml
steps:
  # Calculate statistics
  - name: feature_stats
    type: view
    depends_on: [raw_features]
    sql: |
      SELECT
        AVG(age) as age_mean,
        STDDEV(age) as age_std,
        AVG(income) as income_mean,
        STDDEV(income) as income_std
      FROM "raw_features"

  # Apply normalization
  - name: normalized_features
    type: table
    depends_on: [raw_features, feature_stats]
    sql: |
      SELECT
        id,
        (age - (SELECT age_mean FROM "feature_stats")) /
         (SELECT age_std FROM "feature_stats") as age_normalized,
        (income - (SELECT income_mean FROM "feature_stats")) /
         (SELECT income_std FROM "feature_stats") as income_normalized
      FROM "raw_features"
```

### 4.3 Categorical Encoding

```yaml
steps:
  # Create label encoding mapping
  - name: category_mapping
    type: table
    depends_on: [raw_data]
    sql: |
      SELECT
        category,
        ROW_NUMBER() OVER (ORDER BY category) - 1 as category_id
      FROM (SELECT DISTINCT category FROM "raw_data")

  # Apply encoding
  - name: encoded_data
    type: table
    depends_on: [raw_data, category_mapping]
    sql: |
      SELECT
        r.id,
        r.value,
        m.category_id
      FROM "raw_data" r
      JOIN "category_mapping" m ON r.category = m.category
```

### 4.4 Temporal Features

```yaml
steps:
  - name: temporal_features
    type: view
    depends_on: [raw_events]
    sql: |
      SELECT
        user_id,
        event_time,
        EXTRACT(HOUR FROM event_time) as hour_of_day,
        EXTRACT(DOW FROM event_time) as day_of_week,
        EXTRACT(MONTH FROM event_time) as month,
        event_time - LAG(event_time) OVER (
          PARTITION BY user_id ORDER BY event_time
        ) as time_since_last_event
      FROM "raw_events"
```

## 5. Dependency Management

Arc-Pipeline automatically handles execution order through the `depends_on` field:

```yaml
steps:
  - name: step_a
    depends_on: []  # Runs first

  - name: step_b
    depends_on: [step_a]  # Runs after step_a

  - name: step_c
    depends_on: [step_a]  # Also runs after step_a (parallel with step_b)

  - name: step_d
    depends_on: [step_b, step_c]  # Runs after both step_b and step_c
```

**Execution order:**
1. `step_a` (no dependencies)
2. `step_b` and `step_c` in parallel (both depend only on `step_a`)
3. `step_d` (depends on both `step_b` and `step_c`)

## 6. Best Practices

### 6.1 Use Views for Intermediate Steps

```yaml
# ✅ Good: Views are lightweight
- name: intermediate_transform
  type: view
  sql: SELECT ...

# ❌ Avoid: Tables for temporary results waste space
- name: intermediate_transform
  type: table
  sql: SELECT ...
```

### 6.2 Idempotent Pipelines

Always drop existing tables before creation:

```yaml
steps:
  - name: drop_old_features
    type: execute
    depends_on: []
    sql: DROP TABLE IF EXISTS "features"

  - name: features
    type: table
    depends_on: [drop_old_features]
    sql: SELECT ...
```

### 6.3 Validate Data After Loading

```yaml
steps:
  - name: load_data
    type: table
    sql: SELECT * FROM read_csv('data.csv')

  - name: data_quality_check
    type: view
    depends_on: [load_data]
    sql: |
      SELECT
        COUNT(*) as total_rows,
        SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END) as null_count,
        MIN(value) as min_value,
        MAX(value) as max_value
      FROM "load_data"
```

### 6.4 Document Complex Transformations

```yaml
steps:
  # Calculate user engagement score as weighted combination of:
  # - Recency: Days since last activity (20% weight)
  # - Frequency: Number of actions in last 30 days (40% weight)
  # - Monetary: Total spend in last 30 days (40% weight)
  - name: user_engagement_score
    type: view
    depends_on: [user_activity]
    sql: |
      SELECT
        user_id,
        (0.2 * recency_score +
         0.4 * frequency_score +
         0.4 * monetary_score) as engagement_score
      FROM ...
```

## 7. Integration with Arc-Graph

Arc-Pipeline prepares features that Arc-Graph models consume:

**Arc-Pipeline output:**
```yaml
outputs: [train_set, val_set]
```

**Arc-Graph input:**
```yaml
# References the train_set table from Arc-Pipeline
inputs:
  features:
    dtype: float32
    shape: [null, 10]
    columns: [age, income, ...]  # Columns from train_set
```

The pipeline ensures that the features match the model's input specification.

## 8. Data Loading Integration

Arc-Pipeline integrates with various data sources. See the built-in knowledge guides for details:

- **[Data Loading Patterns](../src/arc/resources/knowledge/data_loading.md)** - CSV, Parquet, JSON
- **[S3 Integration](s3-setup.md)** - Load from S3 buckets
- **[Snowflake Integration](snowflake-setup.md)** - Query Snowflake tables

## 9. Extending with ML Knowledge

Arc-Pipeline is a general ETL tool, but **Arc-Knowledge** is what makes it ML-aware. You can extend Arc's pipeline generation by adding custom ML patterns to `~/.arc/knowledge/`:

```yaml
# ~/.arc/knowledge/metadata.yaml
financial_features:
  name: "Financial Feature Engineering"
  description: "Stock market and trading-specific feature patterns"
  phases: ["data"]
```

```bash
# Create knowledge file
touch ~/.arc/knowledge/financial_features.md
```

When you ask Arc to generate pipelines, it consults this knowledge to apply your domain-specific ML patterns.

See **[Arc-Knowledge](arc-knowledge.md)** for complete details on extending the knowledge system.

## 10. Additional Resources

- **[Arc-Graph Specification](arc-graph.md)** - Model architecture YAML schema
- **[Arc-Knowledge](arc-knowledge.md)** - How to extend Arc with your own patterns
- **[Feature Engineering Guide](../src/arc/resources/knowledge/ml_data_preparation.md)** - ML-specific transformation patterns
