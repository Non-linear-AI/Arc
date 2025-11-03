# Data Loading with DuckDB

This guide covers patterns for loading external data files into DuckDB tables. Use this guidance when you need to import CSV, Parquet, JSON, or other data formats into the database.

## Overview

DuckDB provides native functions for loading external data files directly into tables without requiring external tools or preprocessing. These functions support:
- **CSV files** (with various delimiters and formats)
- **Parquet files** (single files or glob patterns)
- **JSON files** (including newline-delimited JSONL)
- **Local files** (relative or absolute paths)
- **Remote files** (HTTP/HTTPS URLs)

## When to Use Data Loading

Use DuckDB's native data loading functions (via ml_data tool) when:
- Importing external datasets into the database
- Loading data from URLs or cloud storage
- Creating raw tables before transformation
- Initial data ingestion into the database

After loading data, use data transformation pipelines (also via ml_data tool) for:
- Feature engineering from existing tables
- Creating train/validation splits
- Normalizing/scaling features
- Joining multiple tables

## CSV Loading

### Basic CSV Loading (Auto-detection)

Use `read_csv_auto()` for quick exploration with automatic schema detection:

```sql
CREATE TABLE ratings AS
SELECT * FROM read_csv_auto('ratings.csv');
```

DuckDB will automatically:
- Detect column names from the header row
- Infer column types from the data
- Handle common delimiters (`,`, `\t`, `|`)
- Quote and escape character detection

### CSV with Explicit Schema

For production pipelines or ambiguous formats, use `read_csv()` with explicit schema:

```sql
CREATE TABLE ratings AS
SELECT * FROM read_csv('ratings.dat',
    delim='::',
    header=false,
    columns={
        'user_id': 'INTEGER',
        'movie_id': 'INTEGER',
        'rating': 'INTEGER',
        'timestamp': 'INTEGER'
    }
);
```

**When to use explicit schema:**
- Files without headers (header=false)
- Non-standard delimiters (.dat files with `::` delimiter)
- Type inference issues (dates, mixed numeric types)
- Production pipelines requiring consistent types

### CSV from URLs

Load data directly from HTTP/HTTPS URLs:

```sql
CREATE TABLE diabetes AS
SELECT * FROM read_csv_auto('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv');
```

### Common CSV Options

When using `read_csv()`, you can specify these options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `delim` | TEXT | `,` | Field delimiter (e.g., `,`, `\t`, `::`, `\|`) |
| `header` | BOOLEAN | `true` | Whether first row is header |
| `columns` | STRUCT | auto | Explicit column names and types |
| `skip` | INTEGER | 0 | Number of rows to skip |
| `quote` | TEXT | `"` | Quote character |
| `escape` | TEXT | `"` | Escape character |
| `null_padding` | BOOLEAN | `false` | Pad missing fields with NULL |
| `encoding` | TEXT | `'UTF-8'` | File encoding (e.g., `'UTF-8'`, `'ISO-8859-1'`, `'UTF-16'`) |
| `ignore_errors` | BOOLEAN | `false` | Skip rows with parsing errors instead of failing |

### CSV Loading Examples

**Tab-delimited file:**
```sql
CREATE TABLE data AS
SELECT * FROM read_csv('data.tsv', delim='\t');
```

**Pipe-delimited file with header:**
```sql
CREATE TABLE data AS
SELECT * FROM read_csv('data.txt',
    delim='|',
    header=true
);
```

**Headerless CSV with explicit column names:**
```sql
CREATE TABLE measurements AS
SELECT * FROM read_csv('measurements.csv',
    header=false,
    columns={
        'timestamp': 'TIMESTAMP',
        'sensor_id': 'INTEGER',
        'value': 'DECIMAL'
    }
);
```

### Handling Encoding Issues

CSV files may have different character encodings. If you encounter Unicode errors like "Invalid unicode (byte sequence mismatch)", use the `encoding` parameter:

**Non-UTF-8 encoded file:**
```sql
CREATE TABLE data AS
SELECT * FROM read_csv('data.csv',
    encoding='ISO-8859-1'  -- Latin-1 encoding
);
```

**Common encoding values:**
- `'UTF-8'` (default): Standard Unicode encoding
- `'ISO-8859-1'` or `'LATIN1'`: Western European characters
- `'UTF-16'`: 16-bit Unicode encoding
- `'WINDOWS-1252'`: Windows Western European encoding

**Skipping malformed rows with ignore_errors:**

When loading real-world data files that may have inconsistent formatting or corrupted rows, use `ignore_errors=true` to skip problematic rows instead of failing:

```sql
CREATE TABLE movies AS
SELECT * FROM read_csv('ml-1m/movies.dat',
    delim='::',
    header=false,
    columns={
        'movie_id': 'INTEGER',
        'title': 'VARCHAR',
        'genres': 'VARCHAR'
    },
    ignore_errors=true  -- Skip rows with parsing errors
);
```

**Combining encoding and error handling:**

For files with both encoding issues and potential malformed rows:

```sql
CREATE TABLE data AS
SELECT * FROM read_csv('legacy_data.csv',
    encoding='ISO-8859-1',
    ignore_errors=true,
    header=true
);
```

**When to use ignore_errors:**
- Loading legacy data with inconsistent formatting
- Files from external sources with unknown quality
- Data migration from other systems
- When you want to load "best effort" and handle validation downstream

**Important:** When using `ignore_errors=true`, always validate the loaded data to check how many rows were skipped and ensure data quality.

## Parquet Loading

Parquet is a columnar format optimized for analytics. DuckDB can read Parquet files efficiently:

### Single Parquet File

```sql
CREATE TABLE events AS
SELECT * FROM read_parquet('events.parquet');
```

### Multiple Parquet Files (Glob Pattern)

```sql
-- Load all Parquet files in a directory
CREATE TABLE logs AS
SELECT * FROM read_parquet('logs/*.parquet');

-- Load Parquet files with pattern
CREATE TABLE events AS
SELECT * FROM read_parquet('data/events_*.parquet');
```

### Parquet from URLs

```sql
CREATE TABLE data AS
SELECT * FROM read_parquet('https://example.com/data.parquet');
```

## JSON Loading

### JSON Auto-detection

Use `read_json_auto()` for automatic schema detection:

```sql
CREATE TABLE user_profiles AS
SELECT * FROM read_json_auto('users.json');
```

### JSONL (Newline-delimited JSON)

For streaming JSON or large datasets:

```sql
CREATE TABLE events AS
SELECT * FROM read_json_auto('events.jsonl', format='newline_delimited');
```

### JSON Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `format` | TEXT | `auto` | `'auto'`, `'newline_delimited'`, or `'array'` |
| `columns` | STRUCT | auto | Explicit column names and types |
| `maximum_object_size` | INTEGER | 16777216 | Maximum size of JSON objects (bytes) |

## Complete Example: MovieLens Dataset

Loading the MovieLens dataset with correct delimiters and schema:

```yaml
steps:
  - name: drop_old_ratings
    type: execute
    depends_on: []
    sql: DROP TABLE IF EXISTS "ratings"

  - name: drop_old_movies
    type: execute
    depends_on: []
    sql: DROP TABLE IF EXISTS "movies"

  - name: ratings
    type: table
    depends_on: [drop_old_ratings]
    sql: |
      CREATE TABLE ratings AS
      SELECT * FROM read_csv('ml-latest-small/ratings.csv',
          header=true,
          columns={
              'userId': 'INTEGER',
              'movieId': 'INTEGER',
              'rating': 'FLOAT',
              'timestamp': 'INTEGER'
          }
      )

  - name: movies
    type: table
    depends_on: [drop_old_movies]
    sql: |
      CREATE TABLE movies AS
      SELECT * FROM read_csv('ml-latest-small/movies.csv',
          header=true,
          columns={
              'movieId': 'INTEGER',
              'title': 'VARCHAR',
              'genres': 'VARCHAR'
          }
      )

outputs: [ratings, movies]
```

## Complete Example: Pima Indians Diabetes Dataset

Loading the Pima Indians diabetes dataset from a URL:

```yaml
steps:
  - name: drop_old_diabetes
    type: execute
    depends_on: []
    sql: DROP TABLE IF EXISTS "diabetes"

  - name: diabetes
    type: table
    depends_on: [drop_old_diabetes]
    sql: |
      CREATE TABLE diabetes AS
      SELECT
        column0 as pregnancies,
        column1 as glucose,
        column2 as blood_pressure,
        column3 as skin_thickness,
        column4 as insulin,
        column5 as bmi,
        column6 as diabetes_pedigree,
        column7 as age,
        column8 as outcome
      FROM read_csv_auto('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv',
          header=false
      )

outputs: [diabetes]
```

**Note:** Since the Pima dataset has no header, DuckDB assigns generic column names (`column0`, `column1`, etc.). We use SELECT to rename them to meaningful names.

## Best Practices for Data Loading

### 1. Always Specify Schema Explicitly When Possible

For production pipelines, use explicit schema to avoid surprises:

```sql
-- ❌ Risky: Auto-detection may misinterpret types
CREATE TABLE data AS
SELECT * FROM read_csv_auto('data.csv');

-- ✅ Better: Explicit schema ensures correctness
CREATE TABLE data AS
SELECT * FROM read_csv('data.csv',
    header=true,
    columns={
        'id': 'INTEGER',
        'timestamp': 'TIMESTAMP',
        'value': 'DECIMAL'
    }
);
```

### 2. Use read_csv_auto() for Exploration, Then Switch to Explicit Schema

Workflow:
1. Use `read_csv_auto()` to explore the data
2. Inspect the inferred schema with `DESCRIBE SELECT * FROM read_csv_auto('file.csv')`
3. Switch to `read_csv()` with explicit schema for final pipeline

### 3. Drop Existing Tables First for Idempotency

Always use `DROP TABLE IF EXISTS` before loading:

```yaml
steps:
  - name: drop_old_data
    type: execute
    depends_on: []
    sql: DROP TABLE IF EXISTS "data"

  - name: data
    type: table
    depends_on: [drop_old_data]
    sql: |
      CREATE TABLE data AS
      SELECT * FROM read_csv_auto('data.csv')
```

### 4. Validate Data After Loading

Add a validation step to check data quality:

```yaml
steps:
  # ... loading steps ...

  - name: validate_data
    type: view
    depends_on: [data]
    sql: |
      SELECT
        COUNT(*) as row_count,
        COUNT(DISTINCT id) as unique_ids,
        SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END) as null_values
      FROM "data"

outputs: [data, validate_data]
```

### 5. Use Relative Paths for Local Files

When loading local files, use relative paths from the current working directory:

```sql
-- ✅ Good: Relative path
CREATE TABLE data AS
SELECT * FROM read_csv('data/ratings.csv');

-- ❌ Avoid: Absolute paths reduce portability
CREATE TABLE data AS
SELECT * FROM read_csv('/Users/username/data/ratings.csv');
```

### 6. Handle Missing or Malformed Data

Use defensive SQL patterns after loading:

```sql
CREATE TABLE clean_data AS
SELECT
  id,
  COALESCE(value, 0) as value,  -- Handle NULL values
  timestamp
FROM "raw_data"
WHERE id IS NOT NULL  -- Filter out invalid rows
  AND timestamp IS NOT NULL;
```

## Data Loading vs Data Transformation

**Use data loading** (this guide) when:
- Importing external files into the database
- Creating raw tables from CSV/Parquet/JSON
- Initial data ingestion

**Use data transformation** (see ml_data_preparation.md for ML-specific patterns) when:
- Feature engineering from existing tables
- Creating derived columns
- Aggregations and joins
- Train/validation splits
- Normalization and scaling

## Troubleshooting

### CSV Parsing Issues

If `read_csv_auto()` fails or produces incorrect results:

1. **Check delimiter:** Specify with `delim` parameter
2. **Check header:** Set `header=false` if no header row
3. **Check quotes:** Adjust `quote` and `escape` parameters
4. **Check encoding:** Use `encoding` parameter if file is not UTF-8 (see "Handling Encoding Issues" section)

### Encoding Errors

**Error: "Invalid unicode (byte sequence mismatch) detected"**

This means the file is not UTF-8 encoded. Solutions:

1. **Specify correct encoding:**
   ```sql
   SELECT * FROM read_csv('file.csv', encoding='ISO-8859-1')
   ```

2. **Use ignore_errors to skip problematic rows:**
   ```sql
   SELECT * FROM read_csv('file.csv', ignore_errors=true)
   ```

3. **Try common encodings:** `'ISO-8859-1'`, `'WINDOWS-1252'`, `'UTF-16'`

### Malformed Data Errors

**Error: "Parser Error: syntax error at or near..."**

This usually means rows have inconsistent number of fields or invalid formatting:

1. **Use ignore_errors to skip bad rows:**
   ```sql
   SELECT * FROM read_csv('file.csv', ignore_errors=true)
   ```

2. **Validate after loading:**
   ```sql
   SELECT COUNT(*) FROM read_csv('file.csv', ignore_errors=true)
   ```
   Compare with expected row count to see how many rows were skipped

### Type Inference Issues

If DuckDB infers wrong types:

1. Use explicit `columns` parameter with correct types
2. Cast columns in the SELECT clause
3. Use TEXT type initially, then CAST in a downstream step

### Memory Issues with Large Files

For very large files:

1. Use Parquet instead of CSV (more efficient)
2. Load data in chunks using OFFSET/LIMIT
3. Use views instead of tables for intermediate steps
4. Consider filtering during loading: `SELECT * FROM read_csv(...) WHERE condition`

## Summary

DuckDB's native data loading functions make it easy to import external data files:
- **CSV**: `read_csv_auto()` for exploration, `read_csv()` for production
- **Parquet**: `read_parquet()` for efficient columnar data
- **JSON**: `read_json_auto()` for structured data

Key principles:
- Use explicit schemas for production pipelines
- Always drop existing tables first for idempotency
- Validate data after loading
- Use relative paths for portability
- Separate data loading (this guide) from data transformation (ml_data_preparation.md)
