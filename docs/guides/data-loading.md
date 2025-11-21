# Data Loading Guide

This guide covers how to load data from various sources into Arc. Arc uses DuckDB as its data engine, providing powerful data loading capabilities for CSV, Parquet, JSON, S3, Snowflake, and more.

## Overview

Arc can load data from:
- **Local files**: CSV, Parquet, JSON, Excel
- **Remote URLs**: HTTPS endpoints
- **AWS S3**: Public and private buckets
- **Snowflake**: Data warehouse tables
- **Databases**: PostgreSQL, MySQL (via DuckDB extensions)

## Quick Start

The easiest way to load data is to ask Arc in natural language:

```
Load the file data.csv into a table called customers
```

Arc will automatically:
1. Detect the file format
2. Infer the schema
3. Create the table
4. Load the data

You can verify with:
```sql
/sql SELECT * FROM customers LIMIT 10
```

## Loading CSV Files

### From Local File

```
Load users.csv into a table called users
```

Or use SQL directly:
```sql
/sql CREATE TABLE users AS SELECT * FROM 'users.csv'
```

### With Custom Delimiter

For files with non-standard delimiters (e.g., tab-separated or pipe-delimited):

```
Load the tab-separated file data.tsv into table data
```

### Large CSV Files

Arc handles large CSV files efficiently. For very large files (GB+), consider:

```
Load large_data.csv into table data using streaming mode
```

### CSV with No Header

If your CSV lacks a header row:

```
Load data.csv (no header) with columns: id, name, age, email
```

## Loading Parquet Files

Parquet is recommended for large datasets due to compression and speed.

```
Load sales_data.parquet into table sales
```

Or with SQL:
```sql
/sql CREATE TABLE sales AS SELECT * FROM 'sales_data.parquet'
```

### Loading Multiple Parquet Files

Use glob patterns to load multiple files:

```
Load all parquet files from the data/ directory into table combined_data
```

Or with SQL:
```sql
/sql CREATE TABLE combined_data AS SELECT * FROM 'data/*.parquet'
```

## Loading JSON Files

### Standard JSON

```
Load products.json into table products
```

### Newline-Delimited JSON (JSONL)

For streaming JSON data:

```
Load events.jsonl into table events
```

## Loading from URLs

Load data directly from HTTPS URLs:

```
Load data from https://example.com/data.csv into table remote_data
```

Arc will download and load the data automatically.

## Loading from S3

Arc supports loading data from AWS S3 buckets. See the [S3 Integration Guide](../integrations/s3.md) for setup details.

### Public S3 Buckets

```
Load data from s3://nyc-tlc/trip data/yellow_tripdata_2023-01.parquet
```

Or with SQL:
```sql
/sql CREATE TABLE taxi AS
     SELECT * FROM 's3://nyc-tlc/trip data/yellow_tripdata_2023-01.parquet'
```

### Private S3 Buckets

After [configuring S3 credentials](../integrations/s3.md):

```
Load data from s3://my-private-bucket/data.parquet
```

### S3 Glob Patterns

Load multiple files from S3:

```sql
/sql CREATE TABLE all_data AS
     SELECT * FROM 's3://my-bucket/data/*.parquet'
```

## Loading from Snowflake

Arc can query Snowflake data warehouses. See the [Snowflake Integration Guide](../integrations/snowflake.md) for setup.

```
Load customer data from Snowflake table PUBLIC.CUSTOMERS
```

Or with SQL:
```sql
/sql CREATE TABLE local_customers AS
     SELECT * FROM snowflake.PUBLIC.CUSTOMERS
     WHERE signup_date >= '2024-01-01'
```

## Checking Loaded Data

After loading data, verify it:

```sql
-- List all tables
/sql SHOW TABLES

-- View table structure
/sql DESCRIBE users

-- Preview data
/sql SELECT * FROM users LIMIT 10

-- Count rows
/sql SELECT COUNT(*) FROM users

-- Check for nulls
/sql SELECT COUNT(*) as null_count
     FROM users
     WHERE column_name IS NULL
```

## Common Data Loading Patterns

### Pattern 1: Load and Explore

```
Load data.csv and show me the first 10 rows
```

### Pattern 2: Load and Transform

```
Load sales.csv, then create a processed_sales table with:
- Convert dates to proper format
- Filter out invalid entries
- Add a month column
```

### Pattern 3: Load Multiple Sources

```
Load customers.csv and orders.csv, then join them on customer_id
```

### Pattern 4: Incremental Loading

```
Load new_data.csv and append it to the existing data table
```

## Data Loading via /ml data Command

For ML workflows, use the `/ml data` command:

```
/ml data --name processed_customers
         --instruction "Load customers.csv and normalize the age and income columns"
         --source-tables customers
```

This creates an Arc-Pipeline specification that can be reused and version-controlled.

## Best Practices

### 1. Use Appropriate File Formats

- **CSV**: Simple, human-readable, but slow for large data
- **Parquet**: Fast, compressed, recommended for large datasets
- **JSON**: Flexible schema, good for nested data

### 2. Validate Data After Loading

Always check your data after loading:
```sql
/sql SELECT COUNT(*),
            COUNT(DISTINCT id),
            MIN(created_at),
            MAX(created_at)
     FROM users
```

### 3. Handle Missing Values

Check for missing values:
```sql
/sql SELECT
       COUNT(*) - COUNT(column_name) as missing_count
     FROM table_name
```

### 4. Use Efficient Loading for Large Files

For files >1GB:
- Use Parquet instead of CSV
- Use glob patterns to parallelize loading
- Consider sampling for exploration first

### 5. Store Raw Data Separately

Keep raw loaded data in separate tables:
```sql
-- Raw data
CREATE TABLE raw_customers AS SELECT * FROM 'customers.csv';

-- Processed data
CREATE TABLE processed_customers AS
SELECT *, UPPER(name) as name_upper
FROM raw_customers;
```

## Troubleshooting

### File Not Found

If Arc can't find your file:
1. Use absolute paths: `/home/user/data.csv`
2. Or start from current directory: `./data.csv`
3. Check file permissions: `ls -l data.csv`

### Schema Detection Issues

If automatic schema detection fails:
```
Load data.csv with explicit schema: id INTEGER, name VARCHAR, age INTEGER
```

### Memory Issues with Large Files

For very large files:
1. Use Parquet instead of CSV
2. Load in chunks with filtering:
   ```sql
   /sql CREATE TABLE chunk1 AS
        SELECT * FROM 'large.csv'
        WHERE date >= '2024-01-01' AND date < '2024-02-01'
   ```
3. Use external tables (query without loading)

### Character Encoding Issues

If you see garbled text:
```
Load data.csv with UTF-8 encoding
```

### Date Parsing Issues

If dates aren't recognized:
```
Load data.csv and parse the date_column as dates in YYYY-MM-DD format
```

## Advanced Topics

### Loading from Databases

Arc can load from databases via DuckDB extensions:

```sql
-- PostgreSQL
/sql CREATE TABLE users AS
     SELECT * FROM postgres_scan('host=localhost db=mydb', 'users')

-- MySQL
/sql CREATE TABLE users AS
     SELECT * FROM mysql_scan('host=localhost db=mydb', 'users')
```

### Loading Excel Files

```
Load the Sheet1 from workbook.xlsx into table data
```

### Loading Compressed Files

Arc automatically handles compressed files:
```
Load data.csv.gz into table data
```

## Next Steps

- **[Feature Engineering Guide](feature-engineering.md)** - Transform your loaded data
- **[Model Training Guide](model-training.md)** - Train models with your data
- **[S3 Integration](../integrations/s3.md)** - Set up S3 data loading
- **[Snowflake Integration](../integrations/snowflake.md)** - Set up Snowflake access

## Related Documentation

- [Arc-Pipeline Specification](../concepts/arc-pipeline.md) - Declarative data processing
- [Arc Knowledge: Data Loading](../../src/arc/resources/knowledge/data_loading.md) - Technical details for Arc's AI
