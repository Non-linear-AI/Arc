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

```
Load users.csv into a table called users
```

Or use SQL directly:
```sql
/sql CREATE TABLE users AS SELECT * FROM 'users.csv'
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

## Loading JSON Files

Arc supports JSON and JSONL formats:

```
Load products.json into table products
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

Verify data with `/sql SHOW TABLES` and `/sql SELECT * FROM table_name LIMIT 10`.

## Next Steps

- **[Feature Engineering Guide](feature-engineering.md)** - Transform your loaded data
- **[Model Training Guide](model-training.md)** - Train models with your data
- **[S3 Integration](../integrations/s3.md)** - Set up S3 data loading
- **[Snowflake Integration](../integrations/snowflake.md)** - Set up Snowflake access

## Related Documentation

- [Arc-Pipeline Specification](../concepts/arc-pipeline.md) - Declarative data processing
- [Arc-Knowledge](../concepts/arc-knowledge.md) - Built-in ML knowledge system
