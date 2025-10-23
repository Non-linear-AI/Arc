# S3 Integration Setup

Arc integrates with AWS S3 (and S3-compatible storage) using DuckDB's S3 extensions. Load data from public buckets, private buckets, or cloud instances with IAM roles - all accessible via standard SQL queries.

## Quick Start

### Option 1: Public S3 Buckets (No Setup Required)

Public buckets work immediately - just start Arc and query:

```bash
uv run arc chat
```

```sql
-- Load AWS Open Data Registry datasets
/sql CREATE TABLE taxi AS
     SELECT * FROM 's3://nyc-tlc/trip data/yellow_tripdata_2023-01.parquet'

-- Query COVID-19 Data Lake
/sql SELECT * FROM read_csv_auto('s3://covid19-lake/enigma-jhu/csv/*.csv')
     LIMIT 100

-- Load multiple Parquet files with wildcards
/sql CREATE TABLE events AS
     SELECT * FROM 's3://public-bucket/data/events/*.parquet'
```

### Option 2: IAM Roles on AWS (No Setup Required)

If Arc is running on **EC2, ECS, or Lambda** with an IAM role, it automatically uses those credentials:

```bash
# On your EC2 instance, ECS task, or Lambda function
uv run arc chat
```

```sql
-- Access private S3 buckets using instance IAM role
/sql CREATE TABLE data AS
     SELECT * FROM 's3://my-private-bucket/data.parquet'
```

**No Arc configuration needed** - just ensure your instance/task/function has an IAM role with S3 permissions.

### Option 3: Private S3 Buckets (Credentials Required)

For private buckets accessed from outside AWS or without IAM roles, configure credentials first.

**Add to `~/.arc/user-settings.json`:**

```json
{
  "apiKey": "sk-...",
  "awsAccessKeyId": "AKIAIOSFODNN7EXAMPLE",
  "awsSecretAccessKey": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
  "awsRegion": "us-east-1"
}
```

**Or use environment variables:**

```bash
export AWS_ACCESS_KEY_ID="AKIAIOSFODNN7EXAMPLE"
export AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/..."
export AWS_REGION="us-east-1"

uv run arc chat
```

**Then query your private data:**

```sql
/sql CREATE TABLE customers AS
     SELECT * FROM 's3://my-private-bucket/data/customers.parquet'

/sql SELECT * FROM 's3://my-bucket/processed/*.csv'
     WHERE date >= '2024-01-01'
```

## Common Use Cases

### Loading Data for ML Training

**Extract once, use many times** (recommended for iterative feature engineering):

```sql
-- Load full dataset locally
/sql CREATE TABLE raw_sales AS
     SELECT * FROM 's3://company-data/sales/2024/*.parquet'

-- Now do feature engineering locally (fast, free)
/sql CREATE TABLE sales_features AS
     SELECT
       customer_id,
       COUNT(*) as purchase_count,
       SUM(amount) as total_spent,
       AVG(amount) as avg_purchase
     FROM raw_sales
     GROUP BY customer_id

-- Use for ML training (data is local, fast iterations)
> Train a model to predict customer churn using sales_features
```

### Querying Data Directly

**For exploration and one-time queries:**

```sql
-- Quick exploration without creating table
/sql SELECT region, COUNT(*), AVG(revenue)
     FROM 's3://analytics/daily-metrics/*.parquet'
     WHERE date >= '2024-01-01'
     GROUP BY region

-- Filter at source for efficiency
/sql SELECT * FROM 's3://logs/app-logs/*.json'
     WHERE timestamp >= '2024-10-01'
       AND error_level = 'ERROR'
     LIMIT 100
```

### Combining S3 with Local Data

```sql
-- Join S3 data with local predictions
/sql SELECT
       s3.customer_id,
       s3.lifetime_value,
       local.churn_prediction
     FROM 's3://warehouse/customers.parquet' s3
     JOIN local_predictions local
       ON s3.customer_id = local.customer_id
     WHERE local.churn_prediction > 0.8
```

### Loading Apache Iceberg Tables

```sql
-- Query Iceberg tables on S3
/sql CREATE TABLE iceberg_data AS
     SELECT * FROM iceberg_scan('s3://data-lake/warehouse/db/table')
     WHERE date >= '2024-01-01'
```

## Configuration Details

### AWS Credentials Priority

Arc uses the standard AWS credential chain in this order:

1. **Environment variables** (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`)
2. **IAM roles** (EC2 instance profiles, ECS task roles, Lambda execution roles)
3. **AWS credential files** (`~/.aws/credentials`)
4. **Arc settings** (`~/.arc/user-settings.json` - `awsAccessKeyId`, `awsSecretAccessKey`)

### S3-Compatible Storage (MinIO, Wasabi, DigitalOcean Spaces)

Configure custom endpoints in `~/.arc/user-settings.json`:

```json
{
  "awsAccessKeyId": "minioadmin",
  "awsSecretAccessKey": "minioadmin",
  "awsRegion": "us-east-1",
  "s3Endpoint": "http://localhost:9000"
}
```

Or via environment variable:

```bash
export AWS_ENDPOINT_URL="http://localhost:9000"
```

Then use standard S3 URLs:

```sql
/sql CREATE TABLE data AS
     SELECT * FROM 's3://my-bucket/data.parquet'
```

### Supported File Formats

- **CSV** - `read_csv_auto('s3://...')` or direct path
- **Parquet** - Direct path (recommended for ML: smaller, faster, columnar)
- **JSON** - `read_json_auto('s3://...')`
- **Apache Iceberg** - `iceberg_scan('s3://...')`

## Best Practices

### 1. Extract Once for Iterative Work

```sql
-- ✅ Good: Load once, iterate locally
/sql CREATE TABLE training_data AS
     SELECT * FROM 's3://bucket/data/*.parquet'
     WHERE date >= '2024-01-01'

-- Then iterate on features (no S3 costs)
/sql CREATE TABLE features_v1 AS SELECT ...
/sql CREATE TABLE features_v2 AS SELECT ...
/sql CREATE TABLE features_v3 AS SELECT ...
```

```sql
-- ❌ Avoid: Querying S3 repeatedly
/sql SELECT COUNT(*) FROM 's3://bucket/data/*.parquet'  -- $$
/sql SELECT AVG(value) FROM 's3://bucket/data/*.parquet'  -- $$
```

### 2. Filter at Source

Push filters to S3 to reduce data transfer:

```sql
-- ✅ Good: Filter before loading
/sql CREATE TABLE recent_orders AS
     SELECT * FROM 's3://orders/*.parquet'
     WHERE order_date >= '2024-10-01'
       AND status = 'completed'

-- ❌ Avoid: Loading everything then filtering
/sql CREATE TABLE all_orders AS SELECT * FROM 's3://orders/*.parquet'
/sql SELECT * FROM all_orders WHERE order_date >= '2024-10-01'
```

### 3. Use Parquet for ML Workloads

Parquet is optimized for analytical queries:

```sql
-- ✅ Parquet: Fast, compact, columnar
/sql CREATE TABLE data AS SELECT * FROM 's3://bucket/data.parquet'

-- ⚠️ CSV: Slower, larger, row-based (but more universal)
/sql CREATE TABLE data AS SELECT * FROM 's3://bucket/data.csv'
```

### 4. Use Wildcards for Partitioned Data

```sql
-- Load specific partitions efficiently
/sql CREATE TABLE q1_sales AS
     SELECT * FROM 's3://sales/year=2024/month={01,02,03}/*.parquet'

-- Or all 2024 data
/sql CREATE TABLE all_2024 AS
     SELECT * FROM 's3://sales/year=2024/**/*.parquet'
```

## Troubleshooting

### "Permission denied" or "Access Denied"

**For IAM roles:**
- Verify the IAM role attached to your EC2/ECS/Lambda has S3 permissions
- Check bucket policies allow access from your role
- Ensure no SCPs (Service Control Policies) block S3 access

**For credentials:**
- Verify `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are correct
- Check the IAM user/role has `s3:GetObject` and `s3:ListBucket` permissions
- Confirm the bucket region matches `awsRegion` configuration

### "No such file or directory" or "File not found"

- Check the S3 path is correct (bucket name, prefix, file name)
- Verify the file exists: `aws s3 ls s3://bucket/path/`
- For wildcards, ensure files match the pattern

### "Unknown extension" or "Extension not loaded"

Arc automatically loads S3 extensions on first connection. If this fails:

```sql
-- Manually install extensions
/sql INSTALL httpfs
/sql INSTALL aws
/sql INSTALL iceberg

/sql LOAD httpfs
/sql LOAD aws
/sql LOAD iceberg
```

### Slow Queries

- **Use Parquet instead of CSV** - much faster for columnar queries
- **Filter at source** - push WHERE clauses to S3
- **Load data locally first** - avoid repeated S3 queries
- **Check network connectivity** - slow internet = slow S3 reads

### Region Mismatch

If you see "PermanentRedirect" errors:

```bash
# Set correct region
export AWS_REGION="us-west-2"  # Match your bucket's region
```

## Architecture & How It Works

### DuckDB S3 Extensions

Arc uses DuckDB's built-in S3 support via three extensions:

- **`httpfs`** - HTTP/HTTPS file system for public buckets
- **`aws`** - AWS authentication and credential management
- **`iceberg`** - Apache Iceberg table format support

These extensions are automatically installed and loaded when you first connect to the database.

### Authentication Flow

1. **DuckDB checks for credentials** in this order:
   - Environment variables (`AWS_ACCESS_KEY_ID`, etc.)
   - AWS credential chain (IAM roles, credential files)
   - Arc settings (`~/.arc/user-settings.json`)

2. **DuckDB creates S3 client** with discovered credentials

3. **Queries are executed** directly against S3 using AWS SDK

### Data Transfer

- **Streaming**: DuckDB streams data from S3 (doesn't download entire files first)
- **Parallelization**: Multiple S3 requests in parallel for better performance
- **Columnar**: Parquet files use column pruning (only read needed columns)
- **Predicate pushdown**: WHERE filters applied during S3 read when possible

### Cost Considerations

- **S3 GET requests**: Charged per request (~$0.0004 per 1,000 requests)
- **Data transfer**: Free from S3 to EC2 (same region), otherwise charged
- **Local caching**: Load data once to DuckDB to avoid repeated S3 costs

### Extension Installation

Extensions are installed from DuckDB's extension repository automatically:

```python
# Arc does this automatically on database initialization
connection.execute("INSTALL httpfs")
connection.execute("INSTALL aws")
connection.execute("INSTALL iceberg")
connection.execute("LOAD httpfs")
connection.execute("LOAD aws")
connection.execute("LOAD iceberg")
```

## Additional Resources

- [DuckDB S3 Documentation](https://duckdb.org/docs/extensions/httpfs)
- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [Apache Iceberg Format](https://iceberg.apache.org/)
- [AWS Open Data Registry](https://registry.opendata.aws/) - Public datasets to explore
