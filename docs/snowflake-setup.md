# Snowflake Integration Setup

Arc integrates with Snowflake data warehouses using DuckDB's Snowflake extension. Query Snowflake tables directly in Arc, join them with local data, and extract data for cost-efficient local feature engineering.

## Quick Start

Once configured (see Configuration section below), Snowflake tables are automatically available when you start Arc:

```bash
uv run arc chat
```

### Query Snowflake Directly

```sql
-- View available tables (Snowflake appears as "snowflake" database)
> What tables are in my database?

-- Query Snowflake tables directly
/sql SELECT * FROM snowflake.public.customers
     WHERE state = 'CA'
     LIMIT 10

-- Aggregate queries (pushed to Snowflake)
/sql SELECT state, COUNT(*) as customer_count
     FROM snowflake.public.customers
     GROUP BY state
```

### Extract Data for Local Analysis (Recommended)

**Best practice for ML workflows**: Extract once, transform locally

```sql
-- 1. Extract relevant data from Snowflake (one-time cost)
/sql CREATE TABLE ca_customers AS
     SELECT * FROM snowflake.public.customers
     WHERE state = 'CA' AND signup_date >= '2024-01-01'

-- 2. Feature engineering runs locally (fast, free iterations)
/sql CREATE TABLE customer_features AS
     SELECT
       customer_id,
       COUNT(*) as order_count,
       SUM(amount) as lifetime_value,
       AVG(amount) as avg_order_value
     FROM ca_customers c
     JOIN snowflake.public.orders o ON c.id = o.customer_id
     GROUP BY customer_id

-- 3. Train models on local data (no Snowflake compute costs)
> Train a model to predict high-value customers using customer_features
```

### Join Across Data Sources

```sql
-- Combine Snowflake + S3 + Local DuckDB tables
/sql CREATE TABLE enriched_customers AS
     SELECT
       sf.customer_id,
       sf.name,
       s3.demographic_score,
       local.predicted_churn
     FROM snowflake.public.customers sf
     JOIN 's3://my-bucket/demographics.parquet' s3
       ON sf.id = s3.customer_id
     JOIN local_ml_predictions local
       ON sf.id = local.customer_id
```

## Configuration

### Step 1: Configure Snowflake Credentials

**Option A: Settings File** (`~/.arc/user-settings.json`):

```json
{
  "apiKey": "sk-...",
  "snowflakeAccount": "mycompany.snowflakecomputing.com",
  "snowflakeUser": "username",
  "snowflakePassword": "password",
  "snowflakeDatabase": "PROD_DB",
  "snowflakeWarehouse": "COMPUTE_WH",
  "snowflakeSchema": "PUBLIC"
}
```

**Option B: Environment Variables** (takes precedence):

```bash
export SNOWFLAKE_ACCOUNT="mycompany.snowflakecomputing.com"
export SNOWFLAKE_USER="username"
export SNOWFLAKE_PASSWORD="password"
export SNOWFLAKE_DATABASE="PROD_DB"
export SNOWFLAKE_WAREHOUSE="COMPUTE_WH"
export SNOWFLAKE_SCHEMA="PUBLIC"  # Optional, defaults to PUBLIC
```

**Required fields**: `account`, `user`, `password`, `database`, `warehouse`

### Step 2: Set Library Path

The ADBC driver includes native libraries that must be discoverable **before** starting Arc.

#### Linux:

```bash
cd /path/to/arc

# Find ADBC library directory
ADBC_LIB_DIR=$(uv run python -c "import adbc_driver_snowflake; from pathlib import Path; print(Path(adbc_driver_snowflake.__file__).parent)")

# Set library path
export LD_LIBRARY_PATH="${ADBC_LIB_DIR}:${LD_LIBRARY_PATH}"

# Start Arc
uv run arc chat
```

#### macOS:

```bash
cd /path/to/arc

# Find ADBC library directory
ADBC_LIB_DIR=$(uv run python -c "import adbc_driver_snowflake; from pathlib import Path; print(Path(adbc_driver_snowflake.__file__).parent)")

# Set library path
export DYLD_LIBRARY_PATH="${ADBC_LIB_DIR}:${DYLD_LIBRARY_PATH}"

# Start Arc
uv run arc chat
```

#### Windows (PowerShell):

```powershell
cd C:\path\to\arc

# Find ADBC library directory
$ADBC_LIB_DIR = uv run python -c "import adbc_driver_snowflake; from pathlib import Path; print(Path(adbc_driver_snowflake.__file__).parent)"

# Add to PATH
$env:PATH = "$ADBC_LIB_DIR;$env:PATH"

# Start Arc
uv run arc chat
```

### Step 3: Create Startup Script (Recommended)

To avoid manual setup each time, create a startup script:

#### Linux/macOS: `start-arc.sh`

```bash
#!/bin/bash

cd /path/to/arc

# Snowflake credentials (or put in settings file)
export SNOWFLAKE_ACCOUNT="mycompany.snowflakecomputing.com"
export SNOWFLAKE_USER="username"
export SNOWFLAKE_PASSWORD="password"
export SNOWFLAKE_DATABASE="PROD_DB"
export SNOWFLAKE_WAREHOUSE="COMPUTE_WH"

# Set ADBC library path
ADBC_LIB_DIR=$(uv run python -c "import adbc_driver_snowflake; from pathlib import Path; print(Path(adbc_driver_snowflake.__file__).parent)")
export LD_LIBRARY_PATH="${ADBC_LIB_DIR}:${LD_LIBRARY_PATH}"

# Start Arc
uv run arc chat
```

Make it executable:

```bash
chmod +x start-arc.sh
./start-arc.sh
```

#### Windows: `start-arc.bat`

```batch
@echo off

cd C:\path\to\arc

REM Snowflake credentials (or put in settings file)
set SNOWFLAKE_ACCOUNT=mycompany.snowflakecomputing.com
set SNOWFLAKE_USER=username
set SNOWFLAKE_PASSWORD=password
set SNOWFLAKE_DATABASE=PROD_DB
set SNOWFLAKE_WAREHOUSE=COMPUTE_WH

REM Get ADBC library directory
for /f "delims=" %%i in ('uv run python -c "import adbc_driver_snowflake; from pathlib import Path; print(Path(adbc_driver_snowflake.__file__).parent)"') do set ADBC_LIB_DIR=%%i

REM Add to PATH
set PATH=%ADBC_LIB_DIR%;%PATH%

REM Start Arc
uv run arc chat
```

## Best Practices

### 1. Extract Once, Transform Locally (ELT Pattern)

**Why**: Snowflake charges for compute time. Extracting data once and doing feature engineering locally is much cheaper and faster for iterative ML workflows.

```sql
-- ✅ Good: Extract relevant data once
/sql CREATE TABLE local_sales AS
     SELECT * FROM snowflake.sales.transactions
     WHERE date >= '2024-01-01'

-- Then perform feature engineering locally (fast and free)
/sql CREATE TABLE sales_features AS
     SELECT customer_id,
            COUNT(*) as purchase_count,
            AVG(amount) as avg_purchase
     FROM local_sales
     GROUP BY customer_id
```

```sql
-- ❌ Avoid: Repeated queries to Snowflake ($$$ compute costs)
/sql SELECT COUNT(*) FROM snowflake.sales.transactions  -- $$
/sql SELECT AVG(amount) FROM snowflake.sales.transactions  -- $$
/sql SELECT MAX(date) FROM snowflake.sales.transactions  -- $$
```

### 2. Use Filters When Extracting

Extract only what you need to minimize data transfer and Snowflake compute:

```sql
-- ✅ Good: Filter at source
/sql CREATE TABLE recent_orders AS
     SELECT * FROM snowflake.orders.fact_orders
     WHERE order_date >= CURRENT_DATE - INTERVAL 30 DAY
       AND status = 'completed'

-- ❌ Avoid: Extracting everything
/sql CREATE TABLE all_orders AS
     SELECT * FROM snowflake.orders.fact_orders  -- Millions of rows!
```

### 3. Check Available Tables First

Use schema discovery to understand what's available:

```text
> What tables are in my Snowflake database?
```

Or use SQL:

```sql
/sql SELECT table_name, table_type
     FROM information_schema.tables
     WHERE table_catalog = 'snowflake'
```

### 4. When to Query Directly vs. Extract

**Query Snowflake directly when:**
- Exploring data (one-time queries)
- Aggregations that can be pushed to Snowflake
- Joining Snowflake tables to determine what data to extract

**Extract to local when:**
- Feature engineering (many iterative transformations)
- Model training (requires local data)
- Repeated access to same dataset

## Troubleshooting

### Snowflake Database Not Appearing

If you don't see `snowflake` in your database list:

**1. Verify credentials are configured:**

```bash
cd /path/to/arc
uv run python -c "from arc.core.config import SettingsManager; print(SettingsManager().get_snowflake_config())"
```

**2. Check library path is set:**

```bash
echo $LD_LIBRARY_PATH  # Linux/macOS
echo %PATH%  # Windows (should contain ADBC library directory)
```

**3. Verify ADBC library exists:**

```bash
cd /path/to/arc
uv run python -c "import adbc_driver_snowflake; from pathlib import Path; p = Path(adbc_driver_snowflake.__file__).parent / 'libadbc_driver_snowflake.so'; print(f'Library exists: {p.exists()} at {p}')"
```

**4. Try manual attach:**

```sql
/sql CREATE SECRET snowflake_secret (
    TYPE snowflake,
    ACCOUNT 'mycompany.snowflakecomputing.com',
    USER 'username',
    PASSWORD 'password',
    DATABASE 'PROD_DB',
    WAREHOUSE 'COMPUTE_WH'
)

/sql ATTACH '' AS snowflake (
    TYPE snowflake,
    SECRET snowflake_secret,
    READ_ONLY
)
```

### "Unknown ADBC error" or "Library not found"

**This means the ADBC native library can't be found:**

1. **Check you set the library path BEFORE starting Arc** (not after)
2. **Restart your terminal** after setting environment variables
3. **Use the startup script approach** (see Step 3 in Configuration)
4. **Verify Arc dependencies are installed:** `uv sync --dev`

### Connection Fails

**Possible causes:**

1. **Invalid credentials** - Check username, password, account identifier
2. **Network issues** - Ensure you can reach Snowflake (check firewall, VPN)
3. **Warehouse not running** - Snowflake warehouse must be active (auto-resume should work)
4. **Insufficient permissions** - User must have access to the database and warehouse

**Test connection outside Arc:**

```bash
uv run python -c "
import adbc_driver_snowflake.dbapi as snowflake
conn = snowflake.connect(
    account='mycompany.snowflakecomputing.com',
    user='username',
    password='password',
    database='PROD_DB',
    warehouse='COMPUTE_WH'
)
print('Connection successful!')
"
```

### Slow Queries

- **Extract data to local** instead of querying Snowflake repeatedly
- **Use WHERE filters** to reduce data scanned in Snowflake
- **Check warehouse size** - larger warehouses = faster queries (but higher cost)
- **Optimize Snowflake tables** - clustering, partitioning (outside Arc scope)

## Architecture & How It Works

### Integration Stack

Arc's Snowflake integration uses:

- **DuckDB Snowflake Extension** (community) - `snowflake` extension from DuckDB's community repository
- **ADBC Driver** (Python) - `adbc-driver-snowflake` package (installed automatically with Arc)
- **Native Library** - Platform-specific library files:
  - Linux: `libadbc_driver_snowflake.so`
  - macOS: `libadbc_driver_snowflake.dylib`
  - Windows: `adbc_driver_snowflake.dll`

### Database Attachment

When Arc starts with Snowflake credentials configured:

1. **DuckDB installs the Snowflake extension** (from community repository)
2. **Creates a DuckDB secret** with Snowflake credentials
3. **Attaches Snowflake as a read-only database** named `snowflake`

Once attached, Snowflake tables appear in DuckDB's `INFORMATION_SCHEMA`, making them:
- **Discoverable** via Arc's schema discovery tool
- **Queryable** using standard SQL
- **Joinable** with local DuckDB tables and S3 data

### Query Execution

**For queries against Snowflake tables:**
1. DuckDB sends query to Snowflake via ADBC
2. Snowflake executes the query on its compute
3. Results are streamed back to DuckDB via Arrow format
4. DuckDB presents results to Arc

**For joins between Snowflake and local data:**
1. DuckDB fetches Snowflake data needed for the join
2. Join is executed locally in DuckDB
3. Results are returned to Arc

### Credential Chain

Arc checks for Snowflake credentials in this order:

1. **Environment variables** (`SNOWFLAKE_*`)
2. **Arc settings file** (`~/.arc/user-settings.json`)

Environment variables take precedence over the settings file.

### Read-Only Access

Snowflake is attached with `READ_ONLY` flag to prevent accidental writes to Snowflake from Arc. This protects production data.

## Security Best Practices

1. **Never commit credentials** to version control
2. **Use environment variables** for CI/CD and shared environments
3. **Rotate passwords regularly**
4. **Use read-only credentials** when possible (least privilege)
5. **Limit database/warehouse access** to only what's needed
6. **Use Snowflake roles** to manage permissions
7. **Enable MFA** on Snowflake accounts

## Additional Resources

- [DuckDB Snowflake Extension](https://duckdb.org/docs/extensions/snowflake)
- [ADBC Documentation](https://arrow.apache.org/adbc/)
- [Snowflake Documentation](https://docs.snowflake.com/)
- [Snowflake Best Practices](https://docs.snowflake.com/en/user-guide/admin-best-practices)
