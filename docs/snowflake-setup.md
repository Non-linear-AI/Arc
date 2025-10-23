# Snowflake Integration Setup

Arc integrates with Snowflake using DuckDB's Snowflake extension and the ADBC (Arrow Database Connectivity) driver. This guide walks you through the complete setup process.

## Prerequisites

Arc automatically installs the required Python package (`adbc-driver-snowflake`) when you install Arc. However, for the integration to work, you need to configure your environment properly.

## Configuration

### Step 1: Configure Snowflake Credentials

Add your Snowflake credentials to `~/.arc/user-settings.json`:

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

Or use environment variables (takes precedence over settings file):

```bash
export SNOWFLAKE_ACCOUNT="mycompany.snowflakecomputing.com"
export SNOWFLAKE_USER="username"
export SNOWFLAKE_PASSWORD="password"
export SNOWFLAKE_DATABASE="PROD_DB"
export SNOWFLAKE_WAREHOUSE="COMPUTE_WH"
export SNOWFLAKE_SCHEMA="PUBLIC"  # Optional, defaults to PUBLIC
```

**Note:** The `snowflakeSchema` field is optional and defaults to `PUBLIC` if not specified.

### Step 2: Set Library Path (Linux/macOS)

The ADBC driver includes native libraries that need to be discoverable by DuckDB. You need to set the library path **before** starting Arc.

#### On Linux:

```bash
# Navigate to Arc directory
cd /path/to/arc

# Find the ADBC library directory
ADBC_LIB_DIR=$(uv run python -c "import adbc_driver_snowflake; from pathlib import Path; print(Path(adbc_driver_snowflake.__file__).parent)")

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${ADBC_LIB_DIR}:${LD_LIBRARY_PATH}"

# Now start Arc
uv run arc chat
```

#### On macOS:

```bash
# Navigate to Arc directory
cd /path/to/arc

# Find the ADBC library directory
ADBC_LIB_DIR=$(uv run python -c "import adbc_driver_snowflake; from pathlib import Path; print(Path(adbc_driver_snowflake.__file__).parent)")

# Set DYLD_LIBRARY_PATH
export DYLD_LIBRARY_PATH="${ADBC_LIB_DIR}:${DYLD_LIBRARY_PATH}"

# Now start Arc
uv run arc chat
```

#### On Windows:

```powershell
# Navigate to Arc directory (PowerShell)
cd C:\path\to\arc

# Find the ADBC library directory
$ADBC_LIB_DIR = uv run python -c "import adbc_driver_snowflake; from pathlib import Path; print(Path(adbc_driver_snowflake.__file__).parent)"

# Add to PATH
$env:PATH = "$ADBC_LIB_DIR;$env:PATH"

# Now start Arc
uv run arc chat
```

### Step 3: Create a Startup Script (Recommended)

To avoid setting the library path manually each time, create a startup script:

#### Linux/macOS: `start-arc.sh`

```bash
#!/bin/bash

# Navigate to Arc directory
cd /path/to/arc

# Set Snowflake credentials
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

REM Navigate to Arc directory
cd C:\path\to\arc

REM Set Snowflake credentials
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

## Usage

Once configured, Snowflake will automatically attach when you start Arc. You can query Snowflake tables directly:

```sql
-- View available schemas (Snowflake will appear as "snowflake" database)
/sql SELECT * FROM duckdb_schemas()

-- Query Snowflake tables directly
/sql SELECT * FROM snowflake.public.customers LIMIT 10

-- Extract data for local processing (recommended for feature engineering)
/sql CREATE TABLE local_customers AS
     SELECT * FROM snowflake.public.customers
     WHERE signup_date >= '2024-01-01'

-- Join Snowflake data with local data
/sql SELECT s.customer_id, s.total_spend, l.prediction
     FROM snowflake.public.customers s
     JOIN local_predictions l ON s.customer_id = l.id
```

## Best Practices

### 1. Extract Once, Transform Locally (ELT Pattern)

For cost efficiency and performance:

```sql
-- ✅ Good: Extract relevant data once
/sql CREATE TABLE local_sales AS
     SELECT * FROM snowflake.sales.transactions
     WHERE date >= '2024-01-01'

-- Then perform feature engineering locally (fast and free)
/sql SELECT customer_id,
            COUNT(*) as purchase_count,
            AVG(amount) as avg_purchase
     FROM local_sales
     GROUP BY customer_id
```

```sql
-- ❌ Avoid: Repeated queries to Snowflake
/sql SELECT COUNT(*) FROM snowflake.sales.transactions  -- $$
/sql SELECT AVG(amount) FROM snowflake.sales.transactions  -- $$
```

### 2. Use Filters When Extracting

Extract only what you need:

```sql
-- ✅ Good: Filter at source
/sql CREATE TABLE recent_orders AS
     SELECT * FROM snowflake.orders.fact_orders
     WHERE order_date >= CURRENT_DATE - INTERVAL 30 DAY
       AND status = 'completed'

-- ❌ Avoid: Extracting everything
/sql CREATE TABLE all_orders AS
     SELECT * FROM snowflake.orders.fact_orders
```

### 3. Check Available Tables First

Use schema discovery to understand what's available:

```text
> What tables are in my Snowflake database?
```

Or use SQL:

```sql
/sql SELECT * FROM information_schema.tables
     WHERE table_catalog = 'snowflake'
```

## Troubleshooting

### Snowflake Database Not Appearing

If you don't see `snowflake` in your database list:

1. **Verify credentials are configured:**
   ```bash
   cd /path/to/arc
   uv run python -c "from arc.core.config import SettingsManager; print(SettingsManager().get_snowflake_config())"
   ```

2. **Check library path is set:**
   ```bash
   echo $LD_LIBRARY_PATH  # Linux/macOS
   echo %PATH%  # Windows
   ```

3. **Verify ADBC library exists:**
   ```bash
   cd /path/to/arc
   uv run python -c "import adbc_driver_snowflake; from pathlib import Path; p = Path(adbc_driver_snowflake.__file__).parent / 'libadbc_driver_snowflake.so'; print(f'Library exists: {p.exists()} at {p}')"
   ```

4. **Try manual attach:**
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

This means the ADBC native library can't be found:

1. **Check you set the library path BEFORE starting Arc**
2. **Restart your terminal after setting environment variables**
3. **Use the startup script approach** (see Step 3 above)

### Connection Fails

1. **Verify credentials:** Check username, password, account name
2. **Check network:** Ensure you can reach Snowflake (firewall, VPN)
3. **Verify warehouse is running:** Snowflake warehouse must be active
4. **Check permissions:** User must have access to the database and warehouse

## Architecture Notes

Arc uses DuckDB's Snowflake extension which connects via ADBC:

- **DuckDB Snowflake Extension** (community): `snowflake` extension
- **ADBC Driver** (Python): `adbc-driver-snowflake` package
- **Native Library**: `libadbc_driver_snowflake.so` (Linux), `.dylib` (macOS), `.dll` (Windows)

The ATTACH command makes Snowflake appear as a schema in DuckDB's catalog, allowing seamless joins between Snowflake and local data.

## Security Best Practices

1. **Never commit credentials** to version control
2. **Use environment variables** instead of settings files for CI/CD
3. **Rotate passwords regularly**
4. **Use read-only credentials** when possible
5. **Limit database/warehouse access** to only what's needed

## Additional Resources

- [DuckDB Snowflake Extension](https://duckdb.org/docs/extensions/snowflake)
- [ADBC Documentation](https://arrow.apache.org/adbc/)
- [Snowflake Documentation](https://docs.snowflake.com/)
