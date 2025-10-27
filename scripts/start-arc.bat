@echo off
REM
REM Arc Startup Script with Auto-configured Library Paths (Windows)
REM
REM This script automatically configures the library path environment variables
REM needed for Snowflake integration and then starts Arc.
REM
REM Usage:
REM   scripts\start-arc.bat
REM   scripts\start-arc.bat [additional arc options]
REM
REM This script is an alternative to the automatic restart mechanism built into
REM the main `arc` command. Use this if you prefer explicit control over the
REM environment setup process.

echo Detecting ADBC library path...

REM Auto-detect ADBC library directory
for /f "delims=" %%i in ('uv run python -c "import adbc_driver_snowflake; from pathlib import Path; print(Path(adbc_driver_snowflake.__file__).parent)" 2^>nul') do set ADBC_LIB_DIR=%%i

if defined ADBC_LIB_DIR (
    echo Found ADBC library at: %ADBC_LIB_DIR%
    echo Setting PATH...
    set PATH=%ADBC_LIB_DIR%;%PATH%
) else (
    echo Warning: ADBC library not found. Snowflake integration may not be available.
)

REM Start Arc with all passed arguments
echo Starting Arc...
uv run arc chat %*
