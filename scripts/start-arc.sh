#!/bin/bash
#
# Arc Startup Script with Auto-configured Library Paths
#
# This script automatically configures the library path environment variables
# needed for Snowflake integration and then starts Arc.
#
# Usage:
#   ./scripts/start-arc.sh
#   ./scripts/start-arc.sh [additional arc options]
#
# This script is an alternative to the automatic restart mechanism built into
# the main `arc` command. Use this if you prefer explicit control over the
# environment setup process.

set -e  # Exit on error

# Auto-detect ADBC library directory
echo "Detecting ADBC library path..."
ADBC_LIB_DIR=$(uv run python -c "import adbc_driver_snowflake; from pathlib import Path; print(Path(adbc_driver_snowflake.__file__).parent)" 2>/dev/null)

if [ -n "$ADBC_LIB_DIR" ]; then
    echo "Found ADBC library at: $ADBC_LIB_DIR"

    # Set library path based on platform
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Setting DYLD_LIBRARY_PATH (macOS)..."
        export DYLD_LIBRARY_PATH="${ADBC_LIB_DIR}:${DYLD_LIBRARY_PATH}"
    else
        echo "Setting LD_LIBRARY_PATH (Linux)..."
        export LD_LIBRARY_PATH="${ADBC_LIB_DIR}:${LD_LIBRARY_PATH}"
    fi
else
    echo "Warning: ADBC library not found. Snowflake integration may not be available."
fi

# Start Arc with all passed arguments
echo "Starting Arc..."
uv run arc chat "$@"
