# Installation

This guide will help you install Arc on your system.

## Prerequisites

- **Python 3.12 or higher** (Python 3.13 is also supported)
- **uv** - Fast Python package manager (we'll install this if you don't have it)

## Installing uv

Arc uses `uv` for dependency management. If you don't have `uv` installed:

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.sh | iex"
```

## Installing Arc

### Option 1: Install from PyPI (Recommended)

Install the latest stable release from PyPI:

```bash
pip install nl-arc
```

**Note:** The package is named `nl-arc` on PyPI (to avoid naming conflicts), but the command you run is still `arc`.

### Option 2: Install from Source (Development)

Clone the repository and install in editable mode:

```bash
# Clone the repository
git clone https://github.com/non-linear-ai/arc
cd arc

# Install dependencies
uv sync

# Install with development dependencies (for contributors)
uv sync --dev
```

## Verifying Installation

Verify Arc is installed correctly:

```bash
# Check version
arc --help

# Should display Arc's help message with available commands
```

You should see output showing Arc's available commands and options.

## Next Steps

Now that Arc is installed, continue to:

- **[Quick Start Tutorial](quickstart.md)** - Build your first model
- **[Configuration](configuration.md)** - Set up your API keys

## Troubleshooting

### Python Version Issues

If you get a Python version error:

```bash
# Check your Python version
python --version

# Must be 3.12 or higher
```

If you need to install Python 3.12+:
- **macOS**: `brew install python@3.12`
- **Ubuntu/Debian**: `sudo apt install python3.12`
- **Windows**: Download from [python.org](https://www.python.org/downloads/)

### uv Installation Issues

If `uv` fails to install:

1. Check the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
2. Try alternative installation methods (pip, cargo, etc.)

### Package Not Found

If `pip install nl-arc` fails:

```bash
# Update pip first
pip install --upgrade pip

# Try again
pip install nl-arc
```

### Import Errors

If you get import errors after installation:

```bash
# Ensure you're in a clean environment
python -c "import arc; print('Success!')"
```

If this fails, try creating a new virtual environment:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install nl-arc
```

## Need Help?

- Check [GitHub Issues](https://github.com/non-linear-ai/arc/issues)
- Open a [new issue](https://github.com/non-linear-ai/arc/issues/new) with details about your system and error messages
