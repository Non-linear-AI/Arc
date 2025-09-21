# Repository Guidelines

## Project Structure & Module Organization
- `src/arc/` houses the runtime packages; e.g. `core/` coordinates pipeline assembly, `ml/` wraps modeling utilities, `database/` manages persistence layers, `ui/` powers the CLI, and `plugins/` extends third-party integrations.
- Tests live in `tests/` mirroring the source tree (`tests/ml`, `tests/ui`, etc.) with shared fixtures under `tests/fixtures`.
- CLI assets and prompts sit in `templates/` and `editing/`; keep generated artefacts (`artifacts/`, `htmlcov/`) out of commits unless they illustrate a change.
- Project-wide configuration resides in `pyproject.toml`, and example user flows are under `examples/`.

## Build, Test, and Development Commands
- `uv sync --dev` installs editable dependencies; rerun after changing `pyproject.toml`.
- `uv run arc chat` or `make run` launches the interactive assistant.
- `uv run pytest` executes the test suite, while `uv run pytest --cov` (or `make test`) generates terminal and HTML coverage in `htmlcov/`.
- `make lint`, `make format`, and `uv run ruff check . --fix` enforce style; `make all` runs format, lint, and tests in one pass.

## Coding Style & Naming Conventions
- Target Python 3.12, 4-space indentation, and Ruff’s 88-character limit.
- Modules and packages use `snake_case`; classes use `PascalCase`; functions, methods, and variables use `snake_case`.
- Prefer type hints and concise docstrings on user-facing APIs; keep imports sorted via Ruff (first-party is `arc`).
- Use `ruff format .` before committing to maintain consistent formatting.

## Testing Guidelines
- Create `pytest` tests beside the code they cover and name files `test_*.py` or `*_test.py` to honour the configured patterns.
- Share test data via fixtures in `tests/fixtures` and mark long-running tests explicitly (`pytest --strict-markers` will enforce declarations).
- Ensure coverage stays high by running `uv run pytest --cov` and reviewing `htmlcov/index.html` when adding complex logic.

## Commit & Pull Request Guidelines
- Write small, imperative commit subjects (e.g., `add streaming prediction`) and include focused bodies when context is needed.
- Group related changes per commit; avoid committing generated artefacts.
- Pull requests should describe intent, list manual verification (commands run, screenshots for UI tweaks), and reference issues or design docs.
- Confirm `make all` passes locally before requesting review to keep CI noise minimal.

## Agent Workflow Tips
- Use `uv run arc --help` to explore CLI options quickly, and `make clean` to reset build outputs when switching branches.
- Document any new environment variables or secrets in the PR so downstream agents can reproduce the setup.
