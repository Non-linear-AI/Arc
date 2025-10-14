"""Command-line parsing utilities for Arc CLI."""


class OptionParsingError(Exception):
    """Raised when option parsing fails."""


def parse_options(
    args: list[str],
    spec: dict[str, bool],
    command_name: str = "",
) -> dict[str, str | bool]:
    """Parse command-line options from argument list.

    Args:
        args: List of argument tokens to parse
        spec: Dict mapping option names to whether they expect values
              Example: {"name": True, "force": False}
              - True: option expects a value (--name value)
              - False: option is a flag (--force)
        command_name: Command name for better error messages

    Returns:
        Dict of parsed options with string values or bool flags
        Example: {"name": "my_model", "force": True}

    Raises:
        OptionParsingError: If parsing fails due to:
            - Unexpected argument (doesn't start with --)
            - Unknown option (not in spec)
            - Missing value for option that expects one

    Example:
        >>> spec = {"name": True, "data": True, "force": False}
        >>> parse_options(["--name", "model1", "--force"], spec)
        {"name": "model1", "force": True}
    """
    options: dict[str, str | bool] = {}
    idx = 0

    while idx < len(args):
        token = args[idx]

        # Check for -- prefix
        if not token.startswith("--"):
            raise OptionParsingError(
                f"Unexpected argument '{token}'{_cmd_suffix(command_name)}"
            )

        # Extract option name
        key = token[2:]

        # Check if option is known
        if key not in spec:
            valid_options = ", ".join(f"--{k}" for k in sorted(spec.keys()))
            raise OptionParsingError(
                f"Unknown option '--{key}'{_cmd_suffix(command_name)}\n"
                f"Valid options: {valid_options}"
            )

        expects_value = spec[key]

        # Handle flag options (no value expected)
        if not expects_value:
            options[key] = True
            idx += 1
            continue

        # Handle options that expect values
        idx += 1
        if idx >= len(args):
            raise OptionParsingError(
                f"Option '--{key}' requires a value{_cmd_suffix(command_name)}"
            )

        options[key] = args[idx]
        idx += 1

    return options


def _cmd_suffix(command_name: str) -> str:
    """Helper to format command name in error messages.

    Args:
        command_name: Name of the command

    Returns:
        Formatted suffix like " in /ml train" or empty string
    """
    return f" in {command_name}" if command_name else ""
