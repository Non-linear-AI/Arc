def main() -> None:
    """Entry point for Arc CLI."""
    import warnings

    # Suppress SyntaxWarning from third-party sql_formatter package
    warnings.filterwarnings("ignore", category=SyntaxWarning, module="sql_formatter")

    from arc.ui.cli import cli

    cli()
