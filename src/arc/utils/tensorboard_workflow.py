"""TensorBoard preference workflow for first-time setup."""


async def prompt_tensorboard_preference(ui) -> str:
    """Show first-time TensorBoard setup and return user preference.

    Args:
        ui: InteractiveInterface instance

    Returns:
        User's preference: "always", "ask", or "never"
    """
    ui._printer.console.print()
    ui._printer.console.print("[bold cyan]📈 TensorBoard Visualization[/bold cyan]")
    ui._printer.console.print()
    ui._printer.console.print(
        "Arc can automatically launch TensorBoard to visualize your training metrics."
    )
    ui._printer.console.print(
        "This opens a local web server at [bold]http://localhost:6006[/bold]"
    )
    ui._printer.console.print()
    ui._printer.console.print("How would you like to handle TensorBoard?")
    ui._printer.console.print()

    choice = await ui._printer.get_choice_async(
        options=[
            ("always", "Always launch automatically"),
            ("ask", "Ask me each time"),
            ("never", "Never launch (manual only)"),
        ],
        default="ask",
    )

    ui._printer.console.print()
    ui._printer.console.print(
        "[dim]Your choice is saved to ~/.arc/user-settings.json "
        "(change with /config tensorboard_mode <mode>)[/dim]"
    )

    return choice
