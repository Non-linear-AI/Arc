"""TensorBoard preference workflow for first-time setup."""


async def prompt_tensorboard_preference(ui) -> tuple[str, bool]:
    """Show TensorBoard dialog with launch and preference options.

    Args:
        ui: InteractiveInterface instance

    Returns:
        Tuple of (preference_mode, should_launch):
            - preference_mode: "always", "ask", or "never"
            - should_launch: True if user wants to launch now, False otherwise
    """
    ui._printer.console.print()
    ui._printer.console.print("[bold cyan]📈 TensorBoard Visualization[/bold cyan]")
    ui._printer.console.print()
    ui._printer.console.print(
        "Arc can launch TensorBoard to visualize your training metrics."
    )
    ui._printer.console.print(
        "This opens a local web server at [bold]http://localhost:6006[/bold]"
    )
    ui._printer.console.print()
    ui._printer.console.print("How would you like to handle TensorBoard?")
    ui._printer.console.print()

    choice = await ui._printer.get_choice_async(
        options=[
            ("yes", "Yes, launch now"),
            ("always", "Always launch automatically"),
            ("no", "No, skip for now"),
        ],
        default="yes",
    )

    ui._printer.console.print()
    ui._printer.console.print(
        "[dim]Your choice is saved to ~/.arc/user-settings.json "
        "(change with /config tensorboard_mode <mode>)[/dim]"
    )

    # Map choice to preference and launch decision
    if choice == "yes":
        return ("ask", True)  # Launch now, ask next time
    elif choice == "always":
        return ("always", True)  # Launch now, always launch
    else:  # "no"
        return ("ask", False)  # Don't launch, ask next time
