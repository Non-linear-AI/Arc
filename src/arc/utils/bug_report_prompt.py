"""Bug report generation using Jinja2 template."""

from pathlib import Path

import jinja2


def _get_template_directory() -> Path:
    """Get the template directory for bug report prompts.

    Returns:
        Path to the templates directory
    """
    return Path(__file__).parent / "templates"


def _load_template(template_name: str) -> jinja2.Template:
    """Load a Jinja2 template.

    Args:
        template_name: Name of the template file

    Returns:
        Loaded Jinja2 template

    Raises:
        Exception: If template loading fails
    """
    template_dir = _get_template_directory()
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template(template_name)


def _render_bug_report_prompt(chat_history: str) -> str:
    """Render the bug report prompt template.

    Args:
        chat_history: Formatted chat history string

    Returns:
        Rendered prompt string
    """
    template = _load_template("bug_report.j2")
    return template.render(chat_history=chat_history)


def format_chat_history(chat_history: list, max_messages: int = 20) -> str:
    """Format chat history for bug report generation.

    Args:
        chat_history: List of chat messages
        max_messages: Maximum number of recent messages to include

    Returns:
        Formatted chat history string
    """
    # Take last N messages
    recent = chat_history[-max_messages:] if len(chat_history) > max_messages else chat_history

    formatted = []
    for msg in recent:
        role = msg.type if hasattr(msg, 'type') else 'unknown'
        content = msg.content if hasattr(msg, 'content') else str(msg)

        # Truncate very long messages
        if len(content) > 2000:
            content = content[:2000] + "... [truncated]"

        formatted.append(f"{role.upper()}: {content}")

    return "\n\n".join(formatted)


async def generate_bug_report(agent, max_messages: int = 20) -> dict[str, str] | None:
    """Generate bug report from chat history using LLM.

    Args:
        agent: ArcAgent instance with chat history
        max_messages: Maximum number of recent messages to analyze

    Returns:
        Dict with keys: title, description, steps, expected, actual, context
        Returns None if generation fails or no issue detected
    """
    if not agent or not hasattr(agent, 'chat_history') or not agent.chat_history:
        return None

    # Format chat history
    chat_text = format_chat_history(agent.chat_history, max_messages)

    # Render prompt from Jinja2 template
    prompt = _render_bug_report_prompt(chat_text)

    # Create messages for LLM
    messages = [
        {"role": "user", "content": prompt}
    ]

    try:
        # Call LLM using agent's client
        response = await agent.client.create_chat_completion(
            messages=messages,
            temperature=0.3,  # Lower temperature for more focused output
        )

        # Extract content
        content = response.choices[0].message.content

        # Parse structured response
        report = parse_bug_report(content)
        return report

    except Exception:
        return None


def parse_bug_report(content: str) -> dict[str, str]:
    """Parse LLM-generated bug report into structured format.

    Args:
        content: LLM response text

    Returns:
        Dict with parsed sections
    """
    sections = {
        "title": "",
        "description": "",
        "steps": "",
        "expected": "",
        "actual": "",
        "context": ""
    }

    # Split by section headers
    lines = content.split('\n')
    current_section = None

    for line in lines:
        line_upper = line.strip().upper()

        if line_upper.startswith('TITLE:'):
            sections['title'] = line.split(':', 1)[1].strip()
            current_section = None
        elif line_upper.startswith('DESCRIPTION:'):
            current_section = 'description'
        elif line_upper.startswith('STEPS TO REPRODUCE:'):
            current_section = 'steps'
        elif line_upper.startswith('EXPECTED BEHAVIOR:'):
            current_section = 'expected'
        elif line_upper.startswith('ACTUAL BEHAVIOR:'):
            current_section = 'actual'
        elif line_upper.startswith('CONTEXT:'):
            current_section = 'context'
        elif current_section and line.strip():
            # Append to current section
            if sections[current_section]:
                sections[current_section] += '\n' + line.strip()
            else:
                sections[current_section] = line.strip()

    return sections


def format_bug_report_for_display(report: dict[str, str]) -> str:
    """Format parsed bug report for user preview.

    Args:
        report: Parsed bug report dict

    Returns:
        Formatted string for display
    """
    parts = []

    if report['title']:
        parts.append(f"Title: {report['title']}")
        parts.append("")

    if report['description']:
        parts.append("Description:")
        parts.append(report['description'])
        parts.append("")

    if report['steps']:
        parts.append("Steps to Reproduce:")
        parts.append(report['steps'])
        parts.append("")

    if report['expected']:
        parts.append("Expected Behavior:")
        parts.append(report['expected'])
        parts.append("")

    if report['actual']:
        parts.append("Actual Behavior:")
        parts.append(report['actual'])
        parts.append("")

    if report['context']:
        parts.append("Context:")
        parts.append(report['context'])

    return "\n".join(parts)
