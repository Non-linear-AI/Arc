#!/usr/bin/env python3
"""Extract model YAML from Arc database.

This script extracts the YAML specification from a saved model and
optionally saves it to a file.

Usage:
    python scripts/extract_model_yaml.py <model_id> [--output FILE]

Examples:
    # Print to stdout
    python scripts/extract_model_yaml.py test-v8

    # Save to file
    python scripts/extract_model_yaml.py test-v8 --output test-v8.yaml

    # Get latest version of a model
    python scripts/extract_model_yaml.py test --output test-latest.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from arc.database import get_database_manager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract model YAML from Arc database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "model_name",
        help="Model name or model ID (e.g., 'test' or 'test-v8')"
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: print to stdout)"
    )

    parser.add_argument(
        "--db-path",
        default="~/.arc/arc_system.db",
        help="Path to Arc database (default: ~/.arc/arc_system.db)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    try:
        # Initialize database
        db_path = Path(args.db_path).expanduser()
        db_manager = get_database_manager(str(db_path))

        # Try to get model by ID first
        model = db_manager.services.models.get_model_by_id(args.model_name)

        # If not found, try to get latest version by name
        if model is None:
            model = db_manager.services.models.get_latest_model_by_name(args.model_name)

        if model is None:
            print(f"Error: Model not found: {args.model_name}", file=sys.stderr)
            return 1

        # Print model info to stderr so it doesn't interfere with YAML output
        print(f"# Model: {model.id}", file=sys.stderr)
        print(f"# Name: {model.name}", file=sys.stderr)
        print(f"# Version: {model.version}", file=sys.stderr)
        print(f"# Description: {model.description}", file=sys.stderr)
        print(f"# Created: {model.created_at}", file=sys.stderr)
        print("", file=sys.stderr)

        # Get YAML content
        yaml_content = model.spec

        # Output YAML
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(yaml_content)
            print(f"✓ YAML saved to: {output_path}", file=sys.stderr)
        else:
            print(yaml_content)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
