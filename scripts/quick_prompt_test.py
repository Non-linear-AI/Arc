#!/usr/bin/env python3
"""
Quick prompt testing script for manual LLM testing.

This script outputs a single rendered prompt to stdout, making it easy to
copy-paste into LLM interfaces for manual testing.

Usage:
    # Test MLP architecture template
    python scripts/quick_prompt_test.py mlp

    # Test Transformer architecture template
    python scripts/quick_prompt_test.py transformer
"""

import argparse
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from jinja2 import Environment, FileSystemLoader


def load_actual_components():
    """Load available components using the exact same logic as ModelGeneratorAgent._get_model_components()."""
    try:
        # Import and use exact same logic as ModelGeneratorAgent._get_model_components()
        from arc.graph.model import CORE_LAYERS, TORCH_FUNCTIONS
        # Combine both CORE_LAYERS (nn.Module classes) and TORCH_FUNCTIONS (functional components)
        all_components = list(CORE_LAYERS.keys()) + list(TORCH_FUNCTIONS.keys())
        return {
            "node_types": all_components,
            "description": "PyTorch components available in Arc-Graph include layers (instantiated once, used in forward pass) and functions (applied as operations). All standard PyTorch neural network components are supported."
        }
    except Exception as e:
        print(f"Warning: Could not load components: {e}")
        # Minimal fallback
        return {
            "node_types": ["torch.nn.Linear", "torch.nn.functional.sigmoid"],
            "description": "PyTorch components available in Arc-Graph include layers and functions."
        }


def load_architecture_content(architecture_types):
    """Load architecture-specific content from files."""
    if isinstance(architecture_types, str):
        architecture_types = [architecture_types]

    architecture_guides = {}
    for arch_type in architecture_types:
        content_path = project_root / "src" / "arc" / "core" / "agents" / "model_generator" / "templates" / "architectures" / f"{arch_type}.md"
        try:
            with open(content_path, 'r') as f:
                architecture_guides[arch_type.upper()] = f.read()
        except FileNotFoundError:
            architecture_guides[arch_type.upper()] = f"*Content not found for {arch_type}*"

    return architecture_guides

def load_examples_content(architecture_type, complexity="simple"):
    """Load relevant examples for the architecture."""
    examples_path = project_root / "src" / "arc" / "core" / "agents" / "model_generator" / "templates" / "examples" / f"{architecture_type}_{complexity}.md"
    try:
        with open(examples_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"## Examples\n*No examples found for {architecture_type}*"

def create_sample_data(architecture_types):
    """Create sample data for template rendering with known column information."""
    if isinstance(architecture_types, str):
        architecture_types = [architecture_types]

    # Architecture display names
    display_names = {
        "mlp": "Multi-Layer Perceptron (Feedforward Neural Network)",
        "transformer": "Transformer (Attention-based Neural Network)",
        "dcn": "Deep & Cross Network",
        "mmoe": "Multi-gate Mixture of Experts"
    }

    # Create display name from multiple architectures
    if len(architecture_types) == 1:
        display_name = display_names.get(architecture_types[0], architecture_types[0].upper())
    else:
        display_name = " + ".join([display_names.get(arch, arch.upper()) for arch in architecture_types])

    return {
        "architecture_type": "+".join(architecture_types),
        "architecture_display_name": display_name,
        "model_name": "churn_predictor",
        "user_intent": "predict customer churn using demographic and behavioral features",
        "data_profile": {
            "table_name": "customers",
            "columns": [
                {"name": "age", "type": "int"},
                {"name": "income", "type": "float"},
                {"name": "credit_score", "type": "int"},
                {"name": "account_balance", "type": "float"},
                {"name": "years_with_bank", "type": "int"},
            ]
        },
        "available_components": load_actual_components(),
        "architecture_guides": load_architecture_content(architecture_types)
    }


def main():
    parser = argparse.ArgumentParser(description="Quick prompt test for manual LLM testing")
    parser.add_argument("architectures", nargs="+",
                       choices=["mlp", "transformer", "dcn", "mmoe"],
                       help="Architecture types (can specify multiple)")

    args = parser.parse_args()

    # Get template directory and set up Jinja2
    template_dir = project_root / "src" / "arc" / "core" / "agents" / "model_generator" / "templates"
    env = Environment(loader=FileSystemLoader(template_dir))

    # Load base template
    template = env.get_template("base.j2")
    context = create_sample_data(args.architectures)

    rendered = template.render(**context)

    arch_names = "+".join(args.architectures).upper()
    print("=" * 80)
    print(f"PROMPT FOR: {arch_names} ARCHITECTURE")
    print("=" * 80)
    print()
    print(rendered)
    print()
    print("=" * 80)
    print("END OF PROMPT")
    print("=" * 80)


if __name__ == "__main__":
    main()