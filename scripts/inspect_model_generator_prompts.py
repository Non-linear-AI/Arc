#!/usr/bin/env python3
"""
Script to inspect model generator prompts without LLM dependency.

This script renders the model generator templates with sample data so you can
manually inspect the generated prompts for different categories and scenarios.

Usage:
    python scripts/inspect_model_generator_prompts.py
    python scripts/inspect_model_generator_prompts.py --category mlp
    python scripts/inspect_model_generator_prompts.py --category transformer
"""

import argparse
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from jinja2 import Environment, FileSystemLoader


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

def create_sample_data_profile():
    """Create sample data profile for template rendering with known column information."""
    return {
        "table_name": "customer_data",
        "columns": [
            {"name": "age", "type": "int"},
            {"name": "income", "type": "float"},
            {"name": "credit_score", "type": "int"},
            {"name": "account_balance", "type": "float"},
            {"name": "years_with_bank", "type": "int"},
            {"name": "num_products", "type": "int"},
            {"name": "has_credit_card", "type": "bool"},
            {"name": "is_active_member", "type": "bool"},
        ]
    }


def create_sample_available_components():
    """Create sample available components."""
    return {
        "node_types": [
            "torch.nn.Linear", "torch.nn.Conv2d", "torch.nn.LSTM", "torch.nn.Dropout",
            "torch.nn.BatchNorm1d", "torch.nn.LayerNorm", "torch.nn.MultiheadAttention",
            "torch.nn.Embedding", "torch.nn.Sequential", "torch.nn.Transformer",
            "torch.nn.functional.relu", "torch.nn.functional.sigmoid", "torch.nn.functional.softmax",
            "torch.nn.functional.tanh", "torch.nn.functional.gelu", "torch.nn.functional.dropout",
            "torch.cat", "torch.stack", "torch.mean", "torch.sum", "torch.matmul",
            "torch.unsqueeze", "torch.flatten", "arc.stack"
        ],
        "description": "PyTorch components available in Arc-Graph include layers (instantiated once, used in forward pass) and functions (applied as operations). All standard PyTorch neural network components are supported."
    }


def create_sample_examples():
    """Create sample examples."""
    return [
        {
            "name": "Simple Binary Classifier",
            "schema": """inputs:
  features:
    dtype: float32
    shape: [null, 10]
    columns: [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10]

graph:
  - name: classifier
    type: torch.nn.Linear
    params:
      in_features: 10
      out_features: 1
      bias: true
    inputs:
      input: features

  - name: sigmoid
    type: torch.nn.functional.sigmoid
    inputs:
      input: classifier.output

outputs:
  prediction: sigmoid.output"""
        }
    ]


def render_template(category: str):
    """Render a specific template with sample data."""
    # Get the template directory directly
    template_dir = project_root / "src" / "arc" / "core" / "agents" / "model_generator" / "templates"

    # Set up Jinja2 environment
    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True
    )

    # Architecture display names
    display_names = {
        "mlp": "Multi-Layer Perceptron (Feedforward Neural Network)",
        "transformer": "Transformer (Attention-based Neural Network)",
        "dcn": "Deep & Cross Network",
        "mmoe": "Multi-gate Mixture of Experts"
    }

    # Use base template for new architecture system, fallback to old templates for legacy
    if category in ["mlp", "transformer", "dcn", "mmoe"]:
        template_path = "base.j2"

        # Prepare template context for new modular system
        data_profile = create_sample_data_profile()
        context = {
            "architecture_type": category,
            "architecture_display_name": display_names.get(category, category.upper()),
            "model_name": "sample_model",
            "user_intent": "predict customer churn probability using demographic and behavioral features",
            "data_profile": data_profile,
            "available_components": create_sample_available_components(),
            "architecture_guides": load_architecture_content([category])
        }
    else:
        # Legacy template support
        if category == "original":
            template_path = "prompt.j2"
        elif category == "tabular":
            template_path = "tabular/deep_tabular.j2"
        elif category == "fallback":
            template_path = "fallback/generic.j2"
        else:
            raise ValueError(f"Unknown category: {category}")

        # Legacy context
        data_profile = create_sample_data_profile()
        context = {
            "model_name": "sample_model",
            "user_intent": "predict customer churn probability using demographic and behavioral features",
            "data_profile": data_profile,
            "available_components": create_sample_available_components(),
            "examples": create_sample_examples(),
        }

    # Load template
    try:
        template = env.get_template(template_path)
    except Exception as e:
        print(f"Error loading template {template_path}: {e}")
        return None

    # Render template
    try:
        rendered = template.render(**context)
        return rendered
    except Exception as e:
        print(f"Error rendering template: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Inspect model generator prompts")
    parser.add_argument(
        "--category",
        choices=["mlp", "transformer", "dcn", "mmoe", "original", "tabular", "fallback", "all"],
        default="all",
        help="Template category to inspect"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("prompt_outputs"),
        help="Directory to save rendered prompts"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(exist_ok=True)

    categories = ["mlp", "transformer", "dcn", "mmoe", "original"] if args.category == "all" else [args.category]

    print("🔍 Inspecting Model Generator Prompts")
    print("=" * 50)

    for category in categories:
        print(f"\n📝 Rendering: {category} template")

        rendered = render_template(category)

        if rendered:
            # Save to file
            filename = f"{category}.txt"
            output_path = args.output_dir / filename

            with open(output_path, "w") as f:
                f.write(rendered)

            print(f"✅ Saved to: {output_path}")

            # Show preview
            lines = rendered.split('\n')
            print(f"📄 Preview (first 20 lines):")
            print("-" * 40)
            for i, line in enumerate(lines[:20]):
                print(f"{i+1:2d}: {line}")
            if len(lines) > 20:
                print(f"... and {len(lines) - 20} more lines")
            print("-" * 40)
        else:
            print(f"❌ Failed to render {category} template")

    print(f"\n✨ All prompts saved to: {args.output_dir}")
    print("\n🔧 Usage examples:")
    print("  # Inspect specific category")
    print("  python scripts/inspect_model_generator_prompts.py --category mlp")
    print("  python scripts/inspect_model_generator_prompts.py --category transformer")
    print("  python scripts/inspect_model_generator_prompts.py --category dcn")
    print("  python scripts/inspect_model_generator_prompts.py --category mmoe")
    print("  # Custom output directory")
    print("  python scripts/inspect_model_generator_prompts.py --output-dir /tmp/prompts")


if __name__ == "__main__":
    main()