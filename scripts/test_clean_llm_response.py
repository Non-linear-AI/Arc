#!/usr/bin/env python3
"""Test script for _clean_llm_response() edge cases."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc.core.agents.shared.base_agent import BaseAgent


# Create a minimal concrete implementation for testing
class TestAgent(BaseAgent):
    def get_template_directory(self):
        return Path(__file__).parent


def test_clean_llm_response():
    """Test various LLM response formats."""
    agent = TestAgent(services=None, api_key="test")

    # Test case 1: Clean YAML (no code fences)
    print("=" * 80)
    print("Test 1: Clean YAML (no code fences)")
    print("=" * 80)
    response1 = """inputs:
  features:
    dtype: float32
    shape: [null, 8]"""

    cleaned1 = agent._clean_llm_response(response1)
    print(f"Input:\n{response1}\n")
    print(f"Output:\n{cleaned1}\n")
    assert "```" not in cleaned1
    print("✓ PASS\n")

    # Test case 2: Code fences at start/end (current working case)
    print("=" * 80)
    print("Test 2: Code fences at start/end (current working case)")
    print("=" * 80)
    response2 = """```yaml
inputs:
  features:
    dtype: float32
    shape: [null, 8]
```"""

    cleaned2 = agent._clean_llm_response(response2)
    print(f"Input:\n{response2}\n")
    print(f"Output:\n{cleaned2}\n")
    assert "```" not in cleaned2
    print("✓ PASS\n")

    # Test case 3: Preamble text before code fence (THE BUG)
    print("=" * 80)
    print("Test 3: Preamble text before code fence (THE BUG)")
    print("=" * 80)
    response3 = """Here's the Arc-Graph model specification for predicting diabetes outcome:

```yaml
inputs:
  features:
    dtype: float32
    shape: [null, 8]
    columns: [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]

graph:
  - name: hidden1
    type: torch.nn.Linear
    params:
      in_features: 8
      out_features: 64
    inputs: { input: features }

outputs:
  prediction: hidden1

loss:
  type: torch.nn.BCEWithLogitsLoss
  inputs:
    input: prediction
    target: outcome
```"""

    cleaned3 = agent._clean_llm_response(response3)
    print(f"Input:\n{response3}\n")
    print(f"Output:\n{cleaned3}\n")

    if "```" in cleaned3:
        print("✗ FAIL: Code fences still present in output!")
        print(f"First line with backticks: {[line for line in cleaned3.split('\\n') if '```' in line][0]}")
        return False
    else:
        print("✓ PASS\n")

    # Test case 4: Explanatory text before AND after
    print("=" * 80)
    print("Test 4: Explanatory text before AND after")
    print("=" * 80)
    response4 = """I'll create a model for you.

```yaml
inputs:
  features:
    dtype: float32
```

This should work well for your use case."""

    cleaned4 = agent._clean_llm_response(response4)
    print(f"Input:\n{response4}\n")
    print(f"Output:\n{cleaned4}\n")

    if "```" in cleaned4:
        print("✗ FAIL: Code fences still present in output!")
        return False
    else:
        print("✓ PASS\n")

    # Test case 5: Multiple code blocks (edge case)
    print("=" * 80)
    print("Test 5: Multiple code blocks (should extract first one)")
    print("=" * 80)
    response5 = """Here are two options:

Option 1:
```yaml
inputs:
  features:
    dtype: float32
```

Option 2:
```yaml
inputs:
  data:
    dtype: int32
```"""

    cleaned5 = agent._clean_llm_response(response5)
    print(f"Input:\n{response5}\n")
    print(f"Output:\n{cleaned5}\n")

    if "```" in cleaned5:
        print("✗ FAIL: Code fences still present in output!")
        return False
    else:
        print("✓ PASS (extracted first block)\n")

    print("=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
    return True


if __name__ == "__main__":
    try:
        success = test_clean_llm_response()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
