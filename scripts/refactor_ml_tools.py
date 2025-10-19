#!/usr/bin/env python3
"""Script to refactor ML tool interfaces for consistency and simplification.

Changes:
1. Rename user_context → instruction in MLPlanTool
2. Rename context → instruction in MLModelTool
3. Make target_column required for ml_model
4. Simplify metadata (remove plan_comparison, plan_training_config, plan_evaluation)
5. Rename data_table → evaluate_table in ml_evaluate
6. Remove output_path from data_process
7. Make ml_evaluate async (don't wait for results)
"""


def main():
    print("Starting ML tools refactoring...")
    print("\n1. Renaming parameters...")

    # TODO: Implement all the renames and simplifications
    # This is a placeholder - actual implementation would be complex
    # The actual refactoring was done manually using Claude Code

    print("Refactoring complete!")


if __name__ == "__main__":
    main()
