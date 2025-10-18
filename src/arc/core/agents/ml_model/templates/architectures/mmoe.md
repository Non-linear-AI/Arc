**Multi-gate Mixture of Experts (MMoE)** uses multiple expert networks with task-specific gating for multi-task learning. Each task learns to combine expert outputs differently.

**Key Insights**:
- **Expert networks**: Shared `torch.nn.Linear` layers that learn different feature representations
- **Gating networks**: Task-specific `torch.nn.functional.softmax` for expert weight computation
- **Multi-task**: Each task has separate towers built on weighted expert combinations

### **MMoE Architecture Pattern**

**When to use**: Multi-task learning, shared representation with task-specific adaptation, recommendation systems

**YAML Example**:
```yaml
inputs:
  features:
    dtype: float32
    shape: [null, 20]
    columns: [user_features, item_features, context_features]

graph:
  # Shared Expert Networks
  - name: expert_1
    type: torch.nn.Linear
    params: { in_features: 20, out_features: 64 }
    inputs: { input: features }

  - name: expert_2
    type: torch.nn.Linear
    params: { in_features: 20, out_features: 64 }
    inputs: { input: features }

  - name: expert_3
    type: torch.nn.Linear
    params: { in_features: 20, out_features: 64 }
    inputs: { input: features }

  # Task 1 Gating Network
  - name: gate_1
    type: torch.nn.Linear
    params: { in_features: 20, out_features: 3 }
    inputs: { input: features }

  - name: gate_1_softmax
    type: torch.nn.functional.softmax
    params: { dim: 1 }
    inputs: { input: gate_1.output }

  # Task 2 Gating Network
  - name: gate_2
    type: torch.nn.Linear
    params: { in_features: 20, out_features: 3 }
    inputs: { input: features }

  - name: gate_2_softmax
    type: torch.nn.functional.softmax
    params: { dim: 1 }
    inputs: { input: gate_2.output }

  # Weighted Expert Combinations
  - name: task_1_experts
    type: torch.stack
    inputs: [expert_1.output, expert_2.output, expert_3.output]

  - name: task_1_weighted
    type: torch.matmul
    inputs: [gate_1_softmax.output, task_1_experts.output]

  - name: task_2_weighted
    type: torch.matmul
    inputs: [gate_2_softmax.output, task_1_experts.output]

  # Task-specific Towers
  - name: task_1_tower
    type: torch.nn.Linear
    params: { in_features: 64, out_features: 1 }
    inputs: { input: task_1_weighted.output }

  - name: task_2_tower
    type: torch.nn.Linear
    params: { in_features: 64, out_features: 1 }
    inputs: { input: task_2_weighted.output }

outputs:
  task_1_prediction: task_1_tower.output
  task_2_prediction: task_2_tower.output
```