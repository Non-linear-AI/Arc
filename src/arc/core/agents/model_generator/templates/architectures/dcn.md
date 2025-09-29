**Deep & Cross Networks (DCN)** combine explicit feature crossing with deep representation learning. The cross network learns bounded-degree feature interactions explicitly through a novel architecture.

**Key Insights**:
- **Cross layers**: Learn explicit feature interactions via `x_{l+1} = x_0 ⊙ (W_l * x_l + b_l) + x_l`
- **Architecture**: Parallel cross and deep networks, concatenated before final prediction
- **Use cases**: Click-through rate prediction, recommendation systems, feature interaction modeling

### **DCN Architecture Pattern**

**When to use**: CTR prediction, recommendation systems, when explicit feature interactions matter more than deep implicit patterns

**Core Mathematics**:
- **Cross Layer**: `x_{l+1} = x_0 ⊙ (W_l * x_l + b_l) + x_l`
  - Preserves dimension across layers
  - Bounded interaction degree
  - Residual connection maintains original features

**YAML Example**:
```yaml
inputs:
  features:
    dtype: float32
    shape: [null, 100]  # Feature dimension
    columns: [user_features, item_features, context_features]

modules:
  # Cross layer module - implements cross network mathematics
  cross_layer:
    inputs: [x_current, x_original]  # Current state and original input
    graph:
      # Linear transformation: W_l * x_l + b_l
      - name: linear_transform
        type: torch.nn.Linear
        params: { in_features: 100, out_features: 100 }
        inputs: { input: x_current }

      # Element-wise product with original: x_0 ⊙ (W_l * x_l + b_l)
      - name: element_wise_product
        type: torch.mul
        inputs: [x_original, linear_transform.output]

      # Residual connection: ... + x_l
      - name: residual_add
        type: torch.add
        inputs: [element_wise_product.output, x_current]

    outputs:
      cross_output: residual_add.output

  # Deep network module - traditional MLP
  deep_network:
    inputs: [input_features]
    graph:
      - name: dense1
        type: torch.nn.Linear
        params: { in_features: 100, out_features: 512 }
        inputs: { input: input_features }

      - name: relu1
        type: torch.nn.functional.relu
        inputs: { input: dense1.output }

      - name: dense2
        type: torch.nn.Linear
        params: { in_features: 512, out_features: 256 }
        inputs: { input: relu1.output }

      - name: relu2
        type: torch.nn.functional.relu
        inputs: { input: dense2.output }

      - name: dense3
        type: torch.nn.Linear
        params: { in_features: 256, out_features: 128 }
        inputs: { input: relu2.output }

    outputs:
      deep_output: dense3.output

graph:
  # Cross Network - stacked cross layers
  - name: cross_network
    type: arc.stack
    params:
      module: cross_layer
      count: 3  # Typically 3-6 layers optimal
    inputs: [features, features]  # Pass features as both current and original

  # Deep Network - implicit pattern learning
  - name: deep_branch
    type: module.deep_network
    inputs: [features]

  # Combine cross and deep outputs
  - name: combined_features
    type: torch.cat
    params: { dim: 1 }
    inputs: [cross_network.output, deep_branch.output]

  # Final prediction layer
  - name: output_layer
    type: torch.nn.Linear
    params: { in_features: 228, out_features: 1 }  # 100 (cross) + 128 (deep)
    inputs: { input: combined_features.output }

outputs:
  logits: output_layer.output

loss:
  type: torch.nn.functional.binary_cross_entropy_with_logits
  inputs:
    input: logits
    target: target_column
```

### **DCN Configuration Guidelines**

**Cross Network Depth**:
- **2-3 layers**: Simple feature interactions, small datasets
- **3-4 layers**: Balanced explicit/implicit learning (recommended)
- **5-6 layers**: Complex interactions, large datasets
- **>6 layers**: Risk of overfitting, diminishing returns

**Deep Network Configuration**:
- **Wide networks**: [512, 256, 128] for rich implicit patterns
- **Deep networks**: More layers for complex non-linear patterns
- **Balanced**: Match cross network capacity

**Architecture Variants**:
1. **Standard DCN**: Both cross and deep networks (recommended)
2. **Cross-only**: Pure explicit interactions (`deep_network` → identity)
3. **Deep-only**: Traditional MLP fallback (remove `cross_network`)

### **Feature Engineering for DCN**

**Categorical Features**: Use embeddings before DCN
```yaml
# Add before main graph
- name: user_embedding
  type: torch.nn.Embedding
  params: { num_embeddings: 10000, embedding_dim: 64 }

- name: item_embedding
  type: torch.nn.Embedding
  params: { num_embeddings: 5000, embedding_dim: 32 }

- name: feature_concat
  type: torch.cat
  inputs: [user_embedding.output, item_embedding.output, numerical_features]
```

**Numerical Features**: Normalize or standardize before input

### **Performance Tips**

**When DCN Excels**:
- Sparse categorical features (user_id, item_id, category)
- Known important feature interactions
- CTR/CVR prediction tasks
- Recommendation systems

**When to Consider Alternatives**:
- Dense numerical features only → MLP
- Sequential data → Transformer
- Very high-dimensional features → Feature selection first

**Training Considerations**:
- Learning rate: 0.001-0.01 typical
- Batch size: 1024+ for stability
- Regularization: L2 on cross network weights