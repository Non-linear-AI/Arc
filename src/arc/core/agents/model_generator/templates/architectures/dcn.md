**Deep & Cross Networks (DCN)** combine explicit feature crossing with deep representation learning. The cross network models feature interactions explicitly while the deep network captures implicit patterns.

**Key Insights**:
- **Cross layers**: Use `torch.matmul` for explicit feature crossing with learned weight vectors
- **Architecture**: Parallel cross and deep networks, concatenated before final layer
- **Use cases**: Recommender systems, CTR prediction, any task requiring feature interactions

### **DCN Architecture Pattern**

**When to use**: Click-through rate prediction, recommendation systems, feature interaction modeling

**YAML Example**:
```yaml
inputs:
  features:
    dtype: float32
    shape: [null, 10]
    columns: [user_id, item_id, category, price, rating, age, gender, location, timestamp, context]

graph:
  # Embedding layers for categorical features
  - name: user_embed
    type: torch.nn.Embedding
    params: { num_embeddings: 10000, embedding_dim: 64 }
    inputs: { input: features }

  - name: item_embed
    type: torch.nn.Embedding
    params: { num_embeddings: 5000, embedding_dim: 64 }
    inputs: { input: features }

  # Cross Network - Explicit feature crossing
  - name: cross_layer_1
    type: torch.matmul
    inputs: [features, cross_weight_1]

  - name: cross_layer_2
    type: torch.matmul
    inputs: [cross_layer_1.output, cross_weight_2]

  # Deep Network - Implicit patterns
  - name: deep_layer_1
    type: torch.nn.Linear
    params: { in_features: 128, out_features: 256 }
    inputs: { input: features }

  - name: deep_relu_1
    type: torch.nn.functional.relu
    inputs: { input: deep_layer_1.output }

  - name: deep_layer_2
    type: torch.nn.Linear
    params: { in_features: 256, out_features: 128 }
    inputs: { input: deep_relu_1.output }

  # Combine cross and deep
  - name: combined
    type: torch.cat
    inputs: [cross_layer_2.output, deep_layer_2.output]

  - name: final_layer
    type: torch.nn.Linear
    params: { in_features: 256, out_features: 1 }
    inputs: { input: combined.output }

  - name: sigmoid
    type: torch.nn.functional.sigmoid
    inputs: { input: final_layer.output }

outputs:
  probability: sigmoid.output
```