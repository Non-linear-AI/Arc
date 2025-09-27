## Transformer Architecture Principles

### **Core Transformer Patterns**
1. **Self-Attention**: Multi-head attention for sequence modeling
2. **Position Encoding**: Positional information for sequences
3. **Layer Normalization**: Pre/post-norm configurations
4. **Residual Connections**: Skip connections around attention blocks
5. **Feedforward Blocks**: Dense layers within transformer blocks

### **Essential Components**
- `torch.nn.MultiheadAttention`: Core attention mechanism
- `torch.nn.LayerNorm`: Normalization for transformer blocks
- `torch.nn.Linear`: Feedforward networks and projections
- `torch.nn.Embedding`: Token and positional embeddings
- `torch.nn.Dropout`: Regularization

### **Input Considerations for Tabular Data**
For tabular data, you may need to reshape or create sequences from features:
```yaml
inputs:
  sequence:                               # For sequential data
    dtype: float32
    shape: [null, seq_len, feature_dim]  # [batch, sequence, features]
    columns: [...]
```

## Transformer Design Patterns

### **Basic Self-Attention Block**
Core building block for transformer architectures:
```yaml
graph:
  # Input projection
  - name: input_proj
    type: torch.nn.Linear
    params: { in_features: input_dim, out_features: d_model }
    inputs: { input: sequence }

  # Self-attention
  - name: self_attention
    type: torch.nn.MultiheadAttention
    params: { embed_dim: d_model, num_heads: 8, dropout: 0.1 }
    inputs: { query: input_proj.output, key: input_proj.output, value: input_proj.output }

  # Residual connection + layer norm
  - name: attn_residual
    type: torch.cat
    inputs: [input_proj.output, self_attention.output.0]
  - name: norm1
    type: torch.nn.LayerNorm
    params: { normalized_shape: [d_model] }
    inputs: { input: attn_residual.output }
```

### **Complete Transformer Block**
Full transformer encoder block with feedforward:
```yaml
  # Self-attention layer
  - name: attention
    type: torch.nn.MultiheadAttention
    params: { embed_dim: 512, num_heads: 8, dropout: 0.1 }
    inputs: { query: input_embeddings, key: input_embeddings, value: input_embeddings }

  # Post-attention norm with residual
  - name: attn_output
    type: torch.cat
    inputs: [input_embeddings, attention.output.0]
  - name: norm1
    type: torch.nn.LayerNorm
    params: { normalized_shape: [512] }
    inputs: { input: attn_output.output }

  # Feedforward network
  - name: ff1
    type: torch.nn.Linear
    params: { in_features: 512, out_features: 2048 }
    inputs: { input: norm1.output }
  - name: ff_activation
    type: torch.nn.functional.gelu
    inputs: { input: ff1.output }
  - name: ff_dropout
    type: torch.nn.Dropout
    params: { p: 0.1 }
    inputs: { input: ff_activation.output }
  - name: ff2
    type: torch.nn.Linear
    params: { in_features: 2048, out_features: 512 }
    inputs: { input: ff_dropout.output }

  # Post-feedforward norm with residual
  - name: ff_output
    type: torch.cat
    inputs: [norm1.output, ff2.output]
  - name: norm2
    type: torch.nn.LayerNorm
    params: { normalized_shape: [512] }
    inputs: { input: ff_output.output }
```

### **Encoder-Only Transformer**
For classification or feature extraction:
```yaml
graph:
  # Input projection
  - name: input_proj
    type: torch.nn.Linear
    params: { in_features: feature_dim, out_features: d_model }
    inputs: { input: sequence }

  # Transformer encoder layers
  - name: encoder_layer1
    type: torch.nn.MultiheadAttention
    params: { embed_dim: d_model, num_heads: 8 }
    inputs: { query: input_proj.output, key: input_proj.output, value: input_proj.output }

  # Global pooling for classification
  - name: global_pool
    type: torch.mean
    inputs: { input: encoder_layer1.output.0, dim: 1 }

  # Classification head
  - name: classifier
    type: torch.nn.Linear
    params: { in_features: d_model, out_features: num_classes }
    inputs: { input: global_pool.output }
```

## Common Transformer Configurations

### **Model Dimensions**
- **Small**: d_model=256, heads=4, ff_dim=1024
- **Base**: d_model=512, heads=8, ff_dim=2048
- **Large**: d_model=768, heads=12, ff_dim=3072

### **Attention Heads**
- **Few heads (4-8)**: Better for small datasets
- **Many heads (12-16)**: Better for complex patterns
- **Head dimension**: d_model / num_heads (typically 64)

### **Regularization**
- **Attention dropout**: 0.1-0.2
- **Feedforward dropout**: 0.1-0.3
- **Layer normalization**: Always use with transformers

## Output Patterns

### **Sequence Classification**
```yaml
  # Pool sequence representations
  - name: cls_token
    type: torch.mean
    inputs: { input: final_layer.output, dim: 1 }
  - name: classifier
    type: torch.nn.Linear
    params: { in_features: d_model, out_features: num_classes }
    inputs: { input: cls_token.output }

outputs:
  predictions: classifier.output
```

### **Sequence-to-Sequence**
```yaml
  # Output projection for each timestep
  - name: output_proj
    type: torch.nn.Linear
    params: { in_features: d_model, out_features: vocab_size }
    inputs: { input: decoder_output }

outputs:
  sequence_logits: output_proj.output
```

## Transformer Design Guidelines

### **Architecture Choices**
1. **Layer Count**: Start with 2-6 layers, increase for complex tasks
2. **Attention Heads**: Use 8 heads as default, adjust based on d_model
3. **Feedforward Ratio**: FF dimension = 4 × d_model
4. **Normalization**: Pre-norm (before attention) or post-norm (after attention)

### **Training Considerations**
- **Gradient Scaling**: Use with deeper transformers
- **Learning Rate**: Lower than MLPs (1e-4 to 1e-5)
- **Warmup**: Gradual learning rate increase