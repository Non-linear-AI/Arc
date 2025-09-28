**Multi-Layer Perceptrons (MLPs)** are feedforward neural networks using `torch.nn.Linear` layers with non-linear activations. They excel at tabular data classification and regression through universal approximation capabilities.

**Key Insights**:
- **Layer sizing**: Start 2-4x input size, gradually decrease (128→64→32)
- **Regularization**: BatchNorm after linear layers, dropout 0.1-0.5 increasing with depth
- **Activations**: ReLU (standard), GELU (deeper networks)
- **Output activations**: Sigmoid (binary classification), Softmax (multi-class classification), None (regression)

### **Pattern 1: Simple Binary Classifier**

The most basic MLP pattern for binary classification. Ideal for baseline models and simple decision boundaries.

**When to use**: Quick prototyping, simple linear-separable data, baseline comparisons

**YAML Example**:
```yaml
inputs:
  features:
    dtype: float32
    shape: [null, 5]
    columns: [age, income, credit_score, account_balance, years_with_bank]

graph:
  - name: classifier
    type: torch.nn.Linear
    params: { in_features: 5, out_features: 1 }
    inputs: { input: features }

  - name: sigmoid
    type: torch.nn.functional.sigmoid
    inputs: { input: classifier.output }

outputs:
  probability: sigmoid.output
```

### **Pattern 2: Two-Layer MLP with Regularization**

Adds a hidden layer with activation and dropout for better pattern learning while preventing overfitting. Use grouped inputs (`torch.cat`) when you have distinct feature types.

**When to use**: Moderate complexity data, when you need some non-linearity but want to keep model simple

**YAML Example**:
```yaml
inputs:
  features:
    dtype: float32
    shape: [null, 8]
    columns: [age, income, credit_score, account_balance, years_with_bank, num_products, has_credit_card, is_active_member]

graph:
  - name: hidden
    type: torch.nn.Linear
    params: { in_features: 8, out_features: 64 }
    inputs: { input: features }

  - name: activation
    type: torch.nn.functional.relu
    inputs: { input: hidden.output }

  - name: dropout
    type: torch.nn.Dropout
    params: { p: 0.3 }
    inputs: { input: activation.output }

  - name: output
    type: torch.nn.Linear
    params: { in_features: 64, out_features: 1 }
    inputs: { input: dropout.output }

outputs:
  prediction: output.output
```

### **Pattern 3: Deep MLP with Multiple Hidden Layers**

Deep networks can learn hierarchical representations and complex non-linear patterns.

**When to use**: Complex datasets, large amounts of training data, when you need to learn hierarchical features

**YAML Example**:
```yaml
inputs:
  features:
    dtype: float32
    shape: [null, 10]
    columns: [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10]

graph:
  - name: layer1
    type: torch.nn.Linear
    params: { in_features: 10, out_features: 128 }
    inputs: { input: features }
  - name: norm1
    type: torch.nn.BatchNorm1d
    params: { num_features: 128 }
    inputs: { input: layer1.output }
  - name: relu1
    type: torch.nn.functional.relu
    inputs: { input: norm1.output }
  - name: dropout1
    type: torch.nn.Dropout
    params: { p: 0.2 }
    inputs: { input: relu1.output }

  - name: layer2
    type: torch.nn.Linear
    params: { in_features: 128, out_features: 64 }
    inputs: { input: dropout1.output }
  - name: norm2
    type: torch.nn.BatchNorm1d
    params: { num_features: 64 }
    inputs: { input: layer2.output }
  - name: relu2
    type: torch.nn.functional.relu
    inputs: { input: norm2.output }
  - name: dropout2
    type: torch.nn.Dropout
    params: { p: 0.3 }
    inputs: { input: relu2.output }

  - name: output
    type: torch.nn.Linear
    params: { in_features: 64, out_features: 3 }
    inputs: { input: dropout2.output }

  - name: softmax
    type: torch.nn.functional.softmax
    params: { dim: 1 }
    inputs: { input: output.output }

outputs:
  probabilities: softmax.output
```


