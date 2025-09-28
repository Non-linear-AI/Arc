**Multi-Layer Perceptrons (MLPs)** are feedforward neural networks using `torch.nn.Linear` layers with non-linear activations. They excel at tabular data classification and regression through their universal approximation capabilities.

**Key Insights**:

  * **Layer Sizing**: Start with a hidden layer 2-4x the input size, then gradually decrease (e.g., 128 → 64 → 32).
  * **Regularization**: Use **`BatchNorm1d`** after the linear layer (and before activation) to stabilize training. Apply **`Dropout`** with a rate of 0.1-0.5, often increasing the rate for deeper layers.
  * **Activations**: **`ReLU`** is a robust default. Consider **`GELU`** or **`SiLU`** for deeper or more complex networks.
  * **Output for Training**: For maximum numerical stability, always output raw values and use a loss function that handles the final transformation:
      * **Binary Classification**: Output **logits** and use **`BCEWithLogitsLoss`**.
      * **Multi-Class Classification**: Output **logits** and use **`CrossEntropyLoss`**.
      * **Regression**: Output a **prediction** and use a loss like **`MSELoss`**.

-----

### **Pattern 1: MLP for Binary Classification**

This is the standard pattern for "yes/no" or "true/false" predictions. It includes a hidden layer with activation and dropout, making it suitable for both simple and moderately complex datasets.

**When to use**: Fraud detection, customer churn prediction, medical diagnosis, or any binary outcome task.

**YAML Example**:

```yaml
inputs:
  features:
    dtype: float32
    shape: [null, 8]
    columns: [age, income, credit_score, account_balance, num_products, has_credit_card, is_active_member, years_with_bank]

graph:
  - name: hidden_layer
    type: torch.nn.Linear
    params: { in_features: 8, out_features: 64 }
    inputs: { input: features }
  - name: activation
    type: torch.nn.functional.relu
    inputs: { input: hidden_layer.output }
  - name: dropout
    type: torch.nn.Dropout
    params: { p: 0.3 }
    inputs: { input: activation.output }
  - name: output_layer
    type: torch.nn.Linear
    params: { in_features: 64, out_features: 1 }
    inputs: { input: dropout.output }

outputs:
  logits: output_layer.output

loss:
  type: torch.nn.functional.binary_cross_entropy_with_logits
  inputs:
    input: logits
    target: churned
```

-----

### **Pattern 2: Deep MLP for Multi-Class Classification**

For classifying items into one of three or more categories. Deeper networks can learn more complex, hierarchical features from the data.

**When to use**: Product categorization, image classification (with flattened images), or any task with multiple exclusive outcomes.

**YAML Example**:

```yaml
inputs:
  features:
    dtype: float32
    shape: [null, 10]
    columns: [age, income, education_years, experience_years, credit_score, debt_ratio, assets_value, employment_status, region_code, risk_score]

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

  - name: output_layer
    type: torch.nn.Linear
    params: { in_features: 64, out_features: 3 }
    inputs: { input: dropout2.output }

outputs:
  logits: output_layer.output

loss:
  type: torch.nn.functional.cross_entropy
  inputs:
    input: logits
    target: product_category
```

-----

### **Pattern 3: MLP for Regression**

This pattern is used for predicting a continuous numerical value instead of a class. The key differences are the single output neuron (with no activation) and the use of a regression-specific loss function like Mean Squared Error (`MSELoss`).

**When to use**: Predicting house prices, stock values, temperature, age, or any other continuous quantity.

**YAML Example**:

```yaml
inputs:
  features:
    dtype: float32
    shape: [null, 6]
    columns: [cylinders, horsepower, weight, acceleration, model_year, origin]

graph:
  - name: hidden_layer_1
    type: torch.nn.Linear
    params: { in_features: 6, out_features: 64 }
    inputs: { input: features }
  - name: activation_1
    type: torch.nn.functional.relu
    inputs: { input: hidden_layer_1.output }

  - name: hidden_layer_2
    type: torch.nn.Linear
    params: { in_features: 64, out_features: 32 }
    inputs: { input: activation_1.output }
  - name: activation_2
    type: torch.nn.functional.relu
    inputs: { input: hidden_layer_2.output }
    
  - name: output_layer
    type: torch.nn.Linear
    params: { in_features: 32, out_features: 1 } # Single output for the predicted value
    inputs: { input: activation_2.output }

outputs:
  prediction: output_layer.output

loss:
  type: torch.nn.functional.mse_loss # Mean Squared Error for regression
  inputs:
    input: prediction
    target: mpg
```