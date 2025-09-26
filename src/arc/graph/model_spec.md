# **The Definitive Model Schema Specification (v1.0)**

## **1. Overview**

Arc-Graph is the definitive, declarative schema for defining PyTorch neural network architectures. It provides a complete, self-contained model definition in a single, human-readable YAML file. The design is optimized to be intuitive for simple models while scaling elegantly to state-of-the-art complex architectures like the Transformer.

The design is guided by these core principles:

* **Declarative & Native**: The entire model architecture is explicitly defined. The schema is a direct, near 1:1 mapping to PyTorch's modular structure and functional API, eliminating the need for custom abstractions.  
* **Modularity & Reusability**: Complex, repeated components (like a ResNet block or Transformer layer) can be defined once as reusable modules and instantiated throughout the graph, promoting clean, DRY (Don't Repeat Yourself) design.  
* **Simplicity & Scalability**: The schema is optimized for the common case, reducing boilerplate for simple models. For deep networks, the special arc.stack node provides a concise, declarative way to express repetition without manually defining every layer.

## **2\. Root Schema Structure**

Every Arc-Graph model architecture file is composed of these primary, top-level sections:

```yaml
# === MODEL INTERFACE ===
# Defines the data entry points for the model
inputs: { ... }

# === REUSABLE COMPONENTS ===
# (Optional) A library of custom, reusable sub-graphs (e.g., a ResNet block)
modules: { ... }

# === MODEL ARCHITECTURE ===
# The directed acyclic graph (DAG) of computation nodes
graph: [ ... ]

# === MODEL OUTPUTS ===
# Designates the final, named outputs of the graph
outputs: { ... }
```

## **3\. Schema Components in Detail**

### **3.1. The Model Interface: inputs and outputs**

The inputs and outputs sections define the model's public contract—what data it accepts and what it returns.

#### **3.1.1. inputs**

This section declares the named entry points for tensors into the model graph.

**Structure:**

```yaml
inputs:
  <input_name_1>: { dtype: <type>, shape: [...], columns: [...] }
  <input_name_2>: { dtype: <type>, shape: [...], columns: [...] }
```

* \<input\_name\>: A unique name for the input tensor that can be referenced by nodes in the graph.  
* dtype: The data type of the tensor (e.g., float32, int64).  
* shape: The expected shape. Use null for variable dimensions like batch size or sequence length (e.g., \[null, 256\]).  
* columns: The specific source column names from the dataset, providing clear data lineage.

#### **3.1.2. outputs**

This section maps the internal outputs of graph nodes to publicly accessible names.

**Structure:**

```yaml
outputs:
  <public_output_name>: "<node_name_in_graph>.output"
```

* \<public\_output\_name\>: The name returned by the model (e.g., logits, probabilities).  
* The value is a string reference to a node's output. To access elements of a tuple output (e.g., from MultiheadAttention), use .0, .1, etc. (e.g., "my\_attn.output.0").

### **3.2. The Core Architecture: graph**

The graph is a list of nodes that define the model's architecture as a Directed Acyclic Graph (DAG). The execution order is determined by how nodes are connected via their inputs, not by their order in the list.

**Node Structure:**

```yaml
- name: <unique_node_name>
  type: <node_type>
  params: { ... }
  inputs: { ... } or [ ... ]
```

* name: A unique name for the node. Its results are referenced as \<unique\_node\_name\>.output.  
* type: The operation the node performs. This is a direct mapping to the PyTorch ecosystem.  
* params: Keyword arguments for the node's type. For torch.nn modules, these are constructor arguments. For PyTorch functions, these are keyword arguments.  
* inputs: Defines the connections from other nodes. It can be a **map** for named arguments ({arg\_name: source\_node.output}) or a **list** for positional arguments (\[source\_1.output, source\_2.output\]).

**Node Types (type)**

| type Value Format | Description | params Behavior | Example |
| :---- | :---- | :---- | :---- |
| **torch.nn.\<Module\>** | Instantiates a PyTorch Layer class. | Passed to the module's \_\_init\_\_ constructor. | type: torch.nn.Linear params: { in\_features: 512, out\_features: 10 } |
| **torch.\<function\>** **torch.nn.functional.\<function\>** | Calls a PyTorch function directly. | Passed as keyword arguments to the function. | type: torch.add type: torch.nn.functional.relu |
| **module.\<ModuleName\>** | Instantiates a custom component defined in the modules section. | N/A (defined within the module). | type: module.FeedForward |
| **arc.stack** | A special node that creates a deep, sequential stack of a module. | Must contain module (name) and count (integer). | type: arc.stack params: { module: TransformerEncoderBlock, count: 6 } |

### **3.3. Modularity: modules**

The optional modules section is a library of reusable sub-graphs. It allows you to define a complex component once and instantiate it anywhere in your main graph, promoting clean and scalable design.

**Structure:**

```yaml
modules:
  <ModuleName>:
    inputs: [<arg1>, <arg2>, ...] # The module's input arguments
    graph:
      - { ... } # The internal graph of the module
    outputs:
      <output_name>: "<internal_node.output>" # The module's return value
```

## **4\. Formal JSON Schema**

This JSON schema provides the formal definition for validating an Arc-Graph YAML file.


```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Arc-Graph Model Specification Schema",
  "description": "A schema for defining a model architecture using a declarative graph of PyTorch components, with support for reusable modules and automated stacking.",
  "type": "object",
  "properties": {
    "inputs": {
      "description": "Defines the input tensors to the model.",
      "type": "object",
      "minProperties": 1,
      "patternProperties": {
        "^[a-zA-Z0-9_]+$": {
          "type": "object",
          "properties": {
            "dtype": {
              "description": "Data type of the input tensor (e.g., float32, int64).",
              "type": "string"
            },
            "shape": {
              "description": "Shape of the input tensor. Use 'null' for variable dimensions.",
              "type": "array",
              "items": {
                "anyOf": [
                  { "type": "integer" },
                  { "type": "null" }
                ]
              }
            },
            "columns": {
              "description": "List of column names corresponding to the features.",
              "type": "array",
              "items": { "type": "string" }
            }
          },
          "required": ["dtype", "shape", "columns"]
        }
      },
      "additionalProperties": false
    },
    "graph": {
      "description": "The main list of nodes representing the end-to-end computation graph.",
      "type": "array",
      "minItems": 1,
      "items": { "$ref": "#/definitions/graphNode" }
    },
    "modules": {
      "description": "A library of reusable sub-graphs (modules) that can be instantiated in the main graph.",
      "type": "object",
      "patternProperties": {
        "^[a-zA-Z0-9_]+$": {
          "type": "object",
          "properties": {
            "inputs": {
              "description": "A list of named inputs for the module.",
              "type": "array",
              "items": { "type": "string" }
            },
            "graph": {
              "description": "The internal computation graph for this module.",
              "type": "array",
              "items": { "$ref": "#/definitions/graphNode" }
            },
            "outputs": {
              "$ref": "#/definitions/outputs"
            }
          },
          "required": ["inputs", "graph", "outputs"]
        }
      }
    },
    "outputs": { "$ref": "#/definitions/outputs" }
  },
  "required": ["inputs", "graph", "outputs"],
  "definitions": {
    "graphNode": {
      "type": "object",
      "properties": {
        "name": {
          "description": "A unique identifier for the node's output.",
          "type": "string",
          "pattern": "^[a-zA-Z0-9_]+$"
        },
        "type": {
          "description": "The operation type. Can be a torch.nn.Module, a torch function, a custom module, or 'arc.stack'.",
          "type": "string",
          "pattern": "^(torch(\\.(nn(\\.(functional)?)?)?\\.\\w+)|arc\\.stack|module\\.\\w+)$"
        },
        "params": {
          "description": "Parameters for the module constructor or keyword arguments for the function.",
          "type": "object",
          "additionalProperties": true
        },
        "inputs": {
          "description": "Specifies connections from other nodes. Can be a map for named arguments or a list for positional arguments.",
          "oneOf": [
            {
              "type": "object",
              "patternProperties": {
                "^[a-zA-Z0-9_]+$": { "type": "string", "pattern": "^[a-zA-Z0-9_]+(\\.\\w+(\\.\\d+)?)?$" }
              },
              "additionalProperties": false
            },
            {
              "type": "array",
              "items": { "type": "string", "pattern": "^[a-zA-Z0-9_]+(\\.\\w+(\\.\\d+)?)?$" }
            }
          ]
        }
      },
      "required": ["name", "type", "inputs"],
      "if": {
        "properties": { "type": { "const": "arc.stack" } }
      },
      "then": {
        "properties": {
          "params": {
            "type": "object",
            "properties": {
              "module": {
                "description": "The name of the module (defined in the 'modules' section) to be repeated.",
                "type": "string"
              },
              "count": {
                "description": "The number of times the module should be stacked sequentially.",
                "type": "integer",
                "minimum": 1
              }
            },
            "required": ["module", "count"]
          }
        }
      }
    },
    "outputs": {
      "description": "A map of meaningful output names to specific node outputs.",
      "type": "object",
      "minProperties": 1,
      "patternProperties": {
        "^[a-zA-Z0-9_]+$": {
          "type": "string",
          "description": "A reference to a node's output, in the format 'node_name.output'. A '.0' can be added to access tuple elements.",
          "pattern": "^[a-zA-Z0-9_]+(\\.\\w+(\\.\\d+)?)?$"
        }
      },
      "additionalProperties": false
    }
  }
}
```

## **5. Complete Examples**

### 5.1. Example 1: Logistic Regression
This is the most fundamental example, showing a binary classification model. It demonstrates a minimal graph with a linear layer and a sigmoid activation function.

```yaml
inputs:
  features:
    dtype: float32
    shape: [null, 128] # e.g., 128 input features
    columns: [age, bmi, glucose_level, ...]

graph:
  - name: linear_layer
    type: torch.nn.Linear
    params: { in_features: 128, out_features: 1 } # Output is a single logit
    inputs: { input: features }

  - name: sigmoid_activation
    type: torch.nn.Sigmoid
    inputs: { input: linear_layer.output }

outputs:
  probability: sigmoid_activation.output
```

### 5.2. Example 2: Multi-Layer Perceptron (MLP)

This example shows a basic multi-class classification model, demonstrating a sequence of layers.

```yaml
inputs:
  features:
    dtype: float32
    shape: [null, 64]
    columns: [col1, col2, ...]

graph:
  - name: hidden_layer_1
    type: torch.nn.Linear
    params: { in_features: 64, out_features: 128 }
    inputs: { input: features }
  - name: activation_1
    type: torch.nn.functional.relu
    inputs: { input: hidden_layer_1.output }
  - name: dropout
    type: torch.nn.Dropout
    params: { p: 0.5 }
    inputs: { input: activation_1.output }
  - name: output_layer
    type: torch.nn.Linear
    params: { in_features: 128, out_features: 10 }
    inputs: { input: dropout.output }

outputs:
  logits: output_layer.output
```

### 5.3. Example 3: Generative Recommender

This advanced example defines a Transformer-based generative model for next-item prediction, based on the Meta RecSys implementation. It showcases the power of combining embeddings, using `torch.cat` for tensor manipulation, and composing the model from modular parts.


```yaml
inputs:
  country_id:
    dtype: int64
    shape: [null]
    columns: [country_id]
  # Other user features (e.g., age) can be added here
  item_sequence_ids:
    dtype: int64
    shape: [null, 256] # sequence length = 256
    columns: [item_sequence_ids]

graph:
  # --- 1. Compose User Context Token ---
  - name: country_embedding
    type: torch.nn.Embedding
    params: { num_embeddings: 50, embedding_dim: 32 }
    inputs: { input: country_id }
  # In a real model, other user embeddings would be concatenated here
  - name: user_features_composition
    type: torch.nn.Linear
    params: { in_features: 32, out_features: 128 }
    inputs: { input: country_embedding.output }
  - name: user_context_token
    type: torch.unsqueeze
    params: { dim: 1 } # Reshape from [B, D] to [B, 1, D] for sequence concat
    inputs: { input: user_features_composition.output }

  # --- 2. Compose Item Sequence Tokens ---
  - name: item_embedding
    type: torch.nn.Embedding
    params: { num_embeddings: 10000, embedding_dim: 128 } # 10k items in catalog
    inputs: { input: item_sequence_ids }
  # A positional encoding module would typically be added here

  # --- 3. Prepend User Context to Item Sequence ---
  - name: full_sequence
    type: torch.cat
    params: { dim: 1 } # Concatenate on the sequence dimension
    inputs: [user_context_token.output, item_embedding.output]

  # --- 4. Process through Transformer Encoder ---
  # This could be a single torch.nn.TransformerEncoder or an arc.stack of blocks
  - name: transformer_encoder
    type: torch.nn.TransformerEncoder
    params:
      encoder_layer:
        _target_: torch.nn.TransformerEncoderLayer
        d_model: 128
        nhead: 8
        batch_first: true
      num_layers: 4
    inputs: { src: full_sequence.output }

  # --- 5. Output Head ---
  - name: output_projection
    type: torch.nn.Linear
    params: { in_features: 128, out_features: 10000 } # Project back to item vocab size
    inputs: { input: transformer_encoder.output }

outputs:
  next_item_logits: output_projection.output
```

### 5.4. Example 4: Full Transformer for Classification

This comprehensive example demonstrates all features of the schema: `modules` for defining reusable components, `arc.stack` for efficient repetition, and a graph that mixes `torch.nn` modules with direct functional calls like `torch.add`.

```yaml
# === REUSABLE COMPONENTS ===
modules:
  FeedForward:
    inputs: [input]
    graph:
      - { name: ffn_linear_1, type: torch.nn.Linear, params: { in_features: 512, out_features: 2048 }, inputs: { input: input } }
      - { name: ffn_activation, type: torch.nn.functional.relu, inputs: { input: ffn_linear_1.output } }
      - { name: ffn_linear_2, type: torch.nn.Linear, params: { in_features: 2048, out_features: 512 }, inputs: { input: ffn_activation.output } }
    outputs:
      output: ffn_linear_2.output

  TransformerEncoderBlock:
    inputs: [x]
    graph:
      - { name: mha, type: torch.nn.MultiheadAttention, params: { embed_dim: 512, num_heads: 8, batch_first: true }, inputs: { query: x, key: x, value: x } }
      - { name: add_1, type: torch.add, inputs: [x, mha.output.0] }
      - { name: norm_1, type: torch.nn.LayerNorm, params: { normalized_shape: 512 }, inputs: { input: add_1.output } }
      - { name: ffn, type: module.FeedForward, inputs: { input: norm_1.output } }
      - { name: add_2, type: torch.add, inputs: [norm_1.output, ffn.output] }
    outputs:
      output: add_2.output

# === MODEL INTERFACE ===
inputs:
  token_ids: { dtype: int64, shape: [null, 256], columns: [token_ids] }

# === MODEL ARCHITECTURE ===
graph:
  - { name: embedding, type: torch.nn.Embedding, params: { num_embeddings: 30522, embedding_dim: 512 }, inputs: { input: token_ids } }
  - name: encoder_stack
    type: arc.stack
    params: { module: TransformerEncoderBlock, count: 6 }
    inputs: { input: embedding.output }
  - { name: pooling, type: torch.mean, params: { dim: 1 }, inputs: [encoder_stack.output] }
  - { name: classifier, type: torch.nn.Linear, params: { in_features: 512, out_features: 10 }, inputs: { input: pooling.output } }

# === MODEL OUTPUTS ===
outputs:
  logits: classifier.output
```