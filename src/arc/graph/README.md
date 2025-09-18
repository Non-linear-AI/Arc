# Arc-Graph: The Definitive Schema Specification (v0.1)

> **Note**: This document describes the conceptual Arc-Graph schema. The current implementation uses a different structure with `data_inspection`, `components`, and `graph` sections. This documentation represents the target schema design.

## 1\. Overview

Arc-Graph is the definitive, declarative schema for defining machine learning workflows. It provides a complete, self-contained model definition in a single, human-readable YAML file, designed to be intuitive for simple models while scaling elegantly to complex, multi-pipeline scenarios.

The design is guided by these principles:

  * **Declarative & Explicit**: The entire workflow is clearly defined, making it easy to read, validate, and generate.
  * **Separation of Concerns**: The schema cleanly separates data preparation (`features`) from model architecture (`model`).
  * **Functional State Handling**: Stateful operations are handled explicitly and functionally to eliminate ambiguity and guarantee train/serve parity.
  * **Simplicity for the Common Case**: The schema is optimized for the most frequent use cases, using sensible defaults to reduce boilerplate.

## 2\. Root Schema Structure

Every Arc-Graph file is composed of these primary, top-level sections, designed for clarity and separation of concerns:

```yaml
# === METADATA ===
version: "0.1"
model_name: "my_model"
description: "A brief description of the model's purpose."

# === DATA PIPELINE ===
# Transforms raw data into model-ready tensors
features:
  feature_columns: [...]
  target_columns: [...]
  processors: [...]

# === MODEL ARCHITECTURE ===
# Defines the neural network structure and interface
model:
  inputs: {...}
  graph: [...]
  outputs: {...}

# === EXECUTION CONFIG ===
# Training and prediction configurations
trainer:
  config: {...}
  optimizer: {...}
  loss: {...}

predictor:
  returns: [...]
```

## 3\. Data Pipeline: The `features` Section

The `features` section defines a unified data preparation pipeline that transforms raw table columns into model-ready tensors, configuration variables, and learned states.

### 3.1. Structure

```yaml
features:
  # A list of raw feature column names from the input table.
  feature_columns: [<col1>, <col2>, ...]

  # (Optional) A list of raw target column names, used only during training.
  target_columns: [<label_col>, ...]

  # The directed acyclic graph (DAG) of processing steps.
  processors:
    - { ... }
```

### 3.2. Processors Pipeline and Variable Namespaces

The `processors` are an ordered list of operations. Data flows from one processor to the next, consuming and producing variables from three distinct namespaces:

| **Aspect** | **`tensors` (The Data 🏞️)** | **`vars` (The Blueprint 📜)** | **`states` (The Learned Tool 🛠️)** |
| :--- | :--- | :--- | :--- |
| **Purpose** | Represents the **data** flowing through the pipeline. | **Configures** the model's static architecture and feature transformations. | Stores **learned parameters** for data transformations. |
| **Origin** | The input table view, transformed by the features pipeline. | Derived from data inspection via `inspect.*` ops. | Learned from training data via `fit.*` ops. |
| **Nature** | Dynamic (changes with every batch). | Static (fixed after inspection). **Saved with the model artifact.** | Learned (fixed after `fit`). **Saved with the model artifact.** |
| **Primary Consumer** | The **`model`** (as input to its layers). | The **`model`** (to define its shape) and the **`features`** pipeline (to configure transforms). | The **`features`** pipeline (to apply transformations). |

Each processor has the following structure:

```yaml
- name: <unique_processor_name>
  op: <operator_name>

  # (Optional) Declares this processor only runs during training.
  train_only: true

  # (Optional) A dictionary mapping operator inputs to available variables.
  inputs:
    <operator_input_name>: <variable_or_column_name>

  # (Optional) A dictionary mapping global variable names to operator outputs.
  outputs:
    <global_variable_name>: <operator_output_name>
```

### 3.3. Operator Types and State Handling

State is handled explicitly and functionally with three types of operators:

  * **`inspect.*` operators**: Run only during training. They consume raw columns or tensors and produce `vars`.
  * **`fit.*` operators**: Run only during training. They consume raw columns or tensors and produce a `state` object.
  * **`transform.*` operators**: Run during both training and prediction. They consume raw columns, tensors, `vars`, and optionally `states` to produce new `tensors`.

## 4\. Model Architecture: The `model` Section

The `model` section defines the neural network architecture and its input/output interface contract.

### 4.1. Structure

```yaml
model:
  inputs:
    <input_port_name>: { dtype: <type>, shape: [...] }

  graph:
    - { ... }

  outputs:
    <output_name>: "<node_name_in_graph>.<port_name>"
```

### 4.2. Data Flow

  * The `model.inputs` are fed by the `tensors` from the `features` pipeline.
  * The `config` parameters of the `model.graph` nodes are configured by `vars` (e.g., `in_features: vars.n_features`) or set directly (e.g., `dropout_rate: 0.1`).

## 5\. Execution Configuration: `trainer` and `predictor` Sections

These sections define how to execute the training and prediction workflows.

### 5.1. `trainer`

```yaml
trainer:
  optimizer:
    type: "AdamW"
    # `config` provides consistent parameter naming across all sections
    config:
      learning_rate: 0.001
  
  loss:
    type: <loss_function_type>
    inputs: { pred: "model.<output_name>", target: "target_columns.<col_name>" }
```

### 5.2. `predictor`

This section is optional. If omitted, all `model.outputs` are returned.

```yaml
predictor:
  returns: [<output_name_from_model>]
```

## 6\. Complete Examples

### 6.1. Example 1: Logistic Regression

```yaml
version: "0.1"
model_name: "diabetes_predictor_final"

features:
  feature_columns: [age, bmi, glucose_level]
  target_columns: [outcome]

  processors:
    - name: assemble_raw_features
      op: transform.assemble_vector
      inputs: { columns: feature_columns }
      outputs: { tensors.raw_features: output }

    - name: get_feature_count
      op: inspect.feature_stats
      train_only: true
      inputs: { tensor: tensors.raw_features }
      outputs: { vars.n_features: n_features }

    - name: learn_scaler_state
      op: fit.standard_scaler
      train_only: true
      inputs: { x: tensors.raw_features }
      outputs: { states.scaler_params: state }

    - name: apply_scaling
      op: transform.standard_scaler
      inputs:
        x: tensors.raw_features
        state: states.scaler_params
      outputs: { tensors.features: output }

model:
  inputs:
    features: { dtype: float32, shape: [null, vars.n_features] }
  
  graph:
    - name: "linear_layer"
      type: "core.Linear"
      params: { in_features: vars.n_features, out_features: 1 }
      inputs: { input: features }
    
    - name: "sigmoid_activation"
      type: "core.Sigmoid"
      inputs: { input: "linear_layer.output" }
  
  outputs:
    probability: "sigmoid_activation.output"

trainer:
  config: { learning_rate: 0.001 }
  
  optimizer:
    type: "AdamW"
  
  loss:
    type: "core.BCELoss"
    inputs: { pred: "model.probability", target: "target_columns.outcome" }

predictor:
  returns: ["probability"]
```

### 6.2. Example 2: Generative Recommender

This example is updated to show a `var` being used by a `transform` operator.

```yaml
version: "0.1"
model_name: "recommender_complex"

features:
  feature_columns: [age, country, device, event_id, event_category, event_features]
  target_columns: [event_id]

  processors:
    # 1. Inspect the raw event data to determine the ideal sequence length.
    - name: get_sequence_length
      op: inspect.sequence_stats
      train_only: true
      inputs: { sequence: event_id }
      outputs: { vars.max_len: p95_length } # e.g., get the 95th percentile length

    # 2. Handle user profile features.
    - name: hash_country
      op: transform.hash_bucket
      inputs: { x: country, num_buckets: 50 }
      outputs: { tensors.country_id: output }
    # ... other hashing ops ...

    # 3. Handle sequence features, using the derived `var` for truncation/padding.
    - name: hash_and_pad_event_id
      op: transform.hash_and_pad
      inputs:
        x: event_id
        num_buckets: 10000
        max_length: vars.max_len # Uses the var during prediction too!
      outputs: { tensors.event_ids: output }
    # ... other hash_and_pad ops ...
    
    # 4. Create the target label.
    - name: create_label
      op: transform.sequence_shift
      inputs: { x: tensors.event_ids, direction: "left", fill_value: 0 }
      outputs: { tensors.next_item_ids: output }

model:
  inputs:
    age: { dtype: float32, shape: [null] }
    country_id: { dtype: int64, shape: [null] }
    event_ids: { dtype: int64, shape: [null, vars.max_len] } # Also uses the var
    # ... other inputs ...
  
  graph:
    # --- Context Token Composition ---
    - name: "country_embedding"
      type: "core.Embedding"
      params: { num_embeddings: 50, embedding_dim: 32 }
      inputs: { input: country_id }
    # ... other context composition ...
        
    # --- Event Token Composition ---
    - name: "event_id_embedding"
      type: "core.Embedding"
      params: { num_embeddings: 10000, embedding_dim: 128 }
      inputs: { input: event_ids }
    # ... other event composition ...
        
    # --- Combine and Process with Transformer ---
    - name: "positional_encoding"
      type: "core.PositionalEncoding"
      params: { max_len: vars.max_len, d_model: 192 }
    # ... rest of the model ...
  
  outputs:
    next_item_logits: "output_head.output"

trainer:
  config: { learning_rate: 0.001 }
  
  optimizer:
    type: "AdamW"
  
  loss:
    type: "core.CrossEntropyLoss"
    inputs: { pred: "model.next_item_logits", target: "tensors.next_item_ids" }
```