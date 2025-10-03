# **The Definitive Trainer Schema Specification (v1.0)**

## **1. Overview**

Arc-Graph Trainer Specification is the declarative schema for defining PyTorch model training configurations. It provides a complete, self-contained training recipe in a single, human-readable YAML file that pairs with an Arc-Graph model specification. The trainer spec defines **how** to train the model (optimizer, hyperparameters, training loop config) while remaining completely separate from **what** the model architecture is.

The design is guided by these core principles:

* **Separation of Concerns**: Training configuration is completely independent of model architecture. The same model can be trained with different trainers (e.g., different optimizers, learning rates, regularization strategies).
* **Model-Aware**: The trainer references a specific model by ID and validates that optimizer/loss configurations are compatible with the model's inputs/outputs.
* **Declarative & Native**: Direct mapping to PyTorch's training components (optimizers, loss functions, schedulers) with no custom abstractions.
* **Reproducibility**: Every training run can be exactly reproduced by storing the trainer spec alongside the trained artifact.

## **2. Root Schema Structure**

Every Arc-Graph trainer specification file is composed of these primary sections:

```yaml
# === MODEL REFERENCE ===
# Links this trainer to a specific model
model_ref: "<model_id>"

# === OPTIMIZER CONFIGURATION ===
# Defines the optimization algorithm and learning rate
optimizer: { ... }

# === TRAINING LOOP CONFIGURATION ===
# Hyperparameters controlling the training process
config: { ... }
```

**Key Design Decision: Loss is NOT in Trainer Spec**

The loss function is defined in the **model spec**, not the trainer spec. This is because:
1. **Loss is part of the model's contract**: The loss function defines what the model is trained to predict (binary classification → BCE, multi-class → CE, regression → MSE)
2. **Loss inputs depend on model architecture**: The loss needs specific outputs from the model graph (e.g., `logits`, `prediction`), which are defined in the model spec
3. **Architectural consistency**: Just as the model spec defines the forward pass (inputs → graph → outputs → loss), it should define the full computational graph including the loss

The trainer spec focuses purely on the optimization strategy.

## **3. Schema Components in Detail**

### **3.1. Model Reference: model_ref**

This section links the trainer to a specific model specification.

**Structure:**

```yaml
model_ref: "<model_id>"
```

* `<model_id>`: The exact ID of the model this trainer is designed for (e.g., `diabetes-logistic-v1`). This must match an entry in the models registry.

**Purpose:**
- Enables validation that optimizer/loss are compatible with the model
- Provides clear lineage: which trainer was used with which model
- Allows multiple trainers for the same model (e.g., `fast_training_v1` vs `high_accuracy_v1`)

**Example:**
```yaml
model_ref: diabetes-logistic-v1
```

### **3.2. Optimizer Configuration: optimizer**

Defines the optimization algorithm and its hyperparameters.

**Structure:**

```yaml
optimizer:
  type: "<optimizer_class>"
  lr: <learning_rate>
  params:
    <param_name>: <value>
    ...
```

**Fields:**
* `type`: PyTorch optimizer class (e.g., `torch.optim.Adam`, `torch.optim.SGD`, `torch.optim.AdamW`)
* `lr`: Learning rate (float, required)
* `params`: (Optional) Additional optimizer-specific parameters

**Supported Optimizer Types:**
- `torch.optim.SGD` - Stochastic Gradient Descent
- `torch.optim.Adam` - Adaptive Moment Estimation
- `torch.optim.AdamW` - Adam with decoupled weight decay
- `torch.optim.RMSprop` - Root Mean Square Propagation
- `torch.optim.Adagrad` - Adaptive Gradient Algorithm
- All PyTorch optimizers in `torch.optim.*`

**Example 1: Basic Adam**
```yaml
optimizer:
  type: torch.optim.Adam
  lr: 0.001
```

**Example 2: SGD with Momentum**
```yaml
optimizer:
  type: torch.optim.SGD
  lr: 0.01
  params:
    momentum: 0.9
    weight_decay: 0.0001
    nesterov: true
```

**Example 3: AdamW with Weight Decay**
```yaml
optimizer:
  type: torch.optim.AdamW
  lr: 0.0003
  params:
    betas: [0.9, 0.999]
    eps: 1.0e-8
    weight_decay: 0.01
```

### **3.3. Training Configuration: config**

Defines all hyperparameters controlling the training loop behavior.

**Structure:**

```yaml
config:
  # Core Training Parameters
  epochs: <int>
  batch_size: <int>
  validation_split: <float>
  shuffle: <bool>

  # Early Stopping (optional)
  early_stopping_patience: <int>
  early_stopping_min_delta: <float>
  early_stopping_monitor: <metric_name>
  early_stopping_mode: <"min"|"max">

  # Checkpointing (optional)
  checkpoint_every: <int>
  save_best_only: <bool>

  # Hardware Configuration (optional)
  device: <"auto"|"cpu"|"cuda"|"mps">
  num_workers: <int>
  pin_memory: <bool>

  # Advanced Training (optional)
  gradient_clip_val: <float>
  gradient_clip_norm: <float>
  accumulate_grad_batches: <int>

  # Logging (optional)
  log_every: <int>
  verbose: <bool>

  # Reproducibility (optional)
  seed: <int>
```

#### **3.3.1. Core Training Parameters**

**Required fields:**

* `epochs` (int): Number of training epochs (passes through the dataset)
* `batch_size` (int): Number of samples per batch
* `validation_split` (float): Fraction of training data to use for validation (0.0 to 1.0)
* `shuffle` (bool): Whether to shuffle training data each epoch

**Example:**
```yaml
config:
  epochs: 100
  batch_size: 32
  validation_split: 0.2
  shuffle: true
```

#### **3.3.2. Early Stopping (Optional)**

Automatically stops training when validation metric stops improving.

* `early_stopping_patience` (int, optional): Number of epochs to wait for improvement before stopping
* `early_stopping_min_delta` (float, optional): Minimum change to qualify as improvement (default: 0.001)
* `early_stopping_monitor` (str, optional): Metric to monitor (default: "val_loss")
* `early_stopping_mode` (str, optional): Whether metric should decrease ("min") or increase ("max") (default: "min")

**Example:**
```yaml
config:
  early_stopping_patience: 10
  early_stopping_min_delta: 0.0001
  early_stopping_monitor: val_loss
  early_stopping_mode: min
```

#### **3.3.3. Checkpointing (Optional)**

Controls model checkpoint saving during training.

* `checkpoint_every` (int, optional): Save checkpoint every N epochs
* `save_best_only` (bool, optional): Only save checkpoints that improve validation metric

**Example:**
```yaml
config:
  checkpoint_every: 5
  save_best_only: true
```

#### **3.3.4. Hardware Configuration (Optional)**

* `device` (str, optional): Training device ("auto", "cpu", "cuda", "mps"). Default: "auto"
* `num_workers` (int, optional): Number of data loading workers. Default: 0
* `pin_memory` (bool, optional): Pin memory for faster GPU transfer. Default: false

**Example:**
```yaml
config:
  device: cuda
  num_workers: 4
  pin_memory: true
```

#### **3.3.5. Advanced Training (Optional)**

* `gradient_clip_val` (float, optional): Clip gradients by value
* `gradient_clip_norm` (float, optional): Clip gradients by norm (e.g., 1.0 for gradient norm clipping)
* `accumulate_grad_batches` (int, optional): Accumulate gradients over N batches (effective batch size = batch_size × N)

**Example:**
```yaml
config:
  gradient_clip_norm: 1.0
  accumulate_grad_batches: 4  # Effective batch size = 32 × 4 = 128
```

#### **3.3.6. Logging (Optional)**

* `log_every` (int, optional): Log metrics every N batches. Default: 10
* `verbose` (bool, optional): Enable detailed logging. Default: true

**Example:**
```yaml
config:
  log_every: 50
  verbose: true
```

#### **3.3.7. Reproducibility (Optional)**

* `seed` (int, optional): Random seed for reproducibility

**Example:**
```yaml
config:
  seed: 42
```

## **4. Complete Examples**

### 4.1. Example 1: Simple Binary Classification Trainer

Basic trainer for a logistic regression model with early stopping.

```yaml
model_ref: diabetes-logistic-v1

optimizer:
  type: torch.optim.Adam
  lr: 0.001

config:
  epochs: 100
  batch_size: 32
  validation_split: 0.2
  shuffle: true
  early_stopping_patience: 10
  checkpoint_every: 10
  save_best_only: true
  device: auto
  log_every: 10
  verbose: true
```

### 4.2. Example 2: SGD with Momentum and Gradient Clipping

Trainer for a deep neural network with aggressive gradient clipping.

```yaml
model_ref: deep-mlp-v2

optimizer:
  type: torch.optim.SGD
  lr: 0.01
  params:
    momentum: 0.9
    weight_decay: 0.0001
    nesterov: true

config:
  epochs: 200
  batch_size: 64
  validation_split: 0.15
  shuffle: true
  gradient_clip_norm: 1.0
  early_stopping_patience: 20
  early_stopping_min_delta: 0.0001
  checkpoint_every: 5
  save_best_only: true
  device: cuda
  num_workers: 4
  pin_memory: true
  log_every: 20
  verbose: true
  seed: 42
```

### 4.3. Example 3: AdamW with Gradient Accumulation

Trainer for large models requiring gradient accumulation to simulate larger batch sizes.

```yaml
model_ref: transformer-large-v1

optimizer:
  type: torch.optim.AdamW
  lr: 0.0003
  params:
    betas: [0.9, 0.999]
    eps: 1.0e-8
    weight_decay: 0.01

config:
  epochs: 50
  batch_size: 16            # Small batch fits in memory
  accumulate_grad_batches: 8  # Effective batch size = 128
  validation_split: 0.1
  shuffle: true
  gradient_clip_norm: 1.0
  early_stopping_patience: 5
  checkpoint_every: 1
  save_best_only: true
  device: cuda
  num_workers: 8
  pin_memory: true
  log_every: 100
  verbose: true
  seed: 123
```

### 4.4. Example 4: Minimal Fast Training

Minimal trainer spec for quick experimentation.

```yaml
model_ref: prototype-model-v1

optimizer:
  type: torch.optim.Adam
  lr: 0.01

config:
  epochs: 5
  batch_size: 128
  validation_split: 0.2
  shuffle: true
```

## **5. Validation Rules**

### 5.1. Model Reference Validation

- `model_ref` must reference an existing model in the models registry
- The referenced model must have a valid `loss` section defined

### 5.2. Optimizer Validation

- `type` must be a valid PyTorch optimizer class
- `lr` must be a positive float
- `params` must match the optimizer's accepted parameters

### 5.3. Config Validation

- `epochs` must be a positive integer
- `batch_size` must be a positive integer
- `validation_split` must be between 0.0 and 1.0 (exclusive of 1.0)
- `device` must be one of: "auto", "cpu", "cuda", "mps"
- `early_stopping_patience` must be a positive integer or null
- `early_stopping_mode` must be "min" or "max"
- `gradient_clip_val` and `gradient_clip_norm` must be positive floats
- `accumulate_grad_batches` must be a positive integer

### 5.4. Cross-Validation with Model Spec

When a trainer is registered, it should validate:
1. The model exists and has the correct ID format
2. The model has a loss function defined
3. (Future) Learning rate schedule is compatible with optimizer

## **6. Design Rationale**

### Why No Loss in Trainer?

**Loss belongs in the model spec because:**

1. **Architectural Dependency**: Loss inputs must reference specific model outputs
   ```yaml
   # In model spec
   outputs:
     logits: classifier.output
     prediction: sigmoid.output

   loss:
     type: torch.nn.functional.binary_cross_entropy
     inputs:
       input: prediction  # References model output
       target: outcome
   ```

2. **Task Definition**: The loss defines what task the model solves (classification vs regression)

3. **Model Variants**: The same base architecture might have different heads/losses
   ```yaml
   # diabetes-classifier-v1 (classification)
   loss:
     type: torch.nn.functional.binary_cross_entropy

   # diabetes-regressor-v1 (same architecture, different task)
   loss:
     type: torch.nn.functional.mse_loss
   ```

4. **Trainer Reusability**: A trainer (optimizer + hyperparams) can be reused across different models as long as they're compatible

### Why Model Reference?

Explicit `model_ref` enables:
- **Validation**: Verify trainer is compatible with model
- **Lineage Tracking**: Know exactly which trainer trained which model
- **Version Control**: Track trainer versions alongside model versions
- **Flexibility**: Multiple trainers per model (fast training vs production training)

### Why This Structure?

This separation enables powerful workflows:

```bash
# One model, different training strategies
/ml create-model --name diabetes --schema diabetes_model.yaml
/ml create-trainer --name diabetes_fast --schema fast_trainer.yaml --model diabetes
/ml create-trainer --name diabetes_production --schema prod_trainer.yaml --model diabetes

# Quick training for experiments
/ml train --model diabetes --trainer diabetes_fast --data train_data --target outcome

# Production training with full hyperparameter tuning
/ml train --model diabetes --trainer diabetes_production --data train_data --target outcome
```

## **7. Future Extensions**

Potential additions to the trainer spec (not in v1.0):

### Learning Rate Schedulers
```yaml
scheduler:
  type: torch.optim.lr_scheduler.CosineAnnealingLR
  params:
    T_max: 100
    eta_min: 0.00001
```

### Mixed Precision Training
```yaml
config:
  mixed_precision: true
  amp_backend: native  # or "apex"
```

### Distributed Training
```yaml
config:
  distributed: true
  world_size: 4
  backend: nccl
```

### Custom Callbacks
```yaml
callbacks:
  - type: arc.callbacks.TensorBoardLogger
    params:
      log_dir: ./logs
  - type: arc.callbacks.ModelCheckpoint
    params:
      monitor: val_accuracy
```

These features are intentionally excluded from v1.0 to keep the initial spec simple and focused.
