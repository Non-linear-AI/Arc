# Multi-Input Model Training Validation Error

## Problem Summary

Training validation fails for models with multiple inputs with this error:

```
Dry-run validation failed at step 4/5 (Forward Pass):
ValueError: Tensor input provided but model requires multiple named inputs
```

## Root Cause

The model has **4 separate inputs**:
- `user_embedding_input` (columns: `[UserID]`)
- `movie_embedding_input` (columns: `[MovieID]`)
- `numeric_features` (columns: `[user_rating_count, user_avg_rating, ...]`)
- `demographic_features` (columns: `[gender_female, gender_male, ...]`)

However, the **data loader returns a single tensor** containing all features concatenated together.

### The Mismatch

**What the model expects** (`src/arc/ml/builder.py:248-253`):
```python
if isinstance(inputs, torch.Tensor):
    if len(self.input_names) != 1:
        raise ValueError(
            "Tensor input provided but model requires multiple named inputs"
        )
    inputs = {self.input_names[0]: inputs}
```

**What the data loader provides** (`src/arc/ml/data.py:23-45`):
```python
class ArcDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor | None = None):
        self.features = features  # Single tensor

    def __getitem__(self, idx: int):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]  # Returns single tensor
        return self.features[idx]
```

The model's forward method checks if it receives a single tensor but has multiple inputs - and raises this exact error.

## Where the Error Occurs

1. **Dry-run validator** (`src/arc/ml/dry_run_validator.py:309-364`):
   - Line 321: `self.model_output = self.model(self.features)`
   - `self.features` is a single tensor from the data loader
   - Model expects a dictionary

2. **Training loop** (same issue will occur):
   - Training loop loads data the same way
   - Will fail with the same error during actual training

## Missing Feature

There is **no mechanism** to split a feature tensor according to the model's input column specifications.

The model spec defines which columns belong to which input:
```yaml
inputs:
  user_embedding_input:
    columns: [UserID]
  movie_embedding_input:
    columns: [MovieID]
  numeric_features:
    columns: [user_rating_count, user_avg_rating, ...]
```

But the data loading code doesn't use this information to split the features.

## Solution Options

### Option 1: Add Feature Splitting to Data Loader (Recommended)

Create a new dataset class that splits features according to model input specs:

```python
class MultiInputDataset(Dataset):
    """Dataset for models with multiple named inputs."""

    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor | None,
        input_spec: dict[str, dict],  # From model spec
        feature_columns: list[str],   # Column order in features tensor
    ):
        self.features = features
        self.targets = targets

        # Compute column indices for each input
        self.input_slices = {}
        for input_name, spec in input_spec.items():
            input_columns = spec.get('columns', [])
            indices = [feature_columns.index(col) for col in input_columns]
            self.input_slices[input_name] = indices

    def __getitem__(self, idx):
        # Split features according to input specs
        input_dict = {}
        for input_name, indices in self.input_slices.items():
            input_dict[input_name] = self.features[idx, indices]

        if self.targets is not None:
            return input_dict, self.targets[idx]
        return input_dict
```

**Pros**:
- Clean separation of concerns
- Works for any multi-input model
- Reusable for training and inference

**Cons**:
- Requires passing model spec to data loading
- More complex data loading code

### Option 2: Add Validation Before Model Registration

Reject models with multiple inputs during registration if they're not supported yet:

```python
# In model registration code
if len(model_spec.inputs) > 1:
    raise ValueError(
        "Models with multiple inputs are not yet supported for training. "
        "Please use a single input that concatenates all features."
    )
```

**Pros**:
- Prevents user confusion
- Clear error message
- Easy to implement

**Cons**:
- Limits functionality
- Users can't use multi-input models (which are useful for embeddings)

### Option 3: Improve Dry-Run Validation Error Detection

Make the dry-run validator detect this case and provide a helpful error:

```python
# In dry_run_validator.py _validate_forward_pass
if isinstance(self.features, torch.Tensor) and len(self.model.input_names) > 1:
    self.report.root_cause_analysis.append(
        f"Model has {len(self.model.input_names)} inputs but data loader "
        f"provides a single tensor. Multi-input models require dictionary inputs."
    )
    self.report.suggested_fixes.append({
        "priority": 1,
        "description": "Use a single input in your model",
        "details": "Combine all inputs into one input with all columns..."
    })
```

**Pros**:
- Better error messages
- Helps users understand the issue

**Cons**:
- Doesn't fix the underlying problem
- Users still can't use multi-input models

## Recommendation

**Implement Option 1** (Feature Splitting) as the proper fix, then **add Option 3** (Better Error Messages) for cases where it's not used correctly.

This provides both:
1. Full functionality for multi-input models
2. Clear guidance when something goes wrong

## Implementation Plan

1. Create `MultiInputDataset` class in `src/arc/ml/data.py`
2. Detect multi-input models in training setup
3. Use `MultiInputDataset` when model has multiple inputs
4. Update dry-run validator to check for this mismatch
5. Add tests for multi-input model training

## Files That Need Changes

1. `src/arc/ml/data.py` - Add `MultiInputDataset` class
2. `src/arc/ml/training_service.py` - Detect and use multi-input dataset
3. `src/arc/ml/dry_run_validator.py` - Better error messages
4. `tests/ml/test_training.py` - Add multi-input model tests
