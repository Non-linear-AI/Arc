"""Dry-run validation for training setup with detailed error reporting."""

from __future__ import annotations

import logging
import time
import traceback
from typing import Any

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Validation error with attached ValidationReport for detailed diagnostics."""

    def __init__(self, message: str, report: ValidationReport):
        """Initialize validation error.

        Args:
            message: Error message
            report: Detailed validation report
        """
        super().__init__(message)
        self.validation_report = report


class ValidationReport:
    """Container for validation results with detailed diagnostic information."""

    def __init__(self):
        self.success = False
        self.failed_at_step = None
        self.step_name = None
        self.validation_history = []
        self.error = None
        self.context = {}
        self.root_cause_analysis = []
        self.suggested_fixes = []

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for tool result."""
        return {
            "success": self.success,
            "failed_at_step": self.failed_at_step,
            "step_name": self.step_name,
            "validation_history": self.validation_history,
            "error": self.error,
            "context": self.context,
            "root_cause_analysis": self.root_cause_analysis,
            "suggested_fixes": self.suggested_fixes,
        }


class DryRunValidator:
    """Incremental dry-run validation with dual reporting.

    Provides:
    - Console output for human readability (via logger)
    - Detailed validation report for agent debugging
    """

    def __init__(self, model, train_loader, training_config, model_loss):
        """Initialize validator.

        Args:
            model: Built PyTorch model
            train_loader: Training data loader
            training_config: Training configuration object
            model_loss: Loss function specification
        """
        self.model = model
        self.train_loader = train_loader
        self.training_config = training_config
        self.model_loss = model_loss
        self.report = ValidationReport()

        # Capture data for later steps
        self.sample_batch = None
        self.features = None
        self.targets = None
        self.model_output = None
        self.batch_size = None

    def validate(self) -> ValidationReport:
        """Run incremental validation with console output and detailed logging.

        Returns:
            ValidationReport with success status and diagnostic information
        """
        try:
            # Step 1: Configuration validation
            if not self._validate_config():
                return self.report

            # Step 2: Optimizer setup validation
            if not self._validate_optimizer():
                return self.report

            # Step 3: Data loading
            if not self._validate_data_loading():
                return self.report

            # Step 4: Model forward pass
            if not self._validate_forward_pass():
                return self.report

            # Step 5: Loss calculation
            if not self._validate_loss():
                return self.report

            # All passed
            self.report.success = True
            logger.info("✓ Dry-run validation passed")

        except Exception as e:
            # Unexpected error during validation
            self._handle_unexpected_error(e)

        return self.report

    def _validate_config(self) -> bool:
        """Validate configuration."""
        step_num = "1/5"
        step_name = "Configuration Validation"
        start_time = time.time()

        logger.info("Validating configuration...")

        try:
            # Basic config checks
            if not self.model:
                raise ValueError("Model is None")
            if not self.train_loader:
                raise ValueError("Train loader is None")
            if not self.model_loss:
                raise ValueError("Model loss is None")

            duration_ms = int((time.time() - start_time) * 1000)
            self.report.validation_history.append(
                {
                    "step": step_num,
                    "name": step_name,
                    "status": "passed",
                    "duration_ms": duration_ms,
                }
            )

            logger.info("✓ Configuration valid")
            return True

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self._record_failure(step_num, step_name, duration_ms, e)
            return False

    def _validate_optimizer(self) -> bool:
        """Validate optimizer setup."""
        step_num = "2/5"
        step_name = "Optimizer Setup"
        start_time = time.time()

        try:
            # Get optimizer configuration
            optimizer_name = self.training_config.optimizer
            optimizer_params = dict(self.training_config.optimizer_params or {})

            # Convert learning_rate to lr if present
            if "learning_rate" in optimizer_params:
                optimizer_params["lr"] = optimizer_params.pop("learning_rate")
            else:
                optimizer_params["lr"] = self.training_config.learning_rate

            # Store config in report for diagnostics
            self.report.context["optimizer_name"] = optimizer_name
            self.report.context["optimizer_params"] = optimizer_params

            # Get optimizer class from plugin system
            from arc.plugins import get_plugin_manager

            pm = get_plugin_manager()
            optimizer_class = pm.get_optimizer(optimizer_name)

            if optimizer_class is None:
                optimizers = pm.get_optimizers()
                raise ValueError(
                    f"Unsupported optimizer: {optimizer_name}. "
                    f"Available: {list(optimizers.keys())}"
                )

            # Try to instantiate the optimizer
            # This will catch parameter type errors (e.g., string instead of float)
            _optimizer = optimizer_class(self.model.parameters(), **optimizer_params)

            # Record success
            duration_ms = int((time.time() - start_time) * 1000)
            self.report.validation_history.append(
                {
                    "step": step_num,
                    "name": step_name,
                    "status": "passed",
                    "duration_ms": duration_ms,
                    "details": {
                        "optimizer_class": optimizer_class.__name__,
                        "optimizer_params": optimizer_params,
                    },
                }
            )

            logger.info(f"✓ Optimizer setup successful ({optimizer_class.__name__})")
            return True

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self._record_failure(step_num, step_name, duration_ms, e)

            # Analyze optimizer error
            self._analyze_optimizer_error(e)

            return False

    def _validate_data_loading(self) -> bool:
        """Validate data loading."""
        import torch

        step_num = "3/5"
        step_name = "Data Loading"
        start_time = time.time()

        try:
            # Get a single batch for validation
            self.sample_batch = next(iter(self.train_loader))

            # Unpack batch (features, targets)
            if (
                isinstance(self.sample_batch, (tuple, list))
                and len(self.sample_batch) == 2
            ):
                self.features, self.targets = self.sample_batch
            else:
                self.features = self.sample_batch
                self.targets = None

            # Validate features are tensors
            if not isinstance(self.features, torch.Tensor):
                raise ValueError(
                    f"Expected features to be torch.Tensor, got "
                    f"{type(self.features).__name__}. This usually means data "
                    f"loading failed - check your feature columns."
                )

            # Validate targets if present
            if self.targets is not None and not isinstance(self.targets, torch.Tensor):
                raise ValueError(
                    f"Expected targets to be torch.Tensor, got "
                    f"{type(self.targets).__name__}. This usually means target "
                    f"column has non-numeric data."
                )

            self.batch_size = self.features.shape[0]

            # Record success
            duration_ms = int((time.time() - start_time) * 1000)
            self.report.validation_history.append(
                {
                    "step": step_num,
                    "name": step_name,
                    "status": "passed",
                    "duration_ms": duration_ms,
                    "details": {
                        "batch_size": self.batch_size,
                        "feature_shape": list(self.features.shape),
                        "feature_dtype": str(self.features.dtype),
                        "target_shape": list(self.targets.shape)
                        if self.targets is not None
                        else None,
                        "target_dtype": str(self.targets.dtype)
                        if self.targets is not None
                        else None,
                    },
                }
            )

            num_features = self.features.shape[1] if len(self.features.shape) > 1 else 1
            logger.info(
                f"✓ Data loaded ({self.features.shape[0]} samples, "
                f"{num_features} features)"
            )
            return True

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self._record_failure(step_num, step_name, duration_ms, e)

            # Add context about what failed
            self.report.context.update(
                {
                    "batch_loaded": self.sample_batch is not None,
                    "features_type": str(type(self.features))
                    if self.features is not None
                    else None,
                    "targets_type": str(type(self.targets))
                    if self.targets is not None
                    else None,
                }
            )

            return False

    def _validate_forward_pass(self) -> bool:
        """Validate model forward pass."""
        import torch

        step_num = "4/5"
        step_name = "Forward Pass"
        start_time = time.time()

        try:
            # Try forward pass
            self.model.eval()  # Set to eval mode for validation
            with torch.no_grad():
                self.model_output = self.model(self.features)

            # Get parameter count
            param_count = sum(p.numel() for p in self.model.parameters())

            # Determine output shape
            if isinstance(self.model_output, dict):
                output_shapes = {k: list(v.shape) for k, v in self.model_output.items()}
            else:
                output_shapes = list(self.model_output.shape)

            # Record success
            duration_ms = int((time.time() - start_time) * 1000)
            self.report.validation_history.append(
                {
                    "step": step_num,
                    "name": step_name,
                    "status": "passed",
                    "duration_ms": duration_ms,
                    "details": {
                        "parameter_count": param_count,
                        "output_shapes": output_shapes,
                        "output_type": type(self.model_output).__name__,
                    },
                }
            )

            logger.info(f"✓ Model built ({param_count:,} parameters)")
            logger.info("✓ Forward pass successful")
            return True

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self._record_failure(step_num, step_name, duration_ms, e)

            # Add context
            self.report.context.update(
                {
                    "input_shape": list(self.features.shape),
                    "model_type": str(type(self.model)),
                }
            )

            return False

    def _validate_loss(self) -> bool:
        """Validate loss calculation."""

        step_num = "5/5"
        step_name = "Loss Calculation"
        start_time = time.time()

        try:
            if self.targets is None:
                logger.info("⚠ No targets available - skipping loss validation")
                duration_ms = int((time.time() - start_time) * 1000)
                self.report.validation_history.append(
                    {
                        "step": step_num,
                        "name": step_name,
                        "status": "skipped",
                        "duration_ms": duration_ms,
                    }
                )
                return True

            # Get loss function
            from arc.graph.model.components import get_component_class_or_function
            from arc.ml.param_converter import convert_params_for_pytorch_module

            loss_fn_class, component_kind = get_component_class_or_function(
                self.model_loss.type
            )
            loss_params = self.model_loss.params or {}

            # Capture loss function type for diagnostics
            self.report.context["loss_fn_class"] = str(loss_fn_class)
            self.report.context["loss_params"] = loss_params
            self.report.context["component_kind"] = component_kind

            # Handle both functional and class-based losses
            if component_kind == "function":
                loss_fn = loss_fn_class
                self.report.context["loss_fn_type"] = "functional"
            else:  # It's a class (module)
                # Automatic parameter conversion based on type annotations
                converted_params = convert_params_for_pytorch_module(
                    loss_fn_class, loss_params
                )
                loss_fn = loss_fn_class(**converted_params)
                self.report.context["loss_fn_type"] = "class-based"

            # Get model output for loss
            if isinstance(self.model_output, dict):
                output_key = getattr(
                    self.training_config,
                    "target_output_key",
                    "logits",
                )
                if output_key in self.model_output:
                    model_output = self.model_output[output_key]
                elif "logits" in self.model_output:
                    model_output = self.model_output["logits"]
                else:
                    model_output = next(iter(self.model_output.values()))
            else:
                model_output = self.model_output

            # Store shapes for diagnostic
            self.report.context["model_output_shape"] = list(model_output.shape)
            self.report.context["model_output_dtype"] = str(model_output.dtype)
            self.report.context["target_shape"] = list(self.targets.shape)
            self.report.context["target_dtype"] = str(self.targets.dtype)

            # Reshape targets if needed
            targets = self.targets
            if getattr(self.training_config, "reshape_targets", False):
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1).float()
                elif targets.dim() == 2 and targets.shape[1] != 1:
                    pass
                else:
                    targets = targets.float()
                self.report.context["target_reshaped"] = True
                self.report.context["target_shape_after_reshape"] = list(targets.shape)

            # Compute loss
            if component_kind == "function":  # Functional loss
                loss = loss_fn(model_output, targets, **loss_params)
            else:  # Class-based loss
                loss = loss_fn(model_output, targets)

            # Record success
            duration_ms = int((time.time() - start_time) * 1000)
            self.report.validation_history.append(
                {
                    "step": step_num,
                    "name": step_name,
                    "status": "passed",
                    "duration_ms": duration_ms,
                    "details": {
                        "loss_value": float(loss.item()),
                        "loss_dtype": str(loss.dtype),
                    },
                }
            )

            logger.info(f"✓ Loss calculation successful (value: {loss.item():.4f})")
            return True

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self._record_failure(step_num, step_name, duration_ms, e)

            # Analyze error and provide fixes
            self._analyze_loss_error(e)

            return False

    def _record_failure(
        self, step_num: str, step_name: str, duration_ms: int, error: Exception
    ):
        """Record validation failure with error details."""
        self.report.failed_at_step = step_num
        self.report.step_name = step_name

        self.report.validation_history.append(
            {
                "step": step_num,
                "name": step_name,
                "status": "failed",
                "duration_ms": duration_ms,
                "error": f"{type(error).__name__}: {str(error)}",
            }
        )

        # Extract error details
        tb = traceback.extract_tb(error.__traceback__)
        if tb:
            last_frame = tb[-1]
            file_name = last_frame.filename.split("/")[-1]  # Get just the filename
            self.report.error = {
                "type": type(error).__name__,
                "message": str(error),
                "file": file_name,
                "line": last_frame.lineno,
                "function": last_frame.name,
            }
        else:
            self.report.error = {
                "type": type(error).__name__,
                "message": str(error),
            }

        logger.info(f"✗ {step_name} failed")

    def _analyze_loss_error(self, error: Exception):
        """Analyze loss calculation error and suggest fixes."""
        error_msg = str(error)
        error_type = type(error).__name__

        # Shape mismatch errors
        if (
            "shape" in error_msg.lower()
            or "size" in error_msg.lower()
            or error_type == "RuntimeError"
        ):
            self.report.root_cause_analysis.append(
                f"Shape mismatch between model output "
                f"{self.report.context.get('model_output_shape')} and target "
                f"{self.report.context.get('target_shape')}"
            )

            output_shape = self.report.context.get("model_output_shape", [])
            target_shape = self.report.context.get("target_shape", [])

            # Binary classification: [N, 1] vs [N]
            if (
                len(output_shape) == 2
                and output_shape[1] == 1
                and len(target_shape) == 1
            ):
                self.report.suggested_fixes.append(
                    {
                        "priority": 1,
                        "description": "Squeeze model output to match target shape",
                        "details": (
                            f"Model outputs {output_shape} but "
                            f"target is {target_shape}. "
                            "Add squeeze operation to model graph "
                            "before loss calculation."
                        ),
                        "yaml_change": (
                            "Add to model graph:\n"
                            "  - name: logits_squeezed\n"
                            "    type: torch.squeeze\n"
                            "    params: {dim: 1}\n"
                            "    inputs: {input: output_layer.output}\n"
                            "Update outputs to use: logits_squeezed.output"
                        ),
                    }
                )

        # Tuple not callable error
        elif "'tuple' object is not callable" in error_msg:
            self.report.root_cause_analysis.append(
                "Loss function was returned as tuple instead of callable. "
                "This is likely an Arc framework bug in loss function instantiation."
            )

            self.report.suggested_fixes.append(
                {
                    "priority": 1,
                    "description": "Use class-based loss instead of functional",
                    "details": (
                        "Functional losses may have parsing issues. "
                        "Switch to class-based loss for reliability."
                    ),
                    "yaml_change": (
                        "loss:\n  type: torch.nn.BCEWithLogitsLoss\n  params: {}"
                    ),
                }
            )

        # Non-numeric data errors (categorical columns)
        elif "can't convert np.ndarray of type numpy.object_" in error_msg:
            self.report.root_cause_analysis.append(
                "Data contains categorical columns (VARCHAR/TEXT) that cannot be "
                "converted to tensors. Categorical features must be encoded as "
                "integers before training."
            )

            self.report.suggested_fixes.append(
                {
                    "priority": 1,
                    "description": "Encode categorical features using ml_data tool",
                    "details": (
                        "Use label encoding in the ml_data pipeline:\n"
                        "1. Add fit.label_encoder step to create vocabulary from "
                        "training data\n"
                        "2. Add transform.label_encode step to convert strings to "
                        "integer indices\n"
                        "3. Use the encoded columns as features in your model spec"
                    ),
                    "yaml_change": (
                        "# In ml_data pipeline:\n"
                        "- name: genre_encoder\n"
                        "  type: fit.label_encoder\n"
                        "  params:\n"
                        "    input_table: train_data\n"
                        "    column: genre\n"
                        "    output_table: genre_vocab\n\n"
                        "- name: train_encoded\n"
                        "  type: transform.label_encode\n"
                        "  params:\n"
                        "    input_table: train_data\n"
                        "    column: genre\n"
                        "    vocabulary_table: genre_vocab\n"
                        "    output_column: genre_encoded"
                    ),
                }
            )

            self.report.suggested_fixes.append(
                {
                    "priority": 2,
                    "description": "Use hash bucketing for high-cardinality features",
                    "details": (
                        "For categorical features with many unique values, use "
                        "transform.hash_bucket to hash strings into fixed-size "
                        "integer buckets. This works without fitting a vocabulary."
                    ),
                    "yaml_change": (
                        "# In ml_data pipeline:\n"
                        "- name: train_hashed\n"
                        "  type: transform.hash_bucket\n"
                        "  params:\n"
                        "    input_table: train_data\n"
                        "    column: user_id\n"
                        "    num_buckets: 10000\n"
                        "    output_column: user_id_hashed"
                    ),
                }
            )

            self.report.suggested_fixes.append(
                {
                    "priority": 3,
                    "description": "Remove categorical columns from features",
                    "details": (
                        "If you don't need the categorical features, exclude them "
                        "from the feature list in your model spec inputs."
                    ),
                }
            )

    def _analyze_optimizer_error(self, error: Exception):
        """Analyze optimizer setup error and suggest fixes."""
        error_msg = str(error)

        # Type comparison errors (string vs float)
        if (
            "'<=' not supported between instances of" in error_msg
            or "'float' and 'str'" in error_msg
        ):
            self.report.root_cause_analysis.append(
                "Optimizer parameter has incorrect type (string instead of "
                "numeric). This is likely due to YAML parsing scientific "
                "notation (e.g., '1e-5') as string."
            )

            # Check which parameter caused the issue
            param_info = []
            optimizer_params = self.report.context.get("optimizer_params", {})
            for key, value in optimizer_params.items():
                if isinstance(value, str):
                    param_info.append(
                        f"  - {key}: '{value}' (string, should be numeric)"
                    )

            if param_info:
                self.report.root_cause_analysis.append(
                    "Parameters with incorrect types:\n" + "\n".join(param_info)
                )

            self.report.suggested_fixes.append(
                {
                    "priority": 1,
                    "description": "Use decimal point in scientific notation",
                    "details": (
                        "YAML 1.1 parses '1e-5' as string. Use '1.0e-5' or "
                        "'0.00001' instead. Update the trainer spec YAML to "
                        "use proper numeric format."
                    ),
                    "yaml_change": (
                        "optimizer:\n"
                        "  type: torch.optim.Adam\n"
                        "  lr: 0.001\n"
                        "  params:\n"
                        "    weight_decay: 1.0e-5  # or 0.00001\n"
                        "    # NOT: weight_decay: 1e-5 (parsed as string)"
                    ),
                }
            )

        # Missing or invalid parameters
        elif "missing" in error_msg.lower() or "required" in error_msg.lower():
            optimizer_name = self.report.context.get("optimizer_name")
            self.report.root_cause_analysis.append(
                f"Optimizer {optimizer_name} is missing required parameters"
            )

            self.report.suggested_fixes.append(
                {
                    "priority": 1,
                    "description": "Add required optimizer parameters",
                    "details": (
                        "Check PyTorch documentation for required parameters. "
                        "Add them to the trainer spec under optimizer.params."
                    ),
                }
            )

        # Invalid parameter values
        elif "invalid" in error_msg.lower() or "out of range" in error_msg.lower():
            self.report.root_cause_analysis.append(
                "Optimizer parameter has invalid value (out of acceptable range)"
            )

            self.report.suggested_fixes.append(
                {
                    "priority": 1,
                    "description": "Fix parameter value ranges",
                    "details": (
                        "Check PyTorch optimizer documentation for valid "
                        "parameter ranges. For example, learning rate should "
                        "be positive, weight_decay should be >= 0."
                    ),
                }
            )

    def _handle_unexpected_error(self, error: Exception):
        """Handle unexpected errors during validation."""
        self.report.success = False
        self.report.error = {
            "type": type(error).__name__,
            "message": str(error),
            "stack_trace": traceback.format_exc(),
        }

        self.report.root_cause_analysis.append(
            f"Unexpected error during validation: {type(error).__name__}"
        )

        logger.error(f"Unexpected validation error: {error}", exc_info=True)
