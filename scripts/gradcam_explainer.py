"""
scripts/gradcam_explainer.py
============================
Isolated Grad-CAM explainability module for NeuroLens MobileNetV2 brain MRI classification.

Authoritative Specification:
- Computes Gradient-Weighted Class Activation Mapping (Grad-CAM) for MobileNetV2.
- Target Layer: 'out_relu' (terminal spatial rectified activation, shape (None, 7, 7, 1280)).
- Model Topology: Sequential model containing a nested MobileNetV2 base model and a 4-layer
  classification head (GlobalAveragePooling2D -> Dense(256) -> Dropout -> Dense(4, softmax)).
- Mathematical Formulation:
  The classification head of the deployed model terminates in a 4-unit Dense layer with softmax
  activation. Therefore, predictions[:, class_idx] represents the model's post-softmax probability
  P(Y = c | x), NOT a pre-softmax logit. The Grad-CAM explainer computes the gradient of this
  selected softmax probability with respect to the feature activation map A_{out_relu}.
  The model architecture and weights are not modified or recompiled.
- Preserves inference mode (training=False) to ensure deterministic predictions and disable dropout.
- Zero dependencies on Streamlit or app.py (pure independent library).
- Safe Explainer Caching & Lifecycle:
  Uses an external weakref.WeakKeyDictionary registry keyed by model. Crucially, the explainer
  does NOT retain a strong reference to the model (only a weak reference), completely eliminating
  cyclic reference traps and allowing the model to be garbage-collected immediately when external
  references are dropped. Absolutely NO attributes are set on the Keras model instance.
- Comprehensive Error Handling:
  - Validates class index range [0, 3].
  - Validates input tensor dimensions (4D, batch=1) and channel count (3 RGB channels).
  - Validates gradient existence (rejects None).
  - Validates gradient finiteness (rejects NaN and Inf).
  - Validates gradient non-zeroness (rejects all-zero gradient tensors).
  - Validates positive feature attributions (rejects all-zero post-ReLU heatmaps).
- Clinical Interpretation Boundary:
  Heatmaps represent statistical feature attributions from the model's receptive fields.
  They are NOT verified anatomical structures, NOT pathological boundaries, and NOT clinical diagnoses.
  MobileNetV2's terminal feature grid is 7x7; bilinear interpolation creates smooth visual maps
  but does not synthesize fine anatomical details.
"""

import time
import weakref
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


# ==============================================================================
# Custom Exceptions for Controlled Error Handling
# ==============================================================================

class GradCAMError(Exception):
    """Base exception for Grad-CAM explainability errors."""
    pass


class GradCAMInputError(GradCAMError, ValueError):
    """Raised when input tensors, original images, or class indices fail validation."""
    pass


class GradCAMComputationError(GradCAMError, RuntimeError):
    """Raised when gradients or activation maps fail numerical validity checks."""
    pass


# ==============================================================================
# Safe External Explainer Registry (Zero Mutation & Zero Strong Cycles)
# ==============================================================================

# WeakKeyDictionary holds weak references to model instances.
# When a model is garbage collected, its registry entry automatically disappears.
_EXPLAINER_REGISTRY = weakref.WeakKeyDictionary()


def get_cached_explainer(model, target_layer_name="out_relu"):
    """
    Retrieves or creates a cached GradCAMExplainer without mutating
    the Keras model instance and without creating cyclic strong references.
    
    Args:
        model: tf.keras.Model or Sequential instance.
        target_layer_name: Target layer name string.
        
    Returns:
        tuple: (explainer, was_cached, setup_time_ms)
    """
    t_start = time.perf_counter()
    try:
        if model not in _EXPLAINER_REGISTRY:
            _EXPLAINER_REGISTRY[model] = {}
        if target_layer_name not in _EXPLAINER_REGISTRY[model]:
            explainer = GradCAMExplainer(model, target_layer_name=target_layer_name)
            _EXPLAINER_REGISTRY[model][target_layer_name] = explainer
            t_setup_ms = (time.perf_counter() - t_start) * 1000.0
            return explainer, False, t_setup_ms
        else:
            explainer = _EXPLAINER_REGISTRY[model][target_layer_name]
            t_setup_ms = (time.perf_counter() - t_start) * 1000.0
            return explainer, True, t_setup_ms
    except TypeError:
        # Fallback if model is not weak-referenceable
        explainer = GradCAMExplainer(model, target_layer_name=target_layer_name)
        t_setup_ms = (time.perf_counter() - t_start) * 1000.0
        return explainer, False, t_setup_ms


def clear_explainer_registry():
    """Explicitly clears the explainer cache registry."""
    _EXPLAINER_REGISTRY.clear()


# ==============================================================================
# Core GradCAMExplainer Class
# ==============================================================================

class GradCAMExplainer:
    """
    Manages intermediate activation extraction and gradient calculation
    for MobileNetV2 Sequential architectures with nested base models.
    Does NOT hold a strong reference to the parent model.
    """

    def __init__(self, model, target_layer_name="out_relu"):
        """
        Initializes the Grad-CAM explainer.
        
        Args:
            model: Loaded tf.keras.Model or Sequential instance.
            target_layer_name: Name of the convolutional or activation layer to inspect.
        """
        self.target_layer_name = target_layer_name
        self.t_setup_ms = 0.0
        # Crucial lifecycle fix: Store weak reference to avoid cyclic retention in WeakKeyDictionary
        self._model_ref = weakref.ref(model)
        self._setup_gradient_model(model)

    def _setup_gradient_model(self, model):
        """
        Extracts the base model and head layers to construct a gradient-connected graph.
        Does not retain a strong reference to model.
        """
        t0 = time.perf_counter()
        # Case 1: Sequential model with nested Functional base model at layer 0
        if isinstance(model, tf.keras.Sequential) and hasattr(model.layers[0], "layers"):
            base_model = model.layers[0]
            self.head_layers = list(model.layers[1:])
            
            # Find target layer in base model
            try:
                target_layer = base_model.get_layer(self.target_layer_name)
            except ValueError:
                raise GradCAMInputError(
                    f"Target layer '{self.target_layer_name}' not found in base model '{base_model.name}'."
                )
                
            # Build intermediate submodel for base: inputs -> [target_layer.output, base_model.output]
            self.base_submodel = tf.keras.Model(
                inputs=base_model.inputs,
                outputs=[target_layer.output, base_model.output],
                name="gradcam_base_submodel"
            )
            self.is_nested = True
        else:
            # Case 2: Flat Functional or Sequential model
            self.is_nested = False
            try:
                target_layer = model.get_layer(self.target_layer_name)
            except ValueError:
                raise GradCAMInputError(
                    f"Target layer '{self.target_layer_name}' not found in model '{model.name}'."
                )
            self.grad_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[target_layer.output, model.output],
                name="gradcam_flat_model"
            )
        self.t_setup_ms = (time.perf_counter() - t0) * 1000.0

    def _validate_inputs(self, preprocessed_tensor, class_idx):
        """
        Validates input tensor geometry, types, and class index bounds.
        """
        if not isinstance(preprocessed_tensor, (np.ndarray, tf.Tensor)):
            raise GradCAMInputError(
                f"preprocessed_tensor must be numpy.ndarray or tf.Tensor, got {type(preprocessed_tensor).__name__}."
            )
        
        shape = preprocessed_tensor.shape
        if len(shape) != 4:
            raise GradCAMInputError(
                f"Input tensor must have 4 dimensions (batch, height, width, channels), got {len(shape)} dimensions with shape {shape}."
            )
        if shape[0] != 1:
            raise GradCAMInputError(
                f"Input tensor batch size must be 1, got batch size {shape[0]} with shape {shape}."
            )
        if shape[-1] != 3:
            raise GradCAMInputError(
                f"Input tensor channel count must be 3 (RGB), got {shape[-1]} channels with shape {shape}."
            )

        if class_idx is not None:
            if not isinstance(class_idx, (int, np.integer)):
                raise GradCAMInputError(
                    f"Target class index must be an integer, got {type(class_idx).__name__}."
                )
            if class_idx < 0 or class_idx > 3:
                raise GradCAMInputError(
                    f"Target class index must be in range [0, 3] (0=Glioma, 1=Meningioma, 2=No Tumor, 3=Pituitary), got {class_idx}."
                )

    def _validate_gradients(self, grads):
        """
        Rigorously checks gradient tensor existence, finiteness, and non-zeroness.
        """
        # 1. Missing gradients check
        if grads is None:
            raise GradCAMComputationError(
                "Gradient computation failed: tf.GradientTape returned None. "
                "The target layer is disconnected from the model output in the computation graph."
            )

        # 2. Gradient finiteness check (no NaNs, no Infs)
        if not bool(tf.reduce_all(tf.math.is_finite(grads)).numpy()):
            raise GradCAMComputationError(
                "Gradient computation failed: gradients contain non-finite values (NaN or Inf)."
            )

        # 3. Gradient non-zeroness check (at least one non-zero gradient entry)
        max_grad_abs = tf.reduce_max(tf.abs(grads))
        if float(max_grad_abs.numpy()) == 0.0:
            raise GradCAMComputationError(
                "Gradient computation failed: all computed gradients are exactly zero. "
                "The target layer has no active gradient flow to the selected class score."
            )

    def _compute_heatmap(self, grads, conv_features):
        """
        Computes Global Average Pooling of gradients, weights feature maps,
        applies ReLU, and normalizes into [0.0, 1.0]. Rejects all-zero activation maps.
        """
        # Global average pooling of gradients: shape (1, channels)
        weights = tf.reduce_mean(grads, axis=(1, 2))

        # Linear combination of weighted feature maps: (1, H, W, C) * (1, 1, 1, C) -> sum over C -> (H, W)
        cam = tf.reduce_sum(
            tf.multiply(weights[:, tf.newaxis, tf.newaxis, :], conv_features),
            axis=-1
        )[0]

        # Apply ReLU: retain only features that contribute positively to the selected class probability
        cam = tf.maximum(cam, 0.0)

        # Check positive activations (reject all-zero post-ReLU heatmaps)
        cam_max = tf.reduce_max(cam)
        cam_max_float = float(cam_max.numpy())
        if cam_max_float <= 0.0 or not np.isfinite(cam_max_float):
            raise GradCAMComputationError(
                "Grad-CAM generation failed: all rectified feature activations are zero. "
                "No positive feature attributions exist for the selected class score."
            )

        # Normalize spatial heatmap strictly to [0.0, 1.0]
        cam = cam / cam_max
        return cam.numpy()

    def explain(self, preprocessed_tensor, class_idx=None):
        """
        Calculates gradients and raw Grad-CAM heatmap.
        
        Args:
            preprocessed_tensor: 4D numpy array or tf.Tensor of shape (1, 224, 224, 3).
            class_idx: Target class index (0=Glioma, 1=Meningioma, 2=No Tumor, 3=Pituitary).
                       If None, top-1 predicted class is used.
            
        Returns:
            dict containing:
                - heatmap_raw: 2D numpy array of shape (7, 7), normalized to [0.0, 1.0].
                - predictions: 1D numpy array of shape (4,) containing softmax probabilities.
                - predicted_class_idx: int.
                - target_class_idx: int.
                - target_score: float (softmax probability of target class).
                - gradient_metrics: dict of verified gradient properties.
                - timing_ms: dict containing component latencies.
        """
        t_start = time.perf_counter()

        # 1. Input Tensor & Class Validation
        self._validate_inputs(preprocessed_tensor, class_idx)
        inputs_tf = tf.cast(preprocessed_tensor, tf.float32)

        # 2. Forward Pass & Gradient Computation
        t_forward_start = time.perf_counter()
        with tf.GradientTape() as tape:
            if self.is_nested:
                conv_features, base_output = self.base_submodel(inputs_tf, training=False)
                tape.watch(conv_features)

                x = base_output
                for layer in self.head_layers:
                    x = layer(x, training=False)
                predictions = x
            else:
                conv_features, predictions = self.grad_model(inputs_tf, training=False)
                tape.watch(conv_features)

            # Resolve target class index if None
            if class_idx is None:
                class_idx = int(tf.argmax(predictions[0]))

            # Differentiate the selected softmax probability score
            target_score = predictions[:, class_idx]

        # Compute gradient of target softmax probability with respect to convolutional feature maps
        grads = tape.gradient(target_score, conv_features)
        t_forward_and_grad = (time.perf_counter() - t_forward_start) * 1000.0

        # 3. Rigorous Gradient Validation (Raises GradCAMComputationError if invalid)
        self._validate_gradients(grads)

        # 4. Global Average Pooling & Linear Combination (Raises GradCAMComputationError if all-zero)
        t_cam_start = time.perf_counter()
        heatmap_raw = self._compute_heatmap(grads, conv_features)
        t_cam_pooling = (time.perf_counter() - t_cam_start) * 1000.0
        t_total_explain = (time.perf_counter() - t_start) * 1000.0

        # 5. Gradient metadata for QA reporting
        grads_np = grads.numpy()
        max_grad_abs = float(tf.reduce_max(tf.abs(grads)).numpy())
        gradient_metrics = {
            "grads_shape": list(grads.shape),
            "grads_exist": True,
            "grads_finite": True,
            "grads_nonzero": True,
            "max_abs_grad": max_grad_abs,
            "min_grad": float(np.min(grads_np)),
            "max_grad": float(np.max(grads_np)),
            "mean_abs_grad": float(np.mean(np.abs(grads_np)))
        }

        return {
            "heatmap_raw": heatmap_raw,
            "predictions": predictions.numpy()[0],
            "predicted_class_idx": int(np.argmax(predictions.numpy()[0])),
            "target_class_idx": class_idx,
            "target_score": float(target_score.numpy()[0]),
            "gradient_metrics": gradient_metrics,
            "timing_ms": {
                "forward_and_grad_ms": t_forward_and_grad,
                "cam_pooling_ms": t_cam_pooling,
                "total_explain_ms": t_total_explain
            }
        }


# ==============================================================================
# Independent Functional Interface with Non-Overlapping Additive Timings
# ==============================================================================

def generate_gradcam_overlay(
    model,
    preprocessed_tensor,
    original_image,
    class_idx=None,
    target_layer_name="out_relu",
    alpha=0.4,
    colormap="jet",
    explainer=None
):
    """
    Independent functional interface to generate a complete Grad-CAM visualization.
    Measures strictly non-overlapping additive stages so that sum(stages) == total_gradcam_ms.

    Args:
        model: Loaded tf.keras.Model or Sequential instance.
        preprocessed_tensor: 4D numpy array or tf.Tensor of shape (1, 224, 224, 3).
        original_image: PIL Image or RGB numpy array representing the original unscaled MRI.
        class_idx: Target class index (0=Glioma, 1=Meningioma, 2=No Tumor, 3=Pituitary).
                   If None, top-1 predicted class is used.
        target_layer_name: Convolutional/activation target layer (default: 'out_relu').
        alpha: Overlay blending weight (0.0 = original MRI only, 1.0 = heatmap only, default: 0.4).
        colormap: Matplotlib colormap identifier (default: 'jet').
        explainer: Optional pre-instantiated GradCAMExplainer instance. If None, retrieves
                   or instantiates via safe external WeakKeyDictionary registry (zero model mutation).

    Returns:
        dict containing heatmap arrays, PIL images, predictions, and additive stage timings.
    """
    t0 = time.perf_counter()

    # Stage 1: Explainer Resolution (First-Time Construction or WeakKeyDictionary Cache Retrieval)
    was_cached = False
    if explainer is None:
        explainer, was_cached, _ = get_cached_explainer(
            model, target_layer_name=target_layer_name
        )
    else:
        was_cached = True
    t1 = time.perf_counter()
    t_stage1_setup_ms = (t1 - t0) * 1000.0

    # Stage 2: Forward Pass & Gradient Computation (autodiff tape)
    explainer._validate_inputs(preprocessed_tensor, class_idx)
    inputs_tf = tf.cast(preprocessed_tensor, tf.float32)

    with tf.GradientTape() as tape:
        if explainer.is_nested:
            conv_features, base_output = explainer.base_submodel(inputs_tf, training=False)
            tape.watch(conv_features)

            x = base_output
            for layer in explainer.head_layers:
                x = layer(x, training=False)
            predictions = x
        else:
            conv_features, predictions = explainer.grad_model(inputs_tf, training=False)
            tape.watch(conv_features)

        if class_idx is None:
            resolved_class_idx = int(tf.argmax(predictions[0]))
        else:
            resolved_class_idx = class_idx

        target_score = predictions[:, resolved_class_idx]

    grads = tape.gradient(target_score, conv_features)
    explainer._validate_gradients(grads)
    del tape
    t2 = time.perf_counter()
    t_stage2_fwd_grad_ms = (t2 - t1) * 1000.0

    # Stage 3: Channel Weighting, Pooling & ReLU Synthesis
    heatmap_raw = explainer._compute_heatmap(grads, conv_features)
    t3 = time.perf_counter()
    t_stage3_pooling_ms = (t3 - t2) * 1000.0

    # Stage 4: Overlay Generation (Original Image Formatting, Bilinear Upsampling, Colormap & Blending)
    if isinstance(original_image, Image.Image):
        orig_pil = original_image.convert("RGB")
    elif isinstance(original_image, np.ndarray):
        if len(original_image.shape) != 3 or original_image.shape[-1] != 3:
            raise GradCAMInputError(
                f"original_image numpy array must have shape (H, W, 3), got {original_image.shape}."
            )
        orig_pil = Image.fromarray(original_image.astype(np.uint8)).convert("RGB")
    else:
        raise GradCAMInputError(
            f"original_image must be PIL.Image or numpy.ndarray, got {type(original_image).__name__}."
        )

    orig_w, orig_h = orig_pil.size
    orig_np = np.array(orig_pil, dtype=np.float32)

    heatmap_pil = Image.fromarray((heatmap_raw * 255.0).astype(np.uint8), mode="L")
    heatmap_resized_pil = heatmap_pil.resize((orig_w, orig_h), resample=Image.Resampling.BILINEAR)
    heatmap_resized = np.array(heatmap_resized_pil, dtype=np.float32) / 255.0

    try:
        cmap = matplotlib.colormaps[colormap]
    except KeyError:
        cmap = matplotlib.colormaps["jet"]

    cam_rgba = cmap(heatmap_resized)
    cam_rgb = (cam_rgba[:, :, :3] * 255.0).astype(np.uint8)

    alpha_clamped = np.clip(float(alpha), 0.0, 1.0)
    overlay_float = alpha_clamped * cam_rgb.astype(np.float32) + (1.0 - alpha_clamped) * orig_np
    overlay_array = np.clip(overlay_float, 0.0, 255.0).astype(np.uint8)
    overlay_image = Image.fromarray(overlay_array, mode="RGB")
    t4 = time.perf_counter()
    t_stage4_overlay_ms = (t4 - t3) * 1000.0

    # Contiguous non-overlapping timing measurements (telescoping sum identity)
    t_compute_and_overlay_ms = (t4 - t1) * 1000.0
    t_total_gradcam_ms = (t4 - t0) * 1000.0

    # Collect gradient metrics
    grads_np = grads.numpy()
    max_grad_abs = float(tf.reduce_max(tf.abs(grads)).numpy())
    gradient_metrics = {
        "grads_shape": list(grads.shape),
        "grads_exist": True,
        "grads_finite": True,
        "grads_nonzero": True,
        "max_abs_grad": max_grad_abs,
        "min_grad": float(np.min(grads_np)),
        "max_grad": float(np.max(grads_np)),
        "mean_abs_grad": float(np.mean(np.abs(grads_np)))
    }
    del grads, grads_np

    return {
        "heatmap_raw": heatmap_raw,
        "heatmap_resized": heatmap_resized,
        "heatmap_colorized": cam_rgb,
        "overlay_image": overlay_image,
        "overlay_array": overlay_array,
        "predictions": predictions.numpy()[0],
        "predicted_class_idx": int(np.argmax(predictions.numpy()[0])),
        "target_class_idx": resolved_class_idx,
        "target_class_score": float(target_score.numpy()[0]),
        "gradient_metrics": gradient_metrics,
        "was_cached": was_cached,
        "timing_ms": {
            "explainer_setup_ms": t_stage1_setup_ms,
            "forward_and_grad_ms": t_stage2_fwd_grad_ms,
            "cam_pooling_ms": t_stage3_pooling_ms,
            "overlay_blending_ms": t_stage4_overlay_ms,
            "gradcam_compute_and_overlay_ms": t_compute_and_overlay_ms,
            "total_gradcam_overhead_ms": t_total_gradcam_ms,
            # Legacy aliases for backward compatibility
            "submodel_setup_ms": t_stage1_setup_ms,
            "resize_and_overlay_ms": t_stage4_overlay_ms,
            "total_gradcam_execution_ms": t_total_gradcam_ms,
        }
    }
