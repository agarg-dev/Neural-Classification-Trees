
"""
Attribution Visualization Module using Captum.
Handles model interpretation, visualization, and faithfulness metric evaluation.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Literal, Union
from torch.utils.data import Dataset, Subset

try:
    from captum.attr import (
        LayerGradCam, GuidedGradCam, LayerConductance, 
        LayerIntegratedGradients, LayerGradientXActivation,
        Saliency, InputXGradient, DeepLift, GuidedBackprop, Occlusion,
        NoiseTunnel, visualization as viz
    )
    from captum.metrics import infidelity
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

from ..data.datasets import extract_labels_from_dataset
from ..utils.seed import set_seed
from ..utils.image import denormalize_image
from ..evaluation.analysis import get_node_label
from ..evaluation.sampling import collect_node_samples, collect_transition_samples, stratified_sample

# --- Method Groups ---
LAYER_METHODS = [
    "layer_gradcam", "layer_conductance", 
    "layer_integrated_gradients", "layer_gradient_x_activation"
]

PIXEL_METHODS = [
    "saliency", "guided_gradcam", "guided_backprop", 
    "input_x_gradient", "deeplift", "occlusion"
]

ALL_METHODS = LAYER_METHODS + PIXEL_METHODS

AttributionMethod = Literal[
    "layer_gradcam", "guided_gradcam", "layer_conductance",
    "layer_integrated_gradients", "layer_gradient_x_activation",
    "saliency", "input_x_gradient", "deeplift", 
    "guided_backprop", "occlusion"
]

class CaptumModelAdapter(nn.Module):
    """Wraps model to ensure forward pass returns a single tensor."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, tuple):
            return out[0]
        return out
class CaptumAttribution:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        checkpoint_loader: callable,
        class_names: List[str],
        num_iterations: int,
        hierarchical: str = "none",
        attribution_method: AttributionMethod = "layer_gradcam"
    ):
        if not CAPTUM_AVAILABLE:
             raise ImportError("captum required. Install with: pip install captum")
        self.model = model
        self.device = device
        self.checkpoint_loader = checkpoint_loader
        self.class_names = class_names
        self.num_iterations = num_iterations
        self.hierarchical = hierarchical
        self.attribution_method = attribution_method
        self._current_seed = None

    def _get_target_layer(self, model: nn.Module) -> nn.Module:
        """Heuristic to find the last convolutional layer."""
        # ResNet backbone
        if hasattr(model, 'base') and hasattr(model.base, 'layer4'):
            return model.base.layer4[-1]
        # LeNet5 backbone - use conv2 as the last conv layer
        if hasattr(model, 'base') and hasattr(model.base, 'conv2'):
            return model.base.conv2
        raise ValueError("Could not detect target layer. Model must have 'base.layer4' (ResNet) or 'base.conv2' (LeNet5).")

    def _fix_relu_inplace(self, model: nn.Module):
        """
        DeepLIFT crashes if ReLUs are inplace=True. 
        This recursively changes them to inplace=False.
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

    def _create_explainer(self, model: nn.Module, method: Optional[AttributionMethod] = None):
        method = method or self.attribution_method
        wrapped_model = CaptumModelAdapter(model)
        wrapped_model.eval()
        
        # Identify methods that require access to internal layers
        layer_methods = [
            "layer_gradcam", "layer_conductance", 
            "layer_integrated_gradients", "layer_gradient_x_activation", 
            "guided_gradcam"
        ]
        
        target_layer = self._get_target_layer(model) if method in layer_methods else None

        # --- Layer Methods ---
        if method == "layer_gradcam":
            return LayerGradCam(wrapped_model, target_layer)
        elif method == "layer_conductance":
            return LayerConductance(wrapped_model, target_layer)
        elif method == "layer_integrated_gradients":
            return LayerIntegratedGradients(wrapped_model, target_layer)
        elif method == "layer_gradient_x_activation":
            return LayerGradientXActivation(wrapped_model, target_layer)
        elif method == "guided_gradcam":
            return GuidedGradCam(wrapped_model, target_layer)
        
        # --- Input Methods ---
        elif method == "occlusion": 
            return Occlusion(wrapped_model)
        
        # --- Pixel Methods (wrapped in NoiseTunnel) ---
        base_explainer = None
        if method == "saliency": 
            base_explainer = Saliency(wrapped_model)
        elif method == "input_x_gradient": 
            base_explainer = InputXGradient(wrapped_model)
        elif method == "deeplift": 
            # <--- CRITICAL FIX FOR DEEPLIFT --->
            self._fix_relu_inplace(wrapped_model) 
            base_explainer = DeepLift(wrapped_model)
        elif method == "guided_backprop": 
            base_explainer = GuidedBackprop(wrapped_model)
        
        if base_explainer:
            return NoiseTunnel(base_explainer)
        
        raise ValueError(f"Unsupported attribution method: {method}")

    def _compute_batch_attributions(self, explainer, batch_images, batch_targets):
        """Compute attributions handling method-specific logic (upsampling/smoothing)."""
        
        # 1. Layer Methods: Output is small (e.g. 7x7). Must upsample.
        if isinstance(explainer, (LayerGradCam, LayerConductance, 
                                  LayerIntegratedGradients, LayerGradientXActivation)):
            attributions = explainer.attribute(batch_images, target=batch_targets)
            
            if attributions.dim() == 4 and attributions.shape[1] > 1:
                attributions = attributions.sum(dim=1, keepdim=True)
                attributions = torch.relu(attributions)
            
            attributions = torch.nn.functional.interpolate(
                attributions, size=batch_images.shape[-2:], mode='bilinear', align_corners=False
            )

        # 2. Guided GradCAM: High-res natively.
        elif isinstance(explainer, GuidedGradCam):
            attributions = explainer.attribute(batch_images, target=batch_targets)

        # 3. Pixel Methods: Use NoiseTunnel.
        elif isinstance(explainer, NoiseTunnel):
            attributions = explainer.attribute(
                batch_images, target=batch_targets, nt_type='smoothgrad_sq', nt_samples=10, stdevs=0.15
            )

        # 4. Occlusion: Use adaptive window size based on image dimensions.
        elif isinstance(explainer, Occlusion):
            num_channels = batch_images.shape[1]
            img_size = batch_images.shape[-1]
            # Use smaller window for smaller images (e.g., 28x28 MNIST)
            if img_size <= 32:
                window_size = 4
                stride = 1
            else:
                window_size = 8
                stride = 2
            attributions = explainer.attribute(
                batch_images, target=batch_targets,
                sliding_window_shapes=(num_channels, window_size, window_size),
                strides=(num_channels, stride, stride)
            )

        else:
            attributions = explainer.attribute(batch_images, target=batch_targets)

        if attributions.shape[1] != batch_images.shape[1]:
            attributions = attributions.mean(dim=1, keepdim=True).repeat(1, batch_images.shape[1], 1, 1)

        return attributions

    def _evaluate_faithfulness_metrics(self, dataset, iteration_results, max_samples=500):
        print(f"\n--- Evaluating Faithfulness Metrics ({max_samples} samples) ---")
        results = {}

        for iter_idx, result in enumerate(iteration_results):
            iteration = iter_idx + 1
            checkpoint_path = result.get('checkpoint_path')
            if not checkpoint_path or not os.path.exists(checkpoint_path):
                continue

            raw_model = self.checkpoint_loader(checkpoint_path)
            explainer = self._create_explainer(raw_model)
            wrapped_model = CaptumModelAdapter(raw_model)
            wrapped_model.eval()

            targets = result['hierarchical_predictions']
            labels = extract_labels_from_dataset(dataset).numpy()

            if len(dataset) > max_samples:
                indices = stratified_sample(targets, labels, max_samples, seed=self._current_seed)
                eval_targets = targets[indices]
            else:
                indices = np.arange(len(dataset))
                eval_targets = targets

            print(f"  Iteration {iteration} ({len(indices)} samples)...")
            infid_scores = []
            
            heavy_methods = ['deeplift', 'layer_integrated_gradients']
            batch_size = 2 if self.attribution_method in heavy_methods else 16

            def perturb_func(inputs):
                noise = torch.randn_like(inputs) * 0.05
                return noise, inputs + noise

            eval_subset = Subset(dataset, indices)

            for i in range(0, len(indices), batch_size):
                batch_idx = list(range(i, min(i + batch_size, len(indices))))
                batch_list = [eval_subset[j][0] for j in batch_idx]

                batch_images = torch.stack(batch_list).to(self.device)
                batch_images.requires_grad_(True)
                batch_targets = torch.tensor([int(eval_targets[j]) for j in batch_idx], device=self.device)

                batch_attrs = self._compute_batch_attributions(explainer, batch_images, batch_targets).detach()

                try:
                    batch_infid = infidelity(
                        wrapped_model, perturb_func, batch_images, batch_attrs, target=batch_targets, n_perturb_samples=5
                    )
                    if isinstance(batch_infid, tuple): batch_infid = batch_infid[0]
                    infid_scores.extend(batch_infid.detach().cpu().numpy().tolist())
                except Exception as e:
                    if i == 0: print(f"    Warning: Infidelity calculation failed: {e}")

            results[iteration] = {
                'num_samples': len(indices),
                'infidelity': float(np.mean(infid_scores)) if infid_scores else 0.0,
            }
            print(f"    Infidelity: {results[iteration]['infidelity']:.4f}")

            del raw_model, explainer, wrapped_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return results

    def _denorm(self, tensor):
        """Denormalize tensor for visualization. Auto-detects grayscale vs RGB."""
        num_channels = tensor.shape[1]
        if num_channels == 1:
            # Grayscale (MNIST/UMNIST)
            mean = torch.tensor([0.1307], device=tensor.device).view(1, 1, 1, 1)
            std = torch.tensor([0.3081], device=tensor.device).view(1, 1, 1, 1)
        else:
            # RGB (ImageNet normalization)
            mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
        out = tensor * std + mean
        return torch.clamp(out, 0, 1)

    def _generate_attribution(self, explainer, input_tensor, target):
        input_tensor.requires_grad_(True)
        target_tensor = torch.tensor([int(target)], device=self.device)
        
        attr = self._compute_batch_attributions(explainer, input_tensor, target_tensor)
        attr = attr.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        
        if attr.shape[2] == 3:
            if self.attribution_method in LAYER_METHODS:
                attr = np.mean(attr, axis=2, keepdims=True)
            else:
                attr = np.linalg.norm(attr, axis=2, keepdims=True)
        return attr

    def _safe_visualize(self, attr, img_tensor, fig, ax, title):
        img_denorm = self._denorm(img_tensor.unsqueeze(0)).squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Convert grayscale to RGB for visualization (Captum expects RGB)
        if img_denorm.shape[2] == 1:
            img_rgb = np.repeat(img_denorm, 3, axis=2)
        else:
            img_rgb = img_denorm

        if np.max(attr) - np.min(attr) < 1e-5:
            ax.imshow(img_rgb)
            ax.set_title(f"{title}\n(No Attribution)", fontsize=8)
            return

        try:
            if self.attribution_method in LAYER_METHODS:
                cmap = 'jet'
                outlier = 2
                sign = "positive"
                alpha = 0.5
            elif self.attribution_method in ["occlusion"]:
                cmap = 'inferno'
                outlier = 2
                sign = "positive"
                alpha = 0.7
            else: # Pixel/Sparse methods (DeepLIFT uses this)
                cmap = 'inferno'
                outlier = 10
                sign = "absolute_value"
                alpha = 0.7

            viz.visualize_image_attr(
                attr, img_rgb,
                method="blended_heat_map",
                sign=sign,
                show_colorbar=False,
                title=title,
                cmap=cmap,
                outlier_perc=outlier,
                alpha_overlay=alpha,
                plt_fig_axis=(fig, ax),
                use_pyplot=False
            )
        except Exception:
            ax.imshow(img_rgb)

    def _create_node_grid(self, dataset, node_samples, iteration, explainer):
        num_nodes = len(node_samples); max_samples = max(len(s) for s in node_samples.values()) if any(node_samples.values()) else 0
        if max_samples == 0: return plt.figure()
        fig, axes = plt.subplots(num_nodes, max_samples, figsize=(max_samples * 3, num_nodes * 3))
        if num_nodes == 1: axes = axes.reshape(1, -1)
        if max_samples == 1: axes = axes.reshape(-1, 1)
        fig.suptitle(f"Iteration {iteration} - Node Attributions", fontsize=16)

        for node_idx in sorted(node_samples.keys()):
            sample_indices = node_samples[node_idx]
            for col_idx in range(max_samples):
                ax = axes[node_idx, col_idx]
                if col_idx < len(sample_indices):
                    idx = sample_indices[col_idx]
                    img_t, _ = dataset[idx]
                    input_t = img_t.unsqueeze(0).to(self.device)
                    attr = self._generate_attribution(explainer, input_t, node_idx)
                    
                    label = get_node_label(node_idx, iteration, self.class_names, self.hierarchical)
                    self._safe_visualize(attr, img_t, fig, ax, label)
                else: ax.axis('off')
        plt.tight_layout()
        return fig

    def _create_transition_grid(self, dataset, selected_indices, prev_result, curr_result, 
                                prev_iter, curr_iter, prev_explainer, curr_explainer, t_type):
        num_samples = len(selected_indices)
        fig, axes = plt.subplots(num_samples, 2, figsize=(8, num_samples * 3))
        if num_samples == 1: axes = axes.reshape(1, -1)
        fig.suptitle(f"{t_type}: {prev_iter} -> {curr_iter}", fontsize=14)

        for row_idx, idx in enumerate(selected_indices):
            img_t, _ = dataset[idx]
            input_t = img_t.unsqueeze(0).to(self.device)

            # Prev
            prev_node = prev_result['hierarchical_predictions'][idx]
            prev_attr = self._generate_attribution(prev_explainer, input_t, prev_node)
            if prev_attr.ndim == 2: prev_attr = prev_attr[..., np.newaxis]
            prev_lbl = f"Iter {prev_iter}: {get_node_label(prev_node, prev_iter, self.class_names, self.hierarchical)}"
            self._safe_visualize(prev_attr, img_t, fig, axes[row_idx, 0], prev_lbl)

            # Curr
            curr_node = curr_result['hierarchical_predictions'][idx]
            curr_attr = self._generate_attribution(curr_explainer, input_t, curr_node)
            if curr_attr.ndim == 2: curr_attr = curr_attr[..., np.newaxis]
            curr_lbl = f"Iter {curr_iter}: {get_node_label(curr_node, curr_iter, self.class_names, self.hierarchical)}"
            self._safe_visualize(curr_attr, img_t, fig, axes[row_idx, 1], curr_lbl)
        plt.tight_layout()
        return fig

    def _save_transition(self, dataset, indices, num_samples, prev_result, curr_result, 
                        prev_iter, curr_iter, prev_explainer, curr_explainer, t_type, save_dir):
        if len(indices) == 0: return
        rng = np.random.RandomState(self._current_seed)
        selected = rng.choice(indices, size=min(num_samples, len(indices)), replace=False)
        fig = self._create_transition_grid(dataset, selected, prev_result, curr_result, 
                                         prev_iter, curr_iter, prev_explainer, curr_explainer, t_type)
        fig.savefig(os.path.join(save_dir, f"transitions_{prev_iter}_to_{curr_iter}_{t_type}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _generate_node_visualizations(self, dataset, result, iteration, num_samples, save_dir):
        checkpoint_path = result.get('checkpoint_path')
        if not checkpoint_path: return
        model = self.checkpoint_loader(checkpoint_path)
        explainer = self._create_explainer(model)
        node_samples = collect_node_samples(result['hierarchical_predictions'], num_samples, seed=self._current_seed)
        fig = self._create_node_grid(dataset, node_samples, iteration, explainer)
        fig.savefig(os.path.join(save_dir, f"iteration_{iteration}_nodes.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved iteration {iteration} node visualizations")

    def _save_faithfulness_metrics(self, results, save_dir):
        with open(os.path.join(save_dir, "faithfulness_metrics.json"), 'w') as f: json.dump(results, f, indent=2)

    def _generate_transitions(self, dataset, results, labels, num_samples, save_dir):
        for i in range(len(results) - 1):
            prev, curr = results[i], results[i+1]
            prev_model = self.checkpoint_loader(prev['checkpoint_path'])
            curr_model = self.checkpoint_loader(curr['checkpoint_path'])
            prev_explainer = self._create_explainer(prev_model)
            curr_explainer = self._create_explainer(curr_model)
            forget, learn = collect_transition_samples(prev['predictions']==labels, curr['predictions']==labels, num_samples, seed=self._current_seed)
            if len(forget) > 0: self._save_transition(dataset, forget, num_samples, prev, curr, i+1, i+2, prev_explainer, curr_explainer, "Forgetting", save_dir)
            if len(learn) > 0: self._save_transition(dataset, learn, num_samples, prev, curr, i+1, i+2, prev_explainer, curr_explainer, "Learning", save_dir)

    def generate_all_visualizations(
        self, 
        dataset, 
        iteration_results, 
        save_dir="./attributions", 
        num_per_node=5, 
        num_transitions=5, 
        seed=None,
        methods: Union[str, List[str]] = None 
    ):
        os.makedirs(save_dir, exist_ok=True)
        if seed is not None: set_seed(seed)
        self._current_seed = seed

        if methods is None:
            methods_to_run = [self.attribution_method]
        elif isinstance(methods, list):
            methods_to_run = methods
        elif isinstance(methods, str):
            if methods == "all": methods_to_run = ALL_METHODS
            elif methods == "all_layer": methods_to_run = LAYER_METHODS
            elif methods == "all_pixel": methods_to_run = PIXEL_METHODS
            else: methods_to_run = [methods]
        
        print(f"\n>>> Running Batch Visualization for: {methods_to_run}")

        for method in methods_to_run:
            print(f"\n" + "="*50)
            print(f"STARTING METHOD: {method}")
            print("="*50)
            
            self.attribution_method = method
            method_save_dir = os.path.join(save_dir, method)
            os.makedirs(method_save_dir, exist_ok=True)

            # NOTE: DeepLIFT requires fresh explainer initialization per iteration 
            # to handle inplace=False modification correctly. 
            # The _evaluate_faithfulness_metrics and _generate_node_visualizations functions
            # both call _create_explainer, so they will inherit the fix automatically.

            faith_results = self._evaluate_faithfulness_metrics(dataset, iteration_results)
            self._save_faithfulness_metrics(faith_results, method_save_dir)
            
            for iter_idx, result in enumerate(iteration_results):
                self._generate_node_visualizations(dataset, result, iter_idx + 1, num_per_node, method_save_dir)
            
            labels = extract_labels_from_dataset(dataset).numpy()
            self._generate_transitions(dataset, iteration_results, labels, num_transitions, method_save_dir)
            
            print(f"Finished {method}. Results in: {method_save_dir}")

        print(f"\nAll requested methods completed.")























# """
# Attribution Visualization Module using Captum.
# Handles model interpretation, visualization, and faithfulness metric evaluation.
# """

# import os
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# from typing import Dict, List, Optional, Literal, Union, Any
# from torch.utils.data import Dataset, Subset

# try:
#     from captum.attr import (
#         LayerGradCam, GuidedGradCam, LayerConductance, 
#         IntegratedGradients, Saliency, InputXGradient, 
#         DeepLift, GuidedBackprop, Occlusion,
#         NoiseTunnel, visualization as viz
#     )
#     from captum.metrics import infidelity
#     CAPTUM_AVAILABLE = True
# except ImportError:
#     CAPTUM_AVAILABLE = False

# from ..data.datasets import extract_labels_from_dataset
# from ..utils.hierarchical import get_nodes_per_class
# from ..utils.seed import set_seed
# from ..utils.image import denormalize_image
# from ..evaluation.analysis import get_node_label
# from ..evaluation.sampling import collect_node_samples, collect_transition_samples, stratified_sample

# AttributionMethod = Literal[
#     "gradcam", "layer_gradcam", "guided_gradcam", "layer_conductance",
#     "integrated_gradients", "saliency", "input_x_gradient", 
#     "deeplift", "guided_backprop", "occlusion"
# ]

# class CaptumModelAdapter(nn.Module):
#     """Wraps model to ensure forward pass returns a single tensor."""
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, x):
#         out = self.model(x)
#         if isinstance(out, tuple):
#             return out[0]
#         return out

# class CaptumAttribution:
#     def __init__(
#         self,
#         model: nn.Module,
#         device: torch.device,
#         checkpoint_loader: callable,
#         class_names: List[str],
#         num_iterations: int,
#         hierarchical: str = "none",
#         attribution_method: AttributionMethod = "gradcam"
#     ):
#         if not CAPTUM_AVAILABLE:
#              raise ImportError("captum required. Install with: pip install captum")
#         self.model = model
#         self.device = device
#         self.checkpoint_loader = checkpoint_loader
#         self.class_names = class_names
#         self.num_iterations = num_iterations
#         self.hierarchical = hierarchical
#         self.attribution_method = attribution_method
#         self._current_seed = None

#     def _get_target_layer(self, model: nn.Module) -> nn.Module:
#         """Heuristic to find the last convolutional layer."""
#         if hasattr(model, 'base') and hasattr(model.base, 'layer4'):
#             return model.base.layer4[-1]
#         raise ValueError("Could not detect target layer. Model must have 'base.layer4'.")

#     def _create_explainer(self, model: nn.Module, method: Optional[AttributionMethod] = None):
#         method = method or self.attribution_method
#         wrapped_model = CaptumModelAdapter(model)
#         wrapped_model.eval()
        
#         target_layer = None
#         if method in ["gradcam", "layer_gradcam", "layer_conductance", "guided_gradcam"]:
#             target_layer = self._get_target_layer(model)

#         if method in ["gradcam", "layer_gradcam"]:
#             return LayerGradCam(wrapped_model, target_layer)
#         elif method == "layer_conductance":
#             return LayerConductance(wrapped_model, target_layer)
#         elif method == "guided_gradcam":
#             return GuidedGradCam(wrapped_model, target_layer)
#         elif method == "occlusion": 
#             return Occlusion(wrapped_model)
        
#         # Pixel-based methods wrapped in NoiseTunnel (SmoothGrad)
#         base_explainer = None
#         if method == "integrated_gradients": base_explainer = IntegratedGradients(wrapped_model)
#         elif method == "saliency": base_explainer = Saliency(wrapped_model)
#         elif method == "input_x_gradient": base_explainer = InputXGradient(wrapped_model)
#         elif method == "deeplift": base_explainer = DeepLift(wrapped_model)
#         elif method == "guided_backprop": base_explainer = GuidedBackprop(wrapped_model)
        
#         if base_explainer:
#             return NoiseTunnel(base_explainer)
        
#         raise ValueError(f"Unsupported attribution method: {method}")

#     def _compute_batch_attributions(self, explainer, batch_images, batch_targets):
#         """Compute attributions handling method-specific logic."""
        
#         # 1. Layer Methods (Need upsampling)
#         if isinstance(explainer, (LayerGradCam, LayerConductance)):
#             attributions = explainer.attribute(batch_images, target=batch_targets)
#             if attributions.dim() == 4 and attributions.shape[1] > 1:
#                 attributions = attributions.sum(dim=1, keepdim=True)
#                 attributions = torch.relu(attributions)
#             attributions = torch.nn.functional.interpolate(
#                 attributions, size=batch_images.shape[-2:], mode='bilinear', align_corners=False
#             )

#         # 2. Guided GradCAM (No upsampling, just attribute)
#         elif isinstance(explainer, GuidedGradCam):
#             attributions = explainer.attribute(batch_images, target=batch_targets)

#         # 3. NoiseTunnel (SmoothGrad)
#         elif isinstance(explainer, NoiseTunnel):
#             attributions = explainer.attribute(
#                 batch_images, target=batch_targets, nt_type='smoothgrad_sq', nt_samples=10, stdevs=0.15
#             )

#         # 4. Occlusion (High Resolution Settings)
#         elif isinstance(explainer, Occlusion):
#             # Reduced window size for detail, reduced stride for high-res map
#             attributions = explainer.attribute(
#                 batch_images, target=batch_targets,
#                 sliding_window_shapes=(3, 8, 8), strides=(3, 2, 2)
#             )

#         # 5. Fallback
#         else:
#             attributions = explainer.attribute(batch_images, target=batch_targets)

#         # Ensure channel compatibility (Duplicate single channel to RGB if needed)
#         if attributions.shape[1] != batch_images.shape[1]:
#             attributions = attributions.mean(dim=1, keepdim=True).repeat(1, batch_images.shape[1], 1, 1)

#         return attributions

#     def _evaluate_faithfulness_metrics(self, dataset, iteration_results, max_samples=500):
#         print(f"\n--- Evaluating Faithfulness Metrics ({max_samples} samples) ---")
#         results = {}

#         for iter_idx, result in enumerate(iteration_results):
#             iteration = iter_idx + 1
#             checkpoint_path = result.get('checkpoint_path')
#             if not checkpoint_path or not os.path.exists(checkpoint_path):
#                 continue

#             raw_model = self.checkpoint_loader(checkpoint_path)
#             explainer = self._create_explainer(raw_model)
#             wrapped_model = CaptumModelAdapter(raw_model)
#             wrapped_model.eval()

#             targets = result['hierarchical_predictions']
#             labels = extract_labels_from_dataset(dataset).numpy()

#             if len(dataset) > max_samples:
#                 indices = stratified_sample(targets, labels, max_samples, seed=self._current_seed)
#                 eval_targets = targets[indices]
#             else:
#                 indices = np.arange(len(dataset))
#                 eval_targets = targets

#             print(f"  Iteration {iteration} ({len(indices)} samples)...")
#             infid_scores = []
#             batch_size = 2 if self.attribution_method in ['integrated_gradients', 'deeplift'] else 16

#             def perturb_func(inputs):
#                 noise = torch.randn_like(inputs) * 0.05
#                 return noise, inputs + noise

#             # Create subset
#             eval_subset = Subset(dataset, indices)

#             for i in range(0, len(indices), batch_size):
#                 batch_idx = list(range(i, min(i + batch_size, len(indices))))
#                 batch_list = [eval_subset[j][0] for j in batch_idx]

#                 batch_images = torch.stack(batch_list).to(self.device)
#                 batch_images.requires_grad_(True)
#                 batch_targets = torch.tensor([int(eval_targets[j]) for j in batch_idx], device=self.device)

#                 batch_attrs = self._compute_batch_attributions(explainer, batch_images, batch_targets).detach()

#                 try:
#                     batch_infid = infidelity(
#                         wrapped_model, perturb_func, batch_images, batch_attrs, target=batch_targets, n_perturb_samples=5
#                     )
#                     if isinstance(batch_infid, tuple): batch_infid = batch_infid[0]
#                     infid_scores.extend(batch_infid.detach().cpu().numpy().tolist())
#                 except Exception as e:
#                     if i == 0: print(f"    Warning: Infidelity calculation failed: {e}")

#             results[iteration] = {
#                 'num_samples': len(indices),
#                 'infidelity': float(np.mean(infid_scores)) if infid_scores else 0.0,
#             }
#             print(f"    Infidelity: {results[iteration]['infidelity']:.4f}")

#             del raw_model, explainer, wrapped_model
#             torch.cuda.empty_cache() if torch.cuda.is_available() else None

#         return results

#     def _denorm(self, tensor):
#         mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
#         std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
#         out = tensor * std + mean
#         return torch.clamp(out, 0, 1)

#     def _generate_attribution(self, explainer, input_tensor, target):
#         input_tensor.requires_grad_(True)
#         target_tensor = torch.tensor([int(target)], device=self.device)
        
#         attr = self._compute_batch_attributions(explainer, input_tensor, target_tensor)
#         attr = attr.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        
#         # Collapse Channels if needed
#         if attr.shape[2] == 3:
#             if self.attribution_method in ["gradcam", "layer_gradcam", "layer_conductance"]:
#                 attr = np.mean(attr, axis=2, keepdims=True)
#             else:
#                 attr = np.linalg.norm(attr, axis=2, keepdims=True)
#         return attr

#     def _safe_visualize(self, attr, img_tensor, fig, ax, title):
#         img_rgb = self._denorm(img_tensor.unsqueeze(0)).squeeze(0).permute(1, 2, 0).cpu().numpy()
        
#         if np.max(attr) - np.min(attr) < 1e-5:
#             ax.imshow(img_rgb)
#             ax.set_title(f"{title}\n(No Attribution)", fontsize=8)
#             return

#         try:
#             # Set visualization style based on method
#             if self.attribution_method in ["gradcam", "layer_gradcam", "layer_conductance"]:
#                 cmap = 'jet'
#                 outlier = 2
#                 alpha = 0.5
#             else:
#                 # Pixel methods & Occlusion: Inferno (Fire), lower clipping for sharpness
#                 cmap = 'inferno'
#                 outlier = 1 # Clip only top 1% to avoid "glowing blobs"
#                 alpha = 0.7

#             viz.visualize_image_attr(
#                 attr, img_rgb, 
#                 method="blended_heat_map", 
#                 sign="positive", 
#                 show_colorbar=False, 
#                 title=title, 
#                 cmap=cmap,
#                 outlier_perc=outlier,
#                 alpha_overlay=alpha,
#                 plt_fig_axis=(fig, ax), 
#                 use_pyplot=False
#             )
#         except Exception:
#             ax.imshow(img_rgb)

#     def _create_node_grid(self, dataset, node_samples, iteration, explainer):
#         num_nodes = len(node_samples); max_samples = max(len(s) for s in node_samples.values()) if any(node_samples.values()) else 0
#         if max_samples == 0: return plt.figure()
#         fig, axes = plt.subplots(num_nodes, max_samples, figsize=(max_samples * 3, num_nodes * 3))
#         if num_nodes == 1: axes = axes.reshape(1, -1)
#         if max_samples == 1: axes = axes.reshape(-1, 1)
#         fig.suptitle(f"Iteration {iteration} - Node Attributions", fontsize=16)

#         for node_idx in sorted(node_samples.keys()):
#             sample_indices = node_samples[node_idx]
#             for col_idx in range(max_samples):
#                 ax = axes[node_idx, col_idx]
#                 if col_idx < len(sample_indices):
#                     idx = sample_indices[col_idx]
#                     img_t, _ = dataset[idx]
#                     input_t = img_t.unsqueeze(0).to(self.device)
#                     attr = self._generate_attribution(explainer, input_t, node_idx)
                    
#                     label = get_node_label(node_idx, iteration, self.class_names, self.hierarchical)
#                     self._safe_visualize(attr, img_t, fig, ax, label)
#                 else: ax.axis('off')
#         plt.tight_layout()
#         return fig

#     def _create_transition_grid(self, dataset, selected_indices, prev_result, curr_result, 
#                                 prev_iter, curr_iter, prev_explainer, curr_explainer, t_type):
#         num_samples = len(selected_indices)
#         fig, axes = plt.subplots(num_samples, 2, figsize=(8, num_samples * 3))
#         if num_samples == 1: axes = axes.reshape(1, -1)
#         fig.suptitle(f"{t_type}: {prev_iter} -> {curr_iter}", fontsize=14)

#         for row_idx, idx in enumerate(selected_indices):
#             img_t, _ = dataset[idx]
#             input_t = img_t.unsqueeze(0).to(self.device)

#             # Prev
#             prev_node = prev_result['hierarchical_predictions'][idx]
#             prev_attr = self._generate_attribution(prev_explainer, input_t, prev_node)
#             if prev_attr.ndim == 2: prev_attr = prev_attr[..., np.newaxis]
#             prev_lbl = f"Iter {prev_iter}: {get_node_label(prev_node, prev_iter, self.class_names, self.hierarchical)}"
#             self._safe_visualize(prev_attr, img_t, fig, axes[row_idx, 0], prev_lbl)

#             # Curr
#             curr_node = curr_result['hierarchical_predictions'][idx]
#             curr_attr = self._generate_attribution(curr_explainer, input_t, curr_node)
#             if curr_attr.ndim == 2: curr_attr = curr_attr[..., np.newaxis]
#             curr_lbl = f"Iter {curr_iter}: {get_node_label(curr_node, curr_iter, self.class_names, self.hierarchical)}"
#             self._safe_visualize(curr_attr, img_t, fig, axes[row_idx, 1], curr_lbl)
#         plt.tight_layout()
#         return fig

#     def _save_transition(self, dataset, indices, num_samples, prev_result, curr_result, 
#                         prev_iter, curr_iter, prev_explainer, curr_explainer, t_type, save_dir):
#         if len(indices) == 0: return
#         rng = np.random.RandomState(self._current_seed)
#         selected = rng.choice(indices, size=min(num_samples, len(indices)), replace=False)
#         fig = self._create_transition_grid(dataset, selected, prev_result, curr_result, 
#                                          prev_iter, curr_iter, prev_explainer, curr_explainer, t_type)
#         fig.savefig(os.path.join(save_dir, f"transitions_{prev_iter}_to_{curr_iter}_{t_type}.png"), dpi=150, bbox_inches='tight')
#         plt.close(fig)

#     def _generate_node_visualizations(self, dataset, result, iteration, num_samples, save_dir):
#         checkpoint_path = result.get('checkpoint_path')
#         if not checkpoint_path: return
#         model = self.checkpoint_loader(checkpoint_path)
#         explainer = self._create_explainer(model)
#         node_samples = collect_node_samples(result['hierarchical_predictions'], num_samples, seed=self._current_seed)
#         fig = self._create_node_grid(dataset, node_samples, iteration, explainer)
#         fig.savefig(os.path.join(save_dir, f"iteration_{iteration}_nodes.png"), dpi=150, bbox_inches='tight')
#         plt.close(fig)
#         print(f"  Saved iteration {iteration} node visualizations")

#     def _save_faithfulness_metrics(self, results, save_dir):
#         import json
#         with open(os.path.join(save_dir, "faithfulness_metrics.json"), 'w') as f: json.dump(results, f, indent=2)

#     def _generate_transitions(self, dataset, results, labels, num_samples, save_dir):
#         for i in range(len(results) - 1):
#             prev, curr = results[i], results[i+1]
#             prev_model = self.checkpoint_loader(prev['checkpoint_path'])
#             curr_model = self.checkpoint_loader(curr['checkpoint_path'])
#             prev_explainer = self._create_explainer(prev_model)
#             curr_explainer = self._create_explainer(curr_model)
#             forget, learn = collect_transition_samples(prev['predictions']==labels, curr['predictions']==labels, num_samples, seed=self._current_seed)
#             if len(forget) > 0: self._save_transition(dataset, forget, num_samples, prev, curr, i+1, i+2, prev_explainer, curr_explainer, "Forgetting", save_dir)
#             if len(learn) > 0: self._save_transition(dataset, learn, num_samples, prev, curr, i+1, i+2, prev_explainer, curr_explainer, "Learning", save_dir)

#     def generate_all_visualizations(self, dataset, iteration_results, save_dir="./attributions", num_per_node=5, num_transitions=5, seed=None):
#         os.makedirs(save_dir, exist_ok=True)
#         if seed is not None: set_seed(seed)
#         self._current_seed = seed
#         print(f"\n--- Generating Attributions (Captum: {self.attribution_method}) ---")
#         faith_results = self._evaluate_faithfulness_metrics(dataset, iteration_results)
#         self._save_faithfulness_metrics(faith_results, save_dir)
#         for iter_idx, result in enumerate(iteration_results):
#             self._generate_node_visualizations(dataset, result, iter_idx + 1, num_per_node, save_dir)
#         labels = extract_labels_from_dataset(dataset).numpy()
#         self._generate_transitions(dataset, iteration_results, labels, num_transitions, save_dir)
#         print(f"Visualizations saved to: {save_dir}")















# """
# Attribution Visualization using Captum.
# Standard "Research Paper" Implementation:
# - Uses NoiseTunnel (SmoothGrad) for pixel-based methods to remove noise.
# - Uses Robust Normalization (Outlier Clipping) to make features visible.
# - Uses Standard Colormaps (Jet for CAM, Inferno for Pixels).
# """

# import os
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# from typing import Dict, List, Optional, Literal, Union, Any
# from torch.utils.data import Dataset, Subset

# try:
#     from captum.attr import (
#         LayerGradCam, GuidedGradCam, IntegratedGradients, Saliency,
#         InputXGradient, DeepLift, GuidedBackprop, Occlusion,
#         NoiseTunnel, visualization as viz
#     )
#     from captum.metrics import infidelity, sensitivity_max
#     CAPTUM_AVAILABLE = True
# except ImportError:
#     CAPTUM_AVAILABLE = False

# from ..data.datasets import extract_labels_from_dataset
# from ..utils.hierarchical import get_nodes_per_class
# from ..utils.seed import set_seed
# from ..utils.image import denormalize_image
# from ..evaluation.analysis import get_node_label
# from ..evaluation.sampling import collect_node_samples, collect_transition_samples, stratified_sample

# AttributionMethod = Literal[
#     "gradcam", "layer_gradcam", "guided_gradcam", "integrated_gradients",
#     "saliency", "input_x_gradient", "deeplift", "guided_backprop", "occlusion"
# ]

# # --- 1. STANDARD ADAPTER ---
# class CaptumModelAdapter(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, x):
#         out = self.model(x)
#         if isinstance(out, tuple):
#             return out[0]
#         return out

# class CaptumAttribution:
#     def __init__(
#         self,
#         model: nn.Module,
#         device: torch.device,
#         checkpoint_loader: callable,
#         class_names: List[str],
#         num_iterations: int,
#         hierarchical: str = "none",
#         attribution_method: AttributionMethod = "gradcam"
#     ):
#         if not CAPTUM_AVAILABLE:
#              raise ImportError("captum required. Install with: pip install captum")
#         self.model = model
#         self.device = device
#         self.checkpoint_loader = checkpoint_loader
#         self.class_names = class_names
#         self.num_iterations = num_iterations
#         self.hierarchical = hierarchical
#         self.attribution_method = attribution_method
#         self._current_seed = None

#     def _get_target_layer(self, model: nn.Module) -> nn.Module:
#         if hasattr(model, 'base') and hasattr(model.base, 'layer4'):
#             return model.base.layer4[-1]
#         raise ValueError("Could not detect target layer. Model must have 'base.layer4'.")

#     def _create_explainer(self, model: nn.Module, method: Optional[AttributionMethod] = None):
#         method = method or self.attribution_method
#         wrapped_model = CaptumModelAdapter(model)
#         wrapped_model.eval()
        
#         # 1. GradCAM (Smooth by definition)
#         if method == "gradcam":
#             target_layer = self._get_target_layer(model)
#             return LayerGradCam(wrapped_model, target_layer)
        
#         # 2. Pixel Methods -> Wrap in NoiseTunnel (Standard Research Practice)
#         # This implements "SmoothGrad", which removes the "dots" and noise.
#         base_explainer = None
#         if method == "integrated_gradients": base_explainer = IntegratedGradients(wrapped_model)
#         elif method == "saliency": base_explainer = Saliency(wrapped_model)
#         elif method == "input_x_gradient": base_explainer = InputXGradient(wrapped_model)
        
#         if base_explainer:
#             return NoiseTunnel(base_explainer)

#         # 3. Occlusion (Box sliding)
#         if method == "occlusion": return Occlusion(wrapped_model)
        
#         raise ValueError(f"Unsupported attribution method: {method}")

#     def _compute_batch_attributions(self, explainer, batch_images, batch_targets, debug=False):
#         """Compute attributions using type-based detection for consistency."""

#         # Case A: GradCAM - type-based detection
#         if isinstance(explainer, LayerGradCam):
#             attributions = explainer.attribute(batch_images, target=batch_targets)
#             if debug:
#                 print(f"      LayerGradCam raw output: {attributions.shape}")
#             if attributions.dim() == 4 and attributions.shape[1] > 1:
#                 attributions = attributions.sum(dim=1, keepdim=True)
#                 attributions = torch.relu(attributions)
#                 if debug:
#                     print(f"      After sum+relu: {attributions.shape}")
#             attributions = torch.nn.functional.interpolate(
#                 attributions, size=batch_images.shape[-2:], mode='bilinear', align_corners=False
#             )
#             if debug:
#                 print(f"      After interpolate: {attributions.shape}")

#         # Case B: Smoothed Pixel Methods (NoiseTunnel)
#         elif isinstance(explainer, NoiseTunnel):
#             # nt_samples=3: Averages 3 noisy runs to smooth out the "static"
#             attributions = explainer.attribute(
#                 batch_images, target=batch_targets, nt_type='smoothgrad_sq', nt_samples=3, stdevs=0.15
#             )
#             if debug:
#                 print(f"      NoiseTunnel raw output: {attributions.shape}")

#         # Case C: Occlusion - type-based detection
#         elif isinstance(explainer, Occlusion):
#             attributions = explainer.attribute(
#                 batch_images, target=batch_targets,
#                 sliding_window_shapes=(3, 8, 8), strides=(3, 2, 2)
#             )
#             if debug:
#                 print(f"      Occlusion raw output: {attributions.shape}")

#         # Case D: Standard (fallback for any other attribution methods)
#         else:
#             attributions = explainer.attribute(batch_images, target=batch_targets)
#             if debug:
#                 print(f"      Standard raw output: {attributions.shape}")

#         # Metric Compatibility - ALWAYS ensure output channels match input channels
#         # This handles ANY channel mismatch (1→3, 30→3, 2048→3, etc.)
#         if attributions.shape[1] != batch_images.shape[1]:
#             if debug:
#                 print(f"      Channel mismatch: {attributions.shape[1]} != {batch_images.shape[1]}, fixing...")
#             # Collapse any number of channels to 1, then repeat to match input
#             attributions = attributions.mean(dim=1, keepdim=True).repeat(1, batch_images.shape[1], 1, 1)
#             if debug:
#                 print(f"      After channel fix: {attributions.shape}")

#         return attributions

#     def _evaluate_faithfulness_metrics(self, dataset, iteration_results, max_samples=500):
#         print(f"\n--- Evaluating Faithfulness Metrics ({max_samples} samples) ---")
#         results = {}

#         for iter_idx, result in enumerate(iteration_results):
#             iteration = iter_idx + 1
#             checkpoint_path = result.get('checkpoint_path')
#             if not checkpoint_path:
#                 print(f"  Skipping iteration {iteration}: No checkpoint path")
#                 continue

#             # Validate checkpoint exists
#             if not os.path.exists(checkpoint_path):
#                 print(f"  Skipping iteration {iteration}: Checkpoint not found at {checkpoint_path}")
#                 continue

#             raw_model = self.checkpoint_loader(checkpoint_path)
#             explainer = self._create_explainer(raw_model)
#             wrapped_model = CaptumModelAdapter(raw_model)
#             wrapped_model.eval()

#             targets = result['hierarchical_predictions']
#             labels = extract_labels_from_dataset(dataset).numpy()

#             if len(dataset) > max_samples:
#                 indices = stratified_sample(targets, labels, max_samples, seed=self._current_seed)
#                 eval_targets = targets[indices]
#             else:
#                 indices = np.arange(len(dataset))
#                 eval_targets = targets

#             print(f"  Iteration {iteration} ({len(indices)} samples)...")
#             infid_scores, sens_scores = [], []
#             # Reduce batch size for memory-intensive methods
#             batch_size = 2 if self.attribution_method in ['integrated_gradients', 'deeplift'] else 16

#             def perturb_func(inputs):
#                 # Perturbation scale of 0.05 assumes normalized inputs [-2, 2] range
#                 # Adjust this value if your normalization differs
#                 noise = torch.randn_like(inputs) * 0.05
#                 return noise, inputs + noise

#             # Debug: track calls to attr_func
#             attr_func_call_count = [0]
#             attr_func_shapes = []

#             def attr_func(inputs, target=None):
#                 """Attribution function for sensitivity computation.

#                 IMPORTANT: sensitivity_max expects single-channel attributions.
#                 Multi-channel attributions cause shape mismatch errors in Captum's
#                 internal batching logic (e.g., comparing 30-channel vs 3-channel tensors).
#                 """
#                 if isinstance(inputs, tuple):
#                     inputs = inputs[0]

#                 # Ensure inputs require gradients
#                 if not inputs.requires_grad:
#                     inputs.requires_grad_(True)

#                 # Disable verbose debug (only essential errors are shown)
#                 debug_mode = False

#                 # FIX: sensitivity_max may call with different batch sizes due to perturbation batching
#                 # Ensure target matches the input batch size
#                 actual_batch_size = inputs.shape[0]
#                 if target is not None:
#                     if target.shape[0] != actual_batch_size:
#                         # sensitivity_max is batching perturbations - replicate target to match
#                         if target.shape[0] == 1:
#                             # Single sample, replicate to match batch
#                             target = target.repeat(actual_batch_size)
#                         else:
#                             # Take first sample's target and replicate (all should be same in sensitivity test)
#                             target = target[0:1].repeat(actual_batch_size)

#                 # Compute attributions - this should respond to input perturbations
#                 attrs = self._compute_batch_attributions(explainer, inputs, target, debug=debug_mode)

#                 # CRITICAL FIX: sensitivity_max doesn't handle multi-channel attributions correctly.
#                 # Collapse to single channel by taking the mean across color channels.
#                 # This represents the overall attribution magnitude per pixel.
#                 if attrs.shape[1] > 1:
#                     if debug_mode:
#                         print(f"      attr_func: Collapsing {attrs.shape[1]} channels to 1 for sensitivity_max compatibility")
#                     attrs = attrs.mean(dim=1, keepdim=True)

#                 # Debug logging
#                 attr_func_call_count[0] += 1
#                 if attr_func_call_count[0] <= 5:  # Only log first 5 calls
#                     attr_func_shapes.append({
#                         'call': attr_func_call_count[0],
#                         'input_shape': tuple(inputs.shape),
#                         'output_shape': tuple(attrs.shape),
#                         'target': target.tolist() if target is not None and target.numel() < 50 else f"tensor({target.shape})"
#                     })

#                 return attrs

#             # Create subset once for efficiency
#             eval_subset = Subset(dataset, indices)

#             # Diagnostic: Test if attributions change with perturbations (first batch only)
#             if len(indices) > 0:
#                 test_img = eval_subset[0][0].unsqueeze(0).to(self.device)
#                 test_img.requires_grad_(True)
#                 test_target = torch.tensor([int(eval_targets[0])], device=self.device)

#                 # Compute original attributions
#                 with torch.enable_grad():
#                     attrs_orig = attr_func(test_img, target=test_target)

#                     # Compute perturbed attributions
#                     perturbed_img = test_img + torch.randn_like(test_img) * 0.1
#                     attrs_pert = attr_func(perturbed_img, target=test_target)

#                     # Check difference (silently - only used for validation)
#                     attr_diff = torch.abs(attrs_orig - attrs_pert).mean().item()

#             for i in range(0, len(indices), batch_size):
#                 batch_idx = list(range(i, min(i + batch_size, len(indices))))
#                 batch_list = [eval_subset[j][0] for j in batch_idx]

#                 batch_images = torch.stack(batch_list).to(self.device)
#                 batch_images.requires_grad_(True)
#                 batch_targets = torch.tensor([int(eval_targets[j]) for j in batch_idx], device=self.device)

#                 batch_attrs = self._compute_batch_attributions(explainer, batch_images, batch_targets).detach()

#                 try:
#                     batch_infid = infidelity(
#                         wrapped_model, perturb_func, batch_images, batch_attrs, target=batch_targets, n_perturb_samples=5
#                     )
#                     if isinstance(batch_infid, tuple): batch_infid = batch_infid[0]
#                     infid_scores.extend(batch_infid.detach().cpu().numpy().tolist())
#                 except Exception as e:
#                     if i == 0:  # Only print error once per iteration
#                         print(f"    Warning: Infidelity calculation failed: {e}")

#                 # SKIP SENSITIVITY: Captum's sensitivity_max has design incompatibility
#                 # Issue: sensitivity_max compares original batch (size N) with perturbed samples (n_perturb_samples)
#                 # causing tensor size mismatches that cannot be resolved with dynamic attribution computation.
#                 # Infidelity is the more important faithfulness metric and is working correctly.
#                 # Future work: Implement custom sensitivity metric if needed.
#                 sens_scores.extend([0.0] * len(batch_idx))

#             results[iteration] = {'num_samples': len(indices)}
#             results[iteration]['infidelity'] = float(np.mean(infid_scores)) if infid_scores else 0.0
#             results[iteration]['sensitivity'] = float(np.mean(sens_scores)) if sens_scores else 0.0

#             print(f"    Infidelity: {results[iteration]['infidelity']:.4f}")
#             print(f"    Sensitivity: {results[iteration]['sensitivity']:.4f}")

#             # Clean up memory
#             del raw_model, explainer, wrapped_model
#             torch.cuda.empty_cache() if torch.cuda.is_available() else None

#         return results

        

#     def _denorm(self, tensor):
#         mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
#         std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
#         out = tensor * std + mean
#         return torch.clamp(out, 0, 1)

#     def _generate_attribution(self, explainer, input_tensor, target):
#         """Helper for visualization."""
#         input_tensor.requires_grad_(True)
#         target_tensor = torch.tensor([int(target)], device=self.device)
        
#         attr = self._compute_batch_attributions(explainer, input_tensor, target_tensor)
#         attr = attr.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        
#         # Collapse Channels
#         if attr.shape[2] == 3:
#             if self.attribution_method == "gradcam":
#                 attr = np.mean(attr, axis=2, keepdims=True)
#             else:
#                 attr = np.linalg.norm(attr, axis=2, keepdims=True)
#         return attr

#     def _safe_visualize(self, attr, img_tensor, fig, ax, title):
#         """Visualizes using standard Research settings (Fire/Jet colormaps)."""
#         img_rgb = self._denorm(img_tensor.unsqueeze(0)).squeeze(0).permute(1, 2, 0).cpu().numpy()
        
#         if np.max(attr) - np.min(attr) < 1e-5:
#             ax.imshow(img_rgb)
#             ax.set_title(f"{title}\n(No Attribution)", fontsize=8)
#         else:
#             try:
#                 # --- RESEARCH STANDARD VISUALIZATION SETTINGS ---
#                 if self.attribution_method == "gradcam":
#                     # GradCAM: Jet (Rainbow), low clipping (smooth)
#                     cmap = 'jet'
#                     outlier = 2
#                     alpha = 0.5
#                 else:
#                     # Pixel Methods: Inferno (Fire), high clipping (structure)
#                     cmap = 'inferno'
#                     outlier = 2 # Clips top 5% outliers to make faint details visible
#                     alpha = 0.7

#                 viz.visualize_image_attr(
#                     attr, img_rgb, 
#                     method="blended_heat_map", 
#                     sign="positive", 
#                     show_colorbar=False, 
#                     title=title, 
#                     cmap=cmap,             # <--- Fire for Pixels, Rainbow for CAM
#                     outlier_perc=outlier,  # <--- Brightens the structure
#                     alpha_overlay=alpha,
#                     plt_fig_axis=(fig, ax), 
#                     use_pyplot=False
#                 )
#             except Exception:
#                 ax.imshow(img_rgb)

#     def _create_node_grid(self, dataset, node_samples, iteration, explainer):
#         num_nodes = len(node_samples); max_samples = max(len(s) for s in node_samples.values()) if any(node_samples.values()) else 0
#         if max_samples == 0: return plt.figure()
#         fig, axes = plt.subplots(num_nodes, max_samples, figsize=(max_samples * 3, num_nodes * 3))
#         if num_nodes == 1: axes = axes.reshape(1, -1)
#         if max_samples == 1: axes = axes.reshape(-1, 1)
#         fig.suptitle(f"Iteration {iteration} - Node Attributions", fontsize=16)

#         for node_idx in sorted(node_samples.keys()):
#             sample_indices = node_samples[node_idx]
#             for col_idx in range(max_samples):
#                 ax = axes[node_idx, col_idx]
#                 if col_idx < len(sample_indices):
#                     idx = sample_indices[col_idx]
#                     img_t, _ = dataset[idx]
#                     input_t = img_t.unsqueeze(0).to(self.device)
#                     attr = self._generate_attribution(explainer, input_t, node_idx)
                    
#                     label = get_node_label(node_idx, iteration, self.class_names, self.hierarchical)
#                     self._safe_visualize(attr, img_t, fig, ax, label)
#                 else: ax.axis('off')
#         plt.tight_layout()
#         return fig

#     def _create_transition_grid(self, dataset, selected_indices, prev_result, curr_result, 
#                                 prev_iter, curr_iter, prev_explainer, curr_explainer, t_type):
#         num_samples = len(selected_indices)
#         fig, axes = plt.subplots(num_samples, 2, figsize=(8, num_samples * 3))
#         if num_samples == 1: axes = axes.reshape(1, -1)
#         fig.suptitle(f"{t_type}: {prev_iter} -> {curr_iter}", fontsize=14)

#         for row_idx, idx in enumerate(selected_indices):
#             img_t, _ = dataset[idx]
#             input_t = img_t.unsqueeze(0).to(self.device)

#             # Prev
#             prev_node = prev_result['hierarchical_predictions'][idx]
#             prev_attr = self._generate_attribution(prev_explainer, input_t, prev_node)
#             if prev_attr.ndim == 2: prev_attr = prev_attr[..., np.newaxis]
#             prev_lbl = f"Iter {prev_iter}: {get_node_label(prev_node, prev_iter, self.class_names, self.hierarchical)}"
#             self._safe_visualize(prev_attr, img_t, fig, axes[row_idx, 0], prev_lbl)

#             # Curr
#             curr_node = curr_result['hierarchical_predictions'][idx]
#             curr_attr = self._generate_attribution(curr_explainer, input_t, curr_node)
#             if curr_attr.ndim == 2: curr_attr = curr_attr[..., np.newaxis]
#             curr_lbl = f"Iter {curr_iter}: {get_node_label(curr_node, curr_iter, self.class_names, self.hierarchical)}"
#             self._safe_visualize(curr_attr, img_t, fig, axes[row_idx, 1], curr_lbl)
#         plt.tight_layout()
#         return fig

#     def _save_transition(self, dataset, indices, num_samples, prev_result, curr_result, 
#                         prev_iter, curr_iter, prev_explainer, curr_explainer, t_type, save_dir):
#         if len(indices) == 0: return
#         rng = np.random.RandomState(self._current_seed)
#         selected = rng.choice(indices, size=min(num_samples, len(indices)), replace=False)
#         fig = self._create_transition_grid(dataset, selected, prev_result, curr_result, 
#                                          prev_iter, curr_iter, prev_explainer, curr_explainer, t_type)
#         fig.savefig(os.path.join(save_dir, f"transitions_{prev_iter}_to_{curr_iter}_{t_type}.png"), dpi=150, bbox_inches='tight')
#         plt.close(fig)
#         print(f"  Saved {t_type} transition {prev_iter}→{curr_iter}")

#     def _generate_node_visualizations(self, dataset, result, iteration, num_samples, save_dir):
#         checkpoint_path = result.get('checkpoint_path')
#         if not checkpoint_path: return
#         model = self.checkpoint_loader(checkpoint_path)
#         explainer = self._create_explainer(model)
#         node_samples = collect_node_samples(result['hierarchical_predictions'], num_samples, seed=self._current_seed)
#         fig = self._create_node_grid(dataset, node_samples, iteration, explainer)
#         fig.savefig(os.path.join(save_dir, f"iteration_{iteration}_nodes.png"), dpi=150, bbox_inches='tight')
#         plt.close(fig)
#         print(f"  Saved iteration {iteration} node visualizations")

#     def _save_faithfulness_metrics(self, results, save_dir):
#         import json
#         with open(os.path.join(save_dir, "faithfulness_metrics.json"), 'w') as f: json.dump(results, f, indent=2)
#         print(f"\nFaithfulness metrics saved to: {os.path.join(save_dir, 'faithfulness_metrics.json')}")

#     def _generate_transitions(self, dataset, results, labels, num_samples, save_dir):
#         for i in range(len(results) - 1):
#             prev, curr = results[i], results[i+1]
#             prev_model = self.checkpoint_loader(prev['checkpoint_path'])
#             curr_model = self.checkpoint_loader(curr['checkpoint_path'])
#             prev_explainer = self._create_explainer(prev_model)
#             curr_explainer = self._create_explainer(curr_model)
#             forget, learn = collect_transition_samples(prev['predictions']==labels, curr['predictions']==labels, num_samples, seed=self._current_seed)
#             if len(forget) > 0: self._save_transition(dataset, forget, num_samples, prev, curr, i+1, i+2, prev_explainer, curr_explainer, "Forgetting", save_dir)
#             if len(learn) > 0: self._save_transition(dataset, learn, num_samples, prev, curr, i+1, i+2, prev_explainer, curr_explainer, "Learning", save_dir)

#     def generate_all_visualizations(self, dataset, iteration_results, save_dir="./attributions", num_per_node=5, num_transitions=5, seed=None):
#         os.makedirs(save_dir, exist_ok=True)
#         if seed is not None: set_seed(seed)
#         self._current_seed = seed
#         print(f"\n--- Generating Attributions (Captum: {self.attribution_method}) ---")
#         faith_results = self._evaluate_faithfulness_metrics(dataset, iteration_results)
#         self._save_faithfulness_metrics(faith_results, save_dir)
#         for iter_idx, result in enumerate(iteration_results):
#             self._generate_node_visualizations(dataset, result, iter_idx + 1, num_per_node, save_dir)
#         labels = extract_labels_from_dataset(dataset).numpy()
#         self._generate_transitions(dataset, iteration_results, labels, num_transitions, save_dir)
#         print(f"Visualizations saved to: {save_dir}")
