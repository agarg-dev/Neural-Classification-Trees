"""Captum-based attribution visualizations and faithfulness metrics."""

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

    def _safe_load_checkpoint(self, checkpoint_path: str, context: str = "") -> Optional[nn.Module]:
        """Load a checkpoint, returning None (with a warning) if missing/corrupted."""
        if not checkpoint_path:
            print(f"  Warning: No checkpoint path provided{' for ' + context if context else ''}, skipping.")
            return None
        if not os.path.exists(checkpoint_path):
            print(f"  Warning: Checkpoint not found{' for ' + context if context else ''}: {checkpoint_path}")
            return None
        try:
            return self.checkpoint_loader(checkpoint_path)
        except (RuntimeError, AttributeError, OSError) as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ['corrupted', 'invalid header', 'nonetype', 'seek']):
                print(f"  Warning: Checkpoint corrupted{' for ' + context if context else ''}: {checkpoint_path}")
                return None
            raise  # Re-raise unexpected errors

    def _get_target_layer(self, model: nn.Module) -> nn.Module:
        """Heuristic to find the last convolutional layer."""
        # ResNet backbone
        if hasattr(model, 'base') and hasattr(model.base, 'layer4'):
            return model.base.layer4[-1]
        # LeNet5 backbone - use conv2 as the last conv layer
        if hasattr(model, 'base') and hasattr(model.base, 'conv2'):
            return model.base.conv2
        # LeNet4 backbone - last Conv2d inside the Sequential convnet
        if hasattr(model, 'base') and hasattr(model.base, 'convnet'):
            last_conv = None
            for m in model.base.convnet:
                if isinstance(m, nn.Conv2d):
                    last_conv = m
            if last_conv is not None:
                return last_conv
        raise ValueError(
            "Could not detect target layer. Model must have 'base.layer4' (ResNet), "
            "'base.conv2' (LeNet5), or 'base.convnet' (LeNet4)."
        )

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

            raw_model = self._safe_load_checkpoint(checkpoint_path, f"iteration {iteration}")
            if raw_model is None:
                continue

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
            # Only set title if provided
            if title:
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
                title=title if title else None,
                cmap=cmap,
                outlier_perc=outlier,
                alpha_overlay=alpha,
                plt_fig_axis=(fig, ax),
                use_pyplot=False
            )
        except Exception:
            ax.imshow(img_rgb)

    def _create_node_grid(self, dataset, node_samples, iteration, explainer):
        # Filter out empty nodes - only show nodes that have samples
        non_empty_nodes = {k: v for k, v in node_samples.items() if len(v) > 0}
        if not non_empty_nodes:
            return plt.figure()

        num_rows = len(non_empty_nodes)
        max_samples = max(len(s) for s in non_empty_nodes.values())

        # Create figure with extra left margin for row labels
        fig, axes = plt.subplots(
            num_rows, max_samples,
            figsize=(max_samples * 3, num_rows * 3),
            squeeze=False
        )

        # Add title with extra top margin to prevent overlap
        fig.suptitle(f"Iteration {iteration} - Node Attributions", fontsize=16, y=0.98)

        for row_idx, (node_idx, sample_indices) in enumerate(sorted(non_empty_nodes.items())):
            # Get row label once (shared for all images in the row)
            row_label = get_node_label(node_idx, iteration, self.class_names, self.hierarchical)

            for col_idx in range(max_samples):
                ax = axes[row_idx, col_idx]
                if col_idx < len(sample_indices):
                    idx = sample_indices[col_idx]
                    img_t, _ = dataset[idx]
                    input_t = img_t.unsqueeze(0).to(self.device)
                    attr = self._generate_attribution(explainer, input_t, node_idx)

                    # Don't add title to each image - use shared row label instead
                    self._safe_visualize(attr, img_t, fig, ax, "")
                else:
                    ax.axis('off')

                # Remove individual titles
                ax.set_title("")

            # Add shared row label on the left side (only first column gets the label)
            axes[row_idx, 0].set_ylabel(
                row_label, fontsize=10, rotation=0,
                labelpad=80, ha='right', va='center'
            )

        # Adjust layout to prevent title overlap and make room for row labels
        plt.tight_layout(rect=[0.12, 0, 1, 0.95])
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
        model = self._safe_load_checkpoint(checkpoint_path, f"iteration {iteration}")
        if model is None:
            return
        explainer = self._create_explainer(model)
        node_samples = collect_node_samples(result['hierarchical_predictions'], num_samples, seed=self._current_seed)
        fig = self._create_node_grid(dataset, node_samples, iteration, explainer)
        fig.savefig(os.path.join(save_dir, f"iteration_{iteration}_nodes.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved iteration {iteration} node visualizations")

        # Clean up to prevent memory accumulation
        del model, explainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def _save_faithfulness_metrics(self, results, save_dir):
        with open(os.path.join(save_dir, "faithfulness_metrics.json"), 'w') as f: json.dump(results, f, indent=2)

    def _generate_transitions(self, dataset, results, labels, num_samples, save_dir):
        for i in range(len(results) - 1):
            prev, curr = results[i], results[i+1]
            prev_iter, curr_iter = i + 1, i + 2

            prev_model = self._safe_load_checkpoint(prev.get('checkpoint_path'), f"iteration {prev_iter}")
            curr_model = self._safe_load_checkpoint(curr.get('checkpoint_path'), f"iteration {curr_iter}")

            if prev_model is None or curr_model is None:
                print(f"  Skipping transitions {prev_iter} -> {curr_iter} due to missing checkpoint")
                continue

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
