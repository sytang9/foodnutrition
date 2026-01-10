"""Utility functions for XAI (Grad-CAM and Counterfactual explanations)."""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from pathlib import Path
from config import (
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, TARGET_COLS, DEVICE,
    GRADCAM_DOWN_SIZE, GRADCAM_Q_LOW, GRADCAM_Q_HIGH, GRADCAM_GAMMA,
    OVERLAY_ALPHA, HEATMAP_CMAP
)


def _normalize_imagenet(x01: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """Normalize image from [0,1] to ImageNet normalized space."""
    mean_t = torch.tensor(mean, dtype=x01.dtype, device=x01.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=x01.dtype, device=x01.device).view(1, 3, 1, 1)
    return (x01 - mean_t) / std_t


def _denormalize_imagenet(x: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """Denormalize from ImageNet normalized space back to [0,1]."""
    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return x * std_t + mean_t


def predict_raw(model, x_norm: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
    """Get raw-unit predictions from normalized input."""
    with torch.no_grad():
        with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
            pred_z = model(x_norm)
        pred = pred_z * target_std.to(DEVICE) + target_mean.to(DEVICE)
    return pred


class GradCAM:
    """Grad-CAM explainer for feature importance visualization."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._hook = None
        self._register_hook()

    def _register_hook(self):
        def fwd_hook(module, inp, out):
            self.activations = out
            if isinstance(out, torch.Tensor) and out.requires_grad:
                out.register_hook(self._save_grad)
        self._hook = self.target_layer.register_forward_hook(fwd_hook)

    def _save_grad(self, grad):
        self.gradients = grad

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def __call__(self, x_norm: torch.Tensor, target_index: int) -> torch.Tensor:
        """Compute Grad-CAM heatmap for a specific output index."""
        self.model.zero_grad(set_to_none=True)
        self.gradients = None
        self.activations = None

        with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
            out_z = self.model(x_norm)
            score = out_z[:, target_index].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError('Grad-CAM did not capture activations/gradients.')

        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam_map = (weights * acts).sum(dim=1, keepdim=True)
        cam_map = F.relu(cam_map)

        b, _, h, w = cam_map.shape
        cam_map = cam_map.view(b, -1)
        cam_min = cam_map.min(dim=1, keepdim=True).values
        cam_max = cam_map.max(dim=1, keepdim=True).values
        cam_map = (cam_map - cam_min) / (cam_max - cam_min + 1e-8)
        cam_map = cam_map.view(b, 1, h, w)
        return cam_map


def _overlay_heatmap(img01: np.ndarray, heat01: np.ndarray, cmap_name: str = 'jet', alpha: float = 0.35) -> np.ndarray:
    """Overlay a normalized heatmap over an RGB image."""
    heat01 = np.clip(heat01, 0.0, 1.0)
    cmap = plt.colormaps.get_cmap(cmap_name)
    heat_rgb = cmap(heat01)[..., :3]
    out = (1.0 - alpha) * img01 + alpha * heat_rgb
    return np.clip(out, 0.0, 1.0)


def _gradcam_like_from_delta(
    delta_gray: np.ndarray,
    down_size: int = GRADCAM_DOWN_SIZE,
    q_low: float = GRADCAM_Q_LOW,
    q_high: float = GRADCAM_Q_HIGH,
    gamma: float = GRADCAM_GAMMA,
) -> np.ndarray:
    """Convert noisy pixel deltas into smooth Grad-CAM-like regions."""
    h, w = delta_gray.shape
    heat_t = torch.tensor(delta_gray, dtype=torch.float32).view(1, 1, h, w)
    heat_t = F.interpolate(heat_t, size=(down_size, down_size), mode='area')
    heat_t = F.interpolate(heat_t, size=(h, w), mode='bilinear', align_corners=False)
    heat = heat_t[0, 0].cpu().numpy()

    lo = float(np.percentile(heat, q_low))
    hi = float(np.percentile(heat, q_high))
    heat = (heat - lo) / (hi - lo + 1e-8)
    heat = np.clip(heat, 0.0, 1.0)
    heat = np.power(heat, gamma)
    return heat


def _normalize_map_noisy(delta_gray: np.ndarray, q_high: float = 99.5) -> np.ndarray:
    """Normalize raw delta map without smoothing."""
    denom = float(np.percentile(delta_gray, q_high))
    denom = max(denom, 1e-8)
    out = delta_gray / denom
    return np.clip(out, 0.0, 1.0)


def total_variation(x01: torch.Tensor) -> torch.Tensor:
    """Compute total variation regularization."""
    tv_h = torch.mean(torch.abs(x01[:, :, 1:, :] - x01[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x01[:, :, :, 1:] - x01[:, :, :, :-1]))
    return tv_h + tv_w


def _safe_logit(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Safe logit for sigmoid parameterization."""
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


def counterfactual_toward_target_calories(
    model,
    x_norm: torch.Tensor,
    target_cal: float,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    cal_idx: int = 0,
    steps: int = 200,
    lr: float = 0.02,
    lam_l2: float = 10.0,
    lam_tv: float = 0.05,
    grad_clip: float = 1.0,
):
    """Generate counterfactual image targeting a specific calorie value."""
    x_norm = x_norm.unsqueeze(0).to(DEVICE)
    x01_0 = _denormalize_imagenet(x_norm).clamp(0, 1).detach().float()

    # Unconstrained parameterization: x01 = sigmoid(u)
    u = _safe_logit(x01_0).detach().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)

    tgt = torch.tensor([float(target_cal)], device=DEVICE, dtype=torch.float32)
    last_good_u = u.detach().clone()

    for t in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)

        x01 = torch.sigmoid(u)
        x_norm_cur = _normalize_imagenet(x01)

        pred_z = model(x_norm_cur.float())
        pred = pred_z * target_std.to(DEVICE) + target_mean.to(DEVICE)
        pred_cal = pred[:, cal_idx]

        loss_target = (pred_cal - tgt).pow(2).mean()
        loss_l2 = (x01 - x01_0).pow(2).mean()
        loss_tv = total_variation(x01)
        loss = loss_target + lam_l2 * loss_l2 + lam_tv * loss_tv

        if not torch.isfinite(loss):
            u = last_good_u.detach().clone().requires_grad_(True)
            break

        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([u], max_norm=grad_clip)

        opt.step()
        last_good_u = u.detach().clone()

    x01_final = torch.sigmoid(u.detach()).clamp(0, 1)
    x_norm_final = _normalize_imagenet(x01_final)

    with torch.no_grad():
        pred0 = predict_raw(model, _normalize_imagenet(x01_0), target_mean, target_std)[0]
        pred1 = predict_raw(model, x_norm_final, target_mean, target_std)[0]

    return x01_0.cpu()[0], x01_final.cpu()[0], pred0.cpu().numpy(), pred1.cpu().numpy()
