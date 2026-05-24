"""Centralized image quality assessment metrics for super-resolution.

Reference-based: PSNR_Y, SSIM_Y (Y channel in YCbCr space), LPIPS, DISTS, FID
Non-reference:  MANIQA, MUSIQ, CLIPIQA

All metrics expect images in [-1, 1] (training domain) and convert to [0, 1]
internally for pyiqa compatibility.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import models as tv_models


# ============================================================
# PyIQA lazy-loading backend
# ============================================================

_pyiqa_metrics = {}

def _get_pyiqa(name, device):
    """Lazy-create and cache pyiqa metric instances."""
    key = (name, str(device))
    if key not in _pyiqa_metrics:
        import pyiqa
        _pyiqa_metrics[key] = pyiqa.create_metric(name, device=device)
    return _pyiqa_metrics[key]


# ============================================================
# Colour-space helpers
# ============================================================

def _to_01(x):
    """Convert [-1,1] to [0,1], clamping edge values."""
    return (x.clamp(-1, 1) + 1) / 2.0


def rgb_to_y(x):
    """Convert RGB [-1,1] to Y (luminance) in [0,1] via ITU-R BT.601."""
    x = _to_01(x)
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]



# ============================================================
# Reference-based metrics
# ============================================================

@torch.no_grad()
def compute_psnr(pred, target):
    """PSNR on Y channel (YCbCr space), per the ISR paper standard.

    Args:
        pred, target: [B, 3, H, W] tensors in [-1, 1]
    Returns:
        float
    """
    y_pred = rgb_to_y(pred)       # [B, 1, H, W] in [0,1]
    y_target = rgb_to_y(target)
    mse = F.mse_loss(y_pred, y_target)
    return float(20 * np.log10(1.0 / np.sqrt(mse.item())) if mse > 0 else 100.0)


@torch.no_grad()
def compute_ssim(pred, target):
    """SSIM on Y channel (YCbCr space), per the ISR paper standard.

    Uses 11x11 Gaussian window, same as Wang et al. 2004.

    Args:
        pred, target: [B, 3, H, W] tensors in [-1, 1]
    Returns:
        float
    """
    y_pred = rgb_to_y(pred)       # [B, 1, H, W] in [0,1]
    y_target = rgb_to_y(target)

    # 11x11 Gaussian-like window via avg_pool (same as existing implementation)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_pred = F.avg_pool2d(y_pred, 11, stride=1, padding=5)
    mu_target = F.avg_pool2d(y_target, 11, stride=1, padding=5)
    sigma_pred = F.avg_pool2d((y_pred - mu_pred) ** 2, 11, stride=1, padding=5).sqrt()
    sigma_target = F.avg_pool2d((y_target - mu_target) ** 2, 11, stride=1, padding=5).sqrt()
    sigma_pt = F.avg_pool2d((y_pred - mu_pred) * (y_target - mu_target), 11, stride=1, padding=5)

    ssim_map = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pt + C2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred ** 2 + sigma_target ** 2 + C2) + 1e-8)
    return float(ssim_map.mean())


@torch.no_grad()
def compute_lpips(pred, target, device=None):
    """LPIPS (Learned Perceptual Image Patch Similarity).

    Uses AlexNet backbone. Lower is better.

    Args:
        pred, target: [B, 3, H, W] tensors in [-1, 1]
    Returns:
        float
    """
    if device is None:
        device = pred.device
    x = _to_01(pred)
    y = _to_01(target)
    m = _get_pyiqa('lpips', device)
    return float(m(x, y).item())


@torch.no_grad()
def compute_dists(pred, target, device=None):
    """DISTS (Deep Image Structure and Texture Similarity).

    Lower is better.

    Args:
        pred, target: [B, 3, H, W] tensors in [-1, 1]
    Returns:
        float
    """
    if device is None:
        device = pred.device
    x = _to_01(pred)
    y = _to_01(target)
    m = _get_pyiqa('dists', device)
    return float(m(x, y).item())


@torch.no_grad()
def compute_fid(pred_batch, target_batch):
    """FID (Frechet Inception Distance) between two sets of images.

    Uses InceptionV3 pool3 features (2048-d). Canonical FID formula:
      ||mu1 - mu2||^2 + Tr(S1 + S2 - 2*sqrt(S1*S2))

    Args:
        pred_batch: [B, 3, H, W] tensors in [-1, 1] (multiple images)
        target_batch: [B, 3, H, W] tensors in [-1, 1] (multiple images)
    Returns:
        float (FID score)
    """
    if pred_batch.shape[0] < 2 or target_batch.shape[0] < 2:
        return float('nan')

    device = pred_batch.device
    inception = _get_inception(device)

    pred_01 = _to_01(pred_batch)
    target_01 = _to_01(target_batch)

    def get_features(img_01):
        # Inception expects [0,1] resized to 299x299
        img_299 = F.interpolate(img_01, size=(299, 299), mode='bilinear', align_corners=False)
        features = inception(img_299)
        return features.view(features.size(0), -1)

    feat_pred = get_features(pred_01)
    feat_target = get_features(target_01)

    mu1 = feat_pred.mean(dim=0)
    mu2 = feat_target.mean(dim=0)
    sigma1 = torch.cov(feat_pred.T)
    sigma2 = torch.cov(feat_target.T)

    diff = mu1 - mu2
    # sqrtm via matrix square root of symmetric positive definite matrix
    covmean = _sqrtm(sigma1 @ sigma2)
    fid = (diff @ diff) + torch.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid.item())


# ============================================================
# Non-reference IQA metrics
# ============================================================

@torch.no_grad()
def compute_maniqa(pred, device=None):
    """MANIQA (Multi-dimension Attention Network for IQA).

    No-reference IQA. Higher is better.

    Args:
        pred: [B, 3, H, W] tensor in [-1, 1]
    Returns:
        float
    """
    if device is None:
        device = pred.device
    x = _to_01(pred)
    m = _get_pyiqa('maniqa', device)
    return float(m(x).item())


@torch.no_grad()
def compute_musiq(pred, device=None):
    """MUSIQ (Multi-scale Image Quality Transformer).

    No-reference IQA. Higher is better.

    Args:
        pred: [B, 3, H, W] tensor in [-1, 1]
    Returns:
        float
    """
    if device is None:
        device = pred.device
    x = _to_01(pred)
    m = _get_pyiqa('musiq', device)
    return float(m(x).item())


@torch.no_grad()
def compute_clipiqa(pred, device=None):
    """CLIPIQA (CLIP-based Image Quality Assessment).

    No-reference IQA. Higher is better.

    Args:
        pred: [B, 3, H, W] tensor in [-1, 1]
    Returns:
        float
    """
    if device is None:
        device = pred.device
    x = _to_01(pred)
    m = _get_pyiqa('clipiqa', device)
    return float(m(x).item())


# ============================================================
# Legacy metrics (kept for backward compatibility)
# ============================================================

@torch.no_grad()
def compute_mse(pred, target):
    """MSE between pred and target in [-1, 1]."""
    return float(F.mse_loss(pred.float(), target.float()).item())


@torch.no_grad()
def compute_ncc(pred, target):
    """Per-channel normalised cross-correlation, averaged over RGB."""
    ncc_vals = []
    for c in range(3):
        a = pred[:, c].float(); b = target[:, c].float()
        a_m, b_m = a.mean(), b.mean()
        a_s, b_s = a.std(), b.std()
        if a_s < 1e-8 or b_s < 1e-8:
            ncc_vals.append(0.0)
        else:
            ncc_vals.append(float(((a - a_m) * (b - b_m)).mean() / (a_s * b_s)))
    return float(np.mean(ncc_vals))


@torch.no_grad()
def compute_edge_ncc(pred, target):
    """Edge-map NCC via 3x3 Sobel gradient, averaged over RGB."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    edge_ncc_vals = []
    for c in range(3):
        a = pred[0:1, c:c+1].float(); b = target[0:1, c:c+1].float()
        ga = torch.sqrt(F.conv2d(a, sobel_x, padding=1)**2 + F.conv2d(a, sobel_y, padding=1)**2).view(-1)
        gb = torch.sqrt(F.conv2d(b, sobel_x, padding=1)**2 + F.conv2d(b, sobel_y, padding=1)**2).view(-1)
        a_m, b_m = ga.mean(), gb.mean()
        a_s, b_s = ga.std(), gb.std()
        if a_s < 1e-8 or b_s < 1e-8:
            edge_ncc_vals.append(0.0)
        else:
            edge_ncc_vals.append(float(((ga - a_m) * (gb - b_m)).mean() / (a_s * b_s)))
    return float(np.mean(edge_ncc_vals))


# ============================================================
# Unified interface
# ============================================================

@torch.no_grad()
def compute_all_metrics(pred, target, device=None, include_nr=True, include_heavy=False):
    """Compute all standard reference-based metrics in one call.

    Args:
        pred:   [B,3,H,W] in [-1, 1]
        target: [B,3,H,W] in [-1, 1]
        device: torch device (auto-detected if None)
        include_nr: if True, include non-reference metrics (MANIQA, MUSIQ, CLIPIQA)
        include_heavy: if True, include LPIPS, DISTS, FID (slower)

    Returns:
        dict of metric_name -> float
    """
    if device is None:
        device = pred.device

    results = dict(
        psnr_y=compute_psnr(pred, target),
        ssim_y=compute_ssim(pred, target),
        mse=compute_mse(pred, target),
        ncc=compute_ncc(pred, target),
        edge_ncc=compute_edge_ncc(pred, target),
    )

    if include_heavy:
        try:
            results['lpips'] = compute_lpips(pred, target, device)
        except Exception:
            results['lpips'] = float('nan')
        try:
            results['dists'] = compute_dists(pred, target, device)
        except Exception:
            results['dists'] = float('nan')
        try:
            results['fid'] = compute_fid(pred, target)
        except Exception:
            results['fid'] = float('nan')

    if include_nr:
        try:
            results['maniqa'] = compute_maniqa(pred, device)
        except Exception:
            results['maniqa'] = float('nan')
        try:
            results['musiq'] = compute_musiq(pred, device)
        except Exception:
            results['musiq'] = float('nan')
        try:
            results['clipiqa'] = compute_clipiqa(pred, device)
        except Exception:
            results['clipiqa'] = float('nan')

    return results


# ============================================================
# InceptionV3 for FID
# ============================================================

_inception = None

def _get_inception(device):
    """Lazy-load InceptionV3 for FID (pool3 layer, 2048-d features)."""
    global _inception
    if _inception is None:
        inception = tv_models.inception_v3(weights=tv_models.Inception_V3_Weights.DEFAULT)
        inception.fc = torch.nn.Identity()  # replace classifier
        inception.aux_logits = False
        inception.eval()
        for p in inception.parameters():
            p.requires_grad = False
        _inception = inception
    return _inception.to(device)


@torch.no_grad()
def _sqrtm(mat, eps=1e-6, max_iter=50):
    """Matrix square root via Newton-Schulz iteration."""
    if mat.ndim == 0:
        return mat.sqrt()
    # Normalise by trace to keep iteration stable
    trace = torch.trace(mat)
    if trace < eps:
        return torch.zeros_like(mat)
    mat = mat / trace
    identity = torch.eye(mat.shape[0], device=mat.device, dtype=mat.dtype)
    y = mat
    z = identity
    for _ in range(max_iter):
        y_half = 0.5 * y
        z_half = 0.5 * z
        y = y_half @ (3.0 * identity - z_half @ y_half)
        z = (3.0 * identity - z_half @ y_half) @ z_half
        if (y @ y - mat).norm() < eps:
            break
    return y * trace.sqrt()


# ============================================================
# Backward-compatible compute_metrics (drop-in replacement)
# ============================================================

@torch.no_grad()
def compute_metrics(pred, target, heavy=False, nr=False):
    """Backward-compatible drop-in for existing compute_metrics functions.

    Returns dict with keys: mse, psnr, ssim, ncc, edge_ncc
    (matching the existing interface used by all trainers).

    When heavy=True, also includes: lpips, dists, fid
    When nr=True, also includes: maniqa, musiq, clipiqa
    """
    results = dict(
        mse=compute_mse(pred, target),
        psnr=compute_psnr(pred, target),
        ssim=compute_ssim(pred, target),
        ncc=compute_ncc(pred, target),
        edge_ncc=compute_edge_ncc(pred, target),
    )
    if heavy:
        try:
            results['lpips'] = compute_lpips(pred, target, pred.device)
        except Exception:
            results['lpips'] = float('nan')
        try:
            results['dists'] = compute_dists(pred, target, pred.device)
        except Exception:
            results['dists'] = float('nan')
        try:
            results['fid'] = compute_fid(pred, target)
        except Exception:
            results['fid'] = float('nan')
    if nr:
        try:
            results['maniqa'] = compute_maniqa(pred, pred.device)
        except Exception:
            results['maniqa'] = float('nan')
        try:
            results['musiq'] = compute_musiq(pred, pred.device)
        except Exception:
            results['musiq'] = float('nan')
        try:
            results['clipiqa'] = compute_clipiqa(pred, pred.device)
        except Exception:
            results['clipiqa'] = float('nan')
    return results
