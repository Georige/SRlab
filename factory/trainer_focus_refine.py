"""FocusRefineTrainer — iterative refinement for coarse-to-fine center focus.

Phase 9: Multi-step refinement with Teacher Forcing, center-contracting masks,
and region-focused losses. Supports baselines: Direct SR, Center Loss Only,
and ablation variants (w/o time embedding, w/o spatial bias).

Training loop (FocusRefine):
    current = bicubic(I_low, size=H)
    for t in range(T):
        mask = center_mask[H, W, region[t]]           # binary
        x = concat([current, mask])                   # [B,4,H,W]
        residual = model(x, t_norm=t/T)
        current = current + residual
        loss += MSE(crop(current, region[t]), crop(I_gt, region[t]))
        current = I_gt  # Teacher Forcing

Inference (autoregressive, no TF):
    current = bicubic(I_low)
    for t in range(T):
        mask = center_mask[H, W, region[t]]
        x = concat([current, mask])
        residual = model(x, t_norm=t/T)
        current = current + residual
    return current
"""

import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from factory.registry import MODEL_REGISTRY
from model.focus_refine import make_gaussian_heatmap
from vit.overfit_plot import update_curves


# ============================================================
# Helpers
# ============================================================

def to_pil(t):
    """Convert [-1,1] tensor [1,3,H,W] to PIL Image."""
    a = t[0].cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.clip((a + 1) * 127.5, 0, 255).astype(np.uint8))


def crop_center(x, region):
    """Extract central rectangular region from [B,C,H,W] tensor.

    Args:
        x: [B, C, H, W]
        region: int (square) or (h, w) tuple for rectangular region

    Returns:
        [B, C, region_h, region_w]
    """
    H, W_n = x.shape[2], x.shape[3]
    if isinstance(region, int):
        h, w = region, region
    else:
        h, w = region
    start_h = (H - h) // 2
    start_w = (W_n - w) // 2
    return x[:, :, start_h:start_h + h, start_w:start_w + w]


def crop_corners(x, size):
    """Extract four corner size×size squares, average metric-friendly.

    Returns list of 4 tensors [B, C, size, size].
    """
    H, W = x.shape[2], x.shape[3]
    return [
        x[:, :, :size, :size],                 # top-left
        x[:, :, :size, W - size:],             # top-right
        x[:, :, H - size:, :size],             # bottom-left
        x[:, :, H - size:, W - size:],         # bottom-right
    ]


def create_center_mask(H, W, region):
    """Create a binary mask [1, 1, H, W] with 1 in the central region.

    Args:
        H, W: full image dimensions
        region: int (square) or (h, w) tuple for rectangular region

    Returns:
        [1, 1, H, W] float tensor, 1.0 in center, 0.0 elsewhere
    """
    if isinstance(region, int):
        region_h, region_w = region, region
    else:
        region_h, region_w = region
    mask = torch.zeros(1, 1, H, W, dtype=torch.float32)
    start_h = (H - region_h) // 2
    start_w = (W - region_w) // 2
    mask[:, :, start_h:start_h + region_h, start_w:start_w + region_w] = 1.0
    return mask


# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def compute_region_metrics(pred, target, center_size=None):
    """Compute full-image and regional metrics.

    Args:
        pred, target: [1, 3, H, W] in [-1, 1]
        center_size: int, tuple (h,w), or None. If not None, also compute center region metrics.

    Returns:
        dict with keys: mse, psnr, ssim, ncc, center_psnr, corner_psnr
    """
    pred_f = pred.float()
    target_f = target.float()
    H, W = pred_f.shape[2], pred_f.shape[3]

    def _mse(a, b):
        return F.mse_loss(a, b).item()

    def _psnr(mse_val):
        return 20 * np.log10(2.0 / np.sqrt(mse_val)) if mse_val > 0 else 100.0

    def _ssim(a, b):
        # a, b: [1, C, H, W]
        vals = []
        for c in range(a.shape[1]):
            ac = a[0:1, c:c+1]; bc = b[0:1, c:c+1]
            mu_a = F.avg_pool2d(ac, 11, stride=1, padding=5)
            mu_b = F.avg_pool2d(bc, 11, stride=1, padding=5)
            sigma_a = F.avg_pool2d((ac - mu_a)**2, 11, stride=1, padding=5).sqrt()
            sigma_b = F.avg_pool2d((bc - mu_b)**2, 11, stride=1, padding=5).sqrt()
            sigma_ab = F.avg_pool2d((ac - mu_a) * (bc - mu_b), 11, stride=1, padding=5)
            C1, C2 = 0.01**2, 0.03**2
            ssim_map = ((2*mu_a*mu_b + C1) * (2*sigma_ab + C2)) / \
                       ((mu_a**2 + mu_b**2 + C1) * (sigma_a**2 + sigma_b**2 + C2) + 1e-8)
            vals.append(ssim_map.mean().item())
        return float(np.mean(vals))

    def _ncc(a, b):
        vals = []
        for c in range(a.shape[1]):
            ac = a[:, c]; bc = b[:, c]
            a_m, b_m = ac.mean(), bc.mean()
            a_s, bc_s = ac.std(), bc.std()
            if a_s < 1e-8 or bc_s < 1e-8:
                vals.append(0.0)
            else:
                vals.append(((ac - a_m) * (bc - b_m)).mean().item() / (a_s * bc_s).item())
        return float(np.mean(vals))

    results = {}
    mse_full = _mse(pred_f, target_f)
    results['mse'] = mse_full
    results['psnr'] = _psnr(mse_full)
    results['ssim'] = _ssim(pred_f, target_f)
    results['ncc'] = _ncc(pred_f, target_f)

    # Center region metrics
    if center_size is not None:
        cp = crop_center(pred_f, center_size)
        ct = crop_center(target_f, center_size)
        results['center_psnr'] = _psnr(_mse(cp, ct))
    else:
        results['center_psnr'] = 0.0

    # Corner region metrics (average of four corner_size×corner_size squares)
    corner_size = 64
    if min(H, W) >= corner_size * 2:
        corners_p = crop_corners(pred_f, corner_size)
        corners_t = crop_corners(target_f, corner_size)
        corner_psnr_vals = []
        for cp, ct in zip(corners_p, corners_t):
            corner_psnr_vals.append(_psnr(_mse(cp, ct)))
        results['corner_psnr'] = float(np.mean(corner_psnr_vals))
    else:
        results['corner_psnr'] = 0.0

    return results


# ============================================================
# FocusRefineTrainer
# ============================================================

class FocusRefineTrainer:
    """Iterative refinement trainer for coarse-to-fine center focus.

    Supports multiple modes via cfg.training.mode:
      - 'iterative':  multi-step FocusRefine with Teacher Forcing
      - 'direct':     one-step Direct SR, full-image MSE
      - 'center_loss': one-step Direct SR, center-weighted MSE

    Usage:
        trainer = FocusRefineTrainer(cfg, device, output_dir)
        trainer.run()
        trainer.run(resume_ckpt='best_model.pt', start_epoch=100)
    """

    def __init__(self, cfg, device, output_dir):
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        self.exp_name = cfg.experiment.name
        self.mode = cfg.training.get('mode', 'iterative')

        # Image size
        self.H = cfg.data.hr_size[0]
        self.W = cfg.data.hr_size[1]

        # Iterative mode settings
        self.num_steps = cfg.training.get('num_steps', 5)
        self.center_regions = self._resolve_center_regions()
        self.teacher_noise_std = cfg.training.get('teacher_noise_std', 0.0)
        self.use_residual = cfg.training.get('use_residual', True)

        # Scheduled sampling
        self.scheduled_sampling = cfg.training.get('scheduled_sampling', False)
        self.ss_max_prob = cfg.training.get('ss_max_prob', 0.5)

        # Center loss mode
        self.center_loss_weight = cfg.training.get('center_loss_weight', 5.0)
        self.center_loss_region = cfg.training.get('center_loss_region', self.H // 4)

        # B-1/B-2: DiT-specific flags
        self.model_type = cfg.model.type
        self.input_7ch = self.model_type == 'dit_multistep'  # 7ch = I_low_up + current + mask

        # B-2 head loss weights
        self.head_loss_weights = cfg.training.get('head_loss_weights', {'c64': 1.0, 'c128': 0.5, 'full': 0.1})

        # Model and data placeholders
        self.model = None
        self.train_images = []  # list of (I_low_up, I_gt) on CPU
        self.val_images = []    # list of (I_low_up, I_gt) on device
        # Pre-computed masks and center_priors for iterative mode (on device)
        self._center_masks = None
        self._center_priors = None  # for B-2: Gaussian heatmaps at different sizes

    def _resolve_center_regions(self):
        """Resolve center region sizes for each refinement step.

        Config can provide a list of int (square), [h, w] pairs, or None for auto.
        Auto: linearly space from (H,W) down to (H//8, W//8), T steps.

        Returns:
            list of (h, w) tuples
        """
        cfg_regions = self.cfg.training.get('center_regions', None)
        if cfg_regions:
            result = []
            for r in cfg_regions:
                if isinstance(r, list):
                    result.append((int(r[0]), int(r[1])))
                else:
                    result.append((int(r), int(r)))
            return result

        # Auto-generate: from (H,W) down to (H//8, W//8)
        T = self.num_steps
        if T == 1:
            return [(self.H, self.W)]
        regions = []
        for i in range(T):
            h = int(self.H - i * (self.H - self.H // 8) / (T - 1))
            w = int(self.W - i * (self.W - self.W // 8) / (T - 1))
            regions.append((h, w))
        return regions

    # ============================================================
    # Data loading
    # ============================================================

    def _load_data(self):
        """Load image pairs, random-crop to H×W, compute I_low_up.

        Supports two dataset formats:
          1. realsr_v3: HR/LR pairs in one directory (e.g., Canon_001_HR.png, Canon_001_LR2.png)
          2. PanoramaDataset: HR/ and LR/X{scale}/ subdirectories

        Pipeline:
          1. Load HR/LR at native resolution
          2. Random crop to cfg.data.hr_size (e.g., 512×1024)
          3. Bicubic upsample LR crop to H×W as I_low_up
        """
        data_dir = self.cfg.data.data_dir
        scale = self.cfg.data.scale
        n_train = self.cfg.data.n_train
        n_val = self.cfg.data.n_val

        # Detect dataset format: realsr_v3 or PanoramaDataset
        is_realsr = self._is_realsr_format(data_dir)

        all_pairs = []
        if is_realsr:
            print(f"Detected realsr_v3 format: {data_dir}")
            all_pairs = self._load_realsr_pairs(data_dir)
        else:
            print(f"Detected PanoramaDataset format: {data_dir}")
            from utility.data import PanoramaDataset
            full_H, full_W = self.cfg.data.get('full_size', [2048, 1024])
            dataset = PanoramaDataset(data_dir, (full_H, full_W), scale)
            all_pairs = [(dataset[i][0].unsqueeze(0), dataset[i][1].unsqueeze(0))
                         for i in range(len(dataset))]

        total = min(n_train + n_val, len(all_pairs))
        print(f"Loading {total} pairs (train={n_train}, val={n_val}) "
              f"→ random crop to {self.H}×{self.W}")

        rng = np.random.RandomState(42)

        for i in range(total):
            lr_full, hr_full = all_pairs[i]  # [1, 3, H_native, W_native]
            full_H_native, full_W_native = hr_full.shape[2], hr_full.shape[3]

            # If crop equals full size, no cropping needed
            if self.H == full_H_native and self.W == full_W_native:
                I_low_up = F.interpolate(lr_full, size=(self.H, self.W),
                                         mode='bicubic', align_corners=False)
                if i < n_train:
                    self.train_images.append((I_low_up, hr_full))
                else:
                    self.val_images.append((I_low_up.to(self.device), hr_full.to(self.device)))
            else:
                crops_per_img = self.cfg.data.get('crops_per_image', 1)
                max_start_h = full_H_native - self.H
                max_start_w = full_W_native - self.W

                for crop_idx in range(crops_per_img):
                    if i < n_train:
                        start_h = rng.randint(0, max_start_h + 1) if max_start_h > 0 else 0
                        start_w = rng.randint(0, max_start_w + 1) if max_start_w > 0 else 0
                    else:
                        start_h = max_start_h // 2 if max_start_h > 0 else 0
                        start_w = max_start_w // 2 if max_start_w > 0 else 0

                    hr_crop = hr_full[:, :, start_h:start_h + self.H, start_w:start_w + self.W]
                    lr_crop = lr_full[:, :,
                              start_h // scale: (start_h + self.H) // scale,
                              start_w // scale: (start_w + self.W) // scale]

                    I_low_up = F.interpolate(lr_crop, size=(self.H, self.W),
                                             mode='bicubic', align_corners=False)

                    if i < n_train:
                        self.train_images.append((I_low_up, hr_crop))
                    elif crop_idx == 0:
                        self.val_images.append((I_low_up.to(self.device), hr_crop.to(self.device)))

        # Trim
        self.train_images = self.train_images[:n_train]

        # Build center masks for iterative mode
        if self.mode == 'iterative':
            self._center_masks = []
            for region in self.center_regions:
                mask = create_center_mask(self.H, self.W, region).to(self.device)
                self._center_masks.append(mask)

        # Build center prior heatmap for B-2 (single-step DiT)
        if self.mode == 'dit_single':
            self._center_prior = make_gaussian_heatmap(self.H, self.W, sigma=0.4).to(self.device)

        print(f"Loaded {len(self.train_images)} train + {len(self.val_images)} val "
              f"crops ({self.H}×{self.W}), {self.num_steps} steps")
        if self.mode == 'iterative':
            print(f"Center regions: {self.center_regions}")

    def _is_realsr_format(self, data_dir):
        """Check if the directory contains paired _HR and _LR files (realsr format)."""
        import os as _os
        files = _os.listdir(data_dir)
        has_hr = any('_HR' in f for f in files)
        has_lr = any('_LR' in f for f in files)
        return has_hr and has_lr

    @staticmethod
    def _load_realsr_pairs(data_dir):
        """Load paired HR/LR images from realsr_v3 format directory.

        Files: Canon_001_HR.png, Canon_001_LR2.png
        Returns: list of (lr_tensor, hr_tensor) each [1, 3, H, W] in [-1, 1].
        """
        import os as _os
        import re
        files = sorted(_os.listdir(data_dir))

        # Group: find all HR files, then match LR by replacing _HR with _LR{scale}
        hr_files = [f for f in files if '_HR' in f and f.endswith('.png')]
        result = []

        for hr_f in hr_files:
            # Canon_001_HR.png → Canon_001
            base = hr_f.replace('_HR.png', '')
            # Find matching LR: Canon_001_LR*.png
            lr_candidates = [f for f in files
                           if f.startswith(base) and '_LR' in f and f.endswith('.png')]
            if not lr_candidates:
                print(f"  WARNING: no LR match for {hr_f}, skipping")
                continue

            hr_path = _os.path.join(data_dir, hr_f)
            lr_path = _os.path.join(data_dir, lr_candidates[0])

            hr_img = Image.open(hr_path).convert('RGB')
            lr_img = Image.open(lr_path).convert('RGB')
            hr_arr = np.array(hr_img, dtype=np.float32) / 127.5 - 1.0
            lr_arr = np.array(lr_img, dtype=np.float32) / 127.5 - 1.0

            hr_t = torch.from_numpy(hr_arr).permute(2, 0, 1).unsqueeze(0)
            lr_t = torch.from_numpy(lr_arr).permute(2, 0, 1).unsqueeze(0)
            result.append((lr_t, hr_t))

        return result

    # ============================================================
    # Model
    # ============================================================

    def _build_model(self):
        self.model = MODEL_REGISTRY[self.model_type](self.cfg, self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model: {n_params:,} params  |  mode={self.mode}")
        return n_params

    def _load_checkpoint(self, ckpt_path):
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        val_m = self._validate()
        mode_label = "AR" if self.mode == 'iterative' else "direct"
        print(f"Resumed from {ckpt_path}, val_mse={val_m['mse']:.6f}, "
              f"psnr={val_m['psnr']:.1f}dB ({mode_label})")
        return val_m['mse']

    # ============================================================
    # Training steps
    # ============================================================

    def _train_step_iterative(self, I_low_up, I_gt, optimizer, global_epoch, total_epochs):
        """One Teacher Forcing training iteration for FocusRefine.

        Per-step backward (not gradient accumulation) to avoid OOM.
        Each step is independent due to Teacher Forcing (current = I_gt).

        Returns:
            total_loss: scalar float (sum of all step losses)
            step_losses: list of per-step MSE losses
        """
        current = I_low_up.clone()
        step_losses = []

        # Determine scheduled sampling probability for this epoch
        if self.scheduled_sampling:
            p_self = min(self.ss_max_prob,
                         global_epoch / max(total_epochs, 1) * self.ss_max_prob)
        else:
            p_self = 0.0

        for t in range(self.num_steps):
            mask = self._center_masks[t]  # [1, 1, H, W]
            region_h, region_w = self.center_regions[t]
            t_norm = torch.tensor([t / max(self.num_steps - 1, 1)], device=self.device)

            # Build input
            if self.input_7ch:
                # B-1 DiT: I_low_up + current + mask = 7 channels
                model_input = torch.cat([I_low_up, current, mask.expand(1, 1, self.H, self.W)], dim=1)
            else:
                # Original FocusRefineUNet: current + mask = 4 channels
                model_input = torch.cat([current, mask.expand(1, 1, self.H, self.W)], dim=1)

            # Forward
            residual = self.model(model_input, t_norm)
            refined = current + residual if self.use_residual else residual

            # Loss on current region
            if t == 0 or (region_h >= self.H and region_w >= self.W):
                loss = F.mse_loss(refined, I_gt)
            else:
                pred_crop = crop_center(refined, (region_h, region_w))
                gt_crop = crop_center(I_gt, (region_h, region_w))
                loss = F.mse_loss(pred_crop, gt_crop)

            # Per-step backward (prevents OOM vs accumulating 5× activations)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_losses.append(loss.item())

            # Teacher Forcing with optional noise and scheduled sampling
            if p_self > 0 and random.random() < p_self:
                next_center = crop_center(refined, (region_h, region_w)).detach()
            else:
                next_center = crop_center(I_gt, (region_h, region_w)).detach()

            if self.teacher_noise_std > 0:
                next_center = next_center + torch.randn_like(next_center) * self.teacher_noise_std

            # Pure TF: current = I_gt
            new_current = I_gt.clone()
            if p_self > 0 or self.teacher_noise_std > 0:
                start_h = (self.H - region_h) // 2
                start_w = (self.W - region_w) // 2
                new_current[:, :, start_h:start_h + region_h, start_w:start_w + region_w] = next_center
            current = new_current

        return sum(step_losses), step_losses

    def _train_step_direct(self, I_low_up, I_gt, optimizer):
        """One-step direct SR training."""
        # model input is I_low_up only [1, 3, H, W]
        optimizer.zero_grad()
        residual = self.model(I_low_up)
        sr = I_low_up + residual if self.use_residual else residual
        loss = F.mse_loss(sr, I_gt)
        loss.backward()
        optimizer.step()
        return loss.item()

    def _train_step_center_loss(self, I_low_up, I_gt, optimizer):
        """One-step direct SR with center-weighted loss."""
        optimizer.zero_grad()
        residual = self.model(I_low_up)
        sr = I_low_up + residual if self.use_residual else residual

        # Full-image MSE
        loss_full = F.mse_loss(sr, I_gt)

        # Center region MSE
        center_size = self.center_loss_region
        sr_center = crop_center(sr, center_size)
        gt_center = crop_center(I_gt, center_size)
        loss_center = F.mse_loss(sr_center, gt_center)

        # Weighted combination
        loss = loss_full + self.center_loss_weight * loss_center
        loss.backward()
        optimizer.step()
        return loss.item()

    def _train_step_dit_single(self, I_low_up, I_gt, optimizer):
        """B-2: Single-step DiT with multi-scale head loss.

        Model receives I_low_up + center_prior, outputs dict of residuals
        for different center regions + full image.
        Loss = sum(λ_k * Charbonnier(head_k, GT_crop_k))
        """
        optimizer.zero_grad()

        # Build center_prior: Gaussian heatmap [1, 1, H, W]
        center_prior = make_gaussian_heatmap(self.H, self.W, sigma=0.4).to(self.device)
        model_input = torch.cat([I_low_up, center_prior], dim=1)  # [1, 4, H, W]

        # Forward: returns dict
        outputs = self.model(model_input)  # e.g., {'c64': [1,3,64,64], 'c128': [1,3,128,128], 'full': [1,3,H,W]}

        # Hierarchical center-focused loss
        total_loss = torch.tensor(0.0, device=self.device)
        for name, pred in outputs.items():
            w = self.head_loss_weights.get(name, 0.1)
            if name == 'full':
                gt_crop = I_gt
            else:
                # Infer region size from prediction tensor dimensions
                rh, rw = pred.shape[2], pred.shape[3]
                gt_crop = crop_center(I_gt, (rh, rw))
            # Charbonnier loss
            loss_k = torch.sqrt((pred - gt_crop) ** 2 + 1e-6).mean()
            total_loss = total_loss + w * loss_k

        total_loss.backward()
        optimizer.step()
        return total_loss.item()

    def _train_step(self, I_low_up, I_gt, optimizer, global_epoch, total_epochs):
        """Dispatch to the appropriate training step."""
        if self.mode == 'iterative':
            total_loss, step_losses = self._train_step_iterative(
                I_low_up, I_gt, optimizer, global_epoch, total_epochs)
            return total_loss, step_losses
        elif self.mode == 'dit_single':
            return self._train_step_dit_single(I_low_up, I_gt, optimizer), None
        elif self.mode == 'center_loss':
            return self._train_step_center_loss(I_low_up, I_gt, optimizer), None
        else:  # direct
            return self._train_step_direct(I_low_up, I_gt, optimizer), None

    # ============================================================
    # Inference & Validation
    # ============================================================

    @torch.no_grad()
    def _infer(self, I_low_up):
        """Run inference.

        iterative (B-1 / FocusRefineUNet): autoregressive 5-step
        dit_single (B-2): single forward → use 'full' head output
        direct/center_loss: one-step forward

        Returns:
            sr: [1, 3, H, W] final super-resolved image
            steps: list of intermediate [1, 3, H, W] (empty for non-iterative)
        """
        if self.mode == 'iterative':
            current = I_low_up.clone()
            steps = [current.clone()]
            for t in range(self.num_steps):
                mask = self._center_masks[t]
                t_norm = torch.tensor([t / max(self.num_steps - 1, 1)], device=self.device)
                if self.input_7ch:
                    model_input = torch.cat([I_low_up, current, mask.expand(1, 1, self.H, self.W)], dim=1)
                else:
                    model_input = torch.cat([current, mask.expand(1, 1, self.H, self.W)], dim=1)
                residual = self.model(model_input, t_norm)
                current = current + residual if self.use_residual else residual
                steps.append(current.clone())
            return current, steps
        elif self.mode == 'dit_single':
            center_prior = make_gaussian_heatmap(self.H, self.W, sigma=0.4).to(self.device)
            model_input = torch.cat([I_low_up, center_prior], dim=1)
            outputs = self.model(model_input)
            sr = I_low_up + outputs['full'] if self.use_residual else outputs['full']
            return sr, []
        else:
            # Direct SR or center_loss
            residual = self.model(I_low_up)
            sr = I_low_up + residual if self.use_residual else residual
            return sr, []

    @torch.no_grad()
    def _validate(self):
        """Run validation: inference + full/center/corner metrics.

        Returns aggregated metrics dict.
        """
        self.model.eval()

        agg = {'mse': 0.0, 'psnr': 0.0, 'ssim': 0.0, 'ncc': 0.0,
               'center_psnr': 0.0, 'corner_psnr': 0.0}

        # Determine center region for metrics reporting
        if self.mode == 'iterative':
            center_metric_size = self.center_regions[-1]  # smallest region
        else:
            center_metric_size = self.H // 4

        for I_low_up, I_gt in self.val_images:
            sr, _ = self._infer(I_low_up)
            m = compute_region_metrics(sr, I_gt, center_size=center_metric_size)
            for k in agg:
                agg[k] += m.get(k, 0.0)

        for k in agg:
            agg[k] /= max(len(self.val_images), 1)

        self.model.train()
        return agg

    @torch.no_grad()
    def _save_progression(self, epoch):
        """Save validation outputs: final image + intermediate step strip."""
        self.model.eval()
        I_low_up, I_gt = self.val_images[0]

        sr, steps = self._infer(I_low_up)

        # Save final output
        to_pil(sr).save(os.path.join(self.output_dir, f"e{epoch:04d}.png"))

        # Save progression strip for iterative mode
        if self.mode == 'iterative' and len(steps) >= 2:
            # Show: I_low_up → step1 → step2 → ... → stepT → GT
            row_imgs = []
            # Only show key steps to keep it compact
            n_steps = len(steps)
            if n_steps <= 6:
                indices = list(range(n_steps))
            else:
                # First, middle, last-1, last (plus GT at end)
                indices = [0]
                if n_steps > 4:
                    mid = n_steps // 2
                    indices.append(mid)
                if n_steps > 3:
                    indices.append(n_steps - 2)
                indices.append(n_steps - 1)

            for idx in indices:
                img = to_pil(steps[idx] if idx < len(steps) else I_gt)
                img = img.resize((256, 256), Image.LANCZOS)
                row_imgs.append(img)
            # Add GT
            gt_img = to_pil(I_gt).resize((256, 256), Image.LANCZOS)
            row_imgs.append(gt_img)

            w_img, h_img = row_imgs[0].size
            strip = Image.new('RGB', (w_img * len(row_imgs), h_img))
            for j, img in enumerate(row_imgs):
                strip.paste(img, (j * w_img, 0))
            strip.save(os.path.join(self.output_dir, f"prog_{epoch:04d}.png"))

        self.model.train()

    # ============================================================
    # Main training loop
    # ============================================================

    def run(self, epochs=None, resume_ckpt=None, start_epoch=0):
        epochs = epochs or self.cfg.training.epochs
        lr = self.cfg.training.lr

        self._load_data()
        self._build_model()

        if resume_ckpt:
            best_val_mse = self._load_checkpoint(resume_ckpt)
            best_epoch = start_epoch
        else:
            # Save reference images
            if self.val_images:
                ref_low, ref_gt = self.val_images[0]
                to_pil(ref_low).save(os.path.join(self.output_dir, "val_bicubic.png"))
                to_pil(ref_gt).save(os.path.join(self.output_dir, "val_gt.png"))
            best_val_mse = float('inf')
            best_epoch = 0

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_losses, val_epochs, val_mses = [], [], []
        val_every = max(20, epochs // 20)

        # Bicubic baseline
        if not resume_ckpt and self.val_images:
            bicubic_m = compute_region_metrics(
                self.val_images[0][0], self.val_images[0][1],
                center_size=(self.center_regions[-1] if self.mode == 'iterative' else self.H // 4))
            print(f"Bicubic baseline: PSNR={bicubic_m['psnr']:.2f}dB, "
                  f"center_PSNR={bicubic_m['center_psnr']:.2f}dB, "
                  f"corner_PSNR={bicubic_m['corner_psnr']:.2f}dB")

        tag = f" (resume from {start_epoch})" if start_epoch > 0 else ""
        print(f"\n{'=' * 60}")
        print(f"Experiment: {self.exp_name}{tag}")
        print(f"Model: {self.model_type}  |  Mode: {self.mode}  |  LR={lr} (cosine)  |  Epochs={epochs}")
        print(f"Resolution: {self.H}×{self.W}")
        if self.mode == 'iterative':
            print(f"Steps: {self.num_steps}  |  Regions: {self.center_regions}")
            print(f"Residual: {self.use_residual}  |  Teacher noise: {self.teacher_noise_std}")
            if self.scheduled_sampling:
                print(f"Scheduled Sampling: ON (max_prob={self.ss_max_prob}, ramp over {epochs} epochs)")
        elif self.mode == 'center_loss':
            print(f"Center loss weight: {self.center_loss_weight}  |  Region: {self.center_loss_region}")
        print(f"Train: {len(self.train_images)}  |  Val: {len(self.val_images)}")
        print(f"Output: {self.output_dir}")
        print(f"{'=' * 60}")

        pbar = tqdm(range(1, epochs + 1), desc=f"[{self.exp_name}]", unit="ep")
        for epoch in pbar:
            global_epoch = start_epoch + epoch
            self.model.train()

            epoch_loss = 0.0
            step_agg = [0.0] * self.num_steps if self.mode == 'iterative' else None

            # Shuffle training images each epoch
            indices = list(range(len(self.train_images)))
            random.shuffle(indices)

            for idx in indices:
                I_low_cpu, I_gt_cpu = self.train_images[idx]
                I_low_up = I_low_cpu.to(self.device)
                I_gt = I_gt_cpu.to(self.device)

                loss_val, step_losses = self._train_step(
                    I_low_up, I_gt, optimizer, global_epoch, epochs)

                epoch_loss += loss_val
                if step_losses and step_agg:
                    for s in range(len(step_losses)):
                        step_agg[s] += step_losses[s]

            epoch_loss /= len(self.train_images)
            if step_agg:
                step_agg = [s / len(self.train_images) for s in step_agg]
            train_losses.append(epoch_loss)
            scheduler.step()

            # Validation
            if epoch == 1 or epoch % val_every == 0 or epoch == epochs:
                val_m = self._validate()
                val_epochs.append(global_epoch)
                val_mses.append(val_m['mse'])

                if val_m['mse'] < best_val_mse:
                    best_val_mse = val_m['mse']
                    best_epoch = global_epoch
                    torch.save(self.model.state_dict(),
                               os.path.join(self.output_dir, "best_model.pt"))

                self._save_progression(global_epoch)

                update_curves(self.exp_name, train_losses, val_epochs, val_mses,
                              log_dir=self.cfg.output.log_dir)

                post = dict(
                    train=f"{epoch_loss:.6f}",
                    val_mse=f"{val_m['mse']:.6f}",
                    val_psnr=f"{val_m['psnr']:.1f}",
                    ctr_psnr=f"{val_m['center_psnr']:.1f}",
                    best_ep=str(best_epoch),
                )
                if self.mode == 'iterative' and step_agg:
                    post['steps'] = '/'.join(f"{s:.4f}" for s in step_agg)
                pbar.set_postfix(post)
            else:
                postfix = {"train": f"{epoch_loss:.6f}"}
                if self.mode == 'iterative' and step_agg:
                    postfix['steps'] = '/'.join(f"{s:.4f}" for s in step_agg)
                pbar.set_postfix(postfix)

        # Final summary
        print(f"\n{'=' * 60}")
        print(f"Experiment {self.exp_name} Results:")
        print(f"  Final train loss: {train_losses[-1]:.6f}")
        for ep, mse_val in zip(val_epochs, val_mses):
            mkr = " <-- BEST" if ep == best_epoch else ""
            print(f"  Epoch {ep:4d}: MSE={mse_val:.6f}{mkr}")

        self.model.load_state_dict(
            torch.load(os.path.join(self.output_dir, "best_model.pt"),
                       map_location=self.device))
        best = self._validate()
        print(f"\n  Best: epoch {best_epoch}")
        print(f"    Full  → MSE={best['mse']:.6f}, PSNR={best['psnr']:.2f}dB, SSIM={best['ssim']:.4f}")
        print(f"    Center→ PSNR={best['center_psnr']:.2f}dB")
        print(f"    Corner→ PSNR={best['corner_psnr']:.2f}dB")
        print(f"{'=' * 60}")

        return best
