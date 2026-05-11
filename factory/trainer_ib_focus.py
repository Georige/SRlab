"""IB-Focus Trainer — Information Bottleneck iterative refinement.

Phase 10: Route B — multi-step refinement with IB bottleneck, Charbonnier loss
on focus regions, VGG perceptual loss on final output, and IB sparsity regularizer.

Training loop:
    current = bicubic(I_low)
    for t in range(T):
        mask = center_mask[H, W, region[t]]
        x = concat([current, mask])                    # [B,4,H,W]
        residual = model(x, t_norm=t/T)
        refined = current + residual
        loss = charbonnier(crop(refined, region[t]), crop(I_gt, region[t]))
        loss += sparse_weight * model.ib_sparsity_loss()
        if t == T-1:  # final step
            loss += perc_weight * vgg_loss(refined, I_gt)
            loss += full_weight * charbonnier(refined, I_gt)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        current = I_gt  # Teacher Forcing
"""

import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from factory.registry import MODEL_REGISTRY
from vit.overfit_plot import update_curves


# ============================================================
# Helpers (reused from FocusRefine trainer)
# ============================================================

def to_pil(t):
    """Convert [-1,1] tensor [1,3,H,W] to PIL Image."""
    a = t[0].cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.clip((a + 1) * 127.5, 0, 255).astype(np.uint8))


def crop_center(x, region):
    """Extract central rectangular region from [B,C,H,W] tensor."""
    H, W_n = x.shape[2], x.shape[3]
    if isinstance(region, int):
        h, w = region, region
    else:
        h, w = region
    start_h = (H - h) // 2
    start_w = (W_n - w) // 2
    return x[:, :, start_h:start_h + h, start_w:start_w + w]


def crop_corners(x, size):
    """Extract four corner size×size squares."""
    H, W = x.shape[2], x.shape[3]
    return [
        x[:, :, :size, :size],
        x[:, :, :size, W - size:],
        x[:, :, H - size:, :size],
        x[:, :, H - size:, W - size:],
    ]


def create_center_mask(H, W, region):
    """Create a binary mask [1, 1, H, W] with 1 in the central region."""
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
# Charbonnier loss (smooth L1 approximation)
# ============================================================

def charbonnier_loss(pred, target, eps=1e-3):
    """Charbonnier loss: ρ(x) = sqrt(x² + ε²).
    Smooth approximation of L1, less sensitive to outliers than MSE.
    """
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()


# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def compute_region_metrics(pred, target, center_size=None):
    """Compute full-image and regional metrics."""
    pred_f = pred.float()
    target_f = target.float()
    H, W = pred_f.shape[2], pred_f.shape[3]

    def _mse(a, b):
        return F.mse_loss(a, b).item()

    def _psnr(mse_val):
        return 20 * np.log10(2.0 / np.sqrt(mse_val)) if mse_val > 0 else 100.0

    def _ssim(a, b):
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

    if center_size is not None:
        cp = crop_center(pred_f, center_size)
        ct = crop_center(target_f, center_size)
        results['center_psnr'] = _psnr(_mse(cp, ct))
    else:
        results['center_psnr'] = 0.0

    corner_size = 64
    if min(H, W) >= corner_size * 2:
        corners_p = crop_corners(pred_f, corner_size)
        corners_t = crop_corners(target_f, corner_size)
        corner_psnr_vals = [_psnr(_mse(cp, ct)) for cp, ct in zip(corners_p, corners_t)]
        results['corner_psnr'] = float(np.mean(corner_psnr_vals))
    else:
        results['corner_psnr'] = 0.0

    return results


# ============================================================
# Perceptual loss factory
# ============================================================

def build_perceptual_loss(loss_type='vgg', device='cuda'):
    """Build perceptual loss module.

    Args:
        loss_type: 'vgg' (built-in VGG19) or 'lpips' (requires pip install lpips)
        device: torch device

    Returns:
        callable: loss_fn(pred, target) → scalar loss
    """
    if loss_type == 'vgg':
        from model.losses import VGGLoss
        vgg = VGGLoss().to(device)
        return vgg
    elif loss_type == 'lpips':
        try:
            import lpips
            vgg_lpips = lpips.LPIPS(net='vgg').to(device)
            # LPIPS expects [-1, 1] input by default
            return lambda pred, target: vgg_lpips(pred, target).mean()
        except ImportError:
            print("WARNING: lpips not installed, falling back to VGGLoss")
            from model.losses import VGGLoss
            return VGGLoss().to(device)
    else:
        raise ValueError(f"Unknown perceptual_type: {loss_type}")


# ============================================================
# IB-Focus Trainer
# ============================================================

class IBFocusTrainer:
    """Iterative refinement trainer with Information Bottleneck.

    Multi-step refinement with:
      - Charbonnier loss on focus region (each step)
      - VGG perceptual loss on final output (last step only)
      - IB sparsity regularization (encourages edge compression)
      - Full-image Charbonnier consistency (last step only, low weight)

    Usage:
        trainer = IBFocusTrainer(cfg, device, output_dir)
        trainer.run()
        trainer.run(resume_ckpt='best_model.pt', start_epoch=100)
    """

    def __init__(self, cfg, device, output_dir):
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        self.exp_name = cfg.experiment.name

        # Image size
        self.H = cfg.data.hr_size[0]
        self.W = cfg.data.hr_size[1]

        # Iterative settings
        self.num_steps = cfg.training.get('num_steps', 5)
        self.center_regions = self._resolve_center_regions()
        self.teacher_noise_std = cfg.training.get('teacher_noise_std', 0.0)
        self.use_residual = cfg.training.get('use_residual', True)

        # Loss weights
        self.char_eps = cfg.training.get('charbonnier_eps', 1e-3)
        self.perceptual_type = cfg.training.get('perceptual_type', 'vgg')
        self.perceptual_weight = cfg.training.get('perceptual_weight', 0.01)
        self.sparse_weight = cfg.training.get('sparse_weight', 1e-6)
        self.full_image_weight = cfg.training.get('full_image_weight', 0.1)
        self.grad_clip = cfg.training.get('grad_clip', 1.0)

        # Model and data placeholders
        self.model = None
        self.model_type = cfg.model.type
        self.train_images = []
        self.val_images = []
        self._center_masks = None
        self._perceptual_loss = None

        # Loss breakdown tracking (reset each epoch)
        self._loss_breakdown = {}

    def _resolve_center_regions(self):
        """Resolve center region sizes for each refinement step."""
        cfg_regions = self.cfg.training.get('center_regions', None)
        if cfg_regions:
            result = []
            for r in cfg_regions:
                if isinstance(r, list):
                    result.append((int(r[0]), int(r[1])))
                else:
                    result.append((int(r), int(r)))
            return result

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
    # Data loading (same as FocusRefineTrainer)
    # ============================================================

    def _load_data(self):
        """Load image pairs, random-crop to H×W, compute I_low_up."""
        data_dir = self.cfg.data.data_dir
        scale = self.cfg.data.scale
        n_train = self.cfg.data.n_train
        n_val = self.cfg.data.n_val

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
            lr_full, hr_full = all_pairs[i]
            full_H_native, full_W_native = hr_full.shape[2], hr_full.shape[3]

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

        self.train_images = self.train_images[:n_train]

        # Build center masks
        self._center_masks = []
        for region in self.center_regions:
            mask = create_center_mask(self.H, self.W, region).to(self.device)
            self._center_masks.append(mask)

        print(f"Loaded {len(self.train_images)} train + {len(self.val_images)} val "
              f"crops ({self.H}×{self.W}), {self.num_steps} steps")
        print(f"Center regions: {self.center_regions}")

    def _is_realsr_format(self, data_dir):
        import os as _os
        files = _os.listdir(data_dir)
        has_hr = any('_HR' in f for f in files)
        has_lr = any('_LR' in f for f in files)
        return has_hr and has_lr

    @staticmethod
    def _load_realsr_pairs(data_dir):
        import os as _os
        files = sorted(_os.listdir(data_dir))
        hr_files = [f for f in files if '_HR' in f and f.endswith('.png')]
        result = []

        for hr_f in hr_files:
            base = hr_f.replace('_HR.png', '')
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
        print(f"Model: {n_params:,} params")
        return n_params

    def _load_checkpoint(self, ckpt_path):
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        val_m = self._validate()
        print(f"Resumed from {ckpt_path}, val_mse={val_m['mse']:.6f}, "
              f"psnr={val_m['psnr']:.1f}dB (AR)")
        return val_m['mse']

    # ============================================================
    # Training step
    # ============================================================

    def _train_step(self, I_low_up, I_gt, optimizer):
        """One Teacher Forcing training iteration with IB losses.

        Per-step: Charbonnier (focus region) + IB sparsity
        Final step: + VGG perceptual + full-image Charbonnier consistency

        Per-step backward (not gradient accumulation) to avoid OOM.
        """
        current = I_low_up.clone()
        step_losses = []
        breakdown = {'char': 0.0, 'sparse': 0.0, 'perc': 0.0, 'full': 0.0}

        for t in range(self.num_steps):
            mask = self._center_masks[t]
            region_h, region_w = self.center_regions[t]
            t_norm = torch.tensor([t / max(self.num_steps - 1, 1)], device=self.device)

            # Build input: concat(current, mask)
            model_input = torch.cat([current, mask.expand(1, 1, self.H, self.W)], dim=1)

            # Forward
            residual = self.model(model_input, t_norm)
            refined = current + residual if self.use_residual else residual

            # Charbonnier loss on focus region
            if t == 0 or (region_h >= self.H and region_w >= self.W):
                loss_char = charbonnier_loss(refined, I_gt, self.char_eps)
            else:
                pred_crop = crop_center(refined, (region_h, region_w))
                gt_crop = crop_center(I_gt, (region_h, region_w))
                loss_char = charbonnier_loss(pred_crop, gt_crop, self.char_eps)

            loss = loss_char
            breakdown['char'] += loss_char.item()

            # IB sparsity: L1 on (1-M) → encourage edge compression
            loss_sparse = self.model.ib_sparsity_loss()
            loss = loss + self.sparse_weight * loss_sparse
            breakdown['sparse'] += (self.sparse_weight * loss_sparse.item())

            # Final step: perceptual + full-image consistency
            if t == self.num_steps - 1:
                loss_perc = self._perceptual_loss(refined, I_gt)
                loss = loss + self.perceptual_weight * loss_perc
                breakdown['perc'] += (self.perceptual_weight * loss_perc.item())

                loss_full = charbonnier_loss(refined, I_gt, self.char_eps)
                loss = loss + self.full_image_weight * loss_full
                breakdown['full'] += (self.full_image_weight * loss_full.item())

            # Per-step backward
            optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            optimizer.step()

            step_losses.append(loss.item())

            # Teacher Forcing: current = I_gt (with optional noise)
            new_current = I_gt.clone()
            if self.teacher_noise_std > 0:
                next_center = crop_center(I_gt, (region_h, region_w))
                next_center = next_center + torch.randn_like(next_center) * self.teacher_noise_std
                start_h = (self.H - region_h) // 2
                start_w = (self.W - region_w) // 2
                new_current[:, :, start_h:start_h + region_h, start_w:start_w + region_w] = next_center
            current = new_current

        return sum(step_losses), step_losses, breakdown

    # ============================================================
    # Inference & Validation
    # ============================================================

    @torch.no_grad()
    def _infer(self, I_low_up):
        """Autoregressive inference (no Teacher Forcing)."""
        current = I_low_up.clone()
        steps = [current.clone()]
        for t in range(self.num_steps):
            mask = self._center_masks[t]
            t_norm = torch.tensor([t / max(self.num_steps - 1, 1)], device=self.device)
            model_input = torch.cat([current, mask.expand(1, 1, self.H, self.W)], dim=1)
            residual = self.model(model_input, t_norm)
            current = current + residual if self.use_residual else residual
            steps.append(current.clone())
        return current, steps

    @torch.no_grad()
    def _validate(self):
        """Run validation: inference + full/center/corner metrics."""
        self.model.eval()

        agg = {'mse': 0.0, 'psnr': 0.0, 'ssim': 0.0, 'ncc': 0.0,
               'center_psnr': 0.0, 'corner_psnr': 0.0}
        center_metric_size = self.center_regions[-1]

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
        """Save validation outputs: final image + progression strip."""
        self.model.eval()
        I_low_up, I_gt = self.val_images[0]

        sr, steps = self._infer(I_low_up)

        to_pil(sr).save(os.path.join(self.output_dir, f"e{epoch:04d}.png"))

        if len(steps) >= 2:
            row_imgs = []
            n_steps = len(steps)
            if n_steps <= 6:
                indices = list(range(n_steps))
            else:
                indices = [0]
                if n_steps > 4:
                    indices.append(n_steps // 2)
                if n_steps > 3:
                    indices.append(n_steps - 2)
                indices.append(n_steps - 1)

            for idx in indices:
                img = to_pil(steps[idx])
                img = img.resize((256, 256), Image.LANCZOS)
                row_imgs.append(img)
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

        # Build perceptual loss
        self._perceptual_loss = build_perceptual_loss(self.perceptual_type, self.device)

        if resume_ckpt:
            best_val_mse = self._load_checkpoint(resume_ckpt)
            best_epoch = start_epoch
        else:
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
                center_size=self.center_regions[-1])
            print(f"Bicubic baseline: PSNR={bicubic_m['psnr']:.2f}dB, "
                  f"center_PSNR={bicubic_m['center_psnr']:.2f}dB, "
                  f"corner_PSNR={bicubic_m['corner_psnr']:.2f}dB")

        tag = f" (resume from {start_epoch})" if start_epoch > 0 else ""
        print(f"\n{'=' * 60}")
        print(f"Experiment: {self.exp_name}{tag}")
        print(f"Model: {self.model_type}  |  LR={lr} (cosine)  |  Epochs={epochs}")
        print(f"Resolution: {self.H}×{self.W}  |  Steps: {self.num_steps}")
        print(f"Regions: {self.center_regions}")
        print(f"Loss: Charbonnier(eps={self.char_eps}) + "
              f"{self.perceptual_weight}*{self.perceptual_type} + "
              f"{self.sparse_weight}*L1(1-M) + "
              f"{self.full_image_weight}*full_img")
        print(f"Grad clip: {self.grad_clip}  |  Teacher noise: {self.teacher_noise_std}")
        print(f"Train: {len(self.train_images)}  |  Val: {len(self.val_images)}")
        print(f"Output: {self.output_dir}")
        print(f"{'=' * 60}")

        pbar = tqdm(range(1, epochs + 1), desc=f"[{self.exp_name}]", unit="ep")
        for epoch in pbar:
            global_epoch = start_epoch + epoch
            self.model.train()

            epoch_loss = 0.0
            step_agg = [0.0] * self.num_steps
            breakdown_agg = {'char': 0.0, 'sparse': 0.0, 'perc': 0.0, 'full': 0.0}

            indices = list(range(len(self.train_images)))
            random.shuffle(indices)

            for idx in indices:
                I_low_cpu, I_gt_cpu = self.train_images[idx]
                I_low_up = I_low_cpu.to(self.device)
                I_gt = I_gt_cpu.to(self.device)

                loss_val, step_losses, bd = self._train_step(I_low_up, I_gt, optimizer)

                epoch_loss += loss_val
                for s in range(len(step_losses)):
                    step_agg[s] += step_losses[s]
                for k in breakdown_agg:
                    breakdown_agg[k] += bd[k]

            epoch_loss /= len(self.train_images)
            step_agg = [s / len(self.train_images) for s in step_agg]
            for k in breakdown_agg:
                breakdown_agg[k] /= len(self.train_images)
            train_losses.append(epoch_loss)
            scheduler.step()

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

                pbar.set_postfix(dict(
                    train=f"{epoch_loss:.6f}",
                    val_mse=f"{val_m['mse']:.6f}",
                    val_psnr=f"{val_m['psnr']:.1f}",
                    ctr_psnr=f"{val_m['center_psnr']:.1f}",
                    ch=f"{breakdown_agg['char']:.4f}",
                    perc=f"{breakdown_agg['perc']:.5f}",
                    best_ep=str(best_epoch),
                ))
            else:
                pbar.set_postfix(dict(
                    train=f"{epoch_loss:.6f}",
                    ch=f"{breakdown_agg['char']:.4f}",
                ))

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
