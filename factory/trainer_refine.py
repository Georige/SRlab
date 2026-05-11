"""RefinementTrainer — iterative refinement training loop with center-first schedule.

Route B: The model works at a SINGLE resolution (1024x2048) throughout.
Each refinement step improves the central region's quality, guided by a time
embedding that tells the model which step it's at. The refinement schedule
grows from center outward, but resolution stays constant.

Key difference from CenterGrowing: all steps work at 1024x2048 (no spatial
growing of output). Time embedding makes each step's processing distinct.
The process is "denoising from center outward" rather than "generating outward".
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
# Helpers
# ============================================================

def to_pil(t):
    """Convert [-1,1] tensor to PIL Image."""
    a = t[0].cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.clip((a + 1) * 127.5, 0, 255).astype(np.uint8))


def crop_center_rect(x, w, h):
    """Extract central wxh region from [B,C,H,W] tensor."""
    H, W = x.shape[2], x.shape[3]
    start_h = (H - h) // 2
    start_w = (W - w) // 2
    return x[:, :, start_h:start_h + h, start_w:start_w + w]


def place_center(canvas, patch):
    """Place patch at the center of a tensor (in-place)."""
    _, _, Hc, Wc = canvas.shape
    _, _, Hp, Wp = patch.shape
    start_h = (Hc - Hp) // 2
    start_w = (Wc - Wp) // 2
    canvas[:, :, start_h:start_h + Hp, start_w:start_w + Wp] = patch


@torch.no_grad()
def compute_metrics(pred, target):
    """MSE, PSNR, SSIM, NCC for [B,3,H,W] tensors in [-1,1]."""
    pred_f = pred.float()
    target_f = target.float()
    mse = F.mse_loss(pred_f, target_f).item()
    psnr = 20 * np.log10(2.0 / np.sqrt(mse)) if mse > 0 else 100.0

    ncc_vals = []
    for c in range(3):
        a = pred_f[:, c]; b = target_f[:, c]
        a_m, b_m = a.mean(), b.mean()
        a_s, b_s = a.std(), b.std()
        if a_s < 1e-8 or b_s < 1e-8:
            ncc_vals.append(0.0)
        else:
            ncc_vals.append(((a - a_m) * (b - b_m)).mean().item() / (a_s * b_s).item())
    ncc = float(np.mean(ncc_vals))

    ssim_vals = []
    for c in range(3):
        a = pred_f[0:1, c:c + 1]; b = target_f[0:1, c:c + 1]
        mu_a = F.avg_pool2d(a, 11, stride=1, padding=5)
        mu_b = F.avg_pool2d(b, 11, stride=1, padding=5)
        sigma_a = F.avg_pool2d((a - mu_a) ** 2, 11, stride=1, padding=5).sqrt()
        sigma_b = F.avg_pool2d((b - mu_b) ** 2, 11, stride=1, padding=5).sqrt()
        sigma_ab = F.avg_pool2d((a - mu_a) * (b - mu_b), 11, stride=1, padding=5)
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)) / \
                   ((mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a ** 2 + sigma_b ** 2 + C2) + 1e-8)
        ssim_vals.append(ssim_map.mean().item())
    ssim = float(np.mean(ssim_vals))

    return {"mse": mse, "psnr": psnr, "ncc": ncc, "ssim": ssim}


# ============================================================
# Trainer
# ============================================================

class RefinementTrainer:
    """Iterative refinement trainer with center-first schedule.

    All refinement steps work at the SAME resolution (1024x2048).
    The model uses a time embedding to know which step it's at.
    Each step refines the annular region between current and next scale.

    Usage:
        trainer = RefinementTrainer(cfg, device, output_dir)
        trainer.run()
    """

    def __init__(self, cfg, device, output_dir):
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        self.exp_name = cfg.experiment.name
        self.H = cfg.data.hr_size[0]
        self.W = cfg.data.hr_size[1]

        # Auto-compute scale sequence from image dimensions
        n_scales = cfg.training.get('n_scales', 0)
        self.scales = self._compute_scales(self.H, self.W, n_scales)

        self.use_residual = cfg.training.get('use_residual', True)

        # === Anti-overfitting settings ===
        # Method 1: step-dependent TF noise
        self.tf_noise_base = cfg.training.get('teacher_noise_base', 0.02)
        self.tf_noise_scale = cfg.training.get('teacher_noise_scale', 0.05)
        # Method 2: random time embedding dropout
        self.time_dropout = cfg.training.get('time_dropout', 0.0)
        # Method 3: data augmentation
        self.use_augment = cfg.training.get('augmentation', False)

        # === Diagnostic options ===
        # Random mask shift: shift center mask to non-center positions (p=0.3)
        self.random_mask_shift = cfg.training.get('random_mask_shift', 0.0)
        # Gradual warmup: freeze bottleneck for first N epochs
        self.warmup_epochs = cfg.training.get('warmup_epochs', 0)

        self.model = None
        self.train_images = []
        self.val_images = []

    def _compute_scales(self, H, W, n_scales=0):
        """Build scale sequence maintaining aspect ratio from tiny to full.

        Args:
            n_scales: if > 0, use exactly this many scales; else auto-compute.
        """
        aspect = W / H
        min_h = 4
        min_w = max(2, int(min_h * aspect))
        scales = []
        for i in range(12):
            h = min(min_h * (2 ** i), H)
            w = min(int(h * aspect), W)
            scales.append((w, h))
            if w >= W and h >= H:
                break
        scales[-1] = (W, H)
        if n_scales > 0 and n_scales < len(scales):
            indices = [int(i * (len(scales) - 1) / (n_scales - 1)) for i in range(n_scales)]
            scales = [scales[i] for i in indices]
            scales[-1] = (W, H)
        return scales

    # ============================================================
    # Data loading (same as Phase 7)
    # ============================================================

    def _load_data(self):
        from utility.data import PanoramaDataset

        data_dir = self.cfg.data.data_dir
        scale = self.cfg.data.scale
        n_train = self.cfg.data.get('n_train', 1)
        n_val = self.cfg.data.get('n_val', 1)
        image_index = self.cfg.data.get('image_index', 0)

        dataset = PanoramaDataset(data_dir, (self.H, self.W), scale)
        total = min(n_train + n_val, len(dataset))
        train_items = []

        for i in range(image_index, image_index + total):
            lr_img, hr_img = dataset[i]
            lr_img = lr_img.unsqueeze(0)
            hr_img = hr_img.unsqueeze(0)
            I_low_up = F.interpolate(lr_img, size=(self.H, self.W),
                                     mode='bicubic', align_corners=False)
            train_items.append((I_low_up, hr_img))

        self.val_images = [(t[0].to(self.device), t[1].to(self.device))
                           for t in train_items[n_train:n_train + n_val]]
        train_items = train_items[:n_train]
        self.train_images = [(t[0], t[1]) for t in train_items]

        # If no val images (single-image case), use first train image for val
        if len(self.val_images) == 0 and len(self.train_images) > 0:
            self.val_images = [(self.train_images[0][0].to(self.device),
                                self.train_images[0][1].to(self.device))]

        print(f"Loaded {len(train_items)} train + {len(self.val_images)} val images "
              f"({self.H}x{self.W})")

    # ============================================================
    # Model
    # ============================================================

    def _build_model(self):
        self.model = MODEL_REGISTRY[self.cfg.model.type](self.cfg, self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        n_steps = len(self.scales) - 1
        print(f"Model: {n_params:,} params, {n_steps} refinement steps")
        return n_params

    def _load_checkpoint(self, ckpt_path):
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        val_m = self._validate()
        print(f"Resumed from {ckpt_path}, val_mse={val_m['mse']:.6f}, "
              f"psnr={val_m['psnr']:.2f}dB")
        return val_m['mse']

    # ============================================================
    # Input construction (same resolution, mask indicates clean region)
    # ============================================================

    def _build_input(self, I_low_up, current_state, w_clean, h_clean, training=False):
        """Build the 7-channel model input for a refinement step.

        Args:
            I_low_up:      [1, 3, H, W] bicubic-upsampled low-res
            current_state: [1, 3, H, W] current refined image
            w_clean, h_clean: size of the currently-clean central region
            training: if True and random_mask_shift > 0, may shift the mask

        Returns:
            model_input: [1, 7, H, W]
        """
        mask = torch.zeros(1, 1, self.H, self.W, device=I_low_up.device)
        if w_clean > 0 and h_clean > 0:
            # Diagnostic: randomly shift mask away from center
            if training and self.random_mask_shift > 0 and random.random() < self.random_mask_shift:
                # Shift to a random position within the image bounds
                max_dh = self.H - h_clean
                max_dw = self.W - w_clean
                start_h = random.randint(0, max(0, max_dh))
                start_w = random.randint(0, max(0, max_dw))
            else:
                start_h = (self.H - h_clean) // 2
                start_w = (self.W - w_clean) // 2
            mask[:, :, start_h:start_h + h_clean,
                 start_w:start_w + w_clean] = 1.0

        return torch.cat([I_low_up, current_state, mask], dim=1)

    # ============================================================
    # Data augmentation (method 3)
    # ============================================================

    def _augment_pair(self, I_low_up, I_gt):
        """Apply random geometric + photometric augmentations to image pair.

        Both images receive identical transforms to maintain alignment.
        Applied per-epoch in training loop.
        """
        # Random horizontal flip (50%)
        if random.random() < 0.5:
            I_low_up = I_low_up.flip(-1)
            I_gt = I_gt.flip(-1)

        # Random vertical flip (50%)
        if random.random() < 0.5:
            I_low_up = I_low_up.flip(-2)
            I_gt = I_gt.flip(-2)

        # Random rotation: 0 / 90 / 180 / 270 (25% each)
        k = random.randint(0, 3)
        if k > 0:
            I_low_up = I_low_up.rot90(k, [2, 3])
            I_gt = I_gt.rot90(k, [2, 3])

        # Random brightness (±10%)
        if random.random() < 0.5:
            delta = random.uniform(-0.1, 0.1)
            I_low_up = I_low_up + delta
            I_gt = I_gt + delta

        # Random contrast (±10%)
        if random.random() < 0.5:
            factor = random.uniform(0.9, 1.1)
            mean_low = I_low_up.mean(dim=[2, 3], keepdim=True)
            mean_gt = I_gt.mean(dim=[2, 3], keepdim=True)
            I_low_up = (I_low_up - mean_low) * factor + mean_low
            I_gt = (I_gt - mean_gt) * factor + mean_gt

        return I_low_up, I_gt

    # ============================================================
    # Validation (autoregressive iterative refinement)
    # ============================================================

    @torch.no_grad()
    def _validate(self):
        """Run full iterative refinement on val images.

        All steps at 1024x2048. Each step refines based on the time embedding.
        """
        self.model.eval()
        all_metrics = {"mse": 0.0, "psnr": 0.0, "ncc": 0.0, "ssim": 0.0}
        T = len(self.scales) - 1

        for I_low_up, I_gt in self.val_images:
            current_state = I_low_up.clone()
            w_clean, h_clean = 0, 0

            for step in range(T):
                w_next, h_next = self.scales[step + 1]

                # Build input with current clean mask
                model_input = self._build_input(
                    I_low_up, current_state, w_clean, h_clean)
                t_norm = torch.tensor([step / T], device=self.device)

                # Forward: model predicts refinement residual
                residual = self.model(model_input, t_norm)

                if self.use_residual:
                    # Apply residual to refine current state
                    current_state = current_state + residual
                else:
                    current_state = residual

                # Advance clean region
                w_clean, h_clean = w_next, h_next

            m = compute_metrics(current_state, I_gt)
            for k in all_metrics:
                all_metrics[k] += m[k]

        for k in all_metrics:
            all_metrics[k] /= max(len(self.val_images), 1)
        self.model.train()
        return all_metrics

    @torch.no_grad()
    def _save_progression(self, epoch):
        """Save final output + progression strip."""
        self.model.eval()
        I_low_up, I_gt = self.val_images[0]
        T = len(self.scales) - 1

        current_state = I_low_up.clone()
        w_clean, h_clean = 0, 0
        steps = []

        for step in range(T):
            w_next, h_next = self.scales[step + 1]

            model_input = self._build_input(
                I_low_up, current_state, w_clean, h_clean)
            t_norm = torch.tensor([step / T], device=self.device)

            residual = self.model(model_input, t_norm)
            if self.use_residual:
                current_state = current_state + residual
            else:
                current_state = residual

            w_clean, h_clean = w_next, h_next
            steps.append(current_state.clone())

        # Save final output
        to_pil(steps[-1]).save(os.path.join(self.output_dir, f"e{epoch:04d}.png"))

        # Save progression strip
        key_indices = [0, 2, 4, 6, 8]
        row = []
        for idx in key_indices:
            if idx < len(steps):
                img = to_pil(steps[idx])
                img = img.resize((1024, 512), Image.LANCZOS)
                row.append(img)
        if row:
            w_img, h_img = row[0].size
            strip = Image.new('RGB', (w_img * len(row), h_img))
            for j, img in enumerate(row):
                strip.paste(img, (j * w_img, 0))
            strip.save(os.path.join(self.output_dir, f"prog_{epoch:04d}.png"))

        self.model.train()

    # ============================================================
    # Training
    # ============================================================

    def run(self, epochs=None, resume_ckpt=None, start_epoch=0):
        epochs = epochs or self.cfg.training.epochs
        lr = self.cfg.training.lr

        self._load_data()
        self._build_model()

        # Config-specified checkpoint
        config_ckpt = self.cfg.training.get('resume_ckpt', None)
        if config_ckpt and not resume_ckpt:
            resume_ckpt = config_ckpt
            if start_epoch == 0:
                start_epoch = self.cfg.training.get('start_epoch', 200)

        if resume_ckpt:
            best_val_mse = self._load_checkpoint(resume_ckpt)
            best_epoch = start_epoch
        else:
            ref_low, ref_gt = self.val_images[0]
            to_pil(ref_low).save(os.path.join(self.output_dir, "val_bicubic.png"))
            to_pil(ref_gt).save(os.path.join(self.output_dir, "val_gt.png"))
            best_val_mse = float('inf')
            best_epoch = 0

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_losses, val_epochs, val_mses = [], [], []
        val_every = max(20, epochs // 20)
        T = len(self.scales) - 1  # 9 refinement steps

        tag = f" (resume from {start_epoch})" if start_epoch > 0 else ""
        print(f"\n{'=' * 60}")
        print(f"Experiment: {self.exp_name}{tag}")
        print(f"Model: {self.cfg.model.type}  |  LR={lr} (cosine)  |  Epochs={epochs}")
        print(f"Resolution: {self.H}x{self.W}  |  Steps: {T}  |  Scales: {self.scales}")
        print(f"Residual: {self.use_residual}  |  TF noise: {self.tf_noise_base}+{self.tf_noise_scale}*t")
        print(f"Time dropout: {self.time_dropout}  |  Augment: {self.use_augment}")
        print(f"Random mask shift: {self.random_mask_shift}  |  Warmup: {self.warmup_epochs}")
        print(f"Train images: {len(self.train_images)}")
        print(f"Output: {self.output_dir}")
        print(f"{'=' * 60}")

        # Gradual warmup: freeze bottleneck parameters
        bottleneck_frozen = False
        if self.warmup_epochs > 0:
            for name, param in self.model.named_parameters():
                if 'bottleneck' in name:
                    param.requires_grad = False
            bottleneck_frozen = True
            print(f"Bottleneck frozen for first {self.warmup_epochs} epochs")

        pbar = tqdm(range(1, epochs + 1), desc=f"[{self.exp_name}]", unit="ep")
        for epoch in pbar:
            global_epoch = start_epoch + epoch

            # Unfreeze bottleneck after warmup
            if bottleneck_frozen and global_epoch > self.warmup_epochs:
                for param in self.model.parameters():
                    param.requires_grad = True
                bottleneck_frozen = False
                print(f"\n  Bottleneck unfrozen at epoch {global_epoch}")

            self.model.train()

            epoch_loss = 0.0
            n_steps = 0

            for I_low_cpu, I_gt_cpu in self.train_images:
                I_low_up = I_low_cpu.to(self.device)
                I_gt = I_gt_cpu.to(self.device)

                # === Method 3: data augmentation (random flips/rotations/brightness) ===
                if self.use_augment:
                    I_low_up, I_gt = self._augment_pair(I_low_up, I_gt)

                # current_state tracks the "refined so far" image (Teacher Forcing)
                current_state = I_low_up.clone()
                w_clean, h_clean = 0, 0

                for step in range(T):
                    w_curr, h_curr = self.scales[step]
                    w_next, h_next = self.scales[step + 1]

                    # Build input (training=True enables random_mask_shift diagnostics)
                    model_input = self._build_input(
                        I_low_up, current_state, w_curr, h_curr, training=True)

                    # === Method 2: random time embedding dropout (30%) ===
                    t_val = step / T
                    if self.time_dropout > 0 and random.random() < self.time_dropout:
                        t_val_input = 0.0
                    else:
                        t_val_input = t_val
                    t_norm = torch.tensor([t_val_input], device=self.device)

                    # Forward
                    residual = self.model(model_input, t_norm)

                    if self.use_residual:
                        refined = current_state + residual
                    else:
                        refined = residual

                    # Loss: on the annular region (next clean region)
                    pred_crop = crop_center_rect(refined, w_next, h_next)
                    gt_crop = crop_center_rect(I_gt, w_next, h_next)
                    loss = F.mse_loss(pred_crop, gt_crop)

                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_steps += 1

                    # === Method 1: step-dependent Teacher Forcing noise ===
                    current_state = I_low_up.clone()
                    gt_clean = crop_center_rect(I_gt, w_next, h_next)
                    noise_std = self.tf_noise_base + self.tf_noise_scale * t_val
                    if noise_std > 0:
                        gt_clean = gt_clean + torch.randn_like(gt_clean) * noise_std
                    place_center(current_state, gt_clean)

                    w_clean, h_clean = w_next, h_next

            epoch_loss /= max(n_steps, 1)
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

                pbar.set_postfix(
                    train=f"{epoch_loss:.6f}",
                    val_mse=f"{val_m['mse']:.6f}",
                    val_psnr=f"{val_m['psnr']:.1f}",
                    val_ssim=f"{val_m['ssim']:.4f}",
                    best_ep=str(best_epoch),
                )
            else:
                pbar.set_postfix(train=f"{epoch_loss:.6f}")

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
        print(f"\n  Best: epoch {best_epoch}, MSE={best['mse']:.6f}, "
              f"PSNR={best['psnr']:.2f}dB, SSIM={best['ssim']:.4f}, NCC={best['ncc']:.4f}")
        print(f"{'=' * 60}")

        return best
