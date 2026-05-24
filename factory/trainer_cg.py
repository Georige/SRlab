"""CenterGrowingTrainer — progressive central growing super-resolution training loop.

Phase 7: dual-anchor + Teacher Forcing + center-loss in pixel space.
Target resolution: 1024×2048 (H×W), low-res input: 512×1024 → bicubic↑ to 1024×2048.

Scales (W,H) with 2:1 aspect ratio:
  (6,3) → (12,6) → ... → (1536,768) → (2048,1024)
Each step predicts the full image, loss only on the central «next» rectangular region.
"""

import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from factory.registry import MODEL_REGISTRY
from utility.metrics import compute_metrics
from vit.overfit_plot import update_curves


# ============================================================
# Helpers
# ============================================================

def to_pil(t):
    """Convert [-1,1] tensor to PIL Image."""
    a = t[0].cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.clip((a + 1) * 127.5, 0, 255).astype(np.uint8))


def crop_center_rect(x, w, h):
    """Extract central w×h region from [B,C,H,W] tensor."""
    H, W = x.shape[2], x.shape[3]
    start_h = (H - h) // 2
    start_w = (W - w) // 2
    return x[:, :, start_h:start_h + h, start_w:start_w + w]


def place_center(canvas, patch):
    """Place patch at the center of a zero-initialized canvas (in-place)."""
    _, _, Hc, Wc = canvas.shape
    _, _, Hp, Wp = patch.shape
    start_h = (Hc - Hp) // 2
    start_w = (Wc - Wp) // 2
    canvas[:, :, start_h:start_h + Hp, start_w:start_w + Wp] = patch


# ============================================================
# Trainer
# ============================================================

class CenterGrowingTrainer:
    """Progressive central growing trainer with dual-anchor conditioning.

    Scales (W,H) grow exponentially maintaining 2:1 aspect ratio:
      (6,3) → (12,6) → ... → (1536,768) → (2048,1024)

    Each step predicts the full 1024×2048 image, loss only on the
    central next_w×next_h rectangular region.

    Usage:
        trainer = CenterGrowingTrainer(cfg, device, output_dir)
        trainer.run()
        trainer.run(resume_ckpt='best_model.pt', start_epoch=400)
    """

    def __init__(self, cfg, device, output_dir):
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        self.exp_name = cfg.experiment.name
        self.H = cfg.data.hr_size[0]   # 1024
        self.W = cfg.data.hr_size[1]   # 2048

        # Scales: central rectangle (W, H), maintaining 2:1 aspect ratio
        # Configurable via cfg.training.scales (list of [W,H] pairs), otherwise default 9-step
        default_scales = [(6, 3), (12, 6), (24, 12), (48, 24), (96, 48),
                          (192, 96), (384, 192), (768, 384), (1536, 768),
                          (2048, 1024)]
        cfg_scales = cfg.training.get('scales', None)
        if cfg_scales:
            self.scales = [(w, h) for w, h in cfg_scales]
        else:
            self.scales = default_scales

        self.use_residual = cfg.training.get('use_residual', True)
        self.teacher_noise_std = cfg.training.get('teacher_noise_std', 0.0)
        self.val_tf = cfg.training.get('val_tf', False)  # Teacher Forcing during validation

        # Scheduled Sampling: train with self-generated content to combat AR↔TF gap
        self.scheduled_sampling = cfg.training.get('scheduled_sampling', False)
        self.ss_max_prob = cfg.training.get('ss_max_prob', 0.5)

        self.model = None
        self.train_images = []   # list of (I_low_up [1,3,H,W], I_gt [1,3,H,W])
        self.val_images = []     # same structure, on device

    # ============================================================
    # Data loading
    # ============================================================

    def _load_data(self):
        """Load image pairs: low-res 512×1024 → bicubic↑ → 1024×2048, GT 1024×2048.

        Phase 7 Stage 0: single-image overfitting (n_train=1).
        Uses PanoramaDataset with scale=2: loads X2 LR (1024×512 → resized to 512×1024).
        """
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
            lr_img = lr_img.unsqueeze(0)   # [1, 3, 512, 1024]
            hr_img = hr_img.unsqueeze(0)   # [1, 3, 1024, 2048]

            # Upsample low-res to match HR spatial size
            I_low_up = F.interpolate(lr_img, size=(self.H, self.W),
                                     mode='bicubic', align_corners=False)
            train_items.append((I_low_up, hr_img))

        # Slice val BEFORE truncating train_items
        self.val_images = [(t[0].to(self.device), t[1].to(self.device))
                           for t in train_items[n_train:n_train + n_val]]
        train_items = train_items[:n_train]
        # Keep training images on CPU, move to GPU per-iteration
        self.train_images = [(t[0], t[1]) for t in train_items]

        print(f"Loaded {len(train_items)} train + {len(self.val_images)} val images "
              f"({self.H}×{self.W})")

    # ============================================================
    # Model
    # ============================================================

    def _build_model(self):
        self.model = MODEL_REGISTRY[self.cfg.model.type](self.cfg, self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        n_steps = len(self.scales) - 1
        print(f"Model: {n_params:,} params, {n_steps} scale steps")
        return n_params

    def _load_checkpoint(self, ckpt_path):
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        val_m = self._validate(use_tf=self.val_tf)
        print(f"Resumed from {ckpt_path}, val_mse={val_m['mse']:.6f}, "
              f"psnr={val_m['psnr']:.2f}dB")
        return val_m['mse']

    # ============================================================
    # Input construction
    # ============================================================

    def _build_input(self, I_low_up, known_high_res, w, h, prev_w, prev_h):
        """Build the 10-channel model input for a given scale step.

        Args:
            I_low_up:        [1, 3, H, W] bicubic-upsampled low-res
            known_high_res:  [1, 3, prev_h, prev_w] or None (first step)
            w, h:            current scale (width, height)
            prev_w, prev_h:  previous scale (0, 0 for first step)

        Returns:
            model_input: [1, 10, H, W]
        """
        # 1. Local low-res crop at current scale (w, h)
        local_low = torch.zeros_like(I_low_up)
        start_h = (self.H - h) // 2
        start_w = (self.W - w) // 2
        local_low[:, :, start_h:start_h + h, start_w:start_w + w] = \
            I_low_up[:, :, start_h:start_h + h, start_w:start_w + w]

        # 2. Known high-res region
        if known_high_res is None:
            high_res_known = torch.zeros_like(I_low_up)
            mask = torch.zeros(1, 1, self.H, self.W, device=I_low_up.device)
        else:
            high_res_known = torch.zeros_like(I_low_up)
            place_center(high_res_known, known_high_res)
            mask = torch.zeros(1, 1, self.H, self.W, device=I_low_up.device)
            start_h_prev = (self.H - prev_h) // 2
            start_w_prev = (self.W - prev_w) // 2
            mask[:, :, start_h_prev:start_h_prev + prev_h,
                 start_w_prev:start_w_prev + prev_w] = 1.0

        # 3. Concatenate → [1, 10, H, W]
        return torch.cat([I_low_up, local_low, high_res_known, mask], dim=1)

    # ============================================================
    # Validation
    # ============================================================

    @torch.no_grad()
    def _validate(self, use_tf=False):
        """Run full inference loop on val images.

        Args:
            use_tf: If True, use GT crops as known_high_res (Teacher Forcing
                    upper bound). If False, use model predictions (autoregressive).
        Returns final full-image metrics.
        """
        mode_str = "TF" if use_tf else "AR"
        self.model.eval()
        all_metrics = {"mse": 0.0, "psnr": 0.0, "ncc": 0.0, "ssim": 0.0}

        for I_low_up, I_gt in self.val_images:
            known_high_res = None
            prev_w, prev_h = 0, 0
            final_sr = None

            for i in range(len(self.scales) - 1):
                w, h = self.scales[i]
                next_w, next_h = self.scales[i + 1]

                model_input = self._build_input(
                    I_low_up, known_high_res, w, h, prev_w, prev_h)

                residual = self.model(model_input)
                sr = I_low_up + residual if self.use_residual else residual

                if use_tf:
                    # Teacher Forcing: use GT crop as next step's condition
                    known_high_res = crop_center_rect(I_gt, next_w, next_h)
                else:
                    pred_crop = crop_center_rect(sr, next_w, next_h)
                    known_high_res = pred_crop
                prev_w, prev_h = next_w, next_h
                final_sr = sr

            m = compute_metrics(final_sr, I_gt)
            for k in all_metrics:
                all_metrics[k] += m[k]

        for k in all_metrics:
            all_metrics[k] /= max(len(self.val_images), 1)
        self.model.train()
        return all_metrics

    @torch.no_grad()
    def _save_progression(self, epoch, use_tf=False):
        """Save validation output: final image + progression strip."""
        self.model.eval()
        I_low_up, I_gt = self.val_images[0]

        known_high_res = None
        prev_w, prev_h = 0, 0
        steps = []

        for i in range(len(self.scales) - 1):
            w, h = self.scales[i]
            next_w, next_h = self.scales[i + 1]

            model_input = self._build_input(
                I_low_up, known_high_res, w, h, prev_w, prev_h)

            residual = self.model(model_input)
            sr = I_low_up + residual if self.use_residual else residual

            if use_tf:
                known_high_res = crop_center_rect(I_gt, next_w, next_h)
            else:
                pred_crop = crop_center_rect(sr, next_w, next_h)
                known_high_res = pred_crop
            prev_w, prev_h = next_w, next_h
            steps.append(sr)

        # Save final output
        to_pil(steps[-1]).save(os.path.join(self.output_dir, f"e{epoch:04d}.png"))

        # Save progression strip (resize to manageable size for visualization)
        key_indices = [0, 2, 4, 6, 8]  # s: (12,6), (48,24), (192,96), (768,384), (2048,1024)
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

        # Config-specified checkpoint (for diagnostic runs like TF bound test)
        config_ckpt = self.cfg.training.get('resume_ckpt', None)
        if config_ckpt and not resume_ckpt:
            resume_ckpt = config_ckpt
            if start_epoch == 0:
                start_epoch = self.cfg.training.get('start_epoch', 200)

        # Save reference images
        if not resume_ckpt and self.val_images:
            ref_low, ref_gt = self.val_images[0]
            to_pil(ref_low).save(os.path.join(self.output_dir, "val_bicubic.png"))
            to_pil(ref_gt).save(os.path.join(self.output_dir, "val_gt.png"))
            best_val_mse = float('inf')
            best_epoch = 0
        elif resume_ckpt:
            best_val_mse = self._load_checkpoint(resume_ckpt)
            best_epoch = start_epoch
        else:
            best_val_mse = float('inf')
            best_epoch = 0

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_losses, val_epochs, val_mses = [], [], []
        val_every = max(20, epochs // 20)
        n_scale_steps = len(self.scales) - 1  # 9 steps per image

        tag = f" (resume from {start_epoch})" if start_epoch > 0 else ""
        print(f"\n{'=' * 60}")
        print(f"Experiment: {self.exp_name}{tag}")
        print(f"Model: {self.cfg.model.type}  |  LR={lr} (cosine)  |  Epochs={epochs}")
        print(f"Resolution: {self.H}×{self.W}  |  Scales: {self.scales}")
        print(f"Residual: {self.use_residual}  |  Teacher noise: {self.teacher_noise_std}  |  Val TF: {self.val_tf}")
        if self.scheduled_sampling:
            print(f"Scheduled Sampling: ON (max_prob={self.ss_max_prob}, ramp over {epochs} epochs)")
        print(f"Train images: {len(self.train_images)}")
        print(f"Output: {self.output_dir}")
        print(f"{'=' * 60}")

        pbar = tqdm(range(1, epochs + 1), desc=f"[{self.exp_name}]", unit="ep")
        for epoch in pbar:
            global_epoch = start_epoch + epoch
            self.model.train()

            epoch_loss = 0.0
            n_steps = 0

            for I_low_cpu, I_gt_cpu in self.train_images:
                I_low_up = I_low_cpu.to(self.device)
                I_gt = I_gt_cpu.to(self.device)

                known_high_res = None
                prev_w, prev_h = 0, 0

                for i in range(n_scale_steps):
                    w, h = self.scales[i]
                    next_w, next_h = self.scales[i + 1]

                    # ---- Build input ----
                    model_input = self._build_input(
                        I_low_up, known_high_res, w, h, prev_w, prev_h)

                    # ---- Forward ----
                    residual = self.model(model_input)
                    sr = I_low_up + residual if self.use_residual else residual

                    # ---- Center loss on next (w,h) region ----
                    pred_crop = crop_center_rect(sr, next_w, next_h)
                    gt_crop = crop_center_rect(I_gt, next_w, next_h)
                    loss = F.mse_loss(pred_crop, gt_crop)

                    # ---- Backward ----
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_steps += 1

                    # ---- Scheduled Sampling / Teacher Forcing ----
                    if self.scheduled_sampling:
                        p_self = min(self.ss_max_prob,
                                     global_epoch / max(epochs, 1) * self.ss_max_prob)
                        if random.random() < p_self:
                            known_high_res = pred_crop.detach().clone()
                        else:
                            known_high_res = gt_crop.detach()
                    else:
                        known_high_res = gt_crop.detach()
                    if self.teacher_noise_std > 0:
                        known_high_res = known_high_res + \
                            torch.randn_like(known_high_res) * self.teacher_noise_std

                    prev_w, prev_h = next_w, next_h

            epoch_loss /= max(n_steps, 1)
            train_losses.append(epoch_loss)
            scheduler.step()

            # ---- Validation ----
            if epoch == 1 or epoch % val_every == 0 or epoch == epochs:
                val_m = self._validate(use_tf=self.val_tf)
                val_epochs.append(global_epoch)
                val_mses.append(val_m['mse'])

                if val_m['mse'] < best_val_mse:
                    best_val_mse = val_m['mse']
                    best_epoch = global_epoch
                    torch.save(self.model.state_dict(),
                               os.path.join(self.output_dir, "best_model.pt"))

                self._save_progression(global_epoch, use_tf=self.val_tf)

                update_curves(self.exp_name, train_losses, val_epochs, val_mses,
                              log_dir=self.cfg.output.log_dir)

                post = dict(
                    train=f"{epoch_loss:.6f}",
                    val_mse=f"{val_m['mse']:.6f}",
                    val_psnr=f"{val_m['psnr']:.1f}",
                    val_ssim=f"{val_m['ssim']:.4f}",
                    val_tf="TF" if self.val_tf else "AR",
                    best_ep=str(best_epoch),
                )
                if self.scheduled_sampling:
                    p_self = min(self.ss_max_prob,
                                 global_epoch / max(epochs, 1) * self.ss_max_prob)
                    post['ss'] = f"{p_self:.2f}"
                pbar.set_postfix(post)
            else:
                pbar.set_postfix(train=f"{epoch_loss:.6f}")

        # ---- Final summary ----
        print(f"\n{'=' * 60}")
        print(f"Experiment {self.exp_name} Results:")
        print(f"  Final train loss: {train_losses[-1]:.6f}")
        for ep, mse_val in zip(val_epochs, val_mses):
            mkr = " <-- BEST" if ep == best_epoch else ""
            print(f"  Epoch {ep:4d}: MSE={mse_val:.6f}{mkr}")

        self.model.load_state_dict(
            torch.load(os.path.join(self.output_dir, "best_model.pt"),
                       map_location=self.device))
        best = self._validate(use_tf=self.val_tf)
        print(f"\n  Best: epoch {best_epoch}, MSE={best['mse']:.6f}, "
              f"PSNR={best['psnr']:.2f}dB, SSIM={best['ssim']:.4f}, NCC={best['ncc']:.4f}")
        print(f"{'=' * 60}")

        return best
