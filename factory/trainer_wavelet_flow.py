"""Wavelet Flow Matching trainer for panorama super-resolution.

Flow: LR → bicubic↑ → DWT → (LL_lr, H_lr)
      HR → DWT → (LL_hr, H_hr)
      Flow matching: x_t = t * H_hr + (1-t) * H_lr
      Model predicts velocity v = dH/dt conditioned on LL_lr and t.
      Inference: Euler integration → IDWT → SR image.
"""

import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from factory.config import DotDict
from factory.registry import MODEL_REGISTRY
from utility.data import PanoramaDataset
from utility.metrics import compute_metrics
from model.dwt_utils import dwt_high_concat, idwt_from_high_concat
from vit.overfit_plot import update_curves, make_progression


# ============================================================
# Image saving
# ============================================================

def to_pil(t):
    a = t[0].cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.clip((a + 1) * 127.5, 0, 255).astype(np.uint8))


# ============================================================
# Augmentation (simplified for wavelet flow: lr_up + hr)
# ============================================================

def augment_wavelet_flow(lr_up, hr, training=True, scale=4,
                         roll_prob=0.5, crop_prob=0.9,
                         color_prob=0.7, noise_std=0.005, noise_prob=0.3):
    """Panorama-aware augmentation for (lr_up, hr) pair.

    All spatial transforms applied identically to both images.
    HR noise is asymmetric (HR only).

    Phase-aligned cropping: crop coordinates are multiples of scale*2
    to preserve DWT grid alignment. Uses nearest-neighbor resize to
    avoid introducing new pixel values that corrupt wavelet coefficients.
    """
    if not training:
        return lr_up, hr

    B, C, H, W = lr_up.shape
    grid = scale * 2  # alignment grid for DWT: 8 for X4, 4 for X2

    # 1. Horizontal cyclic roll (360° panorama wrapping)
    #    Pure shift, no interpolation — DWT-safe.
    if random.random() < roll_prob:
        shift = random.randint(0, W - 1)
        lr_up = torch.roll(lr_up, shifts=shift, dims=-1)
        hr = torch.roll(hr, shifts=shift, dims=-1)

    # 2. Phase-aligned vertical crop + nearest-neighbor resize
    #    Crop amount is snapped to grid (scale*2) so DWT subbands
    #    remain perfectly aligned. Nearest-neighbor resize preserves
    #    original pixel values — no interpolated artifacts.
    if random.random() < crop_prob:
        crop_frac = random.uniform(0, 0.10)
        crop_h = int(H * crop_frac)
        # Snap to phase-aligned grid
        crop_h = max(0, (crop_h // grid) * grid)
        if crop_h > 0:
            lr_up = lr_up[:, :, crop_h:H - crop_h, :]
            hr = hr[:, :, crop_h:H - crop_h, :]
            lr_up = F.interpolate(lr_up, size=(H, W),
                                  mode='nearest')
            hr = F.interpolate(hr, size=(H, W),
                               mode='nearest')

    # 3. Brightness + contrast jitter (magnitude-only, DWT-safe)
    if random.random() < color_prob:
        brightness = random.uniform(-0.1, 0.1)
        contrast = random.uniform(0.9, 1.1)
        for img in [lr_up, hr]:
            mean = img.mean(dim=(-1, -2), keepdim=True)
            img_new = (img - mean) * contrast + mean + brightness
            img.copy_(img_new.clamp(-1, 1))

    # 4. HR noise (regularization, asymmetric)
    if random.random() < noise_prob:
        noise = torch.randn_like(hr) * noise_std
        hr = (hr + noise).clamp(-1, 1)

    return lr_up, hr


# ============================================================
# Data loading (simplified: no SHT/ISHT)
# ============================================================

class WaveletDataBuilder:
    """Build train/val datasets with bicubic-upsampled LR + HR pairs."""

    def __init__(self, cfg, device):
        self.device = device
        self.H, self.W = cfg.data.hr_size
        self.scale = cfg.data.scale
        self.data_dir = cfg.data.data_dir
        self.n_train = cfg.data.n_train
        self.n_val = cfg.data.n_val

    def build(self):
        dataset = PanoramaDataset(self.data_dir, (self.H, self.W), self.scale)
        n = min(self.n_train + self.n_val, len(dataset))

        train_data, val_data = [], []

        for i in range(n):
            lr_img, hr_img = dataset[i]
            lr_img = lr_img.unsqueeze(0).to(self.device)
            hr_img = hr_img.unsqueeze(0).to(self.device)
            lr_up = F.interpolate(lr_img, size=(self.H, self.W),
                                  mode='bicubic', align_corners=False)

            item = {
                'lr_up': lr_up, 'hr': hr_img, 'idx': i,
            }
            (train_data if i < self.n_train else val_data).append(item)

        return train_data, val_data


# ============================================================
# Metrics (pixel-space, post-IDWT)
# ============================================================

# ============================================================
# Trainer
# ============================================================

class WaveletFlowTrainer:
    """Flow Matching + DWT super-resolution trainer.

    Usage:
        trainer = WaveletFlowTrainer(cfg, device, output_dir)
        trainer.run()
    """

    def __init__(self, cfg, device, output_dir):
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir

        self.exp_name = cfg.experiment.name
        self.scale = cfg.data.scale
        self.infer_steps = getattr(cfg.training, 'infer_steps', 4)

        # Flow loss config
        self.l1_weight = getattr(cfg.training, 'l1_weight', 0.1)

        # Teacher-forcing gap mitigation
        self.tf_noise_std = getattr(cfg.training, 'tf_noise_std', 0.0)
        self.pixel_weight = getattr(cfg.training, 'pixel_weight', 0.0)

        # Augmentation params
        aug = getattr(cfg.training, 'augmentation', {})
        if isinstance(aug, (dict, DotDict)):
            self.aug_kwargs = dict(aug.__dict__) if isinstance(aug, DotDict) else dict(aug)
        else:
            self.aug_kwargs = {}

        self.model = None
        self.train_data = None
        self.val_data = None

    # ---- Data ----

    def _load_data(self):
        builder = WaveletDataBuilder(self.cfg, self.device)
        self.train_data, self.val_data = builder.build()
        print(f"Loaded {len(self.train_data)} train + {len(self.val_data)} val images")

    # ---- Model ----

    def _build_model(self):
        self.model = MODEL_REGISTRY[self.cfg.model.type](self.cfg, self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model: {n_params:,} params")
        return n_params

    def _load_checkpoint(self, ckpt_path):
        self.model.load_state_dict(torch.load(ckpt_path))
        val_m = self._validate()
        print(f"Resumed from {ckpt_path}, val_mse={val_m['mse']:.6f}")
        return val_m['mse']

    # ---- Euler inference ----

    @torch.no_grad()
    def _euler_infer(self, lr_up, steps=None):
        """Euler integration: flow H_lr → H_pred over N steps.

        Args:
            lr_up: [B, 3, H, W] bicubic-upsampled LR.
            steps: number of Euler steps (default: self.infer_steps).

        Returns:
            sr: [B, 3, H, W] super-resolved image.
        """
        if steps is None:
            steps = self.infer_steps

        LL_lr, H_lr = dwt_high_concat(lr_up)
        x_t = H_lr
        dt = 1.0 / steps

        for i in range(steps):
            t_val = i * dt
            t_tensor = torch.full((lr_up.shape[0],), t_val,
                                  device=lr_up.device, dtype=lr_up.dtype)
            v_pred = self.model(x_t, LL_lr, lr_up, t_tensor)
            x_t = x_t + v_pred * dt

        H_pred = x_t
        sr = idwt_from_high_concat(LL_lr, H_pred)
        return sr.clamp(-1, 1)

    # ---- Validation ----

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        agg = {"mse": 0.0, "ncc": 0.0, "edge_ncc": 0.0, "psnr": 0.0, "ssim": 0.0}
        for item in self.val_data:
            sr = self._euler_infer(item['lr_up'])
            m = compute_metrics(sr, item['hr'])
            for k in agg:
                agg[k] += m[k]
        for k in agg:
            agg[k] /= len(self.val_data)
        self.model.train()
        return agg

    @torch.no_grad()
    def _save_sample(self, epoch):
        self.model.eval()
        sr = self._euler_infer(self.val_data[0]['lr_up'])
        self.model.train()
        to_pil(sr).save(os.path.join(self.output_dir, f"e{epoch:04d}.png"))

    # ---- Training loop ----

    def run(self, epochs=None, resume_ckpt=None, start_epoch=0):
        epochs = epochs or self.cfg.training.epochs
        lr = self.cfg.training.lr

        self._load_data()
        self._build_model()

        if resume_ckpt:
            best_val_mse = self._load_checkpoint(resume_ckpt)
            best_epoch = start_epoch
        else:
            ref = self.val_data[0]
            to_pil(ref['hr']).save(os.path.join(self.output_dir, "val_hr.png"))
            to_pil(ref['lr_up']).save(os.path.join(self.output_dir, "val_bicubic.png"))
            best_val_mse = float('inf')
            best_epoch = 0

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_losses, val_epochs, val_metrics_list = [], [], []
        val_every = max(20, epochs // 20)

        tag = f" (resume from {start_epoch})" if start_epoch > 0 else ""
        print(f"\n{'='*60}")
        print(f"Experiment: {self.exp_name}{tag}")
        print(f"Model: {self.cfg.model.type}  |  LR={lr} (cosine)  |  Epochs={epochs}")
        msg = f"Flow matching: MSE + {self.l1_weight}*L1  |  Euler steps: {self.infer_steps}"
        if self.tf_noise_std > 0:
            msg += f"  |  TF noise: {self.tf_noise_std}"
        if self.pixel_weight > 0:
            msg += f"  |  Pixel loss: {self.pixel_weight}"
        print(msg)
        print(f"Aug: {self.aug_kwargs}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}")

        pbar = tqdm(range(1, epochs + 1), desc=f"[{self.exp_name}]", unit="ep")
        for epoch in pbar:
            global_epoch = start_epoch + epoch
            self.model.train()

            epoch_loss = 0.0

            for item in self.train_data:
                # Augmentation (before DWT)
                lr_up_aug, hr_aug = augment_wavelet_flow(
                    item['lr_up'].clone(), item['hr'].clone(),
                    training=True, scale=self.scale, **self.aug_kwargs)

                # DWT decomposition
                LL_lr, H_lr = dwt_high_concat(lr_up_aug)
                LL_hr, H_hr = dwt_high_concat(hr_aug)

                X0 = H_lr  # LR high-freq
                X1 = H_hr  # HR high-freq

                # Flow matching: random t ~ U(0, 1)
                t = torch.rand(lr_up_aug.shape[0], device=self.device,
                               dtype=lr_up_aug.dtype)

                # Straight-line interpolation x_t = t * X1 + (1-t) * X0
                x_t = t.view(-1, 1, 1, 1) * X1 + (1 - t.view(-1, 1, 1, 1)) * X0

                # Bridge TF→AR gap: inject noise into intermediate states.
                # During inference the model sees its own (noisy) predictions,
                # so training with clean x_t creates a distribution gap.
                if self.tf_noise_std > 0:
                    x_t = x_t + torch.randn_like(x_t) * self.tf_noise_std

                # Target velocity v = X1 - X0
                v_target = X1 - X0

                # Predicted velocity
                v_pred = self.model(x_t, LL_lr, lr_up_aug, t)

                # --- Flow matching loss (wavelet domain) ---
                loss_mse = F.mse_loss(v_pred, v_target)
                loss_l1 = F.l1_loss(v_pred, v_target)
                loss = loss_mse + self.l1_weight * loss_l1

                # --- Pixel-domain auxiliary loss (bridges wavelet→pixel gap) ---
                if self.pixel_weight > 0:
                    # Quick 2-step Euler → pixel space
                    x_p = X0
                    for s in range(2):
                        t_s = torch.full_like(t, s / 2.0)
                        v_s = self.model(x_p, LL_lr, lr_up_aug, t_s)
                        x_p = x_p + v_s * 0.5
                    sr_pred = idwt_from_high_concat(LL_lr, x_p).clamp(-1, 1)
                    loss_pixel = F.mse_loss(sr_pred, hr_aug)
                    loss = loss + self.pixel_weight * loss_pixel

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(self.train_data)
            train_losses.append(epoch_loss)
            scheduler.step()

            # Validation
            if epoch == 1 or epoch % val_every == 0 or epoch == epochs:
                val_m = self._validate()
                val_epochs.append(global_epoch)
                val_metrics_list.append(val_m)

                if val_m['mse'] < best_val_mse:
                    best_val_mse = val_m['mse']
                    best_epoch = global_epoch
                    torch.save(self.model.state_dict(),
                               os.path.join(self.output_dir, "best_model.pt"))

                self._save_sample(global_epoch)

                val_mses = [m["mse"] for m in val_metrics_list]
                update_curves(self.exp_name, train_losses, val_epochs, val_mses,
                              log_dir=self.cfg.output.log_dir)
                make_progression(self.output_dir)

                pbar.set_postfix(
                    train=f"{epoch_loss:.6f}",
                    val_mse=f"{val_m['mse']:.6f}",
                    val_psnr=f"{val_m['psnr']:.1f}",
                    val_ssim=f"{val_m['ssim']:.4f}",
                    best_ep=str(best_epoch),
                )
            else:
                pbar.set_postfix({"train": f"{epoch_loss:.6f}"})

        # ---- Final summary ----
        print(f"\n{'='*60}")
        print(f"Experiment {self.exp_name} Results:")
        print(f"  Final train loss: {train_losses[-1]:.6f}")
        for ep, m in zip(val_epochs, val_metrics_list):
            mkr = " <-- BEST" if ep == best_epoch else ""
            print(f"  Epoch {ep:4d}: MSE={m['mse']:.6f}  PSNR={m['psnr']:.2f}  "
                  f"SSIM={m['ssim']:.4f}  NCC={m['ncc']:.4f}  ENCC={m['edge_ncc']:.4f}{mkr}")

        self.model.load_state_dict(
            torch.load(os.path.join(self.output_dir, "best_model.pt")))
        best = self._validate()

        print(f"\n  Best checkpoints saved to "
              f"{os.path.join(self.output_dir, 'best_model.pt')}")
        print(f"\n  Best: epoch {best_epoch}, MSE={best['mse']:.6f}, "
              f"PSNR={best['psnr']:.2f}dB, SSIM={best['ssim']:.4f}, "
              f"NCC={best['ncc']:.4f}, ENCC={best['edge_ncc']:.4f}")
        print(f"{'='*60}")

        return best


# ============================================================
# Direct Wavelet Regression Trainer
# ============================================================

class WaveletDirectTrainer:
    """Direct wavelet regression trainer — no time, no flow, no Euler.

    Model predicts H_hr from LL_lr + lr_up in a single forward pass.
    Inference is deterministic: SR = IDWT(LL_lr, H_pred).

    Usage:
        trainer = WaveletDirectTrainer(cfg, device, output_dir)
        trainer.run()
    """

    def __init__(self, cfg, device, output_dir):
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir

        self.exp_name = cfg.experiment.name
        self.scale = cfg.data.scale

        # Loss config
        self.l1_weight = getattr(cfg.training, 'l1_weight', 0.1)

        # Augmentation params
        aug = getattr(cfg.training, 'augmentation', {})
        if isinstance(aug, (dict, DotDict)):
            self.aug_kwargs = dict(aug.__dict__) if isinstance(aug, DotDict) else dict(aug)
        else:
            self.aug_kwargs = {}

        self.model = None
        self.train_data = None
        self.val_data = None

    # ---- Data ----

    def _load_data(self):
        builder = WaveletDataBuilder(self.cfg, self.device)
        self.train_data, self.val_data = builder.build()
        print(f"Loaded {len(self.train_data)} train + {len(self.val_data)} val images")

    # ---- Model ----

    def _build_model(self):
        self.model = MODEL_REGISTRY[self.cfg.model.type](self.cfg, self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model: {n_params:,} params")
        return n_params

    def _load_checkpoint(self, ckpt_path):
        self.model.load_state_dict(torch.load(ckpt_path))
        val_m = self._validate()
        print(f"Resumed from {ckpt_path}, val_mse={val_m['mse']:.6f}")
        return val_m['mse']

    # ---- Direct inference ----

    @torch.no_grad()
    def _direct_infer(self, lr_up):
        """Single forward pass: LL_lr + lr_up → H_pred → IDWT → SR.

        Args:
            lr_up: [B, 3, H, W] bicubic-upsampled LR.

        Returns:
            sr: [B, 3, H, W] super-resolved image.
        """
        LL_lr, _ = dwt_high_concat(lr_up)
        H_pred = self.model(LL_lr, lr_up)
        sr = idwt_from_high_concat(LL_lr, H_pred)
        return sr.clamp(-1, 1)

    # ---- Validation ----

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        agg = {"mse": 0.0, "ncc": 0.0, "edge_ncc": 0.0, "psnr": 0.0, "ssim": 0.0}
        for item in self.val_data:
            sr = self._direct_infer(item['lr_up'])
            m = compute_metrics(sr, item['hr'])
            for k in agg:
                agg[k] += m[k]
        for k in agg:
            agg[k] /= len(self.val_data)
        self.model.train()
        return agg

    @torch.no_grad()
    def _save_sample(self, epoch):
        self.model.eval()
        sr = self._direct_infer(self.val_data[0]['lr_up'])
        self.model.train()
        to_pil(sr).save(os.path.join(self.output_dir, f"e{epoch:04d}.png"))

    # ---- Training loop ----

    def run(self, epochs=None, resume_ckpt=None, start_epoch=0):
        epochs = epochs or self.cfg.training.epochs
        lr = self.cfg.training.lr

        self._load_data()
        self._build_model()

        if resume_ckpt:
            best_val_mse = self._load_checkpoint(resume_ckpt)
            best_epoch = start_epoch
        else:
            ref = self.val_data[0]
            to_pil(ref['hr']).save(os.path.join(self.output_dir, "val_hr.png"))
            to_pil(ref['lr_up']).save(os.path.join(self.output_dir, "val_bicubic.png"))
            best_val_mse = float('inf')
            best_epoch = 0

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_losses, val_epochs, val_metrics_list = [], [], []
        val_every = max(20, epochs // 20)

        tag = f" (resume from {start_epoch})" if start_epoch > 0 else ""
        print(f"\n{'='*60}")
        print(f"Experiment: {self.exp_name}{tag}")
        print(f"Model: {self.cfg.model.type}  |  LR={lr} (cosine)  |  Epochs={epochs}")
        print(f"Direct regression: MSE + {self.l1_weight}*L1  |  Single forward pass")
        print(f"Aug: {self.aug_kwargs}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}")

        pbar = tqdm(range(1, epochs + 1), desc=f"[{self.exp_name}]", unit="ep")
        for epoch in pbar:
            global_epoch = start_epoch + epoch
            self.model.train()

            epoch_loss = 0.0

            for item in self.train_data:
                # Augmentation (before DWT)
                lr_up_aug, hr_aug = augment_wavelet_flow(
                    item['lr_up'].clone(), item['hr'].clone(),
                    training=True, scale=self.scale, **self.aug_kwargs)

                # DWT decomposition
                LL_lr, _ = dwt_high_concat(lr_up_aug)
                LL_hr, H_hr = dwt_high_concat(hr_aug)
                # Use LL_lr for IDWT later — we predict H, keep LL_lr as-is

                # Direct prediction: LL_lr + lr_up → H_pred
                H_pred = self.model(LL_lr, lr_up_aug)

                # Wavelet-domain loss: MSE + L1 between H_pred and H_hr
                loss_mse = F.mse_loss(H_pred, H_hr)
                loss_l1 = F.l1_loss(H_pred, H_hr)
                loss = loss_mse + self.l1_weight * loss_l1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(self.train_data)
            train_losses.append(epoch_loss)
            scheduler.step()

            # Validation
            if epoch == 1 or epoch % val_every == 0 or epoch == epochs:
                val_m = self._validate()
                val_epochs.append(global_epoch)
                val_metrics_list.append(val_m)

                if val_m['mse'] < best_val_mse:
                    best_val_mse = val_m['mse']
                    best_epoch = global_epoch
                    torch.save(self.model.state_dict(),
                               os.path.join(self.output_dir, "best_model.pt"))

                self._save_sample(global_epoch)

                val_mses = [m["mse"] for m in val_metrics_list]
                update_curves(self.exp_name, train_losses, val_epochs, val_mses,
                              log_dir=self.cfg.output.log_dir)
                make_progression(self.output_dir)

                pbar.set_postfix(
                    train=f"{epoch_loss:.6f}",
                    val_mse=f"{val_m['mse']:.6f}",
                    val_psnr=f"{val_m['psnr']:.1f}",
                    val_ssim=f"{val_m['ssim']:.4f}",
                    best_ep=str(best_epoch),
                )
            else:
                pbar.set_postfix({"train": f"{epoch_loss:.6f}"})

        # ---- Final summary ----
        print(f"\n{'='*60}")
        print(f"Experiment {self.exp_name} Results:")
        print(f"  Final train loss: {train_losses[-1]:.6f}")
        for ep, m in zip(val_epochs, val_metrics_list):
            mkr = " <-- BEST" if ep == best_epoch else ""
            print(f"  Epoch {ep:4d}: MSE={m['mse']:.6f}  PSNR={m['psnr']:.2f}  "
                  f"SSIM={m['ssim']:.4f}  NCC={m['ncc']:.4f}  ENCC={m['edge_ncc']:.4f}{mkr}")

        self.model.load_state_dict(
            torch.load(os.path.join(self.output_dir, "best_model.pt")))
        best = self._validate()

        print(f"\n  Best checkpoints saved to "
              f"{os.path.join(self.output_dir, 'best_model.pt')}")
        print(f"\n  Best: epoch {best_epoch}, MSE={best['mse']:.6f}, "
              f"PSNR={best['psnr']:.2f}dB, SSIM={best['ssim']:.4f}, "
              f"NCC={best['ncc']:.4f}, ENCC={best['edge_ncc']:.4f}")
        print(f"{'='*60}")

        return best
