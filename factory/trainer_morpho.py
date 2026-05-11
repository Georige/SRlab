"""MorphoTrainer — Neural Cellular Automata super-resolution training loop.

Handles: RealSR data loading, crop-based training, NCA iterative steps with
stochastic masking, overflow penalty, progressive loss stages (L1→SSIM→VGG→GAN),
validation with deterministic inference, and progression visualization.
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
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

    return {"mse": mse, "ncc": ncc, "psnr": psnr, "ssim": ssim}


def ssim_loss(pred, target, window=11):
    """Differentiable 1 - SSIM loss using avg_pool window.

    Args:
        pred, target: [B, 3, H, W] in [-1, 1]
        window: pooling kernel size (default 11)

    Returns:
        scalar loss = 1 - mean(SSIM)
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(pred, window, stride=1, padding=window // 2)
    mu_y = F.avg_pool2d(target, window, stride=1, padding=window // 2)

    sigma_x = F.avg_pool2d(pred ** 2, window, stride=1, padding=window // 2) - mu_x ** 2
    sigma_y = F.avg_pool2d(target ** 2, window, stride=1, padding=window // 2) - mu_y ** 2
    sigma_xy = F.avg_pool2d(pred * target, window, stride=1, padding=window // 2) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2) + 1e-8)

    return 1.0 - ssim_map.mean()


# ============================================================
# RealSR Dataset
# ============================================================

class RealSRDataset:
    """Loads RealSR paired images with both real and bicubic LR.

    Each item is a dict:
        lr_real:    [3, H/scale, W/scale] real sensor LR
        lr_bicubic: [3, H/scale, W/scale] clean bicubic-downsampled LR
        hr:         [3, H, W] ground truth HR

    The dataset is small (12 pairs), so we load everything into GPU memory.
    """

    def __init__(self, data_dir, hr_size=(800, 1400), scale=2, n_train=10, n_val=2,
                 image_index=0, device='cuda'):
        self.hr_h, self.hr_w = hr_size
        self.scale = scale
        self.lr_h, self.lr_w = self.hr_h // scale, self.hr_w // scale
        self.device = device

        hr_dir = os.path.join(data_dir, 'HR')
        lr_dir = os.path.join(data_dir, 'LR', f'X{scale}')

        hr_files = sorted(os.listdir(hr_dir))

        # Load all images into GPU memory (12 images × ~15 MB = ~180 MB)
        self.samples = []
        for f in hr_files:
            hr_path = os.path.join(hr_dir, f)
            lr_real_path = os.path.join(lr_dir, f)

            if not os.path.exists(lr_real_path):
                continue

            # Load HR: resize to target size
            hr_img = self._load_img(hr_path, (self.hr_w, self.hr_h)).to(device)

            # Load real LR: resize to target LR size
            lr_real = self._load_img(lr_real_path, (self.lr_w, self.lr_h)).to(device)

            # Generate bicubic LR from HR (clean reference)
            lr_bicubic = F.interpolate(
                hr_img.unsqueeze(0),
                size=(self.lr_h, self.lr_w),
                mode='bicubic', align_corners=False
            ).squeeze(0)

            self.samples.append({
                'lr_real': lr_real,
                'lr_bicubic': lr_bicubic,
                'hr': hr_img,
            })

        # Split into train/val
        total = len(self.samples)
        # Use image_index for single-image mode
        if n_train == 1 and n_val == 1:
            idx = min(image_index, total - 1)
            self.train_samples = [self.samples[idx]]
            self.val_samples = [self.samples[idx]]
        else:
            n_train_actual = min(n_train, total - n_val)
            indices = list(range(total))
            random.shuffle(indices)
            self.train_samples = [self.samples[i] for i in indices[:n_train_actual]]
            self.val_samples = [self.samples[i] for i in indices[n_train_actual:n_train_actual + n_val]]

        print(f"RealSRDataset: {len(self.train_samples)} train + {len(self.val_samples)} val "
              f"from {total} images ({self.lr_h}×{self.lr_w} → {self.hr_h}×{self.hr_w})")

    @staticmethod
    def _load_img(path, target_size):
        """Load image, resize with LANCZOS, normalize to [-1, 1]."""
        img = Image.open(path).convert('RGB').resize(
            (target_size[0], target_size[1]), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)


# ============================================================
# MorphoTrainer
# ============================================================

class MorphoTrainer:
    """NCA-based super-resolution trainer.

    Key features:
      - Crop-based training for memory efficiency
      - NCA loop with stochastic masking and random step counts
      - Overflow penalty on hidden cell channels
      - Progressive loss stages: L1 → SSIM → VGG → GAN
      - Deterministic validation on full images
      - Progression visualization (intermediate NCA states)

    Usage:
        trainer = MorphoTrainer(cfg, device, output_dir)
        trainer.run()
    """

    def __init__(self, cfg, device, output_dir):
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        self.exp_name = cfg.experiment.name

        # ---- Image dimensions ----
        self.H, self.W = cfg.data.hr_size
        self.scale = cfg.data.scale

        # ---- Crop settings ----
        crop = cfg.training.get('crop_size', None)
        self.crop_size = tuple(crop) if crop else None

        # ---- NCA parameters ----
        self.n_steps_fixed = cfg.training.get('n_steps', 15)
        self.n_steps_min = cfg.training.get('n_steps_min', 10)
        self.n_steps_max = cfg.training.get('n_steps_max', 20)
        self.random_steps = cfg.training.get('random_steps', False)
        self.p_update = cfg.training.get('p_update', 0.5)

        # ---- Conditioning ----
        self.use_condition = cfg.training.get('use_condition', False)

        # ---- Residual mode (Stage 1.5+): sr = condition + model_residual ----
        self.residual_mode = cfg.training.get('residual_mode', False)

        # ---- Ensemble validation: average N runs with p_update=0.5 ----
        self.val_ensemble = cfg.training.get('val_ensemble', 1)

        # ---- Pre-encoder ----
        self.use_pre_encoder = cfg.model.get('use_pre_encoder', False)

        # ---- Loss weights ----
        self.overflow_weight = cfg.training.get('overflow_weight', 0.0)
        self.ssim_weight = cfg.training.get('ssim_weight', 0.0)
        self.vgg_weight = cfg.training.get('vgg_weight', 0.0)
        self.gan_weight = cfg.training.get('gan_weight', 0.0)
        self.gan_type = cfg.training.get('gan_type', 'lsgan')

        # ---- Batch settings ----
        self.batches_per_epoch = cfg.training.get('batches_per_epoch', 100)

        # ---- VGG loss (lazy init) ----
        self._vgg_loss = None

        # ---- GAN components (lazy init) ----
        self._discriminator = None
        self._d_optimizer = None

        # ---- Data and model ----
        self.train_data = None
        self.val_data = None
        self.model = None

    # ============================================================
    # Data loading
    # ============================================================

    def _load_data(self):
        dataset = RealSRDataset(
            data_dir=self.cfg.data.data_dir,
            hr_size=(self.H, self.W),
            scale=self.scale,
            n_train=self.cfg.data.n_train,
            n_val=self.cfg.data.n_val,
            image_index=self.cfg.data.get('image_index', 0),
            device=self.device,
        )
        self.train_data = dataset.train_samples
        self.val_data = dataset.val_samples

    # ============================================================
    # Model building
    # ============================================================

    def _build_model(self):
        self.model = MODEL_REGISTRY[self.cfg.model.type](self.cfg, self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model: {n_params:,} params")

        # Lazy-init VGG loss
        if self.vgg_weight > 0:
            from model.losses import VGGLoss
            self._vgg_loss = VGGLoss().to(self.device)
            print(f"VGG loss enabled (weight={self.vgg_weight})")

        # Lazy-init GAN
        if self.gan_weight > 0:
            from model.losses import PatchGANDiscriminator, gan_loss_d, gan_loss_g
            self._discriminator = PatchGANDiscriminator(in_ch=3, base_ch=64).to(self.device)
            self._d_optimizer = torch.optim.Adam(
                self._discriminator.parameters(),
                lr=self.cfg.training.lr
            )
            self._gan_loss_d = gan_loss_d
            self._gan_loss_g = gan_loss_g
            d_params = sum(p.numel() for p in self._discriminator.parameters())
            print(f"PatchGAN discriminator: {d_params:,} params (type={self.gan_type})")

        return n_params

    # ============================================================
    # Crop-based batch iteration
    # ============================================================

    def _iter_train_batches(self):
        """Yield training batches with random crops.

        Each batch:
            lr_crop:   [B, 3, lr_h, lr_w]  LR input (real or bicubic)
            cond_crop: [B, 3, crop_h, crop_w] bicubic-upsampled LR (scaffold)
            hr_crop:   [B, 3, crop_h, crop_w] ground truth HR
        """
        batch_size = 1  # NCA is memory-efficient; batch=1 is fine with random crops

        for _ in range(self.batches_per_epoch):
            # Sample random image from training set
            idx = random.randint(0, len(self.train_data) - 1)
            sample = self.train_data[idx]

            hr = sample['hr']
            lr_bicubic = sample['lr_bicubic']
            lr_real = sample['lr_real']

            if self.crop_size:
                crop_h, crop_w = self.crop_size
                h, w = self.H, self.W
                top = random.randint(0, max(0, h - crop_h))
                left = random.randint(0, max(0, w - crop_w))

                # HR crop
                hr_crop = hr[:, top:top + crop_h, left:left + crop_w]

                # LR crops (scaled coordinates)
                lr_top, lr_left = top // self.scale, left // self.scale
                lr_h, lr_w = crop_h // self.scale, crop_w // self.scale

                lr_real_crop = lr_real[:, lr_top:lr_top + lr_h, lr_left:lr_left + lr_w]
                lr_bicubic_crop = lr_bicubic[:, lr_top:lr_top + lr_h, lr_left:lr_left + lr_w]

                # Condition: bicubic-upsampled LR (always uses clean bicubic)
                cond_crop = F.interpolate(
                    lr_bicubic_crop.unsqueeze(0),
                    size=(crop_h, crop_w), mode='bicubic', align_corners=False
                ).squeeze(0)
            else:
                # Full image mode (Stage 1 single-image)
                hr_crop = hr
                lr_real_crop = lr_real
                lr_bicubic_crop = lr_bicubic
                cond_crop = F.interpolate(
                    lr_bicubic.unsqueeze(0),
                    size=(self.H, self.W), mode='bicubic', align_corners=False
                ).squeeze(0)

            # Choose LR source
            lr_input = lr_real_crop if self.use_pre_encoder else lr_bicubic_crop

            # Add batch dimension
            yield (
                lr_input.unsqueeze(0),
                cond_crop.unsqueeze(0) if self.use_condition else None,
                hr_crop.unsqueeze(0),
            )

    # ============================================================
    # Validation
    # ============================================================

    @torch.no_grad()
    def _validate(self):
        """Run ensemble NCA inference on validation set.

        Uses p_update=0.5 for natural stochasticity, averages N runs.
        Ensemble reduces stochastic variance and preserves natural texture.
        """
        self.model.eval()
        agg = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0, "ncc": 0.0}

        for sample in self.val_data:
            hr = sample['hr']
            lr_bicubic = sample['lr_bicubic']
            lr_real = sample['lr_real']

            # Choose LR source
            lr_input = lr_real if self.use_pre_encoder else lr_bicubic

            # Build condition
            if self.use_condition:
                condition = F.interpolate(
                    lr_bicubic.unsqueeze(0),
                    size=(self.H, self.W), mode='bicubic', align_corners=False
                )
            else:
                condition = None

            # Ensemble: run N times with p_update=0.5, average
            sr_ensemble = None
            for _ in range(self.val_ensemble):
                sr_i, _ = self.model(
                    lr_input.unsqueeze(0),
                    n_steps=self.n_steps_fixed,
                    condition=condition,
                    p_update=0.5,  # match training stochasticity
                    return_progression=False,
                )
                if self.residual_mode and condition is not None:
                    sr_i = condition + sr_i
                if sr_ensemble is None:
                    sr_ensemble = sr_i.float()
                else:
                    sr_ensemble = sr_ensemble + sr_i.float()
            sr = sr_ensemble / self.val_ensemble

            m = compute_metrics(sr, hr.unsqueeze(0))
            for k in agg:
                agg[k] += m[k]

        for k in agg:
            agg[k] /= len(self.val_data)
        self.model.train()
        return agg

    # ============================================================
    # Sample saving
    # ============================================================

    @torch.no_grad()
    def _save_sample(self, epoch):
        """Save SR output for the first validation image."""
        self.model.eval()
        sample = self.val_data[0]
        hr = sample['hr']
        lr_bicubic = sample['lr_bicubic']
        lr_real = sample['lr_real']

        lr_input = lr_real if self.use_pre_encoder else lr_bicubic

        if self.use_condition:
            condition = F.interpolate(
                lr_bicubic.unsqueeze(0),
                size=(self.H, self.W), mode='bicubic', align_corners=False
            )
        else:
            condition = None

        sr, _ = self.model(
            lr_input.unsqueeze(0),
            n_steps=self.n_steps_fixed,
            condition=condition,
            p_update=1.0,
            return_progression=False,
        )

        if self.residual_mode and condition is not None:
            sr = condition + sr

        self.model.train()
        to_pil(sr).save(os.path.join(self.output_dir, f"e{epoch:04d}.png"))

    @torch.no_grad()
    def _save_progression(self, epoch):
        """Save NCA progression frames as a horizontal strip.

        Collects intermediate RGB frames during the NCA loop and stitches
        them into a single wide image for visual inspection.
        """
        self.model.eval()
        sample = self.val_data[0]
        hr = sample['hr']
        lr_bicubic = sample['lr_bicubic']
        lr_real = sample['lr_real']

        lr_input = lr_real if self.use_pre_encoder else lr_bicubic

        if self.use_condition:
            condition = F.interpolate(
                lr_bicubic.unsqueeze(0),
                size=(self.H, self.W), mode='bicubic', align_corners=False
            )
        else:
            condition = None

        sr, progression, _ = self.model(
            lr_input.unsqueeze(0),
            n_steps=self.n_steps_fixed,
            condition=condition,
            p_update=1.0,
            return_progression=True,
        )

        self.model.train()

        # Build progression strip: [bicubic_input | frame_1 | frame_2 | ... | sr | hr]
        bicubic_up = F.interpolate(
            lr_bicubic.unsqueeze(0),
            size=(self.H, self.W), mode='bicubic', align_corners=False
        )

        # In residual mode, add scaffold to raw NCA outputs
        if self.residual_mode and condition is not None:
            sr = condition + sr
            if condition.shape[0] == 1 and bicubic_up.shape[0] == 1:
                progression = [bicubic_up + f for f in progression]

        frames = [bicubic_up] + progression + [sr, hr.unsqueeze(0)]
        strips = [to_pil(f) for f in frames]

        # Resize for compact display
        thumb_h, thumb_w = 200, 400
        strips = [s.resize((thumb_w, thumb_h), Image.LANCZOS) for s in strips]

        total_w = thumb_w * len(strips)
        canvas = Image.new('RGB', (total_w, thumb_h))
        for i, s in enumerate(strips):
            canvas.paste(s, (i * thumb_w, 0))

        canvas.save(os.path.join(self.output_dir, f"prog_e{epoch:04d}.png"))

    # ============================================================
    # Main training loop
    # ============================================================

    def run(self, epochs=None, resume_ckpt=None, start_epoch=0):
        epochs = epochs or self.cfg.training.epochs
        lr = self.cfg.training.lr

        self._load_data()
        self._build_model()

        if resume_ckpt:
            self.model.load_state_dict(torch.load(resume_ckpt))
            val_m = self._validate()
            print(f"Resumed from {resume_ckpt}, val_mse={val_m['mse']:.6f}, "
                  f"val_psnr={val_m['psnr']:.2f}")
            best_val_mse = val_m['mse']
            best_epoch = start_epoch
        else:
            # Save reference images
            ref = self.val_data[0]
            to_pil(ref['hr'].unsqueeze(0)).save(os.path.join(self.output_dir, "val_hr.png"))
            bicubic_up = F.interpolate(
                ref['lr_bicubic'].unsqueeze(0),
                size=(self.H, self.W), mode='bicubic', align_corners=False
            )
            to_pil(bicubic_up).save(os.path.join(self.output_dir, "val_bicubic.png"))
            best_val_mse = float('inf')
            best_epoch = 0

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_losses, val_epochs, val_metrics_list = [], [], []
        val_every = max(20, epochs // 20)

        # Print header
        tag = f" (resume from {start_epoch})" if start_epoch > 0 else ""
        print(f"\n{'=' * 60}")
        print(f"Experiment: {self.exp_name}{tag}")
        print(f"Model: {self.cfg.model.type}  |  LR={lr}  |  Epochs={epochs}")
        print(f"NCA steps: {'random ' + str(self.n_steps_min) + '-' + str(self.n_steps_max) if self.random_steps else self.n_steps_fixed}  |  p_update={self.p_update}")
        print(f"Condition: {self.use_condition}  |  Pre-encoder: {self.use_pre_encoder}")
        loss_parts = ["L1"]
        if self.ssim_weight > 0:
            loss_parts.append(f"{self.ssim_weight}*SSIM")
        if self.overflow_weight > 0:
            loss_parts.append(f"{self.overflow_weight}*overflow")
        if self.vgg_weight > 0:
            loss_parts.append(f"{self.vgg_weight}*VGG")
        if self.gan_weight > 0:
            loss_parts.append(f"{self.gan_weight}*GAN({self.gan_type})")
        print(f"Loss: {' + '.join(loss_parts)}")
        print(f"Crop: {self.crop_size}  |  Batches/epoch: {self.batches_per_epoch}")
        print(f"Output: {self.output_dir}")
        print(f"{'=' * 60}")

        pbar = tqdm(range(1, epochs + 1), desc=f"[{self.exp_name}]", unit="ep")
        for epoch in pbar:
            global_epoch = start_epoch + epoch
            self.model.train()

            epoch_loss = 0.0
            epoch_l1 = 0.0
            n_batches = 0

            for lr_input, condition, hr_gt in self._iter_train_batches():
                # ---- Determine n_steps ----
                if self.random_steps:
                    n_steps = random.randint(self.n_steps_min, self.n_steps_max)
                else:
                    n_steps = self.n_steps_fixed

                # ---- NCA forward ----
                sr, final_state = self.model(
                    lr_input,
                    n_steps=n_steps,
                    condition=condition,
                    p_update=self.p_update,
                    return_progression=False,
                )

                # ---- Residual mode: SR = bicubic scaffold + learned residual ----
                if self.residual_mode and condition is not None:
                    sr = condition + sr

                # ---- Loss computation ----
                l1 = F.l1_loss(sr, hr_gt)
                loss = l1
                epoch_l1 += l1.item()

                # SSIM loss
                if self.ssim_weight > 0:
                    ssim_l = ssim_loss(sr, hr_gt)
                    loss = loss + self.ssim_weight * ssim_l

                # Overflow penalty on hidden channels (4-15)
                if self.overflow_weight > 0:
                    overflow = final_state[:, 4:16].pow(2).mean()
                    loss = loss + self.overflow_weight * overflow

                # VGG perceptual loss
                if self.vgg_weight > 0 and self._vgg_loss is not None:
                    vgg_l = self._vgg_loss(sr, hr_gt)
                    loss = loss + self.vgg_weight * vgg_l

                # GAN training
                if self.gan_weight > 0 and self._discriminator is not None:
                    # Train discriminator
                    d_loss = self._gan_loss_d(
                        self._discriminator(hr_gt),
                        self._discriminator(sr.detach()),
                        self.gan_type,
                    )
                    self._d_optimizer.zero_grad()
                    d_loss.backward()
                    self._d_optimizer.step()

                    # Generator adversarial loss
                    g_adv = self._gan_loss_g(self._discriminator(sr), self.gan_type)
                    loss = loss + self.gan_weight * g_adv

                # ---- Backward ----
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            epoch_loss /= max(n_batches, 1)
            train_losses.append(epoch_loss)
            scheduler.step()

            # ---- Validation and checkpointing ----
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
                if epoch == 1 or epoch % (val_every * 5) == 0 or epoch == epochs:
                    self._save_progression(global_epoch)

                val_mses = [m["mse"] for m in val_metrics_list]
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

        # ---- Final summary ----
        print(f"\n{'=' * 60}")
        print(f"Experiment {self.exp_name} Results:")
        print(f"  Final train loss: {train_losses[-1]:.6f}")
        for ep, m in zip(val_epochs, val_metrics_list):
            mkr = " <-- BEST" if ep == best_epoch else ""
            print(f"  Epoch {ep:4d}: MSE={m['mse']:.6f}  PSNR={m['psnr']:.2f}  "
                  f"SSIM={m['ssim']:.4f}  NCC={m['ncc']:.4f}{mkr}")

        self.model.load_state_dict(
            torch.load(os.path.join(self.output_dir, "best_model.pt")))
        best = self._validate()

        print(f"\n  Best: epoch {best_epoch}, MSE={best['mse']:.6f}, "
              f"PSNR={best['psnr']:.2f}dB, SSIM={best['ssim']:.4f}, NCC={best['ncc']:.4f}")
        print(f"{'=' * 60}")

        return best
