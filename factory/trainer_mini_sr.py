"""MiniSR trainer with patch extraction, overlap stitching, wavelet loss, and DDP."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob

from factory.registry import MODEL_REGISTRY
from model.dwt_utils import dwt_haar
from utility.metrics import compute_metrics
from vit.overfit_plot import update_curves, make_progression


# ============================================================
# Patch utilities
# ============================================================

def make_blend_weight(patch_size, border_ratio=0.25):
    border = int(patch_size * border_ratio)
    w = torch.ones(patch_size)
    ramp = 0.5 - 0.5 * torch.cos(torch.linspace(0, 1, border) * np.pi)
    ramp = ramp * 0.99 + 0.01
    w[:border] = ramp
    w[-border:] = ramp.flip(0)
    w2d = w[:, None] * w[None, :]
    return w2d.view(1, 1, patch_size, patch_size)


def extract_patches(img_1hw, patch_size, stride):
    C, H, W = img_1hw.shape
    patches, positions = [], []
    y_starts = list(range(0, H - patch_size, stride)) + [H - patch_size]
    x_starts = list(range(0, W - patch_size, stride)) + [W - patch_size]
    for y in sorted(set(y_starts)):
        for x in sorted(set(x_starts)):
            patches.append(img_1hw[:, y:y + patch_size, x:x + patch_size])
            positions.append((y, x))
    return torch.stack(patches, dim=0), positions


def stitch_patches(patches, positions, H, W, blend_weight):
    C = patches.shape[1]
    device = patches.device
    patch_size = patches.shape[2]
    output = torch.zeros(1, C, H, W, device=device)
    weight_sum = torch.zeros(1, 1, H, W, device=device)
    bw = blend_weight.to(device)
    for i, (y, x) in enumerate(positions):
        output[:, :, y:y + patch_size, x:x + patch_size] += patches[i:i + 1] * bw
        weight_sum[:, :, y:y + patch_size, x:x + patch_size] += bw
    return output / weight_sum.clamp(min=1e-8)


# ============================================================
# Metrics
# ============================================================

# ============================================================
# Wavelet loss
# ============================================================

def wavelet_loss(sr, hr, weights=None):
    if weights is None:
        weights = {'LL': 0.1, 'LH': 1.0, 'HL': 1.0, 'HH': 1.0}

    def w(k, default=0.0):
        try:
            return getattr(weights, k) if hasattr(weights, k) else weights.get(k, default)
        except Exception:
            return default

    w_ll, w_lh, w_hl, w_hh = w('LL', 0.1), w('LH', 1.0), w('HL', 1.0), w('HH', 1.0)
    sr_ll, sr_lh, sr_hl, sr_hh = dwt_haar(sr)
    hr_ll, hr_lh, hr_hl, hr_hh = dwt_haar(hr)

    loss = torch.tensor(0.0, device=sr.device)
    if w_ll > 0: loss = loss + w_ll * F.l1_loss(sr_ll, hr_ll)
    if w_lh > 0: loss = loss + w_lh * F.l1_loss(sr_lh, hr_lh)
    if w_hl > 0: loss = loss + w_hl * F.l1_loss(sr_hl, hr_hl)
    if w_hh > 0: loss = loss + w_hh * F.l1_loss(sr_hh, hr_hh)
    return loss


# ============================================================
# Image helpers
# ============================================================

def load_image(path, target_size=None):
    img = Image.open(path).convert("RGB")
    if target_size is not None:
        img = img.resize((target_size[1], target_size[0]), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def to_pil(t):
    a = t[0].cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.clip((a + 1) * 127.5, 0, 255).astype(np.uint8))


# ============================================================
# MiniSR Trainer
# ============================================================

class MiniSRTrainer:
    def __init__(self, cfg, device, output_dir, ddp_rank=0, ddp_world_size=1):
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        self.exp_name = cfg.experiment.name
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.is_main = (ddp_rank == 0)

        self.patch_size = cfg.training.get('patch_size', 512)
        self.patch_stride = cfg.training.get('patch_stride', 256)
        self.blend_weight = make_blend_weight(self.patch_size)

        self.loss_type = cfg.training.get('loss', 'mse')
        self.wavelet_weights = cfg.training.get('wavelet_weights', None)
        self.mse_weight = cfg.training.get('mse_weight', 1.0)
        self.input_noise_std = cfg.training.get('input_noise_std', 0.0)
        self.aug_roll = cfg.training.get('aug_roll', 0.0)

        self.data_mode = cfg.data.get('mode', 'bicubic')
        self.crop_size = cfg.training.get('crop_size', None)

        self.patches_per_image = cfg.training.get('patches_per_image', None)
        self.patch_batch = cfg.training.get('patch_batch', 2)

        self.raw_model = None   # underlying model (no DDP wrap)
        self.model = None       # DDP-wrapped model (for forward)
        self.train_pairs = []   # (hr_path, lr_path_or_None) tuples
        self.val_paths = []

    def _log(self, msg):
        if self.is_main:
            print(msg)

    # ----------------------------------------------------------
    # Data
    # ----------------------------------------------------------

    def _load_data(self):
        if self.data_mode == 'paired_real':
            self._load_data_paired()
        else:
            self._load_data_bicubic()

    def _load_data_paired(self):
        hr_suffix = self.cfg.data.get('hr_suffix', '_HR.png')
        lr_suffix = self.cfg.data.get('lr_suffix', '_LR2.png')
        train_dirs = [str(d) for d in self.cfg.data.train_dirs]
        val_dirs = [str(d) for d in self.cfg.data.val_dirs]

        all_train_pairs = []
        for d in train_dirs:
            hr_files = sorted(glob(os.path.join(d, f'*{hr_suffix}')))
            for hr_path in hr_files:
                base = hr_path.rsplit(hr_suffix, 1)[0]
                lr_path = base + lr_suffix
                if os.path.exists(lr_path):
                    all_train_pairs.append((hr_path, lr_path))

        if not all_train_pairs:
            raise FileNotFoundError(f"No HR/LR pairs found in {train_dirs}")

        # Shuffle to mix camera models across DDP ranks (use fixed seed so all ranks get same order)
        rng_state = np.random.get_state()
        np.random.seed(42)
        np.random.shuffle(all_train_pairs)
        np.random.set_state(rng_state)

        # Trim to multiple of world_size so every rank has identical forward/backward count
        if self.ddp_world_size > 1:
            per_rank = len(all_train_pairs) // self.ddp_world_size
            all_train_pairs = all_train_pairs[:per_rank * self.ddp_world_size]
            start = self.ddp_rank * per_rank
            end = start + per_rank if self.ddp_rank < self.ddp_world_size - 1 else len(all_train_pairs)
            self.train_pairs = all_train_pairs[start:end]
            self._log(f"[Rank {self.ddp_rank}] train shard: {len(self.train_pairs)} pairs (indices {start}:{end})")
        else:
            self.train_pairs = all_train_pairs

        # Build val pairs
        val_pairs = []
        for d in val_dirs:
            hr_files = sorted(glob(os.path.join(d, f'*{hr_suffix}')))
            for hr_path in hr_files:
                base = hr_path.rsplit(hr_suffix, 1)[0]
                lr_path = base + lr_suffix
                if os.path.exists(lr_path):
                    val_pairs.append((hr_path, lr_path))

        # Pre-load val images at native resolution (limit to n_val if set for speed)
        self.val_hr, self.val_lr_up = [], []
        n_val = self.cfg.data.get('n_val', len(val_pairs))
        val_pairs = val_pairs[:n_val]
        for hr_path, lr_path in val_pairs:
            hr = load_image(hr_path)
            lr_img = load_image(lr_path)
            lr_up = F.interpolate(lr_img.unsqueeze(0), size=(hr.shape[1], hr.shape[2]),
                                  mode='bicubic', align_corners=False).squeeze(0)
            self.val_hr.append(hr)
            self.val_lr_up.append(lr_up)

        self._log(f"Loaded {len(self.train_pairs)} train + {len(val_pairs)} val pairs "
                  f"(mode=paired_real, crop={self.crop_size}, "
                  f"patch={self.patch_size}, stride={self.patch_stride})")

    def _load_data_bicubic(self):
        data_dir = self.cfg.data.data_dir
        hr_size = tuple(self.cfg.data.hr_size)

        hr_dir = os.path.join(data_dir, "HR")
        all_files = sorted(glob(os.path.join(hr_dir, "*.*")))
        all_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        if not all_files:
            raise FileNotFoundError(f"No images found in {hr_dir}")

        n_train = self.cfg.data.get('n_train', 1)
        n_val = self.cfg.data.get('n_val', 1)

        train_all = all_files[:n_train]
        self.val_paths = all_files[n_train:n_train + n_val]

        # Trim to multiple of world_size so every rank has identical forward/backward count
        if self.ddp_world_size > 1:
            per_rank = len(train_all) // self.ddp_world_size
            train_all = train_all[:per_rank * self.ddp_world_size]
            start = self.ddp_rank * per_rank
            end = start + per_rank if self.ddp_rank < self.ddp_world_size - 1 else len(train_all)
            self.train_pairs = [(p, None) for p in train_all[start:end]]
            self._log(f"[Rank {self.ddp_rank}] train shard: {len(self.train_pairs)} images (indices {start}:{end})")
        else:
            self.train_pairs = [(p, None) for p in train_all]

        # Load val images into memory (all ranks)
        self.val_hr, self.val_lr_up = [], []
        for p in self.val_paths:
            hr = load_image(p, hr_size)
            lr = F.interpolate(hr.unsqueeze(0), scale_factor=0.5, mode='bicubic',
                              align_corners=False).squeeze(0)
            lr_up = F.interpolate(lr.unsqueeze(0), size=hr_size, mode='bicubic',
                                 align_corners=False).squeeze(0)
            self.val_hr.append(hr)
            self.val_lr_up.append(lr_up)

        self._log(f"Loaded {len(self.train_pairs)} train + {len(self.val_paths)} val images "
                  f"(patch={self.patch_size}, stride={self.patch_stride})")

    # ----------------------------------------------------------
    # Model
    # ----------------------------------------------------------

    def _build_model(self):
        self.raw_model = MODEL_REGISTRY[self.cfg.model.type](self.cfg, self.device)

        if self.ddp_world_size > 1:
            self.model = DDP(self.raw_model, device_ids=[self.device.index])
        else:
            self.model = self.raw_model

        n_params = sum(p.numel() for p in self.raw_model.parameters())
        self._log(f"Model: {n_params:,} params  |  DDP: {self.ddp_world_size} GPUs")
        return n_params

    def _model_state_dict(self):
        """Get state dict of underlying model (handles DDP wrap)."""
        if self.ddp_world_size > 1:
            return self.model.module.state_dict()
        return self.model.state_dict()

    def _load_model_state_dict(self, sd):
        if self.ddp_world_size > 1:
            self.model.module.load_state_dict(sd)
        else:
            self.model.load_state_dict(sd)

    # ----------------------------------------------------------
    # Loss
    # ----------------------------------------------------------

    def _compute_loss(self, pred, target):
        if self.loss_type == 'wavelet':
            return wavelet_loss(pred, target, self.wavelet_weights)
        elif self.loss_type == 'mse+wavelet':
            mse = F.mse_loss(pred, target)
            wl = wavelet_loss(pred, target, self.wavelet_weights)
            return self.mse_weight * mse + (1 - self.mse_weight) * wl
        else:
            return F.mse_loss(pred, target)

    # ----------------------------------------------------------
    # Image loading
    # ----------------------------------------------------------

    def _load_image_pair(self, pair):
        hr_path, lr_path = pair

        if lr_path is not None:
            # Paired real LR: load both at native resolution
            hr = load_image(hr_path)
            lr_img = load_image(lr_path)
            lr_up = F.interpolate(lr_img.unsqueeze(0), size=(hr.shape[1], hr.shape[2]),
                                  mode='bicubic', align_corners=False).squeeze(0)
        else:
            # Bicubic mode: generate LR from HR
            hr_size = tuple(self.cfg.data.hr_size)
            hr = load_image(hr_path, hr_size)
            lr = F.interpolate(hr.unsqueeze(0), scale_factor=0.5, mode='bicubic',
                              align_corners=False).squeeze(0)
            lr_up = F.interpolate(lr.unsqueeze(0), size=hr_size, mode='bicubic',
                                 align_corners=False).squeeze(0)

        # Random crop during training
        if self.crop_size and hr.shape[1] >= self.crop_size and hr.shape[2] >= self.crop_size:
            max_y = hr.shape[1] - self.crop_size
            max_x = hr.shape[2] - self.crop_size
            y0 = np.random.randint(0, max_y + 1)
            x0 = np.random.randint(0, max_x + 1)
            lr_up = lr_up[:, y0:y0 + self.crop_size, x0:x0 + self.crop_size]
            hr = hr[:, y0:y0 + self.crop_size, x0:x0 + self.crop_size]

        # Horizontal roll augmentation (panorama only)
        if self.aug_roll > 0 and np.random.random() < self.aug_roll:
            shift = np.random.randint(0, hr.shape[2])
            lr_up = torch.roll(lr_up, shifts=shift, dims=-1)
            hr = torch.roll(hr, shifts=shift, dims=-1)

        return lr_up, hr

    # ----------------------------------------------------------
    # Forward (full-image stitching for val)
    # ----------------------------------------------------------

    def _forward_full_image(self, lr_up_1hw, patch_batch=2):
        _, H, W = lr_up_1hw.shape
        patches, positions = extract_patches(lr_up_1hw, self.patch_size, self.patch_stride)
        out_list = []
        for i in range(0, len(patches), patch_batch):
            batch = patches[i:i + patch_batch].to(self.device)
            out_list.append(self.model(batch).cpu())
        out_patches = torch.cat(out_list, dim=0)
        return stitch_patches(out_patches, positions, H, W, self.blend_weight)

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        agg = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0}
        for hr, lr_up in zip(self.val_hr, self.val_lr_up):
            sr = self._forward_full_image(lr_up).to(self.device)
            m = compute_metrics(sr, hr.unsqueeze(0).to(self.device))
            for k in agg:
                agg[k] += m[k]
        for k in agg:
            agg[k] /= len(self.val_hr)
        self.model.train()
        return agg

    @torch.no_grad()
    def _save_sample(self, epoch):
        self.model.eval()
        sr = self._forward_full_image(self.val_lr_up[0])
        self.model.train()
        to_pil(sr).save(os.path.join(self.output_dir, f"e{epoch:04d}.png"))

    # ----------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------

    def run(self, epochs=None, resume_ckpt=None, start_epoch=0):
        epochs = epochs or self.cfg.training.epochs
        lr = self.cfg.training.lr

        self._load_data()
        self._build_model()

        if resume_ckpt:
            sd = torch.load(resume_ckpt, map_location=self.device)
            self._load_model_state_dict(sd)
            best_val_mse = self._validate()["mse"] if self.is_main else float('inf')
            best_epoch = start_epoch
            self._log(f"Resumed from {resume_ckpt}, val_mse={best_val_mse:.6f}")
        else:
            if self.is_main:
                to_pil(self.val_hr[0].unsqueeze(0)).save(os.path.join(self.output_dir, "val_hr.png"))
                to_pil(self.val_lr_up[0].unsqueeze(0)).save(os.path.join(self.output_dir, "val_bicubic.png"))
            best_val_mse = float('inf')
            best_epoch = 0

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_losses, val_epochs, val_metrics_list = [], [], []
        val_every = max(10, epochs // 20)

        self._log(f"\n{'=' * 60}")
        self._log(f"Experiment: {self.exp_name}")
        self._log(f"Model: {self.cfg.model.type}  |  LR={lr} (cosine)  |  Epochs={epochs}")
        ppi = self.patches_per_image or 'all'
        self._log(f"Data: {self.data_mode}  |  Loss: {self.loss_type}  |  "
                  f"Patch: {self.patch_size}x{self.patch_size}, stride={self.patch_stride}, ppi={ppi}")
        if self.crop_size:
            self._log(f"Random crop: {self.crop_size}x{self.crop_size}")
        if self.input_noise_std > 0:
            self._log(f"Input noise: std={self.input_noise_std}")
        if self.aug_roll > 0:
            self._log(f"Augmentation: random horizontal roll (p={self.aug_roll})")
        self._log(f"GPUs: {self.ddp_world_size}  |  Output: {self.output_dir}")
        self._log(f"{'=' * 60}")

        pbar = tqdm(range(1, epochs + 1), desc=f"[{self.exp_name}]", unit="ep",
                    disable=not self.is_main)
        for epoch in pbar:
            global_epoch = start_epoch + epoch
            self.model.train()

            epoch_loss = 0.0
            n_steps = 0

            for pair in self.train_pairs:
                lr_up, hr = self._load_image_pair(pair)
                hr = hr.to(self.device)

                lr_patches, _ = extract_patches(lr_up, self.patch_size, self.patch_stride)
                hr_patches, _ = extract_patches(hr, self.patch_size, self.patch_stride)

                if self.patches_per_image and len(lr_patches) > self.patches_per_image:
                    idx = torch.randperm(len(lr_patches))[:self.patches_per_image]
                    lr_patches = lr_patches[idx]
                    hr_patches = hr_patches[idx]

                pb = self.patch_batch
                for i in range(0, len(lr_patches), pb):
                    batch_lr = lr_patches[i:i + pb].to(self.device)
                    batch_hr = hr_patches[i:i + pb].to(self.device)
                    if self.input_noise_std > 0:
                        batch_lr = batch_lr + torch.randn_like(batch_lr) * self.input_noise_std
                    loss = self._compute_loss(self.model(batch_lr), batch_hr)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_steps += 1

            epoch_loss /= max(n_steps, 1)
            train_losses.append(epoch_loss)
            scheduler.step()

            # Validation (main rank only) — skip epoch 1 to avoid warmup timeout
            do_val = (epoch % val_every == 0 or epoch == epochs)
            if self.ddp_world_size == 1:
                do_val = do_val or epoch == 1

            if do_val and self.is_main:
                val_m = self._validate()
                val_epochs.append(global_epoch)
                val_metrics_list.append(val_m)

                if val_m['mse'] < best_val_mse:
                    best_val_mse = val_m['mse']
                    best_epoch = global_epoch
                    torch.save(self._model_state_dict(),
                              os.path.join(self.output_dir, "best_model.pt"))

                self._save_sample(global_epoch)

                val_mses = [m["mse"] for m in val_metrics_list]
                val_psnrs = [m["psnr"] for m in val_metrics_list]
                val_ssims = [m["ssim"] for m in val_metrics_list]
                update_curves(self.exp_name, train_losses, val_epochs, val_mses,
                             log_dir=self.cfg.output.log_dir,
                             val_psnrs=val_psnrs, val_ssims=val_ssims)
                make_progression(self.output_dir)

                pbar.set_postfix(
                    train=f"{epoch_loss:.6f}",
                    val_mse=f"{val_m['mse']:.6f}",
                    val_psnr=f"{val_m['psnr']:.1f}",
                    val_ssim=f"{val_m['ssim']:.4f}",
                    best_ep=str(best_epoch),
                )
            else:
                if self.is_main:
                    pbar.set_postfix({"train": f"{epoch_loss:.6f}"})

            # Barrier: keep ranks in sync after validation
            if self.ddp_world_size > 1:
                dist.barrier()

        # Final summary (main only)
        if self.is_main:
            print(f"\n{'=' * 60}")
            print(f"Experiment {self.exp_name} Results:")
            print(f"  Final train loss: {train_losses[-1]:.6f}")
            for ep_idx, m in zip(val_epochs, val_metrics_list):
                mkr = " <-- BEST" if ep_idx == best_epoch else ""
                print(f"  Epoch {ep_idx:4d}: MSE={m['mse']:.6f}  PSNR={m['psnr']:.2f}  "
                      f"SSIM={m['ssim']:.4f}{mkr}")

            self._load_model_state_dict(
                torch.load(os.path.join(self.output_dir, "best_model.pt"), map_location=self.device))
            best = self._validate()
            print(f"\n  Best: epoch {best_epoch}, MSE={best['mse']:.6f}, "
                  f"PSNR={best['psnr']:.2f}dB, SSIM={best['ssim']:.4f}")
            print(f"{'=' * 60}")
            return best
        return None
