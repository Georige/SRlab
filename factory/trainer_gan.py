"""GAN + LoRA trainer for texture sharpening on top of a frozen DirectUNet base.

Multi-stage training pipeline (config-driven):
  Stage 1: L1 warmup only (l1_weight=1.0, vgg_weight=0, gan_weight=0) — ~30 epochs
  Stage 2: L1 + VGG perceptual (vgg_weight=0.05) — ~100 epochs
  Stage 3: L1 + VGG + GAN (gan_weight=0.01) — ~300 epochs

Each stage loads the previous stage's best_model.pt checkpoint.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from factory.augment import augment_panorama
from factory.config import DotDict
from factory.registry import MODEL_REGISTRY
from model.lora import inject_lora_to_direct_unet, get_lora_params, count_lora_params
from model.losses import (
    VGGLoss, PatchGANDiscriminator, TVLoss, CompositeLoss,
    gan_loss_d,
)
from utility.data import PanoramaDataset
from utility.metrics import compute_metrics
from vit.overfit_plot import update_curves, make_progression


def to_pil(t):
    a = t[0].cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.clip((a + 1) * 127.5, 0, 255).astype(np.uint8))


# ============================================================
# Data loading (same as DirectTrainer's DataBuilder)
# ============================================================

class DataBuilder:
    def __init__(self, cfg, device):
        self.device = device
        self.H, self.W = cfg.data.hr_size
        self.scale = cfg.data.scale
        self.data_dir = cfg.data.data_dir
        self.l_cond = cfg.data.l_cond
        self.ms_cond = [(L, f) for L, f in cfg.data.ms_cond]
        self.n_train = cfg.data.n_train
        self.n_val = cfg.data.n_val

        from torch_harmonics import RealSHT, InverseRealSHT

        def _make(lmax):
            s = RealSHT(self.H, self.W, lmax=lmax, mmax=lmax).to(device)
            i = InverseRealSHT(self.H, self.W, lmax=lmax, mmax=lmax).to(device)
            return s, i

        self.sht_cond, self.isht_cond = _make(self.l_cond)
        self.ms_pairs = {}
        for L_val, factor in self.ms_cond:
            self.ms_pairs[(L_val, factor)] = _make(L_val)

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

            base = self.isht_cond(self.sht_cond(lr_up))
            cond = torch.cat([lr_up, base], dim=1)

            ms_isht = {}
            for L_val, factor in self.ms_cond:
                sht, isht = self.ms_pairs[(L_val, factor)]
                ms_img = isht(sht(lr_up))
                h_t, w_t = self.H // factor, self.W // factor
                ms_img = F.interpolate(ms_img, size=(h_t, w_t),
                                       mode='bilinear', align_corners=False)
                ms_isht[f'enc{factor}'] = ms_img

            item = {
                'cond': cond, 'hr': hr_img, 'ms_isht': ms_isht,
                'lr_up': lr_up, 'base': base, 'idx': i,
            }
            (train_data if i < self.n_train else val_data).append(item)

        return train_data, val_data


# ============================================================
# GAN + LoRA Trainer
# ============================================================

class GANTrainer:
    """Freeze base DirectUNet, inject LoRA, train with composite loss + optional GAN.

    Supports 3-stage config-driven pipeline:
      - training.l1_weight:  1.0 (always)
      - training.vgg_weight: 0.05 (0 = skip VGG stage)
      - training.gan_weight: 0.01 (0 = skip GAN stage)
      - training.tv_weight:  1e-6

    Usage:
        trainer = GANTrainer(cfg, device, output_dir)
        trainer.run(resume_ckpt='path/to/base_best_model.pt')
        # Stage 2: reload with vgg_weight=0.05
        # Stage 3: reload with gan_weight=0.01
    """

    def __init__(self, cfg, device, output_dir):
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        self.exp_name = cfg.experiment.name

        # LoRA config
        self.lora_r = getattr(cfg.training, 'lora_r', 4)
        self.lora_alpha = getattr(cfg.training, 'lora_alpha', 1.0)

        # Loss weights from config
        self.l1_weight = getattr(cfg.training, 'l1_weight', 1.0)
        self.vgg_weight = getattr(cfg.training, 'vgg_weight', 0.0)
        self.gan_weight = getattr(cfg.training, 'gan_weight', 0.0)
        self.tv_weight = getattr(cfg.training, 'tv_weight', 0.0)
        self.gan_type = getattr(cfg.training, 'gan_type', 'lsgan')

        # Optimizer config
        self.lr_g = getattr(cfg.training, 'lr_g', 2e-4)
        self.lr_d = getattr(cfg.training, 'lr_d', 2e-4)
        self.betas_g = tuple(getattr(cfg.training, 'betas_g', [0.5, 0.999]))
        self.betas_d = tuple(getattr(cfg.training, 'betas_d', [0.5, 0.999]))

        # Augmentation
        aug = getattr(cfg.training, 'augmentation', {})
        self.aug_kwargs = dict(aug.__dict__) if isinstance(aug, DotDict) else (aug if isinstance(aug, dict) else {})

        # Model components (built in run())
        self.model = None
        self.discriminator = None
        self.composite_loss = None
        self.opt_g = None
        self.opt_d = None
        self.train_data = None
        self.val_data = None

    @property
    def use_gan(self):
        return self.gan_weight > 0

    @property
    def use_vgg(self):
        return self.vgg_weight > 0

    def _load_data(self):
        builder = DataBuilder(self.cfg, self.device)
        self.train_data, self.val_data = builder.build()
        print(f"Loaded {len(self.train_data)} train + {len(self.val_data)} val images")

    def _build_model(self, base_ckpt):
        """Build DirectUNet, load base weights, freeze, inject LoRA.

        Handles two checkpoint types:
          - Pure DirectUNet (no LoRA keys): load weights → inject fresh LoRA
          - LoRA-wrapped checkpoint (has 'lora_' in keys): build → inject → load all
        """
        self.model = MODEL_REGISTRY[self.cfg.model.type](self.cfg, self.device)
        n_total = sum(p.numel() for p in self.model.parameters())

        ckpt = torch.load(base_ckpt)
        state = ckpt.get('model', ckpt)  # support both raw state_dict and wrapped ckpt

        has_lora = any('lora_' in k for k in state.keys())

        if has_lora:
            # Checkpoint already has LoRA — inject matching LoRA, then load all weights
            self.model, n_wrapped = inject_lora_to_direct_unet(
                self.model, r=self.lora_r, alpha=self.lora_alpha)
            # Strip discriminator keys if present, keep only model weights
            model_state = {k: v for k, v in state.items() if not k.startswith(('discriminator', 'opt_'))}
            missing, unexpected = self.model.load_state_dict(model_state, strict=False)
            if missing:
                print(f"  Note: {len(missing)} missing keys (expected for fresh LoRA)")
            if unexpected:
                print(f"  Note: {len(unexpected)} unexpected keys (discriminator/optimizer filtered)")
        else:
            # Pure base model — load weights, then inject fresh (zero-init) LoRA
            self.model.load_state_dict(state)
            self.model, n_wrapped = inject_lora_to_direct_unet(
                self.model, r=self.lora_r, alpha=self.lora_alpha)

        n_lora = count_lora_params(self.model)

        # Freeze everything except LoRA
        for n, p in self.model.named_parameters():
            if 'lora_' not in n:
                p.requires_grad = False

        print(f"Loaded base: {n_total:,} total params")
        print(f"LoRA: {n_wrapped} layers, {n_lora:,} trainable ({100*n_lora/n_total:.1f}%)")

    def _build_losses(self):
        """Build composite loss and discriminator."""
        self.composite_loss = CompositeLoss(
            vgg_weight=self.vgg_weight,
            gan_weight=self.gan_weight,
            tv_weight=self.tv_weight,
            gan_type=self.gan_type,
        )
        # Override: VGGLoss already instantiated inside CompositeLoss
        # L1 is handled inside CompositeLoss

        if self.use_gan:
            disc_in_ch = getattr(self.cfg.training, 'disc_in_ch', 3)
            self.discriminator = PatchGANDiscriminator(
                in_ch=disc_in_ch, base_ch=64, n_layers=3,
            ).to(self.device)

    def _build_optimizers(self):
        """Adam optimizers for generator (LoRA only) and discriminator."""
        lora_params = get_lora_params(self.model)
        self.opt_g = torch.optim.Adam(lora_params, lr=self.lr_g, betas=self.betas_g)
        if self.use_gan:
            self.opt_d = torch.optim.Adam(
                self.discriminator.parameters(), lr=self.lr_d, betas=self.betas_d)

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        agg = {"mse": 0.0, "ncc": 0.0, "edge_ncc": 0.0, "psnr": 0.0, "ssim": 0.0}
        for item in self.val_data:
            pred_residual = self.model(item['cond'], item['ms_isht'])
            sr = item['base'] + pred_residual
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
        pred = self.model(self.val_data[0]['cond'], self.val_data[0]['ms_isht'])
        sr = self.val_data[0]['base'] + pred
        self.model.train()
        to_pil(sr).save(os.path.join(self.output_dir, f"e{epoch:04d}.png"))

    def run(self, epochs=None, resume_ckpt=None, start_epoch=0):
        """Main training loop.

        Args:
            epochs: override training epochs
            resume_ckpt: path to base model checkpoint (required for first run)
            start_epoch: epoch offset for continued training
        """
        epochs = epochs or self.cfg.training.epochs

        if resume_ckpt is None:
            raise ValueError("GANTrainer requires --resume to specify base model checkpoint")

        self._load_data()
        self._build_model(resume_ckpt)
        self._build_losses()
        self._build_optimizers()

        # Save reference images (only if fresh start)
        if start_epoch == 0:
            ref = self.val_data[0]
            to_pil(ref['hr']).save(os.path.join(self.output_dir, "val_hr.png"))
            to_pil(ref['lr_up']).save(os.path.join(self.output_dir, "val_bicubic.png"))
            to_pil(ref['base']).save(os.path.join(self.output_dir, "val_base.png"))

        best_val_mse = float('inf')
        best_epoch = 0
        train_losses, val_epochs, val_metrics_list = [], [], []
        val_every = max(20, epochs // 20)

        tag = f" (resume from {start_epoch})" if start_epoch > 0 else ""
        print(f"\n{'='*60}")
        print(f"Experiment: {self.exp_name}{tag}")
        print(f"LoRA: r={self.lora_r}, alpha={self.lora_alpha}  |  "
              f"LR: G={self.lr_g}, D={self.lr_d}")
        print(f"Loss: l1={self.l1_weight}  vgg={self.vgg_weight}  "
              f"gan={self.gan_weight}  tv={self.tv_weight}")
        print(f"Aug: {self.aug_kwargs}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}")

        pbar = tqdm(range(1, epochs + 1), desc=f"[{self.exp_name}]", unit="ep")
        for epoch in pbar:
            global_epoch = start_epoch + epoch
            self.model.train()
            if self.discriminator:
                self.discriminator.train()

            epoch_g_loss, epoch_d_loss = 0.0, 0.0
            bal_d = {'d_real': 0.0, 'd_fake': 0.0}

            for item in self.train_data:
                # --- Augmentation ---
                cond_aug, hr_aug, base_aug = augment_panorama(
                    item['cond'].clone(), item['hr'].clone(), item['base'].clone(),
                    {k: v.clone() for k, v in item['ms_isht'].items()},
                    training=True, **self.aug_kwargs)

                residual_true = hr_aug - base_aug

                # === Discriminator update (if GAN active) ===
                if self.use_gan:
                    with torch.no_grad():
                        pred_for_d = self.model(cond_aug, item['ms_isht'])
                        sr_fake = base_aug + pred_for_d

                    real_logits = self.discriminator(hr_aug)
                    fake_logits = self.discriminator(sr_fake.detach())

                    d_loss = gan_loss_d(real_logits, fake_logits, self.gan_type)
                    self.opt_d.zero_grad()
                    d_loss.backward()
                    self.opt_d.step()
                    epoch_d_loss += d_loss.item()
                    bal_d['d_real'] += real_logits.mean().item()
                    bal_d['d_fake'] += fake_logits.mean().item()

                # === Generator update (LoRA only) ===
                pred_residual = self.model(cond_aug, item['ms_isht'])
                sr_pred = base_aug + pred_residual

                g_loss, breakdown = self.composite_loss(
                    sr_pred, hr_aug,
                    discriminator=self.discriminator if self.use_gan else None)

                self.opt_g.zero_grad()
                g_loss.backward()
                self.opt_g.step()
                epoch_g_loss += g_loss.item()

            n_items = len(self.train_data)
            epoch_g_loss /= n_items
            if self.use_gan:
                epoch_d_loss /= n_items
                bal_d = {k: v/n_items for k, v in bal_d.items()}

            train_losses.append(epoch_g_loss)

            # Validation
            if epoch == 1 or epoch % val_every == 0 or epoch == epochs:
                val_m = self._validate()
                val_epochs.append(global_epoch)
                val_metrics_list.append(val_m)

                if val_m['mse'] < best_val_mse:
                    best_val_mse = val_m['mse']
                    best_epoch = global_epoch
                    self._save_checkpoint()

                self._save_sample(global_epoch)

                val_mses = [m["mse"] for m in val_metrics_list]
                update_curves(self.exp_name, train_losses, val_epochs, val_mses,
                              log_dir=self.cfg.output.log_dir)
                make_progression(self.output_dir)

                pbar.set_postfix(
                    g_loss=f"{epoch_g_loss:.4f}",
                    d_loss=f"{epoch_d_loss:.4f}" if self.use_gan else "",
                    val_mse=f"{val_m['mse']:.6f}",
                    val_psnr=f"{val_m['psnr']:.1f}",
                    best_ep=str(best_epoch),
                )
            else:
                postfix = {"g_loss": f"{epoch_g_loss:.4f}"}
                if self.use_gan:
                    postfix["d_loss"] = f"{epoch_d_loss:.4f}"
                pbar.set_postfix(postfix)

        # Final summary
        self._load_best_checkpoint()
        best = self._validate()

        print(f"\n{'='*60}")
        print(f"Experiment {self.exp_name} Results:")
        print(f"  Final G loss: {train_losses[-1]:.6f}")
        for ep, m in zip(val_epochs, val_metrics_list):
            mkr = " <-- BEST" if ep == best_epoch else ""
            print(f"  Epoch {ep:4d}: MSE={m['mse']:.6f}  PSNR={m['psnr']:.2f}  "
                  f"SSIM={m['ssim']:.4f}  NCC={m['ncc']:.4f}{mkr}")

        print(f"\n  Best saved to {os.path.join(self.output_dir, 'best_model.pt')}")
        print(f"\n  Best: epoch {best_epoch}, MSE={best['mse']:.6f}, "
              f"PSNR={best['psnr']:.2f}dB, SSIM={best['ssim']:.4f}")
        print(f"{'='*60}")

        return best

    def _save_checkpoint(self):
        ckpt = {
            'model': self.model.state_dict(),
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
        }
        if self.discriminator is not None:
            ckpt['discriminator'] = self.discriminator.state_dict()
        ckpt['opt_g'] = self.opt_g.state_dict()
        if self.opt_d is not None:
            ckpt['opt_d'] = self.opt_d.state_dict()
        torch.save(ckpt, os.path.join(self.output_dir, "best_model.pt"))

    def _load_best_checkpoint(self):
        ckpt = torch.load(os.path.join(self.output_dir, "best_model.pt"))
        self.model.load_state_dict(ckpt['model'])
        if self.discriminator is not None and 'discriminator' in ckpt:
            self.discriminator.load_state_dict(ckpt['discriminator'])
