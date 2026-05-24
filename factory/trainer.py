"""Unified direct regression trainer for panorama super-resolution.

Handles: DirectUNet training with residual prediction, augmentation,
multi-scale ISHT condition, PolarMoE balance loss, validation, checkpointing.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch_harmonics import RealSHT, InverseRealSHT
from tqdm import tqdm

from factory.augment import augment_panorama
from factory.config import DotDict
from factory.registry import MODEL_REGISTRY
from utility.data import PanoramaDataset
from utility.metrics import compute_metrics
from vit.overfit_plot import update_curves, make_progression


def to_pil(t):
    a = t[0].cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.clip((a + 1) * 127.5, 0, 255).astype(np.uint8))


# ============================================================
# Data loading
# ============================================================

class DataBuilder:
    """Build train/val datasets with pre-computed SHT/ISHT conditions."""

    def __init__(self, cfg, device):
        self.device = device
        self.H, self.W = cfg.data.hr_size
        self.scale = cfg.data.scale
        self.data_dir = cfg.data.data_dir
        self.l_cond = cfg.data.l_cond
        self.ms_cond = [(L, f) for L, f in cfg.data.ms_cond]
        self.n_train = cfg.data.n_train
        self.n_val = cfg.data.n_val

        self.sht_cond, self.isht_cond = self._make_sht_isht(self.l_cond)
        self.ms_pairs = {}
        for L_val, factor in self.ms_cond:
            sht, isht = self._make_sht_isht(L_val)
            self.ms_pairs[(L_val, factor)] = (sht, isht)

    def _make_sht_isht(self, lmax):
        sht = RealSHT(self.H, self.W, lmax=lmax, mmax=lmax).to(self.device)
        isht = InverseRealSHT(self.H, self.W, lmax=lmax, mmax=lmax).to(self.device)
        return sht, isht

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
                coeffs = sht(lr_up)
                ms_img = isht(coeffs)
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
# Trainer
# ============================================================

class DirectTrainer:
    """Unified trainer for DirectUNet residual prediction experiments.

    Usage:
        trainer = DirectTrainer(cfg, device, output_dir)
        trainer.run()                          # fresh training
        trainer.run(resume_ckpt='best_model.pt', start_epoch=400)  # resume
    """

    def __init__(self, cfg, device, output_dir):
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir

        self.exp_name = cfg.experiment.name
        self.use_polar_moe = cfg.model.use_polar_moe
        self.balance_weight = getattr(cfg.training, 'balance_weight', 0.0)
        self.ms_injection = cfg.model.ms_injection
        self.ms_isht_none = self.ms_injection == 'none'
        self.residual_l2_weight = getattr(cfg.training, 'residual_l2_weight', 1e-4)

        # Augmentation params from config
        aug = getattr(cfg.training, 'augmentation', {})
        if isinstance(aug, (dict, DotDict)):
            self.aug_kwargs = dict(aug.__dict__) if isinstance(aug, DotDict) else aug
        else:
            self.aug_kwargs = {}

        # Model and data
        self.model = None
        self.train_data = None
        self.val_data = None

    def _load_data(self):
        builder = DataBuilder(self.cfg, self.device)
        self.train_data, self.val_data = builder.build()
        print(f"Loaded {len(self.train_data)} train + {len(self.val_data)} val images")

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

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        agg = {"mse": 0.0, "ncc": 0.0, "edge_ncc": 0.0, "psnr": 0.0, "ssim": 0.0}
        for item in self.val_data:
            ms_isht = item['ms_isht'] if not self.ms_isht_none else None
            pred_residual = self.model(item['cond'], ms_isht)
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
        ms_isht = self.val_data[0]['ms_isht'] if not self.ms_isht_none else None
        pred = self.model(self.val_data[0]['cond'], ms_isht)
        sr = self.val_data[0]['base'] + pred
        self.model.train()
        to_pil(sr).save(os.path.join(self.output_dir, f"e{epoch:04d}.png"))

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
            to_pil(ref['base']).save(os.path.join(self.output_dir, "val_base.png"))
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
        msg = f"Loss: MSE(residual) + {self.residual_l2_weight}*L2(residual)"
        if self.balance_weight > 0:
            msg += f" + {self.balance_weight}*bal"
        print(msg)
        print(f"Aug: {self.aug_kwargs}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}")

        pbar = tqdm(range(1, epochs + 1), desc=f"[{self.exp_name}]", unit="ep")
        for epoch in pbar:
            global_epoch = start_epoch + epoch
            self.model.train()

            epoch_loss, epoch_bal = 0.0, 0.0

            for item in self.train_data:
                cond_aug, hr_aug, base_aug = augment_panorama(
                    item['cond'].clone(), item['hr'].clone(), item['base'].clone(),
                    {k: v.clone() for k, v in item['ms_isht'].items()},
                    training=True, **self.aug_kwargs)

                residual_true = hr_aug - base_aug
                ms_isht_in = None if self.ms_isht_none else item['ms_isht']
                pred_residual = self.model(cond_aug, ms_isht_in)

                mse_loss = F.mse_loss(pred_residual, residual_true)
                l2_reg = pred_residual.pow(2).mean()
                loss = mse_loss + self.residual_l2_weight * l2_reg

                if self.balance_weight > 0 and self.use_polar_moe:
                    bal_loss = self.model.load_balance_loss()
                    epoch_bal += float(bal_loss)
                    loss = loss + self.balance_weight * bal_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(self.train_data)
            if self.use_polar_moe:
                epoch_bal /= len(self.train_data)
            train_losses.append(epoch_loss)
            scheduler.step()

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
                postfix = {"train": f"{epoch_loss:.6f}"}
                if self.use_polar_moe:
                    postfix["bal"] = f"{epoch_bal:.4f}"
                pbar.set_postfix(postfix)

        # ---- Final summary ----
        print(f"\n{'='*60}")
        print(f"Experiment {self.exp_name} Results:")
        print(f"  Final train loss: {train_losses[-1]:.6f}")
        for ep, m in zip(val_epochs, val_metrics_list):
            mkr = " <-- BEST" if ep == best_epoch else ""
            print(f"  Epoch {ep:4d}: MSE={m['mse']:.6f}  PSNR={m['psnr']:.2f}  "
                  f"SSIM={m['ssim']:.4f}  NCC={m['ncc']:.4f}  ENCC={m['edge_ncc']:.4f}{mkr}")

        self.model.load_state_dict(torch.load(os.path.join(self.output_dir, "best_model.pt")))
        best = self._validate()

        print(f"\n  Best checkpoints saved to {os.path.join(self.output_dir, 'best_model.pt')}")
        print(f"\n  Best: epoch {best_epoch}, MSE={best['mse']:.6f}, "
              f"PSNR={best['psnr']:.2f}dB, SSIM={best['ssim']:.4f}, "
              f"NCC={best['ncc']:.4f}, ENCC={best['edge_ncc']:.4f}")
        print(f"{'='*60}")

        return best
