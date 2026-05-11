"""MultiScaleTrainer — single forward pass training with multi-scale center loss.

Route A: The model is a standard U-Net that takes I_low_up and predicts a
residual in one forward pass. During training, auxiliary decoder heads are
supervised with center-cropped losses at different scales. At inference,
only the main output head is used — no autoregressive steps.

Data: 512x1024 LR -> bicubic up -> 1024x2048, single-image overfit.
"""

import os
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

class MultiScaleTrainer:
    """Single-forward-pass trainer with multi-scale auxiliary center loss.

    The model predicts a residual from I_low_up in one shot. During training,
    auxiliary decoder heads provide multi-scale supervision on central crops.
    At inference, only the main head is used.

    Usage:
        trainer = MultiScaleTrainer(cfg, device, output_dir)
        trainer.run()
    """

    def __init__(self, cfg, device, output_dir):
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        self.exp_name = cfg.experiment.name
        self.H = cfg.data.hr_size[0]   # 1024
        self.W = cfg.data.hr_size[1]   # 2048

        self.use_residual = cfg.training.get('use_residual', True)

        # Multi-scale loss config
        aux_w = cfg.training.get('aux_weights', None)
        if aux_w is not None:
            d = getattr(aux_w, '__dict__', aux_w)
            self.aux_weights = {k: float(v) for k, v in d.items()}
        else:
            self.aux_weights = {'dec3': 0.3, 'dec2': 0.15, 'dec1': 0.05}

        cc = cfg.training.get('center_crop_ratios', None)
        if cc is not None:
            d = getattr(cc, '__dict__', cc)
            self.center_crop_ratios = {k: float(v) for k, v in d.items()}
        else:
            self.center_crop_ratios = {'dec3': 0.5, 'dec2': 0.6, 'dec1': 0.8}

        self.model = None
        self.train_images = []
        self.val_images = []

    # ============================================================
    # Data loading
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

        print(f"Loaded {len(train_items)} train + {len(self.val_images)} val images "
              f"({self.H}x{self.W})")

    # ============================================================
    # Model
    # ============================================================

    def _build_model(self):
        self.model = MODEL_REGISTRY[self.cfg.model.type](self.cfg, self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model: {n_params:,} params")
        print(f"Aux weights: {self.aux_weights}")
        print(f"Center crop ratios: {self.center_crop_ratios}")
        return n_params

    def _load_checkpoint(self, ckpt_path):
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        val_m = self._validate()
        print(f"Resumed from {ckpt_path}, val_mse={val_m['mse']:.6f}, "
              f"psnr={val_m['psnr']:.2f}dB")
        return val_m['mse']

    # ============================================================
    # Training step
    # ============================================================

    def _compute_loss(self, I_low_up, I_gt):
        """Forward pass + multi-scale loss computation.

        Returns:
            total_loss: scalar tensor
            loss_dict:  {'main': float, 'dec3': float, 'dec2': float, 'dec1': float}
        """
        output = self.model(I_low_up)
        residual_true = I_gt - I_low_up if self.use_residual else I_gt
        main_pred = output['main']

        # Main loss (full image)
        main_loss = F.mse_loss(main_pred, residual_true)
        total_loss = main_loss
        loss_dict = {'main': main_loss.item()}

        # Auxiliary center-cropped losses
        for name, aux_pred, native_H, native_W in output['aux']:
            # Downsample residual_true to aux prediction resolution
            residual_ds = F.interpolate(residual_true, size=(native_H, native_W),
                                        mode='bilinear', align_corners=False)

            # Center crop
            ratio = self.center_crop_ratios.get(name, 1.0)
            crop_H = max(1, int(native_H * ratio))
            crop_W = max(1, int(native_W * ratio))
            aux_pred_crop = crop_center_rect(aux_pred, crop_W, crop_H)
            residual_crop = crop_center_rect(residual_ds, crop_W, crop_H)

            aux_loss = F.mse_loss(aux_pred_crop, residual_crop)
            weight = self.aux_weights.get(name, 0.0)
            total_loss = total_loss + weight * aux_loss
            loss_dict[name] = aux_loss.item()

        return total_loss, loss_dict

    # ============================================================
    # Validation
    # ============================================================

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        agg = {"mse": 0.0, "psnr": 0.0, "ncc": 0.0, "ssim": 0.0}
        for I_low_up, I_gt in self.val_images:
            output = self.model(I_low_up)
            sr = I_low_up + output['main'] if self.use_residual else output['main']
            m = compute_metrics(sr, I_gt)
            for k in agg:
                agg[k] += m[k]
        for k in agg:
            agg[k] /= max(len(self.val_images), 1)
        self.model.train()
        return agg

    @torch.no_grad()
    def _save_sample(self, epoch):
        self.model.eval()
        I_low_up, I_gt = self.val_images[0]
        output = self.model(I_low_up)
        sr = I_low_up + output['main'] if self.use_residual else output['main']
        to_pil(sr).save(os.path.join(self.output_dir, f"e{epoch:04d}.png"))
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
            ref_low, ref_gt = self.val_images[0]
            to_pil(ref_low).save(os.path.join(self.output_dir, "val_bicubic.png"))
            to_pil(ref_gt).save(os.path.join(self.output_dir, "val_gt.png"))
            best_val_mse = float('inf')
            best_epoch = 0

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_losses, val_epochs, val_mses = [], [], []
        val_every = max(20, epochs // 20)

        print(f"\n{'=' * 60}")
        print(f"Experiment: {self.exp_name}")
        print(f"Model: {self.cfg.model.type}  |  LR={lr} (cosine)  |  Epochs={epochs}")
        print(f"Resolution: {self.H}x{self.W}  |  Residual: {self.use_residual}")
        print(f"Aux weights: {self.aux_weights}")
        print(f"Train images: {len(self.train_images)}")
        print(f"Output: {self.output_dir}")
        print(f"{'=' * 60}")

        pbar = tqdm(range(1, epochs + 1), desc=f"[{self.exp_name}]", unit="ep")
        for epoch in pbar:
            global_epoch = start_epoch + epoch
            self.model.train()

            epoch_loss = 0.0

            for I_low_cpu, I_gt_cpu in self.train_images:
                I_low_up = I_low_cpu.to(self.device)
                I_gt = I_gt_cpu.to(self.device)

                total_loss, loss_dict = self._compute_loss(I_low_up, I_gt)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()

            epoch_loss /= max(len(self.train_images), 1)
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

                self._save_sample(global_epoch)

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
