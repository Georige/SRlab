"""MultiScaleCenterTrainer — extends MultiScaleTrainer with center-weighted loss.

Used by Route A Scheme 1 (center mask conditioning). The main MSE loss is
weighted so that the central region contributes more to the gradient than
peripheral regions. The weight map is a 2D Gaussian centered on the image.

Schemes 2 and 3 use the base MultiScaleTrainer directly (registered as
'multiscale') since the models handle FiLM/coord embedding internally.
"""

import torch
import torch.nn.functional as F

from factory.trainer_multiscale import MultiScaleTrainer


class MultiScaleCenterTrainer(MultiScaleTrainer):
    """MultiScaleTrainer with center-weighted main loss for Scheme 1.

    Config options:
        training.center_weight_boost:  float, extra weight at center (default 1.0)
        training.center_weight_sigma:  float, Gaussian sigma as fraction of H,W
                                       (default 0.25, i.e. H/4, W/4)
    """

    def __init__(self, cfg, device, output_dir):
        super().__init__(cfg, device, output_dir)
        self.center_weight_boost = cfg.training.get('center_weight_boost', 1.0)
        self.center_weight_sigma = cfg.training.get('center_weight_sigma', 0.25)
        self._weight_map = None  # built lazily on first use

    def _build_weight_map(self, H, W):
        """Build a 2D Gaussian weight map peaking at image center.

        weight[i,j] = 1 + boost * exp(-dx^2/(2*sx^2) - dy^2/(2*sy^2))
        """
        device = self.device
        sy = self.center_weight_sigma * H
        sx = self.center_weight_sigma * W
        y = torch.arange(H, device=device, dtype=torch.float32)
        x = torch.arange(W, device=device, dtype=torch.float32)
        cy, cx = (H - 1) / 2, (W - 1) / 2
        gy = torch.exp(-((y - cy) ** 2) / (2 * sy ** 2))
        gx = torch.exp(-((x - cx) ** 2) / (2 * sx ** 2))
        weight = 1.0 + self.center_weight_boost * torch.outer(gy, gx)
        return weight[None, None, :, :]  # [1, 1, H, W]

    # ============================================================
    # Training step (overrides to add center-weighted main loss)
    # ============================================================

    def _compute_loss(self, I_low_up, I_gt):
        """Forward pass + multi-scale loss with center-weighted main loss."""
        output = self.model(I_low_up)
        residual_true = I_gt - I_low_up if self.use_residual else I_gt
        main_pred = output['main']
        _, _, H, W = main_pred.shape

        # Build center weight map (lazy, once per resolution)
        if self._weight_map is None or self._weight_map.shape[2] != H:
            self._weight_map = self._build_weight_map(H, W)

        # Center-weighted main loss
        diff_sq = (main_pred - residual_true) ** 2
        main_loss = (self._weight_map * diff_sq).mean()
        total_loss = main_loss
        loss_dict = {'main': main_loss.item()}

        # Auxiliary center-cropped losses (same as base)
        from factory.trainer_multiscale import crop_center_rect

        for name, aux_pred, native_H, native_W in output['aux']:
            residual_ds = F.interpolate(residual_true, size=(native_H, native_W),
                                        mode='bilinear', align_corners=False)

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
