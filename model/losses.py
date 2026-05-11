"""Loss functions for direct panorama super-resolution.

- VGGLoss: perceptual loss using VGG19 features
- PatchGANDiscriminator: 70x70 receptive field discriminator
- TVLoss: total variation regularizer
- CompositeLoss: combines L1 + VGG + GAN + TV
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# VGG Perceptual Loss
# ============================================================

class VGGLoss(nn.Module):
    """VGG19 perceptual loss on selected feature layers.

    Uses pretrained VGG19, frozen. Compares L1 distance in feature space.
    Default layers: relu1_2, relu2_2, relu3_4, relu4_4, relu5_4.
    """

    def __init__(self, layers=None, normalize_input=True):
        super().__init__()
        if layers is None:
            layers = ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']
        self.layers = layers
        self.normalize_input = normalize_input
        self._device = None

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False

        # Collect all ReLU indices and name them properly
        relu_indices = [i for i, layer in enumerate(vgg.features)
                        if isinstance(layer, nn.ReLU)]
        relu_map = {}  # name -> (start_idx, end_idx)
        block, pos = 1, 1
        prev_relu = -1
        for idx in relu_indices:
            # New block starts after a MaxPool2d
            for j in range(max(0, prev_relu + 1), idx):
                if isinstance(vgg.features[j], nn.MaxPool2d):
                    block += 1
                    pos = 1
                    break
            name = f'relu{block}_{pos}'
            start = prev_relu + 1  # right after previous ReLU
            relu_map[name] = (start, idx)
            pos += 1
            prev_relu = idx

        # Build cumulative feature slices for requested layers
        self.slices = nn.ModuleList()
        for name in layers:
            if name in relu_map:
                _start, end = relu_map[name]
                self.slices.append(vgg.features[0:end+1])  # cumulative from input
        if not self.slices:
            raise ValueError(f"No VGG layers matched. Requested: {layers}. "
                             f"Available: {sorted(relu_map.keys())}")

    def _preprocess(self, x):
        """Convert [-1,1] RGB to VGG ImageNet normalization."""
        # x: [B, 3, H, W] in [-1, 1]
        x = (x + 1) / 2.0  # [0, 1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def forward(self, pred, target):
        # Auto-move VGG slices to input device on first call
        if self._device != pred.device:
            self.slices = self.slices.to(pred.device)
            self._device = pred.device

        if self.normalize_input:
            pred = self._preprocess(pred)
            target = self._preprocess(target)

        loss = 0.0
        for s in self.slices:
            loss += F.l1_loss(s(pred), s(target))
        return loss / len(self.slices)


# ============================================================
# PatchGAN Discriminator
# ============================================================

class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator (from pix2pix/ESRGAN).

    Input: image [B, 3, H, W] or condition+image [B, 9, H, W].
    Output: [B, 1, H/16, W/16] logits (each element judges a 70x70 patch).
    """

    def __init__(self, in_ch=3, base_ch=64, n_layers=3):
        super().__init__()

        # First layer: no normalization
        layers = [
            nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        ch = base_ch
        for i in range(1, n_layers):
            next_ch = min(ch * 2, 512)
            layers += [
                nn.Conv2d(ch, next_ch, 4, stride=2, padding=1),
                nn.InstanceNorm2d(next_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = next_ch

        # Penultimate layer
        layers += [
            nn.Conv2d(ch, ch, 4, stride=1, padding=1),
            nn.InstanceNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Output: 1-channel logits
        layers += [nn.Conv2d(ch, 1, 4, stride=1, padding=1)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ============================================================
# Total Variation Loss
# ============================================================

class TVLoss(nn.Module):
    """Anisotropic total variation for mild artifact suppression."""

    def forward(self, x):
        # x: [B, C, H, W] in [-1, 1]
        dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return dh + dw


# ============================================================
# GAN Loss (LSGAN / Hinge)
# ============================================================

def gan_loss_d(real_logits, fake_logits, gan_type='lsgan'):
    """Discriminator loss.

    Args:
        real_logits: D(real) logits
        fake_logits: D(fake) logits
        gan_type: 'lsgan' (least squares) or 'hinge'
    """
    if gan_type == 'lsgan':
        real_loss = F.mse_loss(real_logits, torch.ones_like(real_logits))
        fake_loss = F.mse_loss(fake_logits, torch.zeros_like(fake_logits))
        return (real_loss + fake_loss) / 2.0
    elif gan_type == 'hinge':
        real_loss = F.relu(1.0 - real_logits).mean()
        fake_loss = F.relu(1.0 + fake_logits).mean()
        return (real_loss + fake_loss) / 2.0
    else:
        raise ValueError(f"Unknown gan_type: {gan_type}")


def gan_loss_g(fake_logits, gan_type='lsgan'):
    """Generator loss (adversarial).

    Args:
        fake_logits: D(G(z)) logits
        gan_type: 'lsgan' or 'hinge'
    """
    if gan_type == 'lsgan':
        return F.mse_loss(fake_logits, torch.ones_like(fake_logits))
    elif gan_type == 'hinge':
        return -fake_logits.mean()
    else:
        raise ValueError(f"Unknown gan_type: {gan_type}")


# ============================================================
# Composite Loss Builder
# ============================================================

class CompositeLoss(nn.Module):
    """Composite loss: L1 + λ_vgg * VGG + λ_gan * GAN + λ_tv * TV.

    Usage during training:
        composite = CompositeLoss(vgg_weight=0.05, gan_weight=0.01, tv_weight=1e-4)
        ...
        l_total, breakdown = composite(pred, target, discriminator=None)
        # breakdown: {'l1': ..., 'vgg': ..., 'gan': ..., 'tv': ...}
    """

    def __init__(self, vgg_weight=0.05, gan_weight=0.01, tv_weight=1e-4,
                 gan_type='lsgan'):
        super().__init__()
        self.vgg_weight = vgg_weight
        self.gan_weight = gan_weight
        self.tv_weight = tv_weight
        self.gan_type = gan_type

        self.vgg = VGGLoss() if vgg_weight > 0 else None
        self.tv = TVLoss() if tv_weight > 0 else None

    def forward(self, pred, target, discriminator=None):
        """Compute composite loss.

        Args:
            pred: model output [B, 3, H, W] in [-1, 1]
            target: ground truth [B, 3, H, W] in [-1, 1]
            discriminator: PatchGANDiscriminator (None = skip GAN loss)

        Returns:
            total_loss, breakdown_dict
        """
        breakdown = {}

        # L1 pixel loss (always active)
        l1 = F.l1_loss(pred, target)
        breakdown['l1'] = l1.item()
        total = l1

        # VGG perceptual loss
        if self.vgg is not None:
            vgg = self.vgg(pred, target)
            total = total + self.vgg_weight * vgg
            breakdown['vgg'] = vgg.item()

        # GAN loss
        if discriminator is not None and self.gan_weight > 0:
            fake_logits = discriminator(pred)
            gan = gan_loss_g(fake_logits, self.gan_type)
            total = total + self.gan_weight * gan
            breakdown['gan'] = gan.item()

        # TV loss
        if self.tv is not None:
            tv = self.tv(pred)
            total = total + self.tv_weight * tv
            breakdown['tv'] = tv.item()

        return total, breakdown
