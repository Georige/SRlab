"""Panorama-specific data augmentation for 360° equirectangular images."""

import random
import torch
import torch.nn.functional as F


def augment_panorama(cond, hr, base, ms_isht, training=True,
                     roll_prob=0.5, crop_prob=0.9, color_prob=0.7,
                     noise_std=0.005, noise_prob=0.3):
    """Apply panorama-aware augmentations during training.

    Args:
        cond:   [B, C, H, W] condition tensor
        hr:     [B, C, H, W] HR target tensor
        base:   [B, C, H, W] ISHT base tensor
        ms_isht: dict of multi-scale ISHT tensors {key: tensor}
        training: if False, return inputs unchanged

    Augmentations:
        1. Horizontal cyclic roll (wraps 360° seamlessly) — roll_prob
        2. Small vertical crop + resize (pitch variation) — crop_prob
        3. Brightness/contrast jitter (±10%) — color_prob
        4. Tiny Gaussian noise on HR target — noise_prob
    """
    if not training:
        return cond, hr, base

    B, C, Hi, Wi = cond.shape

    # 1. Horizontal cyclic roll
    if random.random() < roll_prob:
        shift_w = random.randint(0, Wi - 1)
        cond = torch.roll(cond, shifts=shift_w, dims=-1)
        hr   = torch.roll(hr,   shifts=shift_w, dims=-1)
        base = torch.roll(base, shifts=shift_w, dims=-1)

    # 2. Vertical crop + resize (0-10% crop)
    if random.random() < crop_prob:
        crop_px = random.randint(0, Hi // 10)
        if crop_px > 1:
            top = random.randint(0, crop_px)
            bottom = Hi - (crop_px - top)
            cond = F.interpolate(cond[:, :, top:bottom, :], size=(Hi, Wi),
                                 mode='bilinear', align_corners=False)
            hr   = F.interpolate(hr[:, :, top:bottom, :], size=(Hi, Wi),
                                 mode='bilinear', align_corners=False)
            base = F.interpolate(base[:, :, top:bottom, :], size=(Hi, Wi),
                                 mode='bilinear', align_corners=False)
            for key in ms_isht:
                h_t, w_t = ms_isht[key].shape[2], ms_isht[key].shape[3]
                ms_isht[key] = F.interpolate(
                    ms_isht[key][:, :, top * h_t // Hi:bottom * h_t // Hi, :],
                    size=(h_t, w_t), mode='bilinear', align_corners=False)

    # 3. Color jitter — brightness ±10%, contrast ±10%
    if random.random() < color_prob:
        b_shift = (random.random() - 0.5) * 0.2
        cond = cond + b_shift
        hr   = hr   + b_shift
        c_scale = 1.0 + (random.random() - 0.5) * 0.2
        cond_mean = cond.mean(dim=(-2, -1), keepdim=True)
        hr_mean   = hr.mean(dim=(-2, -1), keepdim=True)
        cond = (cond - cond_mean) * c_scale + cond_mean
        hr   = (hr   - hr_mean)   * c_scale + hr_mean
        cond = torch.clamp(cond, -1, 1)
        hr   = torch.clamp(hr,   -1, 1)

    # 4. Gaussian noise on HR target
    if random.random() < noise_prob:
        hr = hr + torch.randn_like(hr) * noise_std
        hr = torch.clamp(hr, -1, 1)

    return cond, hr, base
