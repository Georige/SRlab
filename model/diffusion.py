"""Pixel-space diffusion with ISHT structural conditioning (multi-scale)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utility.schedule import make_beta_schedule


class PixelDiffusion(nn.Module):
    """DDPM/DDIM diffusion with ISHT multi-scale conditioning.

    Training: sample t, add noise to HF residual (HR - ISHT_base), predict noise.
    Inference: DDIM deterministic sampling → residual + ISHT_base → SR.

    Multi-scale ISHT conditions are injected at each U-Net encoder level:
      enc2 (256×512): ISHT(L=128),  enc3 (128×256): ISHT(L=64),
      bottleneck (64×128): ISHT(L=32)
    """

    def __init__(self, unet, sht_cond, isht_cond, ms_sht_isht,
                 use_hf_residual=False, sht_hf=None, isht_hf=None,
                 use_latitude_weight=False, pole_weight=1.0,
                 use_laplacian=False, lp_lambdas=(1.0, 0.3, 0.1),
                 T=1000, hr_size=(512, 1024)):
        super().__init__()
        self.unet = unet
        self.T = T
        self.hr_size = hr_size
        self.sht_cond = sht_cond
        self.isht_cond = isht_cond
        self.ms_sht_isht = ms_sht_isht
        self.use_hf_residual = use_hf_residual
        self.sht_hf = sht_hf
        self.isht_hf = isht_hf
        self.use_latitude_weight = use_latitude_weight
        self.pole_weight = pole_weight
        self.use_laplacian = use_laplacian
        self.lp_lambdas = lp_lambdas

        betas = make_beta_schedule(T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    # ---- Conditioning ----
    def make_condition(self, lr_imgs):
        """Build main condition, ISHT base, and multi-scale ISHT conditions.

        Returns:
            cond:    [B, C_cond, H, W] — bicubic↑ + ISHT base [+ HF residual]
            base:    [B, 3, H, W]      — ISHT reconstruction (fidelity layer)
            ms_isht: dict str→[B,3,h,w] — multi-scale ISHT for U-Net encoder levels
        """
        H, W = self.hr_size
        bicubic = F.interpolate(lr_imgs, size=(H, W), mode='bicubic', align_corners=False)

        # Main condition: L=255 ISHT
        coeffs = self.sht_cond(bicubic)
        base = self.isht_cond(coeffs)
        parts = [bicubic, base]

        # Optional: HF residual guide (bicubic - ISHT(L_lo))
        if self.use_hf_residual and self.sht_hf is not None:
            hf_residual = bicubic - self.isht_hf(self.sht_hf(bicubic))
            parts.append(hf_residual)

        cond = torch.cat(parts, dim=1)

        # Multi-scale ISHT conditions
        ms_isht = {}
        for sht, isht, factor in self.ms_sht_isht:
            c = sht(bicubic)
            ms_img = isht(c)
            h_t, w_t = H // factor, W // factor
            ms_img = F.interpolate(ms_img, size=(h_t, w_t),
                                   mode='bilinear', align_corners=False)
            ms_isht[f'enc{factor}'] = ms_img

        return cond, base, ms_isht

    # ---- Training ----
    def forward(self, lr_imgs, hr_imgs):
        """One training step: noise prediction on HF residual (HR - ISHT_base).

        When use_laplacian=True: builds Laplacian pyramid from residual,
        predicts noise at each level for multi-scale supervision.
        """
        B = lr_imgs.shape[0]
        cond, base, ms_isht = self.make_condition(lr_imgs)

        residual = hr_imgs - base
        t = torch.randint(0, self.T, (B,), device=hr_imgs.device)
        alpha_t = self.sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
        sigma_t = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)

        if not self.use_laplacian:
            noise = torch.randn_like(residual)
            noisy = alpha_t * residual + sigma_t * noise
            pred_noise = self.unet(noisy, cond, t.float() / self.T, ms_isht)
            loss = F.mse_loss(pred_noise, noise, reduction='none')
        else:
            # Build Gaussian pyramid
            G0 = residual                                           # [B,3,H,W]
            G1 = F.interpolate(G0, scale_factor=0.5, mode='bilinear',
                               align_corners=False)                 # [B,3,H/2,W/2]
            G2 = F.interpolate(G1, scale_factor=0.5, mode='bilinear',
                               align_corners=False)                 # [B,3,H/4,W/4]

            # Build Laplacian pyramid
            L0 = G0 - F.interpolate(G1, size=G0.shape[2:], mode='bilinear',
                                     align_corners=False)           # [B,3,H,W]   finest
            L1 = G1 - F.interpolate(G2, size=G1.shape[2:], mode='bilinear',
                                     align_corners=False)           # [B,3,H/2,W/2] medium
            L2 = G2                                                 # [B,3,H/4,W/4] coarsest

            # Independent noise per level, same timestep
            eps_L0 = torch.randn_like(L0)
            eps_L1 = torch.randn_like(L1)
            eps_L2 = torch.randn_like(L2)

            noisy_L0 = alpha_t * L0 + sigma_t * eps_L0

            pred_full, pred_L1, pred_L2 = self.unet(noisy_L0, cond,
                                                     t.float() / self.T, ms_isht)

            loss_L0 = F.mse_loss(pred_full, eps_L0, reduction='none')
            loss_L1 = F.mse_loss(pred_L1, eps_L1, reduction='none')
            loss_L2 = F.mse_loss(pred_L2, eps_L2, reduction='none')

            loss = (self.lp_lambdas[0] * loss_L0 +
                    self.lp_lambdas[1] * F.interpolate(loss_L1, size=loss_L0.shape[2:],
                                                       mode='bilinear', align_corners=False) +
                    self.lp_lambdas[2] * F.interpolate(loss_L2, size=loss_L0.shape[2:],
                                                       mode='bilinear', align_corners=False))

        if self.use_latitude_weight:
            H = residual.shape[2]
            y = torch.arange(H, device=residual.device, dtype=residual.dtype)
            w = 1.0 + self.pole_weight * (1.0 - torch.sin(torch.pi * y / H))
            loss = loss * w.view(1, 1, H, 1)
        return loss.mean()

    # ---- Inference ----
    @torch.no_grad()
    def sample(self, lr_imgs, steps=50):
        """DDIM sampling: generate HF residual, add ISHT base."""
        B = lr_imgs.shape[0]
        H, W = self.hr_size
        cond, base, ms_isht = self.make_condition(lr_imgs)
        x = torch.randn(B, 3, H, W, device=lr_imgs.device)

        indices = torch.linspace(0, self.T - 1, steps, dtype=torch.long, device=x.device)
        indices_prev = torch.cat([indices[1:], torch.tensor([-1], device=x.device)])

        for i in tqdm(range(steps - 1, -1, -1), desc='DDIM', leave=False):
            t, tp = indices[i], indices_prev[i]
            t_norm = torch.full((B,), t.item() / self.T, device=x.device)

            out = self.unet(x, cond, t_norm, ms_isht)
            eps = out[0] if self.use_laplacian else out
            alpha_t = self.sqrt_alphas_cumprod[t]
            sigma_t = self.sqrt_one_minus_alphas_cumprod[t]
            x0_pred = (x - sigma_t * eps) / alpha_t

            if tp >= 0:
                alpha_p = self.sqrt_alphas_cumprod[tp]
                sigma_p = self.sqrt_one_minus_alphas_cumprod[tp]
                x = alpha_p * x0_pred + sigma_p * eps
            else:
                x = x0_pred

        return base + x
