"""Quick test: train HR-target model, then run Fixed DDIM sampling.

Usage:
  python phase4_fix_test.py -g 7
"""
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from utility.data import PanoramaDataset
from utility.schedule import make_beta_schedule
from model.unet import PixelUNet

H, W = 512, 1024
SCALE = 4
T = 1000
LR = 2e-4
DATA_DIR = "lau_dataset/sun_test"
OUTPUT_DIR = "phase4_output/exp8_fix_test"


def to_pil(t):
    a = t[0].cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.clip((a + 1) * 127.5, 0, 255).astype(np.uint8))


@torch.no_grad()
def ddim_sample_fixed(unet, cond, device, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, steps=50):
    """CORRECTED DDIM: indices high->low, proper iterative denoising."""
    B = cond.shape[0]
    x = torch.randn(B, 3, H, W, device=device)
    indices = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=device)

    for i in range(steps):
        t = indices[i]
        tp = indices[i + 1] if i + 1 < steps else -1
        t_norm = torch.full((B,), t.item() / T, device=device)

        eps = unet(x, cond, t_norm)
        alpha_t = sqrt_alphas_cumprod[t]
        sigma_t = sqrt_one_minus_alphas_cumprod[t]
        x0_pred = (x - sigma_t * eps) / alpha_t

        if tp >= 0:
            alpha_p = sqrt_alphas_cumprod[tp]
            sigma_p = sqrt_one_minus_alphas_cumprod[tp]
            x = alpha_p * x0_pred + sigma_p * eps
        else:
            x = x0_pred

    return x


def main(gpu=0):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load single image
    dataset = PanoramaDataset(DATA_DIR, (H, W), SCALE)
    lr_img, hr_img = dataset[0]
    lr_img = lr_img.unsqueeze(0).to(device)
    hr_img = hr_img.unsqueeze(0).to(device)

    # Baselines
    to_pil(hr_img).save(os.path.join(OUTPUT_DIR, "hr.png"))
    lr_up = F.interpolate(lr_img, size=(H, W), mode='bicubic', align_corners=False)
    to_pil(lr_up).save(os.path.join(OUTPUT_DIR, "bicubic.png"))

    # ISHT base
    from torch_harmonics import RealSHT, InverseRealSHT
    sht = RealSHT(H, W, lmax=255, mmax=255).to(device)
    isht = InverseRealSHT(H, W, lmax=255, mmax=255).to(device)
    base = isht(sht(lr_up))
    to_pil(base).save(os.path.join(OUTPUT_DIR, "base.png"))

    # Condition
    cond = torch.cat([lr_up, base], dim=1)  # [1, 6, H, W]

    # U-Net: match exp_base config (BASE_CH=32, all OFF)
    unet = PixelUNet(
        in_ch=3, cond_ch=6, base_ch=32, time_dim=256,
        use_polar_moe=False, use_laplacian=False,
        use_circular_conv=False, use_coord_embed=False,
        use_spherical_attn=False, hr_size=(H, W),
    ).to(device)

    betas = make_beta_schedule(T).to(device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    n_params = sum(p.numel() for p in unet.parameters())
    print(f"Model: {n_params:,} params, base_ch=32, all innovations OFF")
    print(f"Target: HR (noise prediction), Epochs: 200, LR: {LR}")

    optimizer = torch.optim.Adam(unet.parameters(), lr=LR)

    # Train 200 epochs (same as Exp 6)
    from tqdm import tqdm
    pbar = tqdm(range(1, 201), desc="[train]", unit="ep")
    for epoch in pbar:
        unet.train()
        t = torch.randint(0, T, (1,), device=device)
        alpha_t = sqrt_alphas_cumprod[t].view(1, 1, 1, 1)
        sigma_t = sqrt_one_minus_alphas_cumprod[t].view(1, 1, 1, 1)

        noise = torch.randn_like(hr_img)
        noisy = alpha_t * hr_img + sigma_t * noise
        pred_noise = unet(noisy, cond, t.float() / T)
        if isinstance(pred_noise, tuple):
            pred_noise = pred_noise[0]
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            pbar.set_postfix(train=f"{loss.item():.4f}")
    print(f"Final train loss: {loss.item():.6f}")

    # Save weights
    weight_path = os.path.join(OUTPUT_DIR, "unet_e200.pt")
    torch.save(unet.state_dict(), weight_path)
    print(f"Saved weights to {weight_path}")

    # ---- Run Fixed DDIM sampling ----
    print("\n=== Running Fixed DDIM (50 steps) ===")
    unet.eval()
    with torch.no_grad():
        sr = ddim_sample_fixed(unet, cond, device, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, steps=50)
    mse = F.mse_loss(sr, hr_img).item()
    print(f"Fixed DDIM MSE: {mse:.6f}")

    out_path = os.path.join(OUTPUT_DIR, "recon_fixed_ddim.png")
    to_pil(sr).save(out_path)
    print(f"Saved reconstruction to {out_path}")

    # Also compare: what would buggy DDIM produce?
    print("\n=== Running Buggy DDIM (50 steps) for comparison ===")
    @torch.no_grad()
    def ddim_sample_buggy():
        B = cond.shape[0]
        x = torch.randn(B, 3, H, W, device=device)
        indices = torch.linspace(0, T - 1, 50, dtype=torch.long, device=device)
        indices_prev = torch.cat([indices[1:], torch.tensor([-1], device=device)])
        for i in range(49, -1, -1):
            t, tp = indices[i], indices_prev[i]
            t_norm = torch.full((B,), t.item() / T, device=device)
            eps = unet(x, cond, t_norm)
            if isinstance(eps, tuple):
                eps = eps[0]
            a_t, s_t = sqrt_alphas_cumprod[t], sqrt_one_minus_alphas_cumprod[t]
            x0_pred = (x - s_t * eps) / a_t
            if tp >= 0:
                a_p, s_p = sqrt_alphas_cumprod[tp], sqrt_one_minus_alphas_cumprod[tp]
                x = a_p * x0_pred + s_p * eps
            else:
                x = x0_pred
        return x

    sr_buggy = ddim_sample_buggy()
    mse_buggy = F.mse_loss(sr_buggy, hr_img).item()
    print(f"Buggy DDIM MSE: {mse_buggy:.6f}")
    to_pil(sr_buggy).save(os.path.join(OUTPUT_DIR, "recon_buggy_ddim.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", type=int, default=0)
    args = parser.parse_args()
    main(gpu=args.gpu)
