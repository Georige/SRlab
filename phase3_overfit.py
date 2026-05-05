"""Phase 3: Single-image overfitting test.

Proves architecture capacity: can the model memorize one panorama to perfection?
If it can't overfit a single image, the architecture itself is the bottleneck.

Usage:
  python phase3_overfit.py                                    # base config
  python phase3_overfit.py -c exp_full -g 1 -n full_overfit   # full innovations
"""
import argparse
import importlib
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch_harmonics import RealSHT, InverseRealSHT
from tqdm import tqdm

from utility.data import PanoramaDataset
from model.unet import PixelUNet
from model.diffusion import PixelDiffusion


def main(gpu: int = 0, exp_name: str = "overfit_base", config_module: str = "diffusion_config"):
    cfg = importlib.import_module(f"config.{config_module}")
    output_dir = os.path.join("phase3_output", exp_name)
    os.makedirs(output_dir, exist_ok=True)

    H, W = cfg.HR_SIZE
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    # --- Load a single panorama ---
    dataset = PanoramaDataset(cfg.DATA_DIR, cfg.HR_SIZE, cfg.SCALE)
    lr_img, hr_img = dataset[0]  # first image only
    lr_img = lr_img.unsqueeze(0).to(device)
    hr_img = hr_img.unsqueeze(0).to(device)

    def to_pil(t):
        a = t[0].cpu().permute(1, 2, 0).numpy()
        return Image.fromarray(np.clip((a + 1) * 127.5, 0, 255).astype(np.uint8))

    # Save ground truth and baselines
    to_pil(hr_img).save(os.path.join(output_dir, "hr.png"))
    lr_up = F.interpolate(lr_img, size=(H, W), mode='bicubic', align_corners=False)
    to_pil(lr_up).save(os.path.join(output_dir, "bicubic.png"))

    # --- ISHT conditioning ---
    sht_cond = RealSHT(H, W, lmax=cfg.L_COND, mmax=cfg.L_COND).to(device)
    isht_cond = InverseRealSHT(H, W, lmax=cfg.L_COND, mmax=cfg.L_COND).to(device)
    base = isht_cond(sht_cond(lr_up))  # ISHT(L=255) reconstruction
    to_pil(base).save(os.path.join(output_dir, "base.png"))

    # Multi-scale ISHT
    ms_sht_isht = []
    for L_val, factor in cfg.MS_COND:
        sht = RealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        isht = InverseRealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        ms_sht_isht.append((sht, isht, factor))

    # Optional: HF residual
    sht_hf, isht_hf = None, None
    cond_ch = 6
    if cfg.USE_HF_RESIDUAL:
        sht_hf = RealSHT(H, W, lmax=cfg.L_HF, mmax=cfg.L_HF).to(device)
        isht_hf = InverseRealSHT(H, W, lmax=cfg.L_HF, mmax=cfg.L_HF).to(device)
        cond_ch += 3

    # --- Model ---
    unet = PixelUNet(in_ch=3, cond_ch=cond_ch, base_ch=cfg.BASE_CH, time_dim=cfg.TIME_DIM,
                     use_polar_moe=cfg.USE_POLAR_MOE,
                     use_laplacian=cfg.USE_LAPLACIAN_PYRAMID,
                     use_circular_conv=cfg.USE_CIRCULAR_CONV,
                     use_coord_embed=cfg.USE_COORD_EMBED,
                     use_spherical_attn=cfg.USE_SPHERICAL_ATTN,
                     hr_size=cfg.HR_SIZE).to(device)

    diffusion = PixelDiffusion(
        unet, sht_cond, isht_cond, ms_sht_isht,
        use_hf_residual=cfg.USE_HF_RESIDUAL, sht_hf=sht_hf, isht_hf=isht_hf,
        use_latitude_weight=cfg.USE_LATITUDE_WEIGHT, pole_weight=cfg.POLE_WEIGHT,
        use_laplacian=cfg.USE_LAPLACIAN_PYRAMID, lp_lambdas=cfg.LP_LAMBDAS,
        T=cfg.TRAIN_TIMESTEPS, hr_size=cfg.HR_SIZE,
    ).to(device)

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=cfg.LR_BASE)
    n_params = sum(p.numel() for p in diffusion.parameters())

    print(f"Phase3 Overfit: {exp_name}  |  Config: {config_module}  |  GPU: {gpu}")
    print(f"Params: {n_params:,}  |  HR: {cfg.HR_SIZE}  |  Epochs: 1000")
    print(f"Innovations: HF_res={cfg.USE_HF_RESIDUAL}, LatW={cfg.USE_LATITUDE_WEIGHT}, "
          f"PolMoE={cfg.USE_POLAR_MOE}, LapPyr={cfg.USE_LAPLACIAN_PYRAMID}, "
          f"SphUNet=({cfg.USE_CIRCULAR_CONV},{cfg.USE_COORD_EMBED},{cfg.USE_SPHERICAL_ATTN})")
    print(f"Output: {output_dir}")

    pbar = tqdm(range(1, 1001), desc=f"[{exp_name}]", unit="ep")
    for epoch in pbar:
        diffusion.train()
        loss = diffusion(lr_img, hr_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Sample every 100 epochs (and at epoch 1)
        if epoch == 1 or epoch % 100 == 0 or epoch == 1000:
            diffusion.eval()
            with torch.no_grad():
                sr = diffusion.sample(lr_img, steps=cfg.INFER_STEPS)
            to_pil(sr).save(os.path.join(output_dir, f"e{epoch:04d}.png"))
            pbar.set_postfix(train=f"{loss.item():.4f}",
                             v_img=f"{F.mse_loss(sr, hr_img).item():.4f}")
        else:
            pbar.set_postfix(train=f"{loss.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase3: Single-image overfitting test")
    parser.add_argument("--config", "-c", type=str, default="diffusion_config")
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--name", "-n", type=str, default="overfit_base")
    args = parser.parse_args()
    main(gpu=args.gpu, exp_name=args.name, config_module=args.config)
