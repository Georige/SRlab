"""Phase 2: Pixel-space Conditional Diffusion with ISHT structure conditioning.

ISHT path (fidelity/condition only):
  LR → bicubic↑ → SHT(L=255) → ISHT(L=255) → base [B,3,512,1024]

Multi-scale ISHT injection:
  L=128 ISHT → enc2 (256×512),  L=64 ISHT → enc3 (128×256),
  L=32  ISHT → bottleneck (64×128)

Diffusion path (residual generation):
  residual = HR - base,  noisy residual + cond → U-Net → ε_pred
  Inference: noise → DDIM 50-step → residual_pred,  SR = base + residual_pred

Auto-resume: looks for latest weight/{exp_name}/diff_e*.pt, loads model+optimizer+scheduler+epoch.

Usage:
  python phase2_diffusion.py                                    # base config
  python phase2_diffusion.py -c exp_hf_residual -g 1 -n hf     # HF residual
  python phase2_diffusion.py -c exp_polar -g 2 -n polar        # polar
  python phase2_diffusion.py -c exp_full -g 3 -n full          # all innovations
"""
import argparse
import glob
import importlib
import re
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
from vit.loss_plotter import update_loss_plot
from vit.monitor import TrainingMonitor, plot_dashboard


def _find_latest_checkpoint(weight_dir):
    """Find the latest checkpoint file, return (path, epoch) or (None, 0)."""
    ckpts = glob.glob(os.path.join(weight_dir, "diff_e*.pt"))
    if not ckpts:
        return None, 0
    latest = max(ckpts, key=lambda p: int(re.findall(r"diff_e(\d+)\.pt", p)[0]))
    epoch = int(re.findall(r"diff_e(\d+)\.pt", latest)[0])
    return latest, epoch


def main(gpu: int = 0, exp_name: str = "default", config_module: str = "diffusion_config"):
    # Load experiment config (dynamic import)
    cfg = importlib.import_module(f"config.{config_module}")
    weight_dir = os.path.join(cfg.WEIGHT_DIR, exp_name)
    output_dir = os.path.join(cfg.OUTPUT_DIR, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)

    # Fixed seed for reproducible train/val split across runs
    torch.manual_seed(42)

    dataset = PanoramaDataset(cfg.DATA_DIR, cfg.HR_SIZE, cfg.SCALE)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1)

    H, W = cfg.HR_SIZE
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    # Main SHT/ISHT: L=255
    sht_cond = RealSHT(H, W, lmax=cfg.L_COND, mmax=cfg.L_COND).to(device)
    isht_cond = InverseRealSHT(H, W, lmax=cfg.L_COND, mmax=cfg.L_COND).to(device)

    # Multi-scale SHT/ISHT for U-Net encoder injection
    ms_sht_isht = []
    for L_val, factor in cfg.MS_COND:
        sht = RealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        isht = InverseRealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        ms_sht_isht.append((sht, isht, factor))

    # Optional: HF residual condition (bicubic - ISHT(L_HF))
    sht_hf, isht_hf = None, None
    cond_ch = 6  # bicubic + ISHT(L=255)
    if cfg.USE_HF_RESIDUAL:
        sht_hf = RealSHT(H, W, lmax=cfg.L_HF, mmax=cfg.L_HF).to(device)
        isht_hf = InverseRealSHT(H, W, lmax=cfg.L_HF, mmax=cfg.L_HF).to(device)
        cond_ch += 3  # + HF residual

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.EPOCHS)
    monitor = TrainingMonitor(exp_name, weight_dir, diffusion, device=device.type)

    # ---- Resume from checkpoint if available ----
    start_epoch = 1
    ckpt_path, ckpt_epoch = _find_latest_checkpoint(weight_dir)
    if ckpt_path:
        print(f"Resuming from checkpoint: {ckpt_path} (epoch {ckpt_epoch})")
        state = torch.load(ckpt_path, map_location=device)
        diffusion.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = ckpt_epoch + 1
        for _ in range(ckpt_epoch):
            scheduler.step()

    n_params = sum(p.numel() for p in diffusion.parameters())
    print(f"Experiment: {exp_name}  |  Config: {config_module}  |  GPU: {gpu}")
    print(f"Weight dir: {weight_dir}  |  Output dir: {output_dir}")
    print(f"HR={cfg.HR_SIZE}, L_cond={cfg.L_COND}, base_ch={cfg.BASE_CH}")
    print(f"Multi-scale ISHT: {[(l, f) for l, f in cfg.MS_COND]}")
    print(f"Innovations: HF_res={cfg.USE_HF_RESIDUAL}, LatW={cfg.USE_LATITUDE_WEIGHT}, PolMoE={cfg.USE_POLAR_MOE}, LapPyr={cfg.USE_LAPLACIAN_PYRAMID}, SphUNet=({cfg.USE_CIRCULAR_CONV},{cfg.USE_COORD_EMBED},{cfg.USE_SPHERICAL_ATTN})")
    print(f"Train: {n_train}, Val: {n_val}, Params: {n_params:,}")
    print(f"Diffusion: {cfg.TRAIN_TIMESTEPS} train steps, {cfg.INFER_STEPS} inference steps")
    if ckpt_path:
        print(f"Resumed at epoch {start_epoch}/{cfg.EPOCHS}")

    pbar = tqdm(range(start_epoch, cfg.EPOCHS + 1), desc=f"[{exp_name}]", unit="ep")
    for epoch in pbar:
        monitor.start_epoch()
        diffusion.train()
        train_loss = 0.0
        n_train_samples = 0
        for lr_imgs, hr_imgs in tqdm(train_loader, desc=f"  Epoch {epoch}", leave=False):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            loss = diffusion(lr_imgs, hr_imgs)
            optimizer.zero_grad()
            loss.backward()
            monitor.record_batch_grad()
            optimizer.step()
            train_loss += loss.item()
            n_train_samples += lr_imgs.shape[0]

        scheduler.step()

        # Validation
        diffusion.eval()
        val_img, bic_mse, base_mse = 0.0, 0.0, 0.0
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                sr_imgs = diffusion.sample(lr_imgs, steps=cfg.INFER_STEPS)
                val_img += F.mse_loss(sr_imgs, hr_imgs).item()

                lr_up = F.interpolate(lr_imgs, size=(H, W), mode='bicubic', align_corners=False)
                bic_mse += F.mse_loss(lr_up, hr_imgs).item()

                cond, base, _ = diffusion.make_condition(lr_imgs)
                base_mse += F.mse_loss(base, hr_imgs).item()

        t_avg = train_loss / len(train_loader)
        v_avg = val_img / len(val_loader)

        pbar.set_postfix(train=f"{t_avg:.4f}", v_img=f"{v_avg:.4f}",
                         bic=f"{bic_mse/len(val_loader):.4f}",
                         base=f"{base_mse/len(val_loader):.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.1e}")

        # Log epoch metrics to CSV and refresh loss plot
        import csv as _csv
        _csv_path = os.path.join(weight_dir, "losses.csv")
        _write_header = not os.path.exists(_csv_path)
        with open(_csv_path, "a") as _f:
            _w = _csv.writer(_f)
            if _write_header:
                _w.writerow(["epoch", "train", "v_img"])
            _w.writerow([epoch, t_avg, v_avg])
        update_loss_plot(exp_name)
        monitor.end_epoch(epoch, t_avg, v_avg, n_train_samples)
        plot_dashboard()

        # Periodic checkpoint + sample
        if epoch % 100 == 0 or epoch == cfg.EPOCHS:
            ckpt = os.path.join(weight_dir, f"diff_e{epoch}.pt")
            torch.save({
                "model": diffusion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            }, ckpt)

            lr_imgs, hr_imgs = next(iter(val_loader))
            lr_imgs, hr_imgs = lr_imgs[:1].to(device), hr_imgs[:1].to(device)

            with torch.no_grad():
                sr_imgs = diffusion.sample(lr_imgs, steps=cfg.INFER_STEPS)

            def to_pil(t):
                a = t[0].cpu().permute(1, 2, 0).numpy()
                return Image.fromarray(np.clip((a + 1) * 127.5, 0, 255).astype(np.uint8))

            to_pil(hr_imgs).save(os.path.join(output_dir, "diff_hr.png"))
            lr_up = F.interpolate(lr_imgs, size=(H, W), mode='bicubic', align_corners=False)
            to_pil(lr_up).save(os.path.join(output_dir, "diff_bicubic.png"))
            to_pil(sr_imgs).save(os.path.join(output_dir, f"diff_e{epoch}.png"))
            print(f"  [checkpoint → {ckpt}, sample saved at epoch {epoch}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixel Diffusion + ISHT Super-Resolution")
    parser.add_argument("--config", "-c", type=str, default="diffusion_config",
                        help="Config module name under config/ (default: diffusion_config)")
    parser.add_argument("--gpu", "-g", type=int, default=0, help="GPU device ID (default: 0)")
    parser.add_argument("--name", "-n", type=str, default="default",
                        help="Experiment name, subdirectories under weight/ and phase2_output/")
    args = parser.parse_args()
    main(gpu=args.gpu, exp_name=args.name, config_module=args.config)
