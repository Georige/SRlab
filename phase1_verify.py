"""Phase 1: SHT/ISHT verification and spectral energy analysis."""
import torch
import numpy as np
from PIL import Image
from torch_harmonics import RealSHT, InverseRealSHT
import matplotlib.pyplot as plt

# Paths
IMG_PATH = "lau_dataset/sun_test/HR/058.jpg"
OUTPUT_DIR = "phase1_output"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_panorama(path, target_H=None, target_W=None):
    """Load ERP panorama, return [1, C, H, W] tensor normalized to [-1,1]."""
    img = Image.open(path).convert("RGB")
    if target_H and target_W:
        img = img.resize((target_W, target_H), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # [H, W, C] -> [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


# ---- Step 2: Reconstruction accuracy at different bandwidths ----
print("=" * 60)
print("Step 2: SHT/ISHT reconstruction accuracy")
print("=" * 60)

img = load_panorama(IMG_PATH)
C, H, W = 3, img.shape[-2], img.shape[-1]
print(f"Image: {H}x{W}")

L_candidates = [32, 64, 128, 256, 511]
for L in L_candidates:
    sht = RealSHT(H, W, lmax=L, mmax=L).to(device)
    isht = InverseRealSHT(H, W, lmax=L, mmax=L).to(device)
    coeffs = sht(img)
    recon = isht(coeffs)
    mae = (img - recon).abs().mean().item()
    max_err = (img - recon).abs().max().item()
    n_coeffs = L * L * 3  # 3 channels
    ratio = n_coeffs / (H * W * 3) * 100
    print(f"  L={L:3d}  coeffs={n_coeffs:8d} ({ratio:5.1f}%)  MAE={mae:.4e}  max_err={max_err:.4e}")

# ---- Step 3: Spherical harmonic energy spectrum ----
print("\n" + "=" * 60)
print("Step 3: Spherical harmonic energy spectrum")
print("=" * 60)

L_full = min(H, (W - 1) // 2)
sht = RealSHT(H, W, lmax=L_full, mmax=L_full).to(device)
coeffs = sht(img)  # [1, 3, L_full^2], complex

# Average energy per degree l across channels
energy_per_l = []
idx = 0
for l in range(L_full):
    num_m = 2 * l + 1
    block = coeffs[0, :, idx:idx + num_m]  # [3, 2l+1]
    energy = (block.abs() ** 2).sum().item() / num_m
    energy_per_l.append(energy)
    idx += num_m

cumsum = np.cumsum(energy_per_l)
total = cumsum[-1]
# Find L_lo that captures 99% and 99.9% of energy
L_99 = int(np.searchsorted(cumsum / total, 0.99))
L_999 = int(np.searchsorted(cumsum / total, 0.999))

print(f"Total energy: {total:.2f}")
print(f"99%  energy at L_lo = {L_99}  ({L_99/L_full*100:.1f}% of L_full)")
print(f"99.9% energy at L_lo = {L_999} ({L_999/L_full*100:.1f}% of L_full)")
print(f"Compression ratio at L_99: {L_99**2 / L_full**2 * 100:.1f}%")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Energy spectrum
ax1.semilogy(range(L_full), energy_per_l, linewidth=0.5)
ax1.set_xlabel("Degree l")
ax1.set_ylabel("Avg energy per mode (log)")
ax1.set_title(f"Spherical Harmonic Energy Spectrum (L_full={L_full})")
ax1.axvline(L_99, color="red", linestyle="--", label=f"99% energy: L={L_99}")
ax1.axvline(L_999, color="orange", linestyle="--", label=f"99.9% energy: L={L_999}")
ax1.legend()

# Cumulative energy
ax2.plot(range(L_full), cumsum / total, linewidth=0.5)
ax2.set_xlabel("Degree l")
ax2.set_ylabel("Cumulative energy fraction")
ax2.set_title("Cumulative Energy")
ax2.axhline(0.99, color="red", linestyle="--")
ax2.axhline(0.999, color="orange", linestyle="--")
ax2.axvline(L_99, color="red", linestyle="--")
ax2.axvline(L_999, color="orange", linestyle="--")

plt.tight_layout()
outpath = os.path.join(OUTPUT_DIR, "spectrum.png")
plt.savefig(outpath, dpi=150)
print(f"\nSaved: {outpath}")
print("Done.")
