"""Phase 2: Spectral Super-Resolution — MLP Baseline (Residual Learning).

coeffs_lo ──→ zero-pad to L_hi ──→ + ──→ ISHT ──→ SR_image
                       ↑
MLP(coeffs_lo): residual ─┘

MLP only learns the high-frequency correction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch_harmonics import RealSHT, InverseRealSHT
from tqdm import tqdm
import os, sys

# ---- Config ----
HR_SIZE = (512, 1024)       # (H, W)
L_LO, L_HI = 64, 128
SCALE = 4
BATCH_SIZE = 4
EPOCHS = 150
LR_BASE = 1e-3
HIDDEN = 512
DROPOUT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "lau_dataset/sun_test"
OUTPUT_DIR = "phase2_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")
print(f"L_lo={L_LO}, L_hi={L_HI}, hidden={HIDDEN}, dropout={DROPOUT}")


# ---- Data ----
class PanoramaDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, hr_size, scale):
        hr_files = sorted(os.listdir(os.path.join(data_dir, "HR")))
        self.samples = []
        for f in hr_files:
            hr_path = os.path.join(data_dir, "HR", f)
            lr_path = os.path.join(data_dir, "LR", f"X{scale}", f)
            if os.path.exists(lr_path):
                self.samples.append((lr_path, hr_path))
        self.hr_size = hr_size
        self.lr_size = (hr_size[0] // scale, hr_size[1] // scale)

    def __len__(self):
        return len(self.samples)

    def _load(self, path, target_size):
        img = Image.open(path).convert("RGB").resize(
            (target_size[1], target_size[0]), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def __getitem__(self, idx):
        lr_path, hr_path = self.samples[idx]
        lr = self._load(lr_path, self.lr_size)
        hr = self._load(hr_path, self.hr_size)
        return lr, hr


# ---- Model: Residual MLP ----
class SpectralSR_ResMLP(nn.Module):
    def __init__(self, L_lo, L_hi, C=3, hidden=512, dropout=0.3):
        super().__init__()
        in_dim = C * L_lo * L_lo * 2      # 3×64×64×2 = 24576
        out_dim = C * L_hi * L_hi * 2     # 3×128×128×2 = 98304
        self.L_lo = L_lo
        self.L_hi = L_hi
        self.C = C
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, coeffs_lo):
        # coeffs_lo: [B, C, L_lo, L_lo] complex
        B = coeffs_lo.shape[0]
        x = torch.view_as_real(coeffs_lo).reshape(B, -1)
        out = self.net(x)
        residual = out.reshape(B, self.C, self.L_hi, self.L_hi, 2)
        residual_c = torch.view_as_complex(residual.contiguous())

        # Zero-pad low-bandwidth coeffs to L_hi size, then add residual
        pad = self.L_hi - self.L_lo
        coeffs_padded = F.pad(coeffs_lo, [0, pad, 0, pad])  # [B, C, L_hi, L_hi]
        return coeffs_padded + residual_c


# ---- Training ----
def main():
    dataset = PanoramaDataset(DATA_DIR, HR_SIZE, SCALE)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE)

    H, W = HR_SIZE
    sht_lo = RealSHT(H, W, lmax=L_LO, mmax=L_LO).to(DEVICE)
    sht_hi = RealSHT(H, W, lmax=L_HI, mmax=L_HI).to(DEVICE)
    isht_hi = InverseRealSHT(H, W, lmax=L_HI, mmax=L_HI).to(DEVICE)

    model = SpectralSR_ResMLP(L_LO, L_HI, hidden=HIDDEN, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_BASE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Train: {n_train}, Val: {n_val}, Params: {n_params:,}")

    pbar = tqdm(range(1, EPOCHS + 1), desc="Training", unit="ep")
    for epoch in pbar:
        model.train()
        train_loss = 0.0
        batch_pbar = tqdm(train_loader, desc=f"  Epoch {epoch}", leave=False, unit="b")
        for lr_imgs, hr_imgs in batch_pbar:
            lr_imgs = lr_imgs.to(DEVICE)
            hr_imgs = hr_imgs.to(DEVICE)

            lr_up = F.interpolate(lr_imgs, size=(H, W), mode="bicubic",
                                  align_corners=False)

            coeffs_lo = sht_lo(lr_up)              # [B, 3, 64, 64]
            coeffs_pred = model(coeffs_lo)          # [B, 3, 128, 128] residual
            sr_imgs = isht_hi(coeffs_pred)          # [B, 3, H, W]

            loss = F.mse_loss(sr_imgs, hr_imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_coeff_loss = 0.0
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs = lr_imgs.to(DEVICE)
                hr_imgs = hr_imgs.to(DEVICE)
                lr_up = F.interpolate(lr_imgs, size=(H, W), mode="bicubic",
                                      align_corners=False)
                coeffs_lo = sht_lo(lr_up)
                coeffs_pred = model(coeffs_lo)
                sr_imgs = isht_hi(coeffs_pred)
                val_loss += F.mse_loss(sr_imgs, hr_imgs).item()

                coeffs_gt = sht_hi(hr_imgs)
                val_coeff_loss += F.mse_loss(
                    torch.view_as_real(coeffs_pred),
                    torch.view_as_real(coeffs_gt)).item()

        pbar.set_postfix(
            train=f"{train_loss/len(train_loader):.4f}",
            val=f"{val_loss/len(val_loader):.4f}",
            coeff=f"{val_coeff_loss/len(val_loader):.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.1e}")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "mlp_residual.pt"))
    print(f"\nModel saved to {OUTPUT_DIR}/mlp_residual.pt")

    # ---- Evaluation ----
    model.eval()
    lr_imgs, hr_imgs = next(iter(val_loader))
    lr_imgs = lr_imgs.to(DEVICE)
    hr_imgs = hr_imgs.to(DEVICE)
    lr_up = F.interpolate(lr_imgs, size=(H, W), mode="bicubic", align_corners=False)

    with torch.no_grad():
        coeffs_lo = sht_lo(lr_up)
        coeffs_pred = model(coeffs_lo)
        sr_imgs = isht_hi(coeffs_pred)

    def to_pil(tensor):
        arr = tensor[0].cpu().permute(1, 2, 0).numpy()
        arr = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    to_pil(lr_up).save(os.path.join(OUTPUT_DIR, "sample_lr_bicubic.png"))
    to_pil(sr_imgs).save(os.path.join(OUTPUT_DIR, "sample_sr_resmlp.png"))
    to_pil(hr_imgs).save(os.path.join(OUTPUT_DIR, "sample_hr_gt.png"))

    bicubic_mse = F.mse_loss(lr_up, hr_imgs).item()
    mlp_mse = F.mse_loss(sr_imgs, hr_imgs).item()
    print(f"\nBicubic MSE:  {bicubic_mse:.4f}")
    print(f"ResMLP   MSE: {mlp_mse:.4f}")
    print(f"Gain:         {(1 - mlp_mse/bicubic_mse)*100:.1f}%  (negative = worse)")


if __name__ == "__main__":
    main()
