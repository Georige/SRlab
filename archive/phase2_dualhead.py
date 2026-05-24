"""Phase 2: Attention + dual head, coefficient-space loss, zero-init output.

coeffs_lo → flatten → Linear(24576→1024) → SelfAttn → FFN(1024→1024→1024)
       → split 1024→[512,512] → ┬→ real_head → residual_real
                                 └→ imag_head → residual_imag
       → complex → + padded_coeffs_lo → coeffs_pred

Loss = MSE(coeffs_pred, sht_hi(HR))     ← 系数空间监督
Output layers zero-initialized           ← 默认行为 = bicubic（残差=0）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch_harmonics import RealSHT, InverseRealSHT
from tqdm import tqdm
import os

# ---- Config ----
HR_SIZE = (512, 1024)
L_LO, L_HI = 64, 128
SCALE = 4
BATCH_SIZE = 4
EPOCHS = 150
LR_BASE = 1e-3
HIDDEN = 1024
DROPOUT = 0.3
LAMBDA_COEFF = 1.0   # coefficient-space loss weight (1.0 = pure coeff loss)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "lau_dataset/sun_test"
OUTPUT_DIR = "phase2_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
        return self._load(lr_path, self.lr_size), self._load(hr_path, self.hr_size)


# ---- Model: Proj → SelfAttn → FFN → Split → Dual Head ----
class SpectralSR_DualHead(nn.Module):
    """24576 → 1024 → SelfAttn(1token×1024dim, 8heads) → FFN → split [512,512] → dual head."""
    def __init__(self, L_lo, L_hi, C=3, hidden=1024, dropout=0.3):
        super().__init__()
        in_dim = C * L_lo * L_lo * 2   # 3×64×64×2 = 24576
        out_per_head = C * L_hi * L_hi  # 3×128×128  = 49152 per head
        half = hidden // 2              # 512

        self.L_lo = L_lo
        self.L_hi = L_hi
        self.C = C
        self.hidden = hidden

        # Step 1: Embed input → 1024
        self.proj_in = nn.Linear(in_dim, hidden)

        # Step 2: Multi-head self-attention, embed_dim=1024, 8 heads
        self.attn = nn.MultiheadAttention(
            hidden, num_heads=8, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden)

        # Step 3-4: FFN 1024→1024→1024
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden)

        # Step 5-6: Split 1024 → [512, 512], then real/imag heads
        self.real_head = nn.Sequential(
            nn.Linear(half, half),
            nn.ReLU(),
            nn.Linear(half, out_per_head),
        )
        self.imag_head = nn.Sequential(
            nn.Linear(half, half),
            nn.ReLU(),
            nn.Linear(half, out_per_head),
        )

        # Zero-init output layers → residual starts at 0 → model = bicubic at init
        for head in [self.real_head, self.imag_head]:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, coeffs_lo):
        B = coeffs_lo.shape[0]

        # Flatten complex coefficients: [B, 3, 64, 64] → [B, 24576]
        x = torch.view_as_real(coeffs_lo).reshape(B, -1)

        # Step 1: Project to 1024
        x = self.proj_in(x)                               # [B, 1024]

        # Step 2: Self-attention (single token, 1024-dim, 8 heads × 128d each)
        x = x.unsqueeze(1)                                # [B, 1, 1024]
        attn_out, _ = self.attn(x, x, x)
        x = self.attn_norm(x + attn_out)                  # residual + norm
        x = x.squeeze(1)                                  # [B, 1024]

        # Step 3-4: FFN with residual
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)                    # [B, 1024]

        # Step 5: Split into real / imag feature halves
        half = self.hidden // 2
        real_feat = x[:, :half]                           # [B, 512]
        imag_feat = x[:, half:]                           # [B, 512]

        # Step 6: Dual head decode → [B, C, L_hi, L_hi]
        real = self.real_head(real_feat).reshape(B, self.C, self.L_hi, self.L_hi)
        imag = self.imag_head(imag_feat).reshape(B, self.C, self.L_hi, self.L_hi)

        residual = torch.complex(real, imag)               # [B, C, 128, 128]

        pad = self.L_hi - self.L_lo
        coeffs_padded = F.pad(coeffs_lo, [0, pad, 0, pad])
        return coeffs_padded + residual


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

    model = SpectralSR_DualHead(L_LO, L_HI, hidden=HIDDEN, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_BASE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {DEVICE}")
    print(f"L_lo={L_LO}, L_hi={L_HI}, hidden={HIDDEN}, dropout={DROPOUT}")
    print(f"Train: {n_train}, Val: {n_val}, Params: {n_params:,}")

    pbar = tqdm(range(1, EPOCHS + 1), desc="Training", unit="ep")
    for epoch in pbar:
        model.train()
        train_loss = 0.0
        batch_pbar = tqdm(train_loader, desc=f"  Epoch {epoch}", leave=False, unit="b")
        for lr_imgs, hr_imgs in batch_pbar:
            lr_imgs = lr_imgs.to(DEVICE)
            hr_imgs = hr_imgs.to(DEVICE)
            lr_up = F.interpolate(lr_imgs, size=(H, W), mode="bicubic", align_corners=False)

            coeffs_lo = sht_lo(lr_up)
            coeffs_pred = model(coeffs_lo)

            # Coefficient-space loss (primary)
            coeffs_gt = sht_hi(hr_imgs)
            loss_coeff = F.mse_loss(
                torch.view_as_real(coeffs_pred), torch.view_as_real(coeffs_gt))

            # Image-space loss (auxiliary)
            sr_imgs = isht_hi(coeffs_pred)
            loss_img = F.mse_loss(sr_imgs, hr_imgs)

            loss = LAMBDA_COEFF * loss_coeff + (1 - LAMBDA_COEFF) * loss_img
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                c=f"{loss_coeff.item():.4f}",
                i=f"{loss_img.item():.4f}")

        scheduler.step()

        # Validation
        model.eval()
        val_img = 0.0
        val_coeff = 0.0
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs = lr_imgs.to(DEVICE)
                hr_imgs = hr_imgs.to(DEVICE)
                lr_up = F.interpolate(lr_imgs, size=(H, W), mode="bicubic", align_corners=False)
                coeffs_lo = sht_lo(lr_up)
                coeffs_pred = model(coeffs_lo)
                sr_imgs = isht_hi(coeffs_pred)

                coeffs_gt = sht_hi(hr_imgs)
                val_coeff += F.mse_loss(
                    torch.view_as_real(coeffs_pred), torch.view_as_real(coeffs_gt)).item()
                val_img += F.mse_loss(sr_imgs, hr_imgs).item()

        pbar.set_postfix(
            train=f"{train_loss/len(train_loader):.4f}",
            v_img=f"{val_img/len(val_loader):.4f}",
            v_coeff=f"{val_coeff/len(val_loader):.6f}",
            lr=f"{scheduler.get_last_lr()[0]:.1e}")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "dualhead.pt"))
    print(f"\nModel saved to {OUTPUT_DIR}/dualhead.pt")

    # ---- Eval ----
    model.eval()
    lr_imgs, hr_imgs = next(iter(val_loader))
    lr_imgs = lr_imgs.to(DEVICE)
    hr_imgs = hr_imgs.to(DEVICE)
    lr_up = F.interpolate(lr_imgs, size=(H, W), mode="bicubic", align_corners=False)

    with torch.no_grad():
        sr_imgs = isht_hi(model(sht_lo(lr_up)))

    def to_pil(tensor):
        arr = tensor[0].cpu().permute(1, 2, 0).numpy()
        arr = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    to_pil(lr_up).save(os.path.join(OUTPUT_DIR, "sample_lr_bicubic.png"))
    to_pil(sr_imgs).save(os.path.join(OUTPUT_DIR, "sample_sr_dualhead.png"))
    to_pil(hr_imgs).save(os.path.join(OUTPUT_DIR, "sample_hr_gt.png"))

    bicubic_mse = F.mse_loss(lr_up, hr_imgs).item()
    mlp_mse = F.mse_loss(sr_imgs, hr_imgs).item()
    print(f"\nBicubic MSE: {bicubic_mse:.4f}")
    print(f"DualHead MSE: {mlp_mse:.4f}")
    print(f"Gain:         {(1 - mlp_mse/bicubic_mse)*100:.1f}%")


if __name__ == "__main__":
    main()
