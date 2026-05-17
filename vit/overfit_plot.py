"""Phase3 overfit visualizations: loss curves + image progression grid."""

import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def update_curves(exp_name, losses, sample_epochs, sample_mses, log_dir="logs",
                  val_psnrs=None, val_ssims=None):
    """Generate loss + metrics curves for overfit experiment.

    When val_psnrs and val_ssims are provided, uses a 2x2 grid:
      - top-left: train loss (log scale)
      - top-right: val MSE
      - bottom-left: val PSNR
      - bottom-right: val SSIM
    Otherwise falls back to the original 2-panel layout.
    """
    os.makedirs(log_dir, exist_ok=True)

    has_extra = val_psnrs is not None and val_ssims is not None

    if has_extra:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        (ax1, ax2), (ax3, ax4) = axes
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(losses, lw=0.5, alpha=0.8)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train Loss")
    ax1.set_title(f"{exp_name} — Train Loss")
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    if sample_epochs:
        ax2.plot(sample_epochs, sample_mses, 'o-', markersize=4, lw=1, color='#e74c3c')
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("MSE vs HR")
        ax2.set_title(f"{exp_name} — Val MSE")
        ax2.grid(True, alpha=0.3)

        if has_extra:
            ax3.plot(sample_epochs, val_psnrs, 'o-', markersize=4, lw=1, color='#2ecc71')
            ax3.set_xlabel("Epoch"); ax3.set_ylabel("PSNR (dB)")
            ax3.set_title(f"{exp_name} — Val PSNR")
            ax3.grid(True, alpha=0.3)

            ax4.plot(sample_epochs, val_ssims, 'o-', markersize=4, lw=1, color='#3498db')
            ax4.set_xlabel("Epoch"); ax4.set_ylabel("SSIM")
            ax4.set_title(f"{exp_name} — Val SSIM")
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(log_dir, f"overfit_{exp_name}_curves.png")
    plt.savefig(path, dpi=100)
    plt.close(fig)
    print(f"  curves → {path}")


def make_progression(output_dir, max_cols=10):
    """Stitch all e*.png samples into a single progression grid."""
    files = sorted(glob.glob(os.path.join(output_dir, "e*.png")))
    if not files:
        return

    images = [Image.open(f) for f in files]
    labels = [os.path.basename(f).replace(".png", "") for f in files]

    n = len(images)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols
    w, h = images[0].size

    # Thumbnail size: 256px wide per frame
    tw = 256
    th = int(h * tw / w)

    grid = Image.new('RGB', (tw * cols + 2 * (cols + 1), th * rows + 24 * rows + 2 * (rows + 1)),
                     color=(40, 40, 40))

    for i, (img, label) in enumerate(zip(images, labels)):
        r, c = i // cols, i % cols
        x = 2 + c * (tw + 2)
        y = 2 + r * (th + 26)
        grid.paste(img.resize((tw, th), Image.LANCZOS), (x, y + 24))
        # Simple text via PIL is limited; the filename serves as label.
        # We'll add epoch number by drawing it on a small strip.
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()
        draw.text((x + 4, y + 4), label, fill=(200, 200, 200), font=font)

    path = os.path.join(output_dir, "progression.png")
    grid.save(path)
    print(f"  progression → {path}")
