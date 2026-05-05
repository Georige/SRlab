"""One-shot: parse existing training logs and plot train/v_img curves."""
import re
import glob
from loss_plotter import update_loss_plot  # noqa: E402

# Currently this only works with CSV-based data (from restarted experiments).
# For the old log format, parse directly and generate the plot.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os as _os
LOG_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "logs")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for log_path in sorted(glob.glob(f"{LOG_DIR}/*.log")):
    name = log_path.split("/")[-1].replace(".log", "")
    epochs, trains, v_imgs = [], [], []

    with open(log_path) as f:
        for line in f:
            m = re.search(r"(\d+)/(\d+).*train=([\d.]+).*v_img=([\d.]+)", line)
            if m:
                epoch = int(m.group(1))
                total = int(m.group(2))
                # Only keep each epoch once (last occurrence = the final bar for that epoch)
                if not epochs or epoch != epochs[-1]:
                    epochs.append(epoch)
                    trains.append(float(m.group(3)))
                    v_imgs.append(float(m.group(4)))
                else:
                    trains[-1] = float(m.group(3))
                    v_imgs[-1] = float(m.group(4))

    if not epochs:
        print(f"  skip {name}: no data")
        continue

    ax1.plot(epochs, trains, alpha=0.7, label=name)
    ax2.plot(epochs, v_imgs, alpha=0.7, label=name)
    print(f"  {name}: {len(epochs)} epochs, train {trains[-1]:.4f}, v_img {v_imgs[-1]:.4f}")

ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train Loss"); ax1.set_title("Training Loss")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("v_img MSE"); ax2.set_title("Validation (Image MSE)")
ax1.legend(fontsize=8); ax2.legend(fontsize=8)
ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{LOG_DIR}/loss_curves.png", dpi=100)
print(f"\nSaved → {LOG_DIR}/loss_curves.png")
