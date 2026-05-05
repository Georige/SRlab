"""Loss curve plotting: read weight/*/losses.csv and generate comparison plot."""
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def update_loss_plot(current_exp=None, log_dir="logs", weight_dir="weight"):
    """Read all {weight_dir}/*/losses.csv and regenerate {log_dir}/loss_curves.png."""
    os.makedirs(log_dir, exist_ok=True)

    csv_files = sorted(glob.glob(f"{weight_dir}/*/losses.csv"))
    if not csv_files:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for f in csv_files:
        name = f.split("/")[1]
        epochs, trains, v_imgs = [], [], []
        with open(f) as fh:
            for line in fh:
                parts = line.strip().split(",")
                if parts[0] == "epoch":
                    continue
                epochs.append(int(parts[0]))
                trains.append(float(parts[1]))
                v_imgs.append(float(parts[2]))
        if not epochs:
            continue
        lw = 1.5 if name == current_exp else 0.7
        ax1.plot(epochs, trains, alpha=0.8, lw=lw, label=name)
        ax2.plot(epochs, v_imgs, alpha=0.8, lw=lw, label=name)

    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train Loss"); ax1.set_title("Training Loss")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("v_img MSE"); ax2.set_title("Validation (Image MSE)")
    ax1.legend(fontsize=8); ax2.legend(fontsize=8)
    ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{log_dir}/loss_curves.png", dpi=100)
    plt.close(fig)
