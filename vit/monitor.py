"""TrainingMonitor: per-epoch metrics logging + performance dashboard."""

import os
import csv
import time
import torch
import glob


class TrainingMonitor:
    """Collects per-epoch training metrics and writes to CSV.

    Metrics:
      - train_loss, v_img: primary loss curves
      - grad_norm: total gradient L2 norm (detect explosion/vanishing)
      - update_ratio: ||lr * grad|| / ||param|| (healthy ~1e-3)
      - epoch_time: wall-clock seconds per epoch
      - samples_per_sec: training throughput
      - gpu_mem_mb: allocated GPU memory
      - gpu_mem_peak_mb: peak allocated GPU memory
    """

    def __init__(self, exp_name, weight_dir, model, device="cuda"):
        self.exp_name = exp_name
        self.csv_path = os.path.join(weight_dir, "metrics.csv")
        self.model = model
        self.device = device
        self._epoch_start = None
        self._max_grad = 0.0

        # Write header if new file
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "train_loss", "v_img", "grad_norm",
                            "update_ratio", "epoch_time_s", "samples_per_sec",
                            "gpu_mem_mb", "gpu_mem_peak_mb"])

    def start_epoch(self):
        self._epoch_start = time.time()
        self._max_grad = 0.0

    def record_batch_grad(self):
        """Call after loss.backward() to track max gradient norm across batches."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        batch_grad_norm = total_norm ** 0.5
        if batch_grad_norm > self._max_grad:
            self._max_grad = batch_grad_norm

    def end_epoch(self, epoch, train_loss, v_img_loss, n_samples):
        """Write epoch metrics to CSV. Returns dict of metrics."""
        elapsed = time.time() - self._epoch_start

        # Parameter update ratio: ||lr * grad|| / ||param|| (approximate)
        param_norm = sum(p.data.norm(2).item() ** 2 for p in self.model.parameters()
                         if p.requires_grad) ** 0.5
        update_ratio = (self._max_grad / param_norm) if param_norm > 0 else 0.0

        # GPU memory
        if self.device.startswith("cuda"):
            gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)
            gpu_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            gpu_mem = gpu_peak = 0.0

        samples_per_sec = n_samples / elapsed if elapsed > 0 else 0.0

        with open(self.csv_path, "a") as f:
            w = csv.writer(f)
            w.writerow([epoch, train_loss, v_img_loss, self._max_grad,
                        update_ratio, elapsed, samples_per_sec, gpu_mem, gpu_peak])

        return {"epoch": epoch, "grad_norm": self._max_grad,
                "update_ratio": update_ratio, "time": elapsed,
                "samples_per_sec": samples_per_sec,
                "gpu_mem": gpu_mem, "gpu_peak": gpu_peak}


def plot_dashboard(log_dir="logs", weight_dir="weight"):
    """Generate performance dashboard from all weight/*/metrics.csv files."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(log_dir, exist_ok=True)
    csv_files = sorted(glob.glob(f"{weight_dir}/*/metrics.csv"))
    if not csv_files:
        return

    # 2-row, 4-col layout: loss | grad_norm | update_ratio | throughput | memory
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    titles = ["Train Loss", "v_img (Val MSE)", "Gradient Norm",
              "Update Ratio", "Epoch Time (s)", "Samples/sec",
              "GPU Memory (MB)", "GPU Peak Memory (MB)"]
    columns = ["train_loss", "v_img", "grad_norm", "update_ratio",
               "epoch_time_s", "samples_per_sec", "gpu_mem_mb", "gpu_mem_peak_mb"]

    for f in csv_files:
        name = f.split("/")[1]
        data = {c: [] for c in columns}
        epochs = []
        with open(f) as fh:
            for row in csv.DictReader(fh):
                epochs.append(int(row["epoch"]))
                for c in columns:
                    data[c].append(float(row[c]))
        if not epochs:
            continue
        for i, c in enumerate(columns):
            axes[i].plot(epochs, data[c], alpha=0.7, lw=0.8, label=name)

    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide empty subplot (8 panels)
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{log_dir}/dashboard.png", dpi=100)
    plt.close(fig)
