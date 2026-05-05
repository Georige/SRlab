"""Panorama dataset loader."""

import os
import numpy as np
import torch
from PIL import Image


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
