"""Configuration loader with nested attr access for YAML experiment configs."""

import os
import yaml
from pathlib import Path

# Absolute path to the repository root (parent of factory/)
REPO_ROOT = Path(__file__).resolve().parent.parent


class DotDict:
    """Nested dict with attribute access: cfg.model.base_ch instead of cfg['model']['base_ch']."""

    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, DotDict(v))
            elif isinstance(v, list):
                setattr(self, k, [DotDict(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __repr__(self):
        items = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return f"DotDict({items})"


def load_config(path):
    """Load YAML config file and return as DotDict with resolved paths."""
    cwd = Path.cwd()

    # Handle relative paths from cwd
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = cwd / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    cfg = DotDict(raw)

    # Resolve output paths relative to repo root
    if hasattr(cfg, 'output'):
        cfg.output.dir = str(REPO_ROOT / cfg.output.dir)
        cfg.output.log_dir = str(REPO_ROOT / cfg.output.log_dir)

    return cfg
