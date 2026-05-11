#!/usr/bin/env python3
"""Single entry point for all panorama SR experiments.

All parameters controlled via YAML configs in factory/configs/.
New modules only need registration in factory/registry.py.

Usage:
  python run.py --cfg factory/configs/phase6_stage1.yaml -g 7
  python run.py --cfg factory/configs/phase6_stage1.yaml -g 7 --resume --epochs 600
  python run.py --list
"""

import argparse
import os
import sys
from glob import glob

import torch

from factory.config import load_config
from factory.registry import MODEL_REGISTRY, TRAINER_REGISTRY, _register_builtin_trainers


def main():
    parser = argparse.ArgumentParser(description="Panorama SR unified trainer")
    parser.add_argument("--cfg", type=str, help="Path to YAML config file")
    parser.add_argument("--list", action="store_true",
                        help="List available configs and registered modules")
    parser.add_argument("--gpu", "-g", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs from config")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from best_model.pt in output dir")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="Epoch offset for resumed training (default: 400)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    # --list mode
    if args.list:
        configs = sorted(glob("factory/configs/*.yaml"))
        print(f"\nAvailable configs ({len(configs)}):")
        for c in configs:
            print(f"  {c}")
        print(f"\nRegistered models: {list(MODEL_REGISTRY.keys())}")
        _register_builtin_trainers()
        print(f"Registered trainers: {list(TRAINER_REGISTRY.keys())}")
        return

    if not args.cfg:
        parser.print_help()
        print("\nTip: use --list to see available configs")
        return

    # Load config
    cfg = load_config(args.cfg)
    exp_name = cfg.experiment.name
    output_dir = args.output_dir or os.path.join(cfg.output.dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cfg.output.log_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs if args.epochs is not None else cfg.training.epochs

    # Resolve checkpoint for resume
    trainer_name = cfg.experiment.get('trainer', 'direct')
    resume_ckpt = None
    start_epoch = 0

    if args.resume:
        ckpt_path = os.path.join(output_dir, "best_model.pt")
        if os.path.exists(ckpt_path):
            resume_ckpt = ckpt_path
            start_epoch = args.start_epoch or 400
        else:
            print(f"ERROR: checkpoint not found: {ckpt_path}")
            sys.exit(1)
    elif trainer_name == 'gan':
        # GAN trainer always needs a base checkpoint to start from
        base_ckpt = cfg.training.get('base_ckpt', None)
        if base_ckpt and os.path.exists(base_ckpt):
            resume_ckpt = base_ckpt
        else:
            print(f"ERROR: GAN trainer requires base_ckpt in config (not found: {base_ckpt})")
            sys.exit(1)

    # Build trainer
    _register_builtin_trainers()
    trainer_cls = TRAINER_REGISTRY[trainer_name]
    trainer = trainer_cls(cfg, device, output_dir)
    trainer.run(epochs=epochs, resume_ckpt=resume_ckpt, start_epoch=start_epoch)


if __name__ == "__main__":
    main()
