#!/usr/bin/env python3
"""Single entry point for all panorama SR experiments.

All parameters controlled via YAML configs in factory/configs/.
New modules only need registration in factory/registry.py.

Usage:
  # Single GPU
  python run.py --cfg factory/configs/mini_sr_stage0.yaml -g 7

  # Multi-GPU (DDP) — use torchrun
  torchrun --nproc_per_node=4 run.py --cfg factory/configs/mini_sr_stage1.yaml

  # List configs
  python run.py --list
"""

import argparse
import os
import sys
from glob import glob

import torch
import torch.distributed as dist

from factory.config import load_config
from factory.registry import MODEL_REGISTRY, TRAINER_REGISTRY, _register_builtin_trainers


def setup_ddp():
    """Initialize DDP from torchrun env vars. Returns (rank, world_size, device)."""
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


def main():
    parser = argparse.ArgumentParser(description="Panorama SR unified trainer")
    parser.add_argument("--cfg", type=str, help="Path to YAML config file")
    parser.add_argument("--list", action="store_true",
                        help="List available configs and registered modules")
    parser.add_argument("--gpu", "-g", type=int, default=None,
                        help="GPU id (single-GPU mode). When using torchrun, uses LOCAL_RANK instead.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs from config")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from best_model.pt in output dir")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="Epoch offset for resumed training")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    # --list mode (no DDP needed)
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

    # Setup DDP (no-op in single-GPU mode)
    ddp_rank, ddp_world_size, device = setup_ddp()
    is_main = (ddp_rank == 0)

    # Override gpu with DDP local rank when using torchrun
    if args.gpu is not None and ddp_world_size == 1:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load config
    cfg = load_config(args.cfg)
    exp_name = cfg.experiment.name
    output_dir = args.output_dir or os.path.join(cfg.output.dir, exp_name)
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cfg.output.log_dir, exist_ok=True)

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
            if is_main:
                print(f"ERROR: checkpoint not found: {ckpt_path}")
            sys.exit(1)
    elif trainer_name == 'gan':
        base_ckpt = cfg.training.get('base_ckpt', None)
        if base_ckpt and os.path.exists(base_ckpt):
            resume_ckpt = base_ckpt
        else:
            if is_main:
                print(f"ERROR: GAN trainer requires base_ckpt in config (not found: {base_ckpt})")
            sys.exit(1)

    # Build trainer
    _register_builtin_trainers()
    trainer_cls = TRAINER_REGISTRY[trainer_name]
    trainer = trainer_cls(cfg, device, output_dir,
                          ddp_rank=ddp_rank, ddp_world_size=ddp_world_size)
    trainer.run(epochs=epochs, resume_ckpt=resume_ckpt, start_epoch=start_epoch)

    if ddp_world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
