"""Experiment factory: unified training framework for panorama SR.

Usage:
  python run.py --cfg factory/configs/phase6_stage1.yaml -g 7
  python run.py --cfg factory/configs/phase6_stage2_polar_moe.yaml -g 5 --resume
"""

from factory.registry import (
    MODEL_REGISTRY, TRAINER_REGISTRY,
    register_model, register_trainer,
)
from factory.trainer import DirectTrainer
from factory.trainer_gan import GANTrainer
from factory.trainer_wavelet_flow import WaveletFlowTrainer
from factory.augment import augment_panorama
