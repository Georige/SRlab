#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/lili/lyn/lab
exec conda run -n spectral-sr python run.py --cfg factory/configs/phase7_stage3c_noise15.yaml -g 7
