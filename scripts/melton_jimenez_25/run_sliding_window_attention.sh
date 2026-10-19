#!/bin/bash
#SBATCH --job-name=sw_attn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250GB
#SBATCH --time=12:00:00
#SBATCH --output=/home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility/logs/sw_attn_%j.out
#SBATCH --error=/home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility/logs/sw_attn_%j.out
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --nice=10000
#
# Attention extraction for the sliding-window ablation sweep (18 models).
#
#   sbatch scripts/melton_jimenez_25/run_sliding_window_attention.sh
#   sbatch scripts/melton_jimenez_25/run_sliding_window_attention.sh --skip-existing
#   sbatch scripts/melton_jimenez_25/run_sliding_window_attention.sh --arms w3000 --seeds 42
#
# GPU is for the forward pass only; the binning is numpy. 250 GB because get_model_output holds a
# 151,595 x max_seq_len float frame (2.1 GB for the w3000 arm) plus the slide's AnnData.
set -euo pipefail
REPO=/home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility
cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
/ictstr01/groups/ml01/workspace/francesca.drummer/tools/apptainer_containers/interscale/run_interscale.sh \
    scripts/melton_jimenez_25/5a_sliding_window_attention.py "$@"
