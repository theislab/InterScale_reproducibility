#!/bin/bash
#SBATCH --job-name=sw_abl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=08:00:00
#SBATCH --output=/home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility/logs/sw_ablation_analysis_%j.out
#SBATCH --error=/home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility/logs/sw_ablation_analysis_%j.out
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=10000
#
# Ablation analysis for the sliding-window sweep. Reads the attention masks written by
# run_sliding_window_attention.sh; no model and no GPU involved.
#
#   sbatch scripts/melton_jimenez_25/run_sliding_window_ablation.sh
#   sbatch scripts/melton_jimenez_25/run_sliding_window_ablation.sh --n-permutations 200
#   sbatch scripts/melton_jimenez_25/run_sliding_window_ablation.sh --validate
#
# CPU partition: the permutation test is numpy bincount and BLAS over per-window blocks. The
# expensive arm is w400, at 3,426 windows x 100 permutations per seed.
set -euo pipefail
REPO=/home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility
cd "$REPO"
/ictstr01/groups/ml01/workspace/francesca.drummer/tools/apptainer_containers/interscale/run_interscale.sh \
    scripts/melton_jimenez_25/5b_sliding_window_ablation.py "$@"
