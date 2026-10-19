#!/bin/bash
#SBATCH --job-name=ovl_flows
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=08:00:00
#SBATCH --output=/home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility/logs/ovl_flows_%j.out
#SBATCH --error=/home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility/logs/ovl_flows_%j.out
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --nice=10000
#
# Zone-to-zone flows for the legnini overlap ladder (12 models).
#   sbatch scripts/legnini_23/run_overlap_ladder_flows.sh
#   sbatch scripts/legnini_23/run_overlap_ladder_flows.sh --steps step1 step2
set -euo pipefail
cd /home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
/ictstr01/groups/ml01/workspace/francesca.drummer/tools/apptainer_containers/interscale/run_interscale.sh \
    scripts/legnini_23/3d_overlap_ladder_flows.py "$@"
