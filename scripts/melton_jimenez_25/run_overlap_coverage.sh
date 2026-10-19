#!/bin/bash
#SBATCH --job-name=sw_overlap
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --output=/home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility/logs/sw_overlap_%j.out
#SBATCH --error=/home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility/logs/sw_overlap_%j.out
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=10000
set -euo pipefail
cd /home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility
/ictstr01/groups/ml01/workspace/francesca.drummer/tools/apptainer_containers/interscale/run_interscale.sh \
    scripts/melton_jimenez_25/5c_window_overlap_coverage.py "$@"
