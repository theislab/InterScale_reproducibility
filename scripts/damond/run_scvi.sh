#! /bin/bash

#SBATCH -o logs/run_interactive_%j.out
#SBATCH -e logs/run_interactive_%j.out
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=500GB
#SBATCH -t 24:00:00
#SBATCH -c 5
#SBATCH --nice=1000

source activate GT_long_range_env

python scvi_damond.py

