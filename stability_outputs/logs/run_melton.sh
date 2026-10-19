#!/bin/bash
# Detached melton25 run: the permutation tests are ~20 min and must survive the caller exiting.
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
REPO=/home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility
cd "$REPO"
/ictstr01/groups/ml01/workspace/francesca.drummer/tools/apptainer_containers/interscale/run_interscale.sh -c "
import sys; sys.path.insert(0, '$REPO')
from src.loading_stability_run import run
run('melton25')
print('MELTON25 RUN COMPLETE', flush=True)
"
