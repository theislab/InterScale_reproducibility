#!/bin/bash
#SBATCH --job-name=IS_seed_stability
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=08:00:00
#SBATCH --output=/home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility/logs/seed_stability_%j.out
#SBATCH --error=/home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility/logs/seed_stability_%j.out
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --nice=10000
#
# Decoder-loading stability across seeds 40-44, for either dataset.
#
#   SPEC=legnini23 sbatch scripts/legnini_23/run_seed_stability.sh      # default
#   SPEC=melton25  sbatch scripts/legnini_23/run_seed_stability.sh      # CosMx pancreas
#   SPEC=melton25  sbatch scripts/legnini_23/run_seed_stability.sh --overwrite
#   COMPARE_ONLY=1 SPEC=melton25 scripts/legnini_23/run_seed_stability.sh   # skip extraction
#
# Three stages inside the interscale apptainer container:
#   4a_seed_stability_extract.py       one gene-level pipeline pass per seed (needs checkpoints)
#   4b_seed_stability_compare.py       dimension-matched cross-seed comparison
#   4d_loading_geometry_stability.py   the matching-free comparisons (gene-gene geometry,
#                                      predicted patterns, up/down-regulation, gene pairings)
#   4e_embedding_and_rank_stability.py what reproduces when the loadings do not: embedding
#                                      cluster coherence, top-dimension gene panels, gene ranks
#
# WHERE THE CHECKPOINTS COME FROM -- the two datasets differ, and it matters:
#
#   legnini23  the seed sweep in the interscale repo, which varied ONLY optim.seed:
#                sbatch scripts/run.sh legnini23 node_reg \
#                    --sweep_cfg config_files/sweeps/seed.yaml --sweep_goal robustness --count 5
#              (wandb sweep zmnw1x9c). Trained at the registry defaults, i.e. n_embed 16.
#
#   melton25   already trained, by scripts/melton_jimenez_25/celltype_reconstruction_mse.py,
#              which retrained the lowest-val_mse trial of sweep sio2fsjm once per seed into
#              results/melton25_revision/best_recon/. n_embed 128, GaussianNLL. There is no
#              seed sweep to run for CosMx -- 4a reapplies that config via SPECS["melton25"].
#
# The compare stages are light (seconds to a couple of minutes) and pure numpy/scipy, so they
# also run fine on a login node with COMPARE_ONLY=1.

set -euo pipefail

CONTAINER_DIR=/ictstr01/groups/ml01/workspace/francesca.drummer/tools/apptainer_containers/interscale
REPO=/home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility
LUSTRE=/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer

SPEC="${SPEC:-legnini23}"

# Kept per spec rather than derived, because 4b takes explicit paths and its defaults are
# legnini's. ARRAYS_DIR must match what 4a writes: <models_dir>/analysis.
case "$SPEC" in
    legnini23)
        ARRAYS_DIR="$LUSTRE/results/legnini23_revision/seed_stability/analysis"
        FIG_DIR="$REPO/figures/legnini23/seed_stability"
        SWEEP_ID="${SWEEP_ID:-zmnw1x9c}"
        ;;
    melton25)
        ARRAYS_DIR="$LUSTRE/results/melton25_revision/best_recon/analysis"
        FIG_DIR="$REPO/figures/melton25/seed_stability"
        # The CosMx models were retrained per seed outside any sweep, so there is no sweep id
        # whose per-seed metrics 4b could merge in.
        SWEEP_ID="${SWEEP_ID:-}"
        ;;
    *)
        echo "ERROR: unknown SPEC '$SPEC' (expected legnini23 or melton25)" >&2
        exit 1
        ;;
esac

mkdir -p "$REPO/logs" "$FIG_DIR"
cd "$REPO"

# Without this, python buffers stdout when it is a file rather than a tty, and a multi-hour
# stage looks hung in the slurm log until it finishes and flushes everything at once.
export PYTHONUNBUFFERED=1

# These nodes report >100 cores and OpenBLAS is built for at most 128 threads, so an unpinned
# KMeans in 4e allocates one thread pool per restart and aborts the process. Slurm gives us 8 cpus.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

# Keep scratch off the 10G-capped /tmp.
export TMPDIR="/localscratch/$USER/interscale_seed_stability"
mkdir -p "$TMPDIR"

echo "======================================================================"
echo " $SPEC node_reg -- decoder loading stability across seeds"
echo "   host   : $(hostname -f)"
echo "   job id : ${SLURM_JOB_ID:-none}"
echo "   arrays : $ARRAYS_DIR"
echo "   figures: $FIG_DIR"
echo "   sweep  : ${SWEEP_ID:-<none>}"
echo "======================================================================"

RUN="$CONTAINER_DIR/run_interscale.sh"

if [[ -z "${COMPARE_ONLY:-}" ]]; then
    "$RUN" scripts/legnini_23/4a_seed_stability_extract.py --spec "$SPEC" "$@"
fi

COMPARE_ARGS=( --arrays-dir "$ARRAYS_DIR" --out-dir "$FIG_DIR" )
[[ -n "$SWEEP_ID" ]] && COMPARE_ARGS+=( --sweep-id "$SWEEP_ID" )
"$RUN" scripts/legnini_23/4b_seed_stability_compare.py "${COMPARE_ARGS[@]}"

"$RUN" scripts/legnini_23/4d_loading_geometry_stability.py \
    --spec "$SPEC" --arrays-dir "$ARRAYS_DIR" --out-dir "$FIG_DIR"

"$RUN" scripts/legnini_23/4e_embedding_and_rank_stability.py \
    --spec "$SPEC" --arrays-dir "$ARRAYS_DIR" --out-dir "$FIG_DIR"
