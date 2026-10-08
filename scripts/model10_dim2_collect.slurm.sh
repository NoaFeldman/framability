#!/bin/bash
# ============================================================
#  SLURM job: collect stages of the 2D-lattice model10 rate pipeline
#  (scripts/model10_dim2_collect.py).
#
#    MODE=xeval   d_ext = 4 / 8 neighbour cross-evaluation (N_PROC processes,
#                 certified with the per-column LP; between the seed stage and
#                 the quick-refine rounds)
#    MODE=plot    the figure (framability panels of model10_rate_panels.png)
#
#  Driven by scripts/submit_model10_dim2.sh; by hand (any time -- missing
#  points are left blank):
#      MODE=plot sbatch --cpus-per-task=1 scripts/model10_dim2_collect.slurm.sh
#
#  Output: $OUT_PNG (default results_model4_rate/model10_dim2_rate_panels.png)
#          $OUT_DIR/model10_dim2_rate_panels.npz
#          $OUT_DIR/model10_seeded/pt_<ix>_<iy>_xeval.npz   (MODE=xeval)
# ============================================================

#SBATCH --job-name=m10d2_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/m10dim2_collect_%x_%A.out
#SBATCH --error=logs/m10dim2_collect_%x_%A.err

MODE=${MODE:?set MODE (xeval | plot)}
DIM=${DIM:-2}
OUT_DIR=${OUT_DIR:-results_model10_rate_dim${DIM}}
OUT_PNG=${OUT_PNG:-results_model4_rate/model10_dim${DIM}_rate_panels.png}
STRIDE=${STRIDE:-1}               # must match the array stages
PROD_STRIDE=${PROD_STRIDE:-$STRIDE}
RADIUS4=${RADIUS4:-2}
RADIUS8=${RADIUS8:-1}
MAX_SWEEPS=${MAX_SWEEPS:-6}
N_PROC=${N_PROC:-${SLURM_CPUS_PER_TASK:-1}}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

case "$MODE" in
xeval) EXTRA="--xeval --no_plot" ;;
plot)  EXTRA="" ;;
*)     echo "unknown MODE=${MODE}"; exit 1 ;;
esac

python scripts/model10_dim2_collect.py \
    --dim         "$DIM" \
    --out_dir     "$OUT_DIR" \
    --out_png     "$OUT_PNG" \
    --stride      "$STRIDE" \
    --prod_stride "$PROD_STRIDE" \
    --radius4     "$RADIUS4" \
    --radius8     "$RADIUS8" \
    --max_sweeps  "$MAX_SWEEPS" \
    --n_proc      "$N_PROC" \
    $EXTRA \
    || { echo "[model10 dim=${DIM} collect] ${MODE} FAILED"; exit 1; }

echo "[model10 dim=${DIM} collect] ${MODE} done"
