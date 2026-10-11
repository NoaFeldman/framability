#!/bin/bash
# ============================================================
#  SLURM job: collect stages of the seeded rate pipeline for any model
#  (scripts/rate_seeded_collect.py).
#
#    MODE=xeval   d_ext = 4 / 8 neighbour cross-evaluation (N_PROC processes,
#                 certified with the per-column LP; between the seed stage and
#                 the quick-refine rounds)
#    MODE=plot    the figure (stabilizer-3, Pauli, opt d_ext = 4 / 8 / 12,
#                 product-state chi = 10 / 40)
#
#  Driven by scripts/submit_rate_seeded.sh; by hand (any time -- missing
#  points are left blank):
#      MODEL=model4 MODE=plot sbatch --cpus-per-task=1 scripts/rate_seeded_collect.slurm.sh
#
#  Output: $OUT_PNG (default results_model4_rate/<model>_seeded_rate_panels.png)
#          $OUT_DIR/<model>_seeded_rate_panels.npz
#          $OUT_DIR/<model>_seeded/pt_<ix>_<iy>_xeval.npz   (MODE=xeval)
# ============================================================

#SBATCH --job-name=rate_seed_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rateseed_collect_%x_%A.out
#SBATCH --error=logs/rateseed_collect_%x_%A.err

MODEL=${MODEL:?set MODEL (e.g. model3 | model4)}
MODE=${MODE:?set MODE (xeval | plot)}
OUT_DIR=${OUT_DIR:-results_${MODEL}_seeded}
OUT_PNG=${OUT_PNG:-results_model4_rate/${MODEL}_seeded_rate_panels.png}
REF_NPZ=${REF_NPZ:-}               # earlier figure data to compare with (optional)
DIM_ARG=${DIM:+--dim $DIM}
STRIDE=${STRIDE:-1}                # must match the array stages
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
plot)  EXTRA="${REF_NPZ:+--ref_npz $REF_NPZ}" ;;
*)     echo "unknown MODE=${MODE}"; exit 1 ;;
esac

python scripts/rate_seeded_collect.py \
    --model       "$MODEL" \
    $DIM_ARG \
    --out_dir     "$OUT_DIR" \
    --out_png     "$OUT_PNG" \
    --stride      "$STRIDE" \
    --prod_stride "$PROD_STRIDE" \
    --radius4     "$RADIUS4" \
    --radius8     "$RADIUS8" \
    --max_sweeps  "$MAX_SWEEPS" \
    --n_proc      "$N_PROC" \
    $EXTRA \
    || { echo "[${MODEL} seeded collect] ${MODE} FAILED"; exit 1; }

echo "[${MODEL} seeded collect] ${MODE} done"
