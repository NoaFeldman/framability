#!/bin/bash
# ============================================================
#  SLURM job: collect the seeded model10 rates, cross-evaluate neighbouring
#  frames (N_PROC processes, certified with the per-column LP), and redraw
#  the model10 rate figure with two rows of new panels appended
#  (scripts/model10_seeded_collect.py).
#
#  Submitted by scripts/submit_model10_seeded.sh with
#      --dependency=afterok:<array job id>
#  By hand (any time; missing points are left blank):
#      sbatch scripts/model10_seeded_collect.slurm.sh
#      NO_XEVAL=1 sbatch scripts/model10_seeded_collect.slurm.sh   # plot only
#
#  Output: $OUT_PNG (default results_model4_rate/model10_rate_panels.png)
#          $OUT_DIR/model10_seeded_rates.npz
#          $OUT_DIR/model10_seeded/pt_<ix>_<iy>_xeval.npz  (improved points)
# ============================================================

#SBATCH --job-name=m10_seed_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/m10seed_collect_%x_%A.out
#SBATCH --error=logs/m10seed_collect_%x_%A.err

OUT_DIR=${OUT_DIR:-results_model10_rate}
OUT_PNG=${OUT_PNG:-results_model4_rate/model10_rate_panels.png}
BASE_IN_DIRS=${BASE_IN_DIRS:-"results_model10_rate results_model4_rate"}
BASE_NPZ=${BASE_NPZ:-"results_model4_rate/model10_rate_panels.npz results_model10_rate/model10_rate_panels.npz"}
STRIDE=${STRIDE:-1}               # must match the seeded worker array
MB_STRIDE=${MB_STRIDE:-5}         # must match the original model10 pipeline
Q_DIR=${Q_DIR:-results_liouvillian_q}
OBS_DIR=${OBS_DIR:-results_observable_q}
Q_LEVELS=${Q_LEVELS:-1}
RADIUS4=${RADIUS4:-2}
RADIUS8=${RADIUS8:-1}
MAX_SWEEPS=${MAX_SWEEPS:-6}
N_PROC=${N_PROC:-${SLURM_CPUS_PER_TASK:-1}}
NO_XEVAL=${NO_XEVAL:-}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python scripts/model10_seeded_collect.py \
    --out_dir      "$OUT_DIR" \
    --out_png      "$OUT_PNG" \
    --base_in_dirs $BASE_IN_DIRS \
    --base_npz     $BASE_NPZ \
    --stride       "$STRIDE" \
    --mb_stride    "$MB_STRIDE" \
    --q_dir        "$Q_DIR" \
    --obs_dir      "$OBS_DIR" \
    --q_levels     $Q_LEVELS \
    --radius4      "$RADIUS4" \
    --radius8      "$RADIUS8" \
    --max_sweeps   "$MAX_SWEEPS" \
    --n_proc       "$N_PROC" \
    ${NO_XEVAL:+--no_xeval}

echo "[model10 seeded collect] done"
