#!/bin/bash
# ============================================================
#  SLURM job: redraw the model10 rate figure with panels 3-4 replaced by the
#  seeded + quick-refined d_ext = 4 / 8 rates
#  (scripts/model10_seeded_panels_collect.py).
#
#  Submitted at the end of scripts/submit_model10_seeded_refine.sh; by hand
#  (any time -- points not reached yet fall back to the earlier optimiser):
#      sbatch scripts/model10_seeded_panels_collect.slurm.sh
#
#  Output: $OUT_PNG (default results_model4_rate/model10_rate_panels.png)
#          $OUT_DIR/model10_seeded_panels.npz
# ============================================================

#SBATCH --job-name=m10_seed_panels
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/m10seedpanels_%x_%A.out
#SBATCH --error=logs/m10seedpanels_%x_%A.err

OUT_DIR=${OUT_DIR:-results_model10_rate}
OUT_PNG=${OUT_PNG:-results_model4_rate/model10_rate_panels.png}
BASE_IN_DIRS=${BASE_IN_DIRS:-"results_model10_rate results_model4_rate"}
BASE_NPZ=${BASE_NPZ:-"results_model4_rate/model10_rate_panels.npz results_model10_rate/model10_rate_panels.npz"}
STRIDE=${STRIDE:-1}               # must match the seeded worker array
MB_STRIDE=${MB_STRIDE:-5}         # must match the original model10 pipeline
Q_DIR=${Q_DIR:-results_liouvillian_q}
OBS_DIR=${OBS_DIR:-results_observable_q}
Q_LEVELS=${Q_LEVELS:-1}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python scripts/model10_seeded_panels_collect.py \
    --out_dir      "$OUT_DIR" \
    --out_png      "$OUT_PNG" \
    --base_in_dirs $BASE_IN_DIRS \
    --base_npz     $BASE_NPZ \
    --stride       "$STRIDE" \
    --mb_stride    "$MB_STRIDE" \
    --q_dir        "$Q_DIR" \
    --obs_dir      "$OBS_DIR" \
    --q_levels     $Q_LEVELS

echo "[model10 seeded panels] done"
