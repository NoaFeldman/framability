#!/bin/bash
# ============================================================
#  SLURM job (one task): collect of the model10 d_ext scan + figure
#    1. scripts/model10_dext_scan_collect.py  -> table (log) and
#       $OUT_DIR/model10_dext_scan/summary.npz
#    2. scripts/model10_opt_rate_vs_dext_plot.py -> $OUT_PNG
#
#  Submitted at the end of scripts/submit_model10_dext_scan.sh; by hand (any
#  time -- unfinished combos just have fewer restarts):
#      sbatch scripts/model10_dext_scan_collect.slurm.sh
#  DELTAS / D_EXTS must match the scan's.
# ============================================================

#SBATCH --job-name=m10_dext_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/m10dext_io_%x_%A.out
#SBATCH --error=logs/m10dext_io_%x_%A.err

OUT_DIR=${OUT_DIR:-results_model10_rate}
DELTAS=${DELTAS:-"0.5 0.6 0.7"}
D_EXTS=${D_EXTS:-"6 10"}
MIN_HITS=${MIN_HITS:-3}
MIN_COLD_HITS=${MIN_COLD_HITS:-1}
TOL_REL=${TOL_REL:-1e-4}
TOL_ABS=${TOL_ABS:-1e-5}
OUT_PNG=${OUT_PNG:-results_model10_rate/model10_opt_rate_vs_dext.png}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

set -e
# shellcheck disable=SC2086
python scripts/model10_dext_scan_collect.py \
    --deltas        $DELTAS \
    --d_exts        $D_EXTS \
    --out_dir       "$OUT_DIR" \
    --min_hits      "$MIN_HITS" \
    --min_cold_hits "$MIN_COLD_HITS" \
    --tol_rel       "$TOL_REL" \
    --tol_abs       "$TOL_ABS"

python scripts/model10_opt_rate_vs_dext_plot.py \
    --scan_npz "$OUT_DIR/model10_dext_scan/summary.npz" \
    --out_png  "$OUT_PNG"

echo "[model10 dext scan] collect + figure: done"
