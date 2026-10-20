#!/bin/bash
# ============================================================
#  SLURM job: collect one KW-asymmetry run (scripts/kw_asym_collect.py) and
#  draw its figure (scripts/kw_asym_plot.py).
#
#  Submitted by scripts/submit_kw_asym.sh with --dependency=afterok:<array>.
#  By hand at any time (missing points stay NaN and are listed in the log):
#      TAG=ring12_local_ghz_s sbatch scripts/kw_asym_collect.slurm.sh
#
#  Output: $OUT_DIR/$TAG/summary.npz, $OUT_DIR/$TAG/kw_asym_$TAG.png
# ============================================================

#SBATCH --job-name=kw_asym_col
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/kwasym_io_%x_%j.out
#SBATCH --error=logs/kwasym_io_%x_%j.err

TAG=${TAG:?set TAG to the run tag, e.g. ring12_local_ghz_s}
OUT_DIR=${OUT_DIR:-results_kw_asym}
T_SHOW=${T_SHOW:-0.25}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

set -e
python scripts/kw_asym_collect.py --tag "$TAG" --in_dir "$OUT_DIR"
python scripts/kw_asym_plot.py \
    --summary "$OUT_DIR/$TAG/summary.npz" \
    --t_show  "$T_SHOW"

echo "[kw asym collect] ${TAG}: done"
