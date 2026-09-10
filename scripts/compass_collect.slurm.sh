#!/bin/bash
# ============================================================
#  SLURM job: collect ONE compass-chain case and draw its eight-panel figure.
#
#  Submitted by scripts/submit_compass_all.sh with
#  --dependency=afterok:<rates array>:<many-body array>.  Can also be run by
#  hand at any time: missing points are left NaN and reported in the log.
#
#  Submit by hand:
#    CASE=jx1.0_jy1.0_hx sbatch scripts/compass_collect.slurm.sh
#
#  Output: results_compass/<CASE>/compass_<CASE>_panels.npz
#          results_compass/<CASE>/compass_<CASE>_panels.png
# ============================================================

#SBATCH --job-name=cmp_col
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cmpcol_%x_%j.out
#SBATCH --error=logs/cmpcol_%x_%j.err

CASE=${CASE:?set CASE to a key of compass_chain.CASES}
IN_DIR=${IN_DIR:-results_compass}
OUT_DIR=${OUT_DIR:-results_compass}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python scripts/compass_collect.py \
    --case    "$CASE" \
    --in_dir  "$IN_DIR" \
    --out_dir "$OUT_DIR"
status=$?

echo "[compass collect] ${CASE}: exit ${status}"
exit "$status"
