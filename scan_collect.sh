#!/bin/bash
# ============================================================
#  SLURM job: aggregate row files and produce colourmap plots.
#  Submitted by submit_scan.sh with a dependency on the full
#  job array (afterok:<array_job_id>).
#
#  Direct submission example:
#    export N_PTS=20 J=1.0 GAMMA_STEP=0.1 OUT_DIR=results
#    sbatch scan_collect.sh
# ============================================================

#SBATCH --job-name=fra_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/collect_%j.out
#SBATCH --error=logs/collect_%j.err

# ── read scan parameters (with defaults) ─────────────────────
N_PTS=${N_PTS:-41}
J=${J:-1.0}
GAMMA_STEP=${GAMMA_STEP:-0.2}
OUT_DIR=${OUT_DIR:-results}

# ── activate Python environment ───────────────────────────────
# Option A: virtualenv (default)
# SLURM_SUBMIT_DIR is the directory from which sbatch was called.
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

# Option B: conda – comment out A and uncomment:
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate framability

# Option C: environment modules – comment out A and uncomment:
# module load python/3.11

# ── silence matplotlib config-dir warning ────────────────────
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

# ── run ──────────────────────────────────────────────────────
echo "Collecting results from ${OUT_DIR}/ ..."

python scan_collect.py \
    --n_pts      "$N_PTS" \
    --J          "$J" \
    --gamma_step "$GAMMA_STEP" \
    --out_dir    "$OUT_DIR"

echo "Collection done."
