#!/bin/bash
# ============================================================
#  SLURM job: collect schro_fra_pt files and regenerate figure.
#  Submitted by submit_schro_fra.sh with a dependency on the
#  full array job (afterok:<array_job_id>).
#
#  Direct submission example:
#    export N_PTS=41 J=1.0 GAMMA_STEP=0.2 OUT_DIR=results
#    sbatch schro_fra_collect.sh
# ============================================================

#SBATCH --job-name=schro_fra_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/schro_fra_collect_%j.out
#SBATCH --error=logs/schro_fra_collect_%j.err

# ── read scan parameters (with defaults) ─────────────────────
N_PTS=${N_PTS:-41}
J=${J:-1.0}
GAMMA_STEP=${GAMMA_STEP:-0.2}
OUT_DIR=${OUT_DIR:-results}

# ── activate Python environment ───────────────────────────────
# Option A: virtualenv (default)
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

# Option B: conda – comment out A and uncomment:
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate framability

# Option C: environment modules – comment out A and uncomment:
# module load python/3.11

# ── silence matplotlib config-dir warning ────────────────────
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

# ── run ──────────────────────────────────────────────────────
echo "Collecting schro_fra_pt files from ${OUT_DIR}/ ..."

python schro_fra_collect.py \
    --n_pts      "$N_PTS" \
    --J          "$J" \
    --gamma_step "$GAMMA_STEP" \
    --out_dir    "$OUT_DIR"

echo "Collection done."
