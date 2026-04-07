#!/bin/bash
# ============================================================
#  SLURM job: iterative neighbor-seeded refinement of outliers.
#  Submitted by submit_scan.sh with a dependency on the collect
#  job (afterok:<collect_job_id>).
#
#  Direct submission example:
#    export N_PTS=20 J=1.0 GAMMA_STEP=0.1 OUT_DIR=results
#    sbatch refine_scan.sh
# ============================================================

#SBATCH --job-name=fra_refine
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/refine_%j.out
#SBATCH --error=logs/refine_%j.err

# ── read scan parameters (with defaults) ─────────────────────
N_PTS=${N_PTS:-20}
J=${J:-1.0}
GAMMA_STEP=${GAMMA_STEP:-0.1}
OUT_DIR=${OUT_DIR:-results}
REL_TOL=${REL_TOL:-0.05}
ABS_TOL=${ABS_TOL:-1e-3}
MAX_PASSES=${MAX_PASSES:-10}
N_RESTARTS=${N_RESTARTS:-3}
MAXFEV=${MAXFEV:-1000}

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
echo "Refining outliers in ${OUT_DIR}/ ..."

python refine_scan.py \
    --n_pts      "$N_PTS" \
    --J          "$J" \
    --gamma_step "$GAMMA_STEP" \
    --out_dir    "$OUT_DIR" \
    --rel_tol    "$REL_TOL" \
    --abs_tol    "$ABS_TOL" \
    --max_passes "$MAX_PASSES" \
    --n_restarts "$N_RESTARTS" \
    --maxfev     "$MAXFEV"

echo "Refinement done."

# ── re-plot with the updated data ────────────────────────────
echo "Re-generating figure ..."

python scan_collect.py \
    --n_pts      "$N_PTS" \
    --J          "$J" \
    --gamma_step "$GAMMA_STEP" \
    --out_dir    "$OUT_DIR"

echo "Figure updated: ${OUT_DIR}/two_qubit_scan.png"
