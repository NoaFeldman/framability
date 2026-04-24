#!/bin/bash
# ============================================================
#  SLURM job-array script: product-state Schrödinger framability
#  chi=30 over the full (gamma, gamma') grid.
#  One task per grid point.
#
#  task_id = ig * N_PTS + igp   (1681 tasks for 41×41 grid)
#
#  Submit via submit_product_schro.sh (which sets --array upper
#  bound and exports N_PTS / J / GAMMA_STEP / OUT_DIR).
#
#  Direct submission example:
#    export N_PTS=41 J=1.0 GAMMA_STEP=0.2 OUT_DIR=results
#    sbatch --array=0-1680 product_schro_array.sh
# ============================================================

#SBATCH --job-name=prod_schro
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prod_schro_%A_%a.out
#SBATCH --error=logs/prod_schro_%A_%a.err

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
echo "Task ${SLURM_ARRAY_TASK_ID}: prod_schro chi=30 (N_PTS=${N_PTS}, J=${J}, step=${GAMMA_STEP})"

python product_schro_worker.py \
    --task_id    "$SLURM_ARRAY_TASK_ID" \
    --n_pts      "$N_PTS" \
    --J          "$J" \
    --gamma_step "$GAMMA_STEP" \
    --out_dir    "$OUT_DIR"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
