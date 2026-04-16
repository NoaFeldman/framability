#!/bin/bash
# ============================================================
#  SLURM job-array script: extra properties (max bond dim,
#  magnetization).  One task per (gamma, gamma') grid POINT.
#
#  task_id = ig * N_PTS + igp   (400 tasks for a 20x20 grid)
#
#  Submit via submit_scan_extra.sh (which sets --array upper
#  bound and exports N_PTS / J / GAMMA_STEP / OUT_DIR / N_QUBITS).
#
#  Direct submission example:
#    export N_PTS=20 J=1.0 GAMMA_STEP=0.1 OUT_DIR=results N_QUBITS=2
#    sbatch --array=0-399 scan_array_extra.sh
# ============================================================

#SBATCH --job-name=fra_extra
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/extra_%A_%a.out
#SBATCH --error=logs/extra_%A_%a.err

# ── read scan parameters (with defaults) ─────────────────────
N_PTS=${N_PTS:-41}
J=${J:-1.0}
GAMMA_STEP=${GAMMA_STEP:-0.2}
OUT_DIR=${OUT_DIR:-results}
N_QUBITS=${N_QUBITS:-2}

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
echo "Task ${SLURM_ARRAY_TASK_ID}: starting extra props (N_PTS=${N_PTS}, J=${J}, step=${GAMMA_STEP}, N=${N_QUBITS})"

python scan_worker_extra.py \
    --task_id    "$SLURM_ARRAY_TASK_ID" \
    --n_pts      "$N_PTS" \
    --J          "$J" \
    --gamma_step "$GAMMA_STEP" \
    --out_dir    "$OUT_DIR" \
    --N          "$N_QUBITS"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
