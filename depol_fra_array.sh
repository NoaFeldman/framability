#!/bin/bash
# ============================================================
#  SLURM job-array script: product-state framability of
#  depolarised gates.
#
#  task_id = gate_idx * N_P + p_idx
#    gate_idx 0  depol(p) ∘ H
#    gate_idx 1  depol(p) ∘ T
#    gate_idx 2  depol(p)⊗² ∘ CNOT
#    N_P = 5   p ∈ {0.05, 0.07, 0.09, 0.11, 0.13}
#
#  Total tasks: 0–14  (3 gates × 5 p values)
#
#  Submit via submit_depol_fra.sh, or directly:
#    export OUT_DIR=results_depol
#    sbatch --array=0-14 depol_fra_array.sh
# ============================================================

#SBATCH --job-name=depol_fra
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/depol_fra_%A_%a.out
#SBATCH --error=logs/depol_fra_%A_%a.err

# ── read parameters ───────────────────────────────────────────
OUT_DIR=${OUT_DIR:-results_depol}

# ── activate Python environment ───────────────────────────────
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

# ── silence matplotlib config-dir warning ────────────────────
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

# ── run ──────────────────────────────────────────────────────
echo "Task ${SLURM_ARRAY_TASK_ID}: starting  (OUT_DIR=${OUT_DIR})"

python depol_fra_worker.py \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --out_dir  "$OUT_DIR"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
