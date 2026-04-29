#!/bin/bash
# ============================================================
#  SLURM job-array script: recompute max LPDO bond entropy
#  with fidelity_threshold=0.99 for gamma' < n_igp*gamma_step.
#
#  Task-id mapping: ig = task_id // N_IGP, igp = task_id % N_IGP
#  Array range: 0 .. N_PTS*N_IGP-1
#
#  Direct submission example:
#    export N_PTS=41 N_IGP=20 J=1.0 GAMMA_STEP=0.2 OUT_DIR=results
#    sbatch --array=0-819 bond_entropy_refine_array.sh
# ============================================================

#SBATCH --job-name=bond_ent_refine
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/bond_ent_refine_%A_%a.out
#SBATCH --error=logs/bond_ent_refine_%A_%a.err

# ── scan parameters (with defaults) ──────────────────────────
N_PTS=${N_PTS:-41}
N_IGP=${N_IGP:-20}
J=${J:-1.0}
GAMMA_STEP=${GAMMA_STEP:-0.2}
OUT_DIR=${OUT_DIR:-results}

# ── activate Python environment ───────────────────────────────
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

# ── silence matplotlib config-dir warning ────────────────────
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

# ── run ──────────────────────────────────────────────────────
echo "Task ${SLURM_ARRAY_TASK_ID}: starting bond-entropy refinement"
echo "  N_PTS=${N_PTS}  N_IGP=${N_IGP}  J=${J}  step=${GAMMA_STEP}  out=${OUT_DIR}"

python bond_entropy_refine_worker.py \
    --task_id    "$SLURM_ARRAY_TASK_ID" \
    --n_pts      "$N_PTS" \
    --n_igp      "$N_IGP" \
    --J          "$J" \
    --gamma_step "$GAMMA_STEP" \
    --out_dir    "$OUT_DIR"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
