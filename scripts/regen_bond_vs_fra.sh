#!/bin/bash
# ============================================================
#  SLURM job: regenerate two_qubit_scan_full_bond_vs_fra.png
#  from the current scan_full.npy (no data recomputation).
#
#  Submitted by submit_refine_full.sh after all refinement
#  steps are complete.
#
#  Direct submission example:
#    export N_PTS=41 J=1.0 GAMMA_STEP=0.2 OUT_DIR=results
#    sbatch scripts/regen_bond_vs_fra.sh
# ============================================================

#SBATCH --job-name=regen_bond_fra
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/regen_bond_fra_%j.out
#SBATCH --error=logs/regen_bond_fra_%j.err

# ── scan parameters (with defaults) ──────────────────────────
N_PTS=${N_PTS:-41}
J=${J:-1.0}
GAMMA_STEP=${GAMMA_STEP:-0.2}
OUT_DIR=${OUT_DIR:-results}
OUT_NAME=${OUT_NAME:-two_qubit_scan_full.png}

# ── activate Python environment ───────────────────────────────
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

# ── silence matplotlib config-dir warning ────────────────────
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

# ── run ──────────────────────────────────────────────────────
echo "Regenerating bond-vs-fra figure from ${OUT_DIR}/scan_full.npy ..."

python regen_bond_vs_fra.py \
    --out_dir    "$OUT_DIR" \
    --n_pts      "$N_PTS" \
    --J          "$J" \
    --gamma_step "$GAMMA_STEP" \
    --out_name   "$OUT_NAME"

echo "Done. Figure saved to ${OUT_DIR}/${OUT_NAME%.png}_bond_vs_fra.png"
