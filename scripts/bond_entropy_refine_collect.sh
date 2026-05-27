#!/bin/bash
# ============================================================
#  SLURM job: collect bond-entropy refinement results and
#  rebuild scan_full.npy.  Submitted by submit_refine_full.sh
#  with a dependency on the bond_entropy_refine array job.
#
#  Direct submission example:
#    export N_PTS=41 N_IGP=20 J=1.0 GAMMA_STEP=0.2 OUT_DIR=results
#    sbatch scripts/bond_entropy_refine_collect.sh
# ============================================================

#SBATCH --job-name=bond_ent_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/bond_ent_collect_%j.out
#SBATCH --error=logs/bond_ent_collect_%j.err

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
echo "Collecting bond-entropy refinement results from ${OUT_DIR}/ ..."

python scripts/bond_entropy_refine_collect.py \
    --n_pts      "$N_PTS" \
    --n_igp      "$N_IGP" \
    --J          "$J" \
    --gamma_step "$GAMMA_STEP" \
    --out_dir    "$OUT_DIR"

echo "Bond-entropy collect done."
