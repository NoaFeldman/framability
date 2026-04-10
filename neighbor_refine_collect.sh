#!/bin/bash
# ============================================================
#  SLURM job: collect neighbor-refine results and regenerate
#  the scan figure.  Submitted by submit_neighbor_refine.sh
#  with a dependency on the array job.
#
#  Direct submission example:
#    export N_PTS=20 J=1.0 GAMMA_STEP=0.1 OUT_DIR=results
#    sbatch neighbor_refine_collect.sh
# ============================================================

#SBATCH --job-name=nb_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/nb_collect_%j.out
#SBATCH --error=logs/nb_collect_%j.err

# ── scan parameters (with defaults) ──────────────────────────
N_PTS=${N_PTS:-20}
J=${J:-1.0}
GAMMA_STEP=${GAMMA_STEP:-0.1}
OUT_DIR=${OUT_DIR:-results}

# ── activate Python environment ───────────────────────────────
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

# ── silence matplotlib config-dir warning ────────────────────
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

# ── run ──────────────────────────────────────────────────────
echo "Collecting neighbor-refine results from ${OUT_DIR}/ ..."

python neighbor_refine_collect.py \
    --n_pts      "$N_PTS" \
    --J          "$J" \
    --gamma_step "$GAMMA_STEP" \
    --out_dir    "$OUT_DIR"

echo "Collect done. Figure updated: ${OUT_DIR}/two_qubit_scan.png"
