#!/bin/bash
# ============================================================
#  SLURM single job: cross-evaluation sweep of the Trotter-scan
#  optimised framabilities (opt_fra_4 / opt_fra_6).
#
#  Frame label-propagation with no optimisation: every point is
#  evaluated in all its 4-connected neighbours' stored frames
#  (one batched LP per frame — a rigorous upper bound), iterated
#  Gauss-Seidel until a full sweep changes nothing.  Erases the
#  framability 'islands' whose neighbour frame transfers exactly
#  and prints a stall-vs-branch diagnostic for the rest.
#
#  The whole grid fits in ONE job (a sweep is ~13 LPs/point, ms
#  each, and the propagation is inherently sequential), so this
#  is not an array; parallelise across MODELs instead.
#
#    MODEL=model1 ROUND=1 sbatch scripts/trotter_scan_island_fix.slurm.sh
# ============================================================

#SBATCH --job-name=trot_xeval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/trotxeval_%x_%A.out
#SBATCH --error=logs/trotxeval_%x_%A.err

MODEL=${MODEL:-model1}
ROUND=${ROUND:-1}
OUT_DIR=${OUT_DIR:-results_trotter_v3}
TOL=${TOL:-1e-9}
FRA_TOL=${FRA_TOL:-1e-6}
MAX_SWEEPS=${MAX_SWEEPS:-200}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[$MODEL] cross-eval sweep round ${ROUND}: starting"

python scripts/trotter_scan_cross_eval.py \
    --model      "$MODEL" \
    --round      "$ROUND" \
    --out_dir    "$OUT_DIR" \
    --tol        "$TOL" \
    --fra_tol    "$FRA_TOL" \
    --max_sweeps "$MAX_SWEEPS"

echo "[$MODEL] cross-eval sweep round ${ROUND}: done"
