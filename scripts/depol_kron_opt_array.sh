#!/bin/bash
# ============================================================
#  SLURM job-array: heavy optimisation of the Heisenberg-picture
#  framability of four fixed 2-qubit gates
#
#     channel = depol_2q(p) . superop(exp(i*(a*XX + b*YY + c*ZZ)))
#
#     g1_p0.00 : (a,b,c) = (sqrt(0.5), exp(-1), pi),  p = 0.00
#     g1_p0.08 : (a,b,c) = (sqrt(0.5), exp(-1), pi),  p = 0.08
#     g2_p0.00 : (a,b,c) = (0.3, 0.3, 0.0),           p = 0.00
#     g2_p0.08 : (a,b,c) = (0.3, 0.3, 0.0),           p = 0.08
#
#  for d_ext_single in {4, 6, 8}.  Each (gate, d) cell is spread over
#  N_BATCHES = 16 independent random-seed batches; the collect step keeps
#  the global minimum over batches.
#
#  task_id = gate_idx * (N_D * N_BATCHES) + d_idx * N_BATCHES + batch_idx
#    gate_idx  0..3   (4 gates)
#    d_idx     0..2   (D_EXT_SINGLES = [4, 6, 8])
#    batch_idx 0..15  (N_BATCHES = 16)
#
#  Total tasks: 4 * 3 * 16 = 192  (task_ids 0..191)
#
#  Submit:
#      mkdir -p logs results_depol_kron_opt
#      sbatch --array=0-191%200 scripts/depol_kron_opt_array.sh
#
#  Every restart's converged frame is recorded (not only the best), so the
#  collect step can pick a robustly reachable optimum (widest near-optimal
#  basin), reporting a reachability score -- not just the framability value.
#
#  Collect (after the array finishes):
#      python scripts/depol_kron_opt_collect.py \
#          --in_dir results_depol_kron_opt --out_dir results_depol_kron_opt \
#          --eps_tol 1e-4 --fp_tol 5e-3
#
#  Budgets are overridable via env vars, e.g.:
#      N_RESTARTS=100 POLISH_ITER=8000 sbatch --array=0-191%200 \
#          scripts/depol_kron_opt_array.sh
# ============================================================

#SBATCH --job-name=depol_kron_opt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --output=logs/depol_kron_opt_%A_%a.out
#SBATCH --error=logs/depol_kron_opt_%A_%a.err

OUT_DIR=${OUT_DIR:-results_depol_kron_opt}
N_RESTARTS=${N_RESTARTS:-60}
MAX_ITER=${MAX_ITER:-2000}
MAXFEV=${MAXFEV:-12000}
POLISH_ITER=${POLISH_ITER:-4000}
N_POLISH=${N_POLISH:-3}
TOL=${TOL:-1e-9}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[depol_kron_opt] task ${SLURM_ARRAY_TASK_ID}: starting  (n_restarts=${N_RESTARTS}, polish_iter=${POLISH_ITER}, n_polish=${N_POLISH})"

python scripts/depol_kron_opt_worker.py \
    --task_id     "$SLURM_ARRAY_TASK_ID" \
    --out_dir     "$OUT_DIR" \
    --n_restarts  "$N_RESTARTS" \
    --max_iter    "$MAX_ITER" \
    --maxfev      "$MAXFEV" \
    --polish_iter "$POLISH_ITER" \
    --n_polish    "$N_POLISH" \
    --tol         "$TOL" \
    --seed        "$SEED"

echo "[depol_kron_opt] task ${SLURM_ARRAY_TASK_ID}: done"
