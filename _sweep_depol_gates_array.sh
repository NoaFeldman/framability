#!/bin/bash
# ============================================================
#  SLURM array: depolarised-gate sweep
#    gates  = CNOT, H, T            (gate_idx 0..2)
#    p      = 0.00, 0.01, ..., 0.07 (p_idx 0..7)
#    task_id = gate_idx * 8 + p_idx
#  Total: 24 tasks.
#
#  Each task computes framability for 5 frame choices plus OTOC,
#  channel stabilizer purity and operator bond entropy.
#
#  Submit:
#    export OUT_DIR=results_depol_sweep
#    sbatch --array=0-23 _sweep_depol_gates_array.sh
# ============================================================
#SBATCH --job-name=depol_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/depol_sweep_%A_%a.out
#SBATCH --error=logs/depol_sweep_%A_%a.err

OUT_DIR=${OUT_DIR:-results_depol_sweep}
N_RESTARTS=${N_RESTARTS:-5}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID}: starting  (OUT_DIR=${OUT_DIR})"
python _sweep_depol_gates_worker.py \
    --task_id    "$SLURM_ARRAY_TASK_ID" \
    --out_dir    "$OUT_DIR" \
    --n_restarts "$N_RESTARTS"
echo "Task ${SLURM_ARRAY_TASK_ID}: done"
