#!/bin/bash
# ============================================================
#  SLURM job-array: oscillation rate  max |Im(lambda)/Re(lambda)|  of the full
#  8-qubit ring Lindbladian, over model3's (gamma, gamma') grid.  Becomes the
#  extra panel on the dtbase-line framability figure.
#
#  The 51x51 = 2601 grid points are split across a 0-199 array (200 tasks, the
#  job cap), ~13 points per task.  Each point builds the sparse 65536x65536
#  Liouvillian and takes the K rightmost eigenvalues via scipy eigs -- dense
#  diagonalization at N=8 is ~69 GB/point and is refused by the library.
#
#  Submit:
#    mkdir -p logs results_osc_rate
#    sbatch scripts/osc_rate.slurm.sh
#    MODEL=model3 K=128 sbatch scripts/osc_rate.slurm.sh     # tighter bound
#    N_QUBITS=6 METHOD=dense sbatch scripts/osc_rate.slurm.sh  # exact, smaller N
#
#  Output: results_osc_rate/<model>/pt_<ix>_<iy>.npz
#
#  RUNTIME IS UNCERTAIN: ARPACK convergence on a 65536-dim non-normal operator
#  varies a lot with (gamma, gamma').  Start with one array, check a log, and
#  raise --time / lower K if tasks are timing out.
# ============================================================

#SBATCH --job-name=osc_rate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/oscrate_%x_%A_%a.out
#SBATCH --error=logs/oscrate_%x_%A_%a.err

MODEL=${MODEL:-model3}
OUT_DIR=${OUT_DIR:-results_osc_rate}
N_CHUNKS=${N_CHUNKS:-200}      # must match the --array size above
STRIDE=${STRIDE:-1}            # 1 = full 51x51, matching the framability panels
N_QUBITS=${N_QUBITS:-8}
METHOD=${METHOD:-sparse}       # 'sparse' required at N=8; 'dense' only for N<=6
K=${K:-64}
WHICH=${WHICH:-LR}
MAXITER=${MAXITER:-10000}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[$MODEL osc_rate N=$N_QUBITS $METHOD k=$K] chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/osc_rate_worker.py \
    --model    "$MODEL" \
    --task_id  "$SLURM_ARRAY_TASK_ID" \
    --n_chunks "$N_CHUNKS" \
    --out_dir  "$OUT_DIR" \
    --stride   "$STRIDE" \
    --n_qubits "$N_QUBITS" \
    --method   "$METHOD" \
    --k        "$K" \
    --which    "$WHICH" \
    --maxiter  "$MAXITER"

echo "[$MODEL osc_rate] chunk ${SLURM_ARRAY_TASK_ID}: done"
