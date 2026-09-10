#!/bin/bash
# ============================================================
#  SLURM job-array: compass-chain many-body panels (7-8) for ONE case --
#  oscillation rate and Lindbladian gap of the full Liouvillian of an open
#  8-qubit compass chain, over compass_chain's (gamma, h) grid.
#
#  One grid point per array task: 11x11 = 121 points -> --array=0-120.  Each
#  task builds the sparse 65536x65536 Liouvillian and takes two partial spectra
#  in ARPACK regular mode (K_OSC rightmost modes for max|Im/Re|, K_GAP for the
#  gap).  Dense diagonalization and shift-invert are both out of reach at N=8
#  (see n_qubit_lindbladian.lindbladian_gap).
#
#  Submit (normally through scripts/submit_compass_all.sh):
#    mkdir -p logs results_compass
#    CASE=jx1.0_jy1.0_hx sbatch scripts/compass_manybody.slurm.sh
#
#  Output: results_compass/<CASE>/manybody/pt_<ig>_<ih>.npz
#
#  RUNTIME IS UNCERTAIN: ARPACK convergence on a 65536-dim non-normal operator
#  varies a lot across the grid.  Check a log before scaling up.
# ============================================================

#SBATCH --job-name=cmp_8q
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-120
#SBATCH --output=logs/cmp8q_%x_%A_%a.out
#SBATCH --error=logs/cmp8q_%x_%A_%a.err

CASE=${CASE:?set CASE to a key of compass_chain.CASES}
OUT_DIR=${OUT_DIR:-results_compass}
N_CHUNKS=${N_CHUNKS:-121}      # must match the --array size above
N_QUBITS=${N_QUBITS:-8}
TOPOLOGY=${TOPOLOGY:-chain}
METHOD=${METHOD:-sparse}       # 'sparse' required at N=8
K_OSC=${K_OSC:-64}
K_GAP=${K_GAP:-12}
WHICH=${WHICH:-LR}
NOISE_FLOOR=${NOISE_FLOOR:-1e-6}
MAXITER=${MAXITER:-10000}
# Empty SIGMA => ARPACK regular mode (matvecs only).  Shift-invert's sparse LU
# does not fit in a job's memory at N=8; set SIGMA only for small N.
SIGMA=${SIGMA:-}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[compass 8q] ${CASE} chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/compass_manybody_worker.py \
    --case        "$CASE" \
    --task_id     "$SLURM_ARRAY_TASK_ID" \
    --n_chunks    "$N_CHUNKS" \
    --out_dir     "$OUT_DIR" \
    --n_qubits    "$N_QUBITS" \
    --topology    "$TOPOLOGY" \
    --method      "$METHOD" \
    --k_osc       "$K_OSC" \
    --k_gap       "$K_GAP" \
    --which       "$WHICH" \
    ${SIGMA:+"--sigma=$SIGMA"} \
    --noise_floor "$NOISE_FLOOR" \
    --maxiter     "$MAXITER"
status=$?

echo "[compass 8q] ${CASE} chunk ${SLURM_ARRAY_TASK_ID}: exit ${status}"
exit "$status"
