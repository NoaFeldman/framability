#!/bin/bash
# ============================================================
#  SLURM job-array: model4 many-body panels (7-8 of the model4 figure) --
#  oscillation rate and Lindbladian gap of the FULL Lindbladian of a 2x4
#  lattice of 8 qubits, over model4's (gamma, gamma') grid.
#
#  Each point builds the sparse 65536x65536 Liouvillian of model4 on the 2x4
#  open-boundary lattice (H = J sum ZZ + 1.5 sum X, jumps sqrt(gamma)|-><+|_i,
#  sqrt(gamma')Z_i) and takes two partial spectra via scipy eigs in ARPACK
#  regular mode: K_OSC rightmost modes for max|Im/Re|, K_GAP for the gap.
#  Dense diagonalization at N=8 is ~69 GB/point and is refused by the library;
#  shift-invert is NOT used (its sparse LU does not fit in a job's memory at
#  this size -- see n_qubit_lindbladian.lindbladian_gap).
#
#  Default STRIDE=5 -> 11x11 = 121 points, one per array task (121 <= 200),
#  matching the model3 item-4 gap pipeline.  If you change STRIDE you MUST
#  edit --array and N_CHUNKS to match: n_grid = len(p1_vals[::STRIDE]),
#  n_total = n_grid**2.  For the full 51x51 grid use
#      STRIDE=1 N_CHUNKS=200 sbatch ...   with #SBATCH --array=0-199
#  (each task then walks ~14 points in a strided sweep).
#
#  Submit:
#    mkdir -p logs results_model4_rate
#    sbatch scripts/model4_manybody.slurm.sh
#    K_OSC=128 sbatch scripts/model4_manybody.slurm.sh      # tighter osc bound
#
#  Output: results_model4_rate/model4_8q/pt_<ix>_<iy>.npz
#
#  RUNTIME IS UNCERTAIN: ARPACK convergence on a 65536-dim non-normal operator
#  varies a lot with (gamma, gamma').  Check a log before scaling up.
# ============================================================

#SBATCH --job-name=m4_8q
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-120
#SBATCH --output=logs/m48q_%x_%A_%a.out
#SBATCH --error=logs/m48q_%x_%A_%a.err

OUT_DIR=${OUT_DIR:-results_model4_rate}
STRIDE=${STRIDE:-5}            # 5 -> 11x11 = 121 points (matches --array=0-120)
N_CHUNKS=${N_CHUNKS:-121}      # must match the --array size above
METHOD=${METHOD:-sparse}       # 'sparse' required at N=8
K_OSC=${K_OSC:-64}
K_GAP=${K_GAP:-12}
WHICH=${WHICH:-LR}
NOISE_FLOOR=${NOISE_FLOOR:-1e-6}
MAXITER=${MAXITER:-10000}
# SIGMA is intentionally EMPTY by default: empty => ARPACK regular mode, which
# touches the operator only through matrix-vector products and so exploits its
# sparsity.  Setting SIGMA turns on shift-invert, whose sparse LU fill-in
# exhausts the job's memory at N=8.  Set it only for small N.
SIGMA=${SIGMA:-}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[model4 8q] chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/model4_manybody_worker.py \
    --task_id     "$SLURM_ARRAY_TASK_ID" \
    --n_chunks    "$N_CHUNKS" \
    --out_dir     "$OUT_DIR" \
    --stride      "$STRIDE" \
    --method      "$METHOD" \
    --k_osc       "$K_OSC" \
    --k_gap       "$K_GAP" \
    --which       "$WHICH" \
    ${SIGMA:+"--sigma=$SIGMA"} \
    --noise_floor "$NOISE_FLOOR" \
    --maxiter     "$MAXITER"

echo "[model4 8q] chunk ${SLURM_ARRAY_TASK_ID}: done"
