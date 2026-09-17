#!/bin/bash
# ============================================================
#  SLURM job-array: random-frame framabilities along the DT_BASE line
#  (scripts/dtbase_randframe_worker.py), for the extra panels of
#  results_dtbase_line/model3_dtbase_extrap.png:
#
#    prod_mix_10,  prod_mix_40    product frame of random MIXED 1-qubit states
#    heis_unit_10, heis_unit_40   Heisenberg D = S(x)S, S = identity + random
#                                 X,Y,Z columns of operator norm exactly 1
#    heis_rnorm_10, heis_rnorm_40 same directions, norms ~ U[SUPPORT_EPS, 1]
#
#  Grid: model3, gamma = 0..10 (51 values) x gamma' = 0..4.2 (22 values)
#  = 1122 points x 10 DT_BASE values = 11220 (point, DT_BASE) pairs, strided
#  over a 0-199 array (~56 pairs per task, every task samples the whole grid).
#  Frames are fixed (seeded) and identical in every task.
#
#  Submit together with the dependent collect job (see
#  scripts/dtbase_randframe_collect.slurm.sh):
#    mkdir -p logs results_dtbase_randframe
#    JID=$(sbatch --parsable scripts/dtbase_randframe.slurm.sh)
#    sbatch --dependency=afterok:${JID} scripts/dtbase_randframe_collect.slurm.sh
#
#  Idempotent per key (the file is rewritten after every key): if some tasks
#  hit the time limit, resubmit the same script -- only the holes are computed.
#
#  Output: results_dtbase_randframe/<tag>/base_<idx>.npz
# ============================================================

#SBATCH --job-name=dtb_randframe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/dtbrandframe_%x_%A_%a.out
#SBATCH --error=logs/dtbrandframe_%x_%A_%a.err

MODEL=${MODEL:-model3}
OUT_DIR=${OUT_DIR:-results_dtbase_randframe}
N_CHUNKS=${N_CHUNKS:-200}     # must match the --array size above
STRIDE=${STRIDE:-1}           # must match the collect job
P2_MAX=${P2_MAX:-4.2}         # largest gamma' swept
DIM=${DIM:-2}                 # same bond convention as trotter_dtbase_line.slurm.sh

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[$MODEL randframe] chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/dtbase_randframe_worker.py \
    --model    "$MODEL" \
    --task_id  "$SLURM_ARRAY_TASK_ID" \
    --n_chunks "$N_CHUNKS" \
    --stride   "$STRIDE" \
    --p2_max   "$P2_MAX" \
    --out_dir  "$OUT_DIR" \
    --dim      "$DIM"

echo "[$MODEL randframe] chunk ${SLURM_ARRAY_TASK_ID}: done"
