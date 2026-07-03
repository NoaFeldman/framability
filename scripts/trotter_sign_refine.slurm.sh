#!/bin/bash
# ============================================================
#  SLURM job-array: neighbour-seeded refinement of the optimised sign problem
#  (sign_opt) for the Trotter scan.
#
#  Run 3 sequential rounds (ROUND=1..3); each round seeds every point's
#  rotation search with the best rotation found so far at the point itself and
#  its 4-connected neighbours, so the winning basis propagates across the grid
#  and the sign_opt map becomes smooth.  Requires the base scan for MODEL.
#
#  Submit (chained):
#    R1=$(MODEL=model1 ROUND=1 sbatch --parsable scripts/trotter_sign_refine.slurm.sh)
#    R2=$(MODEL=model1 ROUND=2 sbatch --parsable --dependency=afterok:$R1 \
#         scripts/trotter_sign_refine.slurm.sh)
#    R3=$(MODEL=model1 ROUND=3 sbatch --parsable --dependency=afterok:$R2 \
#         scripts/trotter_sign_refine.slurm.sh)
#  then merge:  python scripts/trotter_sign_refine_collect.py --model model1
# ============================================================

#SBATCH --job-name=trot_sign
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=10:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/trotsign_%x_%A_%a.out
#SBATCH --error=logs/trotsign_%x_%A_%a.err

MODEL=${MODEL:-model1}
ROUND=${ROUND:-1}
OUT_DIR=${OUT_DIR:-results_trotter}
N_CHUNKS=${N_CHUNKS:-200}
N_RESTARTS=${N_RESTARTS:-20}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[$MODEL] sign chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS} round ${ROUND}: starting"

python scripts/trotter_sign_refine_worker.py \
    --model      "$MODEL" \
    --task_id    "$SLURM_ARRAY_TASK_ID" \
    --n_chunks   "$N_CHUNKS" \
    --round      "$ROUND" \
    --out_dir    "$OUT_DIR" \
    --n_restarts "$N_RESTARTS" \
    --seed       "$SEED"

echo "[$MODEL] sign chunk ${SLURM_ARRAY_TASK_ID} round ${ROUND}: done"
