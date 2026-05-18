#!/bin/bash
# ============================================================
#  SLURM single-task: collect unified neighbor-refine results.
#  Required env vars: VARIANT, ROUND, OUT_DIR
#
#  Submit:
#    VARIANT=free6 ROUND=1 OUT_DIR=results_free6 \
#      sbatch --dependency=afterok:<arrayid> unified_nb_refine_collect.sh
# ============================================================

#SBATCH --job-name=unb_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/unb_collect_%j.out
#SBATCH --error=logs/unb_collect_%j.err

VARIANT=${VARIANT:?Must set VARIANT}
ROUND=${ROUND:?Must set ROUND}
OUT_DIR=${OUT_DIR:?Must set OUT_DIR}
N_PTS=${N_PTS:-41}
N_IGP=${N_IGP:-21}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "Collecting ${VARIANT} round ${ROUND} from ${OUT_DIR}/ ..."

python unified_nb_refine_collect.py \
    --variant "$VARIANT" \
    --round   "$ROUND" \
    --n_pts   "$N_PTS" \
    --n_igp   "$N_IGP" \
    --out_dir "$OUT_DIR" \
    --cleanup

echo "Collect done."
