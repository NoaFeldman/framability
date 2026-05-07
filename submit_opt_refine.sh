#!/bin/bash
# ============================================================
#  Submit neighbor-seeded opt refinement on SLURM.
#  Submits the array job, then a dependent collect job.
#
#  Usage:
#    bash submit_opt_refine.sh [--n_pts N] [--out_dir DIR]
#                              [--n_restarts R] [--maxfev M]
#                              [--max_concurrent C]
#                              [--after_job JOBID]
# ============================================================
set -euo pipefail

N_PTS=41
J=1.0
GAMMA_STEP=0.2
OUT_DIR=results_opt
N_RESTARTS=5
MAXFEV=1000
MAX_CONCURRENT=200
AFTER_JOB=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n_pts)         N_PTS="$2";          shift 2 ;;
        --out_dir)       OUT_DIR="$2";        shift 2 ;;
        --n_restarts)    N_RESTARTS="$2";     shift 2 ;;
        --maxfev)        MAXFEV="$2";         shift 2 ;;
        --max_concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --after_job)     AFTER_JOB="$2";      shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

ARRAY_END=$(( N_PTS * N_PTS - 1 ))

mkdir -p logs

export N_PTS J GAMMA_STEP OUT_DIR N_RESTARTS MAXFEV

ARRAY_ARGS=(
    --parsable
    "--array=0-${ARRAY_END}%${MAX_CONCURRENT}"
    "--export=N_PTS,J,GAMMA_STEP,OUT_DIR,N_RESTARTS,MAXFEV"
)
if [[ -n "$AFTER_JOB" ]]; then
    ARRAY_ARGS+=( "--dependency=afterok:${AFTER_JOB}" )
fi

ARRAY_JOB=$(sbatch "${ARRAY_ARGS[@]}" opt_refine_array.sh)
echo "Submitted opt_refine array job: ${ARRAY_JOB}"

COLLECT_JOB=$(sbatch \
    --parsable \
    --job-name=opt_refine_collect \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=4G \
    --time=00:15:00 \
    --output="logs/opt_refine_collect_%j.out" \
    --error="logs/opt_refine_collect_%j.err" \
    "--dependency=afterok:${ARRAY_JOB}" \
    --wrap="cd \"\${SLURM_SUBMIT_DIR}\" && \
            source .venv/bin/activate && \
            python opt_refine_collect.py \
                --out_dir \"${OUT_DIR}\" \
                --n_pts ${N_PTS} \
                --gamma_step ${GAMMA_STEP}")
echo "Submitted opt_refine collect job: ${COLLECT_JOB} (depends on ${ARRAY_JOB})"
