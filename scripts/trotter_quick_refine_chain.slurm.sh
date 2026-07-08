#!/bin/bash
# ============================================================
#  Chain controller for convergence-driven quick refinement of the
#  Trotter Lindbladian scan.
#
#  Submitted with --dependency=afterany:<round-R array> and ROUND=R.
#  It checks whether round R changed any opt_fra value
#  (scripts/trotter_quick_refine_check.py):
#    - CHANGED and ROUND < MAX_ROUNDS -> submit the round R+1 array
#      (trotter_scan_quick_refine.slurm.sh, 200 tasks) plus a new copy
#      of this controller chained after it.
#    - CONVERGED (or MAX_ROUNDS hit)  -> submit the final collect job
#      (trotter_scan_refine_collect.py) that merges all rounds and
#      rebuilds the figure.
#
#  Environment (inherited by every job it submits via sbatch --export=ALL):
#    MODEL (required), ROUND (required), OUT_DIR (results_trotter_v3),
#    MAX_ROUNDS (100), CHECK_TOL (1e-9),
#    plus anything the quick-refine array reads
#    (N_RESTARTS, FRA_MAXFEV_4/6, FRA_TOL, SEED).
#
#  Do not submit this by hand — use scripts/submit_trotter_quick_refine_conv.sh
# ============================================================

#SBATCH --job-name=trot_qchain
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=logs/trotqchain_%x_%A.out
#SBATCH --error=logs/trotqchain_%x_%A.err

MODEL=${MODEL:?MODEL must be set}
ROUND=${ROUND:?ROUND must be set}
OUT_DIR=${OUT_DIR:-results_trotter_v3}
MAX_ROUNDS=${MAX_ROUNDS:-100}
CHECK_TOL=${CHECK_TOL:-1e-9}

cd "${SLURM_SUBMIT_DIR}"
source "${PWD}/.venv/bin/activate"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

submit_collect () {
    COLLECT=$(sbatch --parsable \
        --job-name=trot_qcollect --ntasks=1 --cpus-per-task=1 --mem=4G \
        --time=00:30:00 --output=logs/trotqcol_%x_%A.out --error=logs/trotqcol_%x_%A.err \
        --wrap="source ${PWD}/.venv/bin/activate; cd ${PWD}; \
                export MPLCONFIGDIR=/tmp/mpl-\$SLURM_JOB_ID; \
                python scripts/trotter_scan_refine_collect.py \
                    --model $MODEL --in_dir $OUT_DIR")
    echo "[$MODEL] collect figure : job $COLLECT"
    echo "[$MODEL] -> results_plots/trotter_${MODEL}.png"
}

echo "[$MODEL] checking quick round $ROUND for changes ..."
python scripts/trotter_quick_refine_check.py \
    --model "$MODEL" --round "$ROUND" --in_dir "$OUT_DIR" --tol "$CHECK_TOL"
RC=$?

if [ "$RC" -eq 0 ]; then
    if [ "$ROUND" -ge "$MAX_ROUNDS" ]; then
        echo "[$MODEL] round $ROUND still changed but MAX_ROUNDS=$MAX_ROUNDS reached" \
             "-> stopping WITHOUT convergence; resubmit to continue."
        submit_collect
        exit 0
    fi
    NEXT=$((ROUND + 1))
    JID=$(MODEL=$MODEL ROUND=$NEXT OUT_DIR=$OUT_DIR \
          sbatch --parsable scripts/trotter_scan_quick_refine.slurm.sh)
    echo "[$MODEL] quick refine round $NEXT : job $JID"
    CHAIN=$(MODEL=$MODEL ROUND=$NEXT OUT_DIR=$OUT_DIR \
            MAX_ROUNDS=$MAX_ROUNDS CHECK_TOL=$CHECK_TOL \
            sbatch --parsable --dependency=afterany:$JID \
                   scripts/trotter_quick_refine_chain.slurm.sh)
    echo "[$MODEL] chain check after round $NEXT : job $CHAIN  (after $JID)"
elif [ "$RC" -eq 3 ]; then
    echo "[$MODEL] converged after quick round $ROUND."
    submit_collect
else
    echo "[$MODEL] ERROR: convergence check exited with code $RC — chain stopped." >&2
    echo "[$MODEL] inspect this log, then resubmit via submit_trotter_quick_refine_conv.sh" >&2
    exit 1
fi
