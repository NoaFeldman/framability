#!/bin/bash
# ============================================================
#  SLURM job-array: reweighted-Pauli rate of the open Dicke model against the
#  smallest allowed weight w_min (scripts/dicke_pauli_weight_worker.py,
#  framability_rate_pauli_weights.py).  Task t takes combos t, t + N_TASKS,
#  ... (one file per combo, written atomically, finished combos skipped), so
#  tasks never collide and a re-submit resumes.
#
#  Submitted by scripts/submit_dicke_pauli_weight.sh (sets --array to the
#  number of combos, capped at 200, and the dependent collect job); by hand:
#    mkdir -p logs
#    N_TASKS=11 sbatch --array=0-10 scripts/dicke_pauli_weight.slurm.sh
#
#  Output: results_dicke_pauli_weight/<family>_N<N>_M<M>/points/<label>.npz
#          results_dicke_pauli_weight/<family>_N<N>_M<M>_w<width>/pt_<ix>_<iy>.npz (GRID=1)
#
#  Knobs (env): FAMILY (dicke_dec), N_ATOMS (100), M_SPECIES (4), POINTS
#  (unset = all labelled points; "U1 D3" = only these; "none" = grid only),
#  GRID (unset; 1 = also every grid point for the w* maps), GRID_WIDTHS
#  ("0 0.5"), LOG10_WMIN_END (-8), N_W (33), N_RANDOM (3), N_BISECT (30),
#  N_TASKS (= array size), OUT_ROOT (results_dicke_pauli_weight), FORCE.
#  Load: a uniform point ~1 min, a disordered point (10 pairs, shared +
#  species) a few min; GRID=1 adds 2 x 336 points over 200 tasks.
# ============================================================

#SBATCH --job-name=dicke_pw
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dicke_pw_%A_%a.out
#SBATCH --error=logs/dicke_pw_%A_%a.err

FAMILY=${FAMILY:-dicke_dec}
N_ATOMS=${N_ATOMS:-100}
M_SPECIES=${M_SPECIES:-4}
POINTS=${POINTS:-}
GRID=${GRID:-}
GRID_WIDTHS=${GRID_WIDTHS:-"0 0.5"}
LOG10_WMIN_END=${LOG10_WMIN_END:--8}
N_W=${N_W:-33}
N_RANDOM=${N_RANDOM:-3}
N_BISECT=${N_BISECT:-30}
N_TASKS=${N_TASKS:-1}              # must match the --array size
OUT_ROOT=${OUT_ROOT:-results_dicke_pauli_weight}
FORCE=${FORCE:-}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

if [ "$POINTS" = "none" ]; then
    POINT_ARGS="--points"
elif [ -n "$POINTS" ]; then
    POINT_ARGS="--points $POINTS"
else
    POINT_ARGS=""
fi

echo "[dicke pauli weight] task ${SLURM_ARRAY_TASK_ID}/${N_TASKS}: ${FAMILY} N=${N_ATOMS} M=${M_SPECIES} points=${POINTS:-all} grid=${GRID:-0}"

python scripts/dicke_pauli_weight_worker.py \
    --task_id        "$SLURM_ARRAY_TASK_ID" \
    --n_tasks        "$N_TASKS" \
    --family         "$FAMILY" \
    --N              "$N_ATOMS" \
    --M              "$M_SPECIES" \
    --grid_widths    $GRID_WIDTHS \
    --log10_wmin_end "$LOG10_WMIN_END" \
    --n_w            "$N_W" \
    --n_random       "$N_RANDOM" \
    --n_bisect       "$N_BISECT" \
    --out_root       "$OUT_ROOT" \
    $POINT_ARGS \
    ${GRID:+--grid} \
    ${FORCE:+--force}

echo "[dicke pauli weight] task ${SLURM_ARRAY_TASK_ID}: done"
