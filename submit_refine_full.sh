#!/bin/bash
# ============================================================
#  Master submission script: fine-grain bond entropy + 3-round
#  neighbor framability refinement, then regenerate the
#  bond-vs-framability figure.
#
#  All work is restricted to gamma' < GP_CUTOFF (default 4.0),
#  i.e. igp < N_IGP = floor(GP_CUTOFF / GAMMA_STEP).
#
#  Pipeline (each step depends on the previous via afterok):
#   1. bond_entropy_refine array      (N_PTS * N_IGP tasks)
#   2. bond_entropy_refine collect    (1 task – updates point_extra + scan_full)
#   3. neighbor refine round 1 array  (N_PTS * N_IGP tasks)
#   4. neighbor refine round 1 collect
#   5. neighbor refine round 2 array  (N_PTS * N_IGP tasks)
#   6. neighbor refine round 2 collect
#   7. neighbor refine round 3 array  (N_PTS * N_IGP tasks)
#   8. neighbor refine round 3 collect
#   9. regen_bond_vs_fra plot
#
#  Usage
#  -----
#    bash submit_refine_full.sh [options]
#
#  Options
#  -------
#    --n_pts          N        Grid size (default: 41)
#    --J              J        Coupling constant (default: 1.0)
#    --gamma_step     S        Grid spacing (default: 0.2)
#    --gp_cutoff      C        Restrict to gamma' < C (default: 4.0)
#    --out_dir        DIR      Output directory (default: results)
#    --n_restarts     K        Framability restarts per point (default: 5)
#    --maxfev         F        Max func-evals per restart (default: 1000)
#    --max_concurrent C        Max simultaneous array tasks (default: 100)
#    --after_job      JOB_ID   Optional dependency for the first array job
#    --out_name       NAME     Figure stem (default: two_qubit_scan_full.png)
# ============================================================

set -euo pipefail

# ── defaults ─────────────────────────────────────────────────
N_PTS=41
J=1.0
GAMMA_STEP=0.2
GP_CUTOFF=4.0
OUT_DIR=results
N_RESTARTS=5
MAXFEV=1000
MAX_CONCURRENT=100
AFTER_JOB=""
OUT_NAME="two_qubit_scan_full.png"

# ── parse arguments ──────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n_pts)          N_PTS="$2";          shift 2 ;;
        --J)              J="$2";              shift 2 ;;
        --gamma_step)     GAMMA_STEP="$2";     shift 2 ;;
        --gp_cutoff)      GP_CUTOFF="$2";      shift 2 ;;
        --out_dir)        OUT_DIR="$2";        shift 2 ;;
        --n_restarts)     N_RESTARTS="$2";     shift 2 ;;
        --maxfev)         MAXFEV="$2";         shift 2 ;;
        --max_concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --after_job)      AFTER_JOB="$2";      shift 2 ;;
        --out_name)       OUT_NAME="$2";       shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── derived parameters ────────────────────────────────────────
# N_IGP = floor(GP_CUTOFF / GAMMA_STEP)
N_IGP=$(python3 -c "print(int(float('${GP_CUTOFF}') / float('${GAMMA_STEP}')))")
BOND_ARRAY_END=$(( N_PTS * N_IGP - 1 ))
NB_ARRAY_END=$(( N_PTS * N_IGP - 1 ))

mkdir -p "${OUT_DIR}" logs

# ── sanity checks ─────────────────────────────────────────────
if [[ ! -f "${OUT_DIR}/scan_full.npy" ]]; then
    echo "ERROR: ${OUT_DIR}/scan_full.npy not found."
    echo "Run the main scan first (submit_scan.sh / build_two_qubit_scan_full.py)."
    exit 1
fi

# Check at least one point_extra file exists
SAMPLE_EXTRA="${OUT_DIR}/point_extra_0000_0000.npy"
if [[ ! -f "$SAMPLE_EXTRA" ]]; then
    echo "ERROR: ${SAMPLE_EXTRA} not found."
    echo "Run scan_worker_extra.py to generate point_extra files first."
    exit 1
fi

echo "========================================================"
echo "  Refinement pipeline: bond entropy + 3x neighbor refine"
echo "  N_PTS        = ${N_PTS}"
echo "  GAMMA_STEP   = ${GAMMA_STEP}"
echo "  GP_CUTOFF    = ${GP_CUTOFF}  ->  N_IGP = ${N_IGP}"
echo "  Array tasks  = ${N_PTS} x ${N_IGP} = $((N_PTS * N_IGP))"
echo "  J            = ${J}"
echo "  OUT_DIR      = ${OUT_DIR}"
echo "  n_restarts   = ${N_RESTARTS}  maxfev = ${MAXFEV}"
echo "  Max concurrent = ${MAX_CONCURRENT}"
if [[ -n "$AFTER_JOB" ]]; then
    echo "  Initial dependency = afterok:${AFTER_JOB}"
fi
echo "========================================================"

# ── helper: submit a job and capture its ID ──────────────────
_submit() {
    # Usage: _submit <sbatch_args...>
    sbatch --parsable "$@"
}

# ── helper: build env exports for array scripts ──────────────
_env() {
    echo "N_PTS=${N_PTS} J=${J} GAMMA_STEP=${GAMMA_STEP} OUT_DIR=${OUT_DIR}"
}

# ── Step 1: bond entropy refinement array ────────────────────
DEPEND_1=""
if [[ -n "$AFTER_JOB" ]]; then
    DEPEND_1="--dependency=afterok:${AFTER_JOB}"
fi

BOND_ARRAY_JOB=$(
    N_PTS="$N_PTS" N_IGP="$N_IGP" J="$J" GAMMA_STEP="$GAMMA_STEP" OUT_DIR="$OUT_DIR" \
    _submit \
        --array="0-${BOND_ARRAY_END}%${MAX_CONCURRENT}" \
        ${DEPEND_1} \
        bond_entropy_refine_array.sh
)
echo "[1] Bond-entropy refine array  job: ${BOND_ARRAY_JOB}  (tasks 0-${BOND_ARRAY_END})"

# ── Step 2: bond entropy collect ─────────────────────────────
BOND_COLLECT_JOB=$(
    N_PTS="$N_PTS" N_IGP="$N_IGP" J="$J" GAMMA_STEP="$GAMMA_STEP" OUT_DIR="$OUT_DIR" \
    _submit \
        --dependency="afterok:${BOND_ARRAY_JOB}" \
        bond_entropy_refine_collect.sh
)
echo "[2] Bond-entropy collect        job: ${BOND_COLLECT_JOB}"

# ── Steps 3-8: three rounds of neighbor framability refine ───
PREV_JOB="${BOND_COLLECT_JOB}"

for ROUND in 1 2 3; do
    # Array job (depends on previous collect)
    NB_ARRAY_JOB=$(
        N_PTS="$N_PTS" N_IGP="$N_IGP" J="$J" GAMMA_STEP="$GAMMA_STEP" \
        OUT_DIR="$OUT_DIR" N_RESTARTS="$N_RESTARTS" MAXFEV="$MAXFEV" \
        _submit \
            --array="0-${NB_ARRAY_END}%${MAX_CONCURRENT}" \
            --dependency="afterok:${PREV_JOB}" \
            neighbor_refine_restricted_array.sh
    )
    echo "[$((ROUND*2+1))] Neighbor-refine round ${ROUND} array  job: ${NB_ARRAY_JOB}  (tasks 0-${NB_ARRAY_END})"

    # Collect job (depends on array job)
    NB_COLLECT_JOB=$(
        N_PTS="$N_PTS" J="$J" GAMMA_STEP="$GAMMA_STEP" OUT_DIR="$OUT_DIR" \
        _submit \
            --dependency="afterok:${NB_ARRAY_JOB}" \
            neighbor_refine_collect.sh
    )
    echo "[$((ROUND*2+2))] Neighbor-refine round ${ROUND} collect job: ${NB_COLLECT_JOB}"

    PREV_JOB="${NB_COLLECT_JOB}"
done

# ── Step 9: regenerate bond-vs-fra figure ────────────────────
PLOT_JOB=$(
    N_PTS="$N_PTS" J="$J" GAMMA_STEP="$GAMMA_STEP" OUT_DIR="$OUT_DIR" OUT_NAME="$OUT_NAME" \
    _submit \
        --dependency="afterok:${PREV_JOB}" \
        regen_bond_vs_fra.sh
)
echo "[9] Regen bond-vs-fra plot      job: ${PLOT_JOB}"

echo ""
echo "========================================================"
echo "  All jobs submitted.  Dependency chain:"
echo "    bond_array(${BOND_ARRAY_JOB})"
echo "      -> bond_collect(${BOND_COLLECT_JOB})"
echo "      -> nb_r1_array -> nb_r1_collect"
echo "      -> nb_r2_array -> nb_r2_collect"
echo "      -> nb_r3_array -> nb_r3_collect"
echo "      -> plot(${PLOT_JOB})"
echo ""
echo "  Monitor:  squeue -u \$USER"
echo "  Output figure: ${OUT_DIR}/${OUT_NAME%.png}_bond_vs_fra.png"
echo "========================================================"
