#!/bin/bash
# ============================================================
#  Master submission: 2 rounds of neighbor refinement for all
#  three variants (d6, d4, free6), gamma' up to 4.
#
#  All three variants run in PARALLEL within each round.
#  Round 2 depends on round 1 collect finishing.
#
#  Usage:
#      bash submit_unified_nb_refine.sh
# ============================================================

set -euo pipefail

N_PTS=41          # gamma points (0..8, step 0.2)
N_IGP=21          # gamma' points (0..4, step 0.2)
GAMMA_STEP=0.2
J=1.0
N_RESTARTS=5
MAXFEV=1000
MAX_CONCURRENT=50

# task count = N_PTS * N_IGP - 1
ARRAY_END=$(( N_PTS * N_IGP - 1 ))   # 860

# Variant configs: (variant, out_dir)
VARIANTS=("d6:results" "d4:results_d4" "free6:results_free6")

mkdir -p logs

echo "========================================================"
echo "  Unified neighbor refinement — 2 rounds × 3 variants"
echo "  N_PTS=${N_PTS}  N_IGP=${N_IGP}  tasks/variant=${ARRAY_END}+1"
echo "  Variants: d6 (results), d4 (results_d4), free6 (results_free6)"
echo "========================================================"

# ── Helper: submit one round for one variant ─────────────────
# Usage: submit_round VARIANT OUT_DIR ROUND [DEPEND_JOB_ID]
# Prints the COLLECT job id (for chaining round 2).
submit_round() {
    local VARIANT="$1"
    local OUT_DIR="$2"
    local ROUND="$3"
    local DEPEND_JOB="${4:-}"

    local DEPEND_FLAG=""
    if [[ -n "$DEPEND_JOB" ]]; then
        DEPEND_FLAG="--dependency=afterok:${DEPEND_JOB}"
    fi

    # Submit array job
    local ARRAY_ID
    ARRAY_ID=$(
        VARIANT="$VARIANT" ROUND="$ROUND" OUT_DIR="$OUT_DIR" \
        N_PTS="$N_PTS" N_IGP="$N_IGP" J="$J" GAMMA_STEP="$GAMMA_STEP" \
        N_RESTARTS="$N_RESTARTS" MAXFEV="$MAXFEV" \
        sbatch --parsable \
               --array="0-${ARRAY_END}%${MAX_CONCURRENT}" \
               ${DEPEND_FLAG} \
               unified_nb_refine_array.sh
    )
    echo "  [${VARIANT} r${ROUND}] array job ${ARRAY_ID}  (${ARRAY_END}+1 tasks)" >&2

    # Submit collect job (depends on array)
    local COLLECT_ID
    COLLECT_ID=$(
        VARIANT="$VARIANT" ROUND="$ROUND" OUT_DIR="$OUT_DIR" \
        N_PTS="$N_PTS" N_IGP="$N_IGP" \
        sbatch --parsable \
               --dependency="afterok:${ARRAY_ID}" \
               unified_nb_refine_collect.sh
    )
    echo "  [${VARIANT} r${ROUND}] collect job ${COLLECT_ID}" >&2

    # Return the collect job id
    echo "$COLLECT_ID"
}

# ── Round 1: all variants in parallel ────────────────────────
echo ""
echo "=== Round 1 ==="
declare -A R1_COLLECT
for spec in "${VARIANTS[@]}"; do
    IFS=':' read -r VAR DIR <<< "$spec"
    R1_COLLECT[$VAR]=$(submit_round "$VAR" "$DIR" 1)
done

# ── Round 2: each variant depends on its own round-1 collect ─
echo ""
echo "=== Round 2 ==="
declare -A R2_COLLECT
for spec in "${VARIANTS[@]}"; do
    IFS=':' read -r VAR DIR <<< "$spec"
    R2_COLLECT[$VAR]=$(submit_round "$VAR" "$DIR" 2 "${R1_COLLECT[$VAR]}")
done

echo ""
echo "========================================================"
echo "  All jobs submitted."
echo ""
echo "  Round 1 collect jobs: ${R1_COLLECT[d6]} (d6)  ${R1_COLLECT[d4]} (d4)  ${R1_COLLECT[free6]} (free6)"
echo "  Round 2 collect jobs: ${R2_COLLECT[d6]} (d6)  ${R2_COLLECT[d4]} (d4)  ${R2_COLLECT[free6]} (free6)"
echo ""
echo "  Monitor: squeue -u \$USER"
echo "  Logs:    logs/unb_refine_*.out  logs/unb_collect_*.out"
echo "========================================================"
