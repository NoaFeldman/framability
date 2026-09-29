#!/bin/bash
# ============================================================
#  model10: product-state rates (chi = 10, 40) + optimised Heisenberg rate at
#  d_ext = 12, then the figure results_model4_rate/model10_rate_panels.png
#
#    row 1 | stabilizer-3 | Pauli | opt Heisenberg d_ext=4 | 8 | 12
#    row 2 | product-state chi=10 | chi=40 | 8q osc rate | 8q gap
#
#  One sequential chain:
#    0. wait for earlier seeded jobs still queued (m10_seed, m10_seed_collect,
#       m10_seed_qref): the d_ext = 12 seeds read the d_ext = 8 results
#    1. product    scripts/model4_product_rate.slurm.sh (MODEL=model10, array 0-199)
#                  only if files are missing; a few tens of seconds per point
#    2. d12 seed   scripts/model10_d12.slurm.sh STAGE=seed (array 0-199)
#                  only if files are missing
#    3. d12 refine N_ROUNDS quick neighbour-refine rounds (STAGE=refine, each
#                  an array 0-199; START_ROUND defaults to one past the highest
#                  round on disk, so a re-run continues)
#    4. figure     scripts/model10_seeded_panels_collect.slurm.sh
#
#  At most one 200-task array is queued at a time, so every stage is
#  submitted with `sbatch --wait`; run the script on the login node:
#
#      nohup bash scripts/submit_model10_d12_prod.sh > logs/submit_m10_d12_prod.log 2>&1 &
#
#  Knobs (env): N_ROUNDS (3), START_ROUND, SKIP_PROD=1, SKIP_D12=1,
#  SKIP_REFINE=1, STRIDE, OUT_DIR, OUT_PNG and every variable of the slurm
#  scripts (FAM_MAXFEV, BUNDLE_ITERS, NM_MAXFEV, REFINE_BUNDLE, ...).
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."               # repo root
export OUT_DIR="${OUT_DIR:-results_model10_rate}"
export STRIDE="${STRIDE:-1}"
export PROD_STRIDE="${PROD_STRIDE:-$STRIDE}"
N_ROUNDS="${N_ROUNDS:-3}"
TAG='[m10 d12+prod]'
mkdir -p logs "$OUT_DIR"

N_SIDE=$(( (51 + STRIDE - 1) / STRIDE ))          # model10 grid: 0..2.5 step 0.05
N_TOTAL=$(( N_SIDE * N_SIDE ))
if [ "$STRIDE" = "1" ]; then SFX=""; else SFX="_s${STRIDE}"; fi
PROD_DIR="$OUT_DIR/model10_product"
D12_DIR="$OUT_DIR/model10_seeded_d12${SFX}"

count() { find "$1" -maxdepth 1 -name "$2" 2>/dev/null | wc -l || true; }
BASE_PAT='pt_[0-9][0-9][0-9]_[0-9][0-9][0-9].npz'

# ---- 0. earlier seeded jobs ------------------------------------------------
for name in m10_seed m10_seed_collect m10_seed_qref; do
    while :; do
        n=$(squeue -h -u "$USER" -n "$name" -o '%r' 2>/dev/null \
            | grep -vc 'DependencyNeverSatisfied' || true)
        [ "${n:-0}" -eq 0 ] && break
        echo "$TAG waiting for ${n} queued ${name} job(s)..."
        sleep 300
    done
done

# ---- 1. product-state rates ------------------------------------------------
n_have=$(count "$PROD_DIR" "$BASE_PAT")
if [ "${SKIP_PROD:-0}" != "1" ] && [ "$n_have" -lt "$N_TOTAL" ]; then
    echo "$TAG stage 1: product-state rates (${n_have}/${N_TOTAL} on disk)"
    MODEL=model10 STRIDE="$PROD_STRIDE" \
        sbatch --wait --job-name=m10_prod scripts/model4_product_rate.slurm.sh \
        || echo "$TAG warning: some product tasks failed; continuing"
    echo "$TAG stage 1: $(count "$PROD_DIR" "$BASE_PAT")/${N_TOTAL} points"
else
    echo "$TAG stage 1 skipped (${n_have}/${N_TOTAL} on disk)"
fi

# ---- 2. d_ext = 12 seed stage ------------------------------------------------
n_have=$(count "$D12_DIR" "$BASE_PAT")
if [ "${SKIP_D12:-0}" != "1" ] && [ "$n_have" -lt "$N_TOTAL" ]; then
    echo "$TAG stage 2: d_ext=12 seed stage (${n_have}/${N_TOTAL} on disk)"
    STAGE=seed sbatch --wait scripts/model10_d12.slurm.sh \
        || echo "$TAG warning: some d12 seed tasks failed; continuing"
    echo "$TAG stage 2: $(count "$D12_DIR" "$BASE_PAT")/${N_TOTAL} points"
else
    echo "$TAG stage 2 skipped (${n_have}/${N_TOTAL} on disk)"
fi

# ---- 3. d_ext = 12 refine rounds ---------------------------------------------
if [ "${SKIP_REFINE:-0}" != "1" ]; then
    if [ -z "${START_ROUND:-}" ]; then
        last=$(find "$D12_DIR" -maxdepth 1 -name '*_qrefine_r[0-9][0-9].npz' 2>/dev/null \
               | sed 's/.*_qrefine_r\([0-9][0-9]\)\.npz$/\1/' | sort -n | tail -1 || true)
        START_ROUND=$(( 10#${last:-0} + 1 ))
    fi
    END_ROUND=$(( START_ROUND + N_ROUNDS - 1 ))
    echo "$TAG stage 3: d_ext=12 refine rounds ${START_ROUND}..${END_ROUND}"
    for round in $(seq "$START_ROUND" "$END_ROUND"); do
        STAGE=refine ROUND="$round" \
            sbatch --wait --time=08:00:00 --job-name=m10_d12_qref \
                   scripts/model10_d12.slurm.sh \
            || echo "$TAG warning: some round-${round} tasks failed; continuing"
        n_new=$(count "$D12_DIR" "*_qrefine_r$(printf '%02d' "$round").npz")
        echo "$TAG round ${round}/${END_ROUND}: ${n_new} point(s) improved"
    done
fi

# ---- 4. figure ------------------------------------------------------------------
pid=$(sbatch --parsable scripts/model10_seeded_panels_collect.slurm.sh)
echo "$TAG stage 4: figure job ${pid%%;*} submitted -> " \
     "${OUT_PNG:-results_model4_rate/model10_rate_panels.png}"
