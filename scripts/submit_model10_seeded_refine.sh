#!/bin/bash
# ============================================================
#  Seeded model10 rates -> 6 quick neighbour-refine rounds -> figure with
#  panels 3-4 ('Opt Heisenberg d_ext=4 / 6') replaced by the seeded
#  d_ext = 4 / 8 rates.  One sequential chain:
#
#    0. wait for any earlier seeded job still queued (m10_seed,
#       m10_seed_collect from scripts/submit_model10_seeded.sh)
#    1. seeded array   scripts/model10_seeded.slurm.sh         (array 0-199)
#       only if worker files are missing; finished points are skipped
#    2. cross-eval     scripts/model10_seeded_collect.slurm.sh  (8 CPUs)
#       neighbour cross-evaluation of every point (its extended figure goes
#       to $OUT_DIR/model10_seeded_extended.png, not over the main figure)
#    3. N_ROUNDS quick-refine rounds, each an array 0-199
#       scripts/model10_seeded_qrefine.slurm.sh (round r reads rounds < r)
#    4. panels         scripts/model10_seeded_panels_collect.slurm.sh
#       -> $OUT_PNG (default results_model4_rate/model10_rate_panels.png)
#
#  Rounds must be sequential and at most one 200-task array may be queued at
#  a time, so every stage is submitted with `sbatch --wait` (as in
#  scripts/submit_model10_rate.sh).  Run the script on the login node inside
#  tmux / nohup:
#
#      nohup bash scripts/submit_model10_seeded_refine.sh > logs/submit_m10_seeded_refine.log 2>&1 &
#
#  Knobs (env): N_ROUNDS (6), START_ROUND (default: one past the highest round
#  on disk, so a re-run continues at 7, 8, ...), OPTIMIZER (rate | global),
#  EARLY_STOP=1 (stop after a round that improves nothing), SKIP_SEEDED=1,
#  SKIP_XEVAL=1, STRIDE, OUT_DIR, OUT_PNG and every variable of the four
#  slurm scripts (sbatch exports the submitting environment).
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."               # repo root
export OUT_DIR="${OUT_DIR:-results_model10_rate}"
export STRIDE="${STRIDE:-1}"
N_ROUNDS="${N_ROUNDS:-6}"
TAG='[m10 seeded refine]'
mkdir -p logs "$OUT_DIR"

if [ "$STRIDE" = "1" ]; then PT_DIR="$OUT_DIR/model10_seeded"
else PT_DIR="$OUT_DIR/model10_seeded_s${STRIDE}"; fi
N_SIDE=$(( (51 + STRIDE - 1) / STRIDE ))          # model10 grid: 0..2.5 step 0.05
N_TOTAL=$(( N_SIDE * N_SIDE ))

count() { find "$PT_DIR" -maxdepth 1 -name "$1" 2>/dev/null | wc -l || true; }

# ---- 0. earlier submissions still in the queue ---------------------------
for name in m10_seed m10_seed_collect; do
    while :; do
        n=$(squeue -h -u "$USER" -n "$name" -o '%r' 2>/dev/null \
            | grep -vc 'DependencyNeverSatisfied' || true)
        [ "${n:-0}" -eq 0 ] && break
        echo "$TAG waiting for ${n} queued ${name} job(s)..."
        sleep 300
    done
done

# ---- 1. seeded worker ------------------------------------------------------
n_have=$(count 'pt_[0-9][0-9][0-9]_[0-9][0-9][0-9].npz')
if [ "${SKIP_SEEDED:-0}" != "1" ] && [ "$n_have" -lt "$N_TOTAL" ]; then
    echo "$TAG stage 1: seeded array (${n_have}/${N_TOTAL} points on disk)"
    sbatch --wait scripts/model10_seeded.slurm.sh \
        || echo "$TAG warning: some seeded tasks failed; continuing"
    echo "$TAG stage 1: $(count 'pt_[0-9][0-9][0-9]_[0-9][0-9][0-9].npz')/${N_TOTAL} points"
else
    echo "$TAG stage 1 skipped (${n_have}/${N_TOTAL} points on disk)"
fi

# ---- 2. whole-grid neighbour cross-evaluation --------------------------------
if [ "${SKIP_XEVAL:-0}" != "1" ]; then
    echo "$TAG stage 2: neighbour cross-evaluation"
    OUT_PNG="$OUT_DIR/model10_seeded_extended.png" \
        sbatch --wait scripts/model10_seeded_collect.slurm.sh \
        || echo "$TAG warning: cross-evaluation job failed; continuing"
fi

# ---- 3. quick-refine rounds ----------------------------------------------------
if [ -z "${START_ROUND:-}" ]; then
    last=$(find "$PT_DIR" -maxdepth 1 -name '*_qrefine_r[0-9][0-9].npz' 2>/dev/null \
           | sed 's/.*_qrefine_r\([0-9][0-9]\)\.npz$/\1/' | sort -n | tail -1 || true)
    START_ROUND=$(( 10#${last:-0} + 1 ))
fi
END_ROUND=$(( START_ROUND + N_ROUNDS - 1 ))
echo "$TAG stage 3: quick-refine rounds ${START_ROUND}..${END_ROUND}"
for round in $(seq "$START_ROUND" "$END_ROUND"); do
    ROUND="$round" sbatch --wait scripts/model10_seeded_qrefine.slurm.sh \
        || echo "$TAG warning: some round-${round} tasks failed; continuing"
    n_new=$(count "*_qrefine_r$(printf '%02d' "$round").npz")
    echo "$TAG round ${round}/${END_ROUND}: ${n_new} point(s) improved"
    if [ "${EARLY_STOP:-0}" = "1" ] && [ "$n_new" -eq 0 ]; then
        echo "$TAG round ${round} improved nothing -- stopping early (EARLY_STOP=1)"
        break
    fi
done

# ---- 4. figure ------------------------------------------------------------------
pid=$(sbatch --parsable scripts/model10_seeded_panels_collect.slurm.sh)
echo "$TAG stage 4: panels job ${pid%%;*} submitted -> " \
     "${OUT_PNG:-results_model4_rate/model10_rate_panels.png}"
