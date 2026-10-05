#!/bin/bash
# ============================================================
#  Randomised refine of the seeded model10 rates (default d_ext = 12) until
#  exhausted, then the figure results_model4_rate/model10_rate_panels.png.
#
#  Each round is one array (scripts/model10_rrefine.slurm.sh): every positive
#  point (TARGETS=all; TARGETS=nonmono: only points with a less-noisy
#  4-neighbour at a lower rate, or a 4-neighbour at rate 0) runs a basin-hopping
#  search with a fresh per-round random seed.  The refinement counts as
#  EXHAUSTED only after PATIENCE consecutive randomised rounds that improve no
#  point (default 1); MAX_ROUNDS caps the chain.
#
#  Rounds are sequential and at most one 200-task array may be queued at a
#  time, so every round uses `sbatch --wait`; run on the login node:
#
#      nohup bash scripts/submit_model10_rrefine.sh > logs/submit_m10_rrefine.log 2>&1 &
#
#  Knobs (env): D_EXT (12), TARGETS (all), MAX_ROUNDS (10), PATIENCE (1),
#  START_ROUND (default one past the highest round on disk), HOPS (6),
#  POLISH_ITERS (100), SCALES ("0.02 0.06 0.15"), MAX_SECONDS (1800 per
#  point), STRIDE, OUT_DIR, OUT_PNG.
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."               # repo root
export OUT_DIR="${OUT_DIR:-results_model10_rate}"
export STRIDE="${STRIDE:-1}"
export D_EXT="${D_EXT:-12}"
MAX_ROUNDS="${MAX_ROUNDS:-10}"
PATIENCE="${PATIENCE:-1}"
TAG="[m10 rrefine d${D_EXT}]"
mkdir -p logs "$OUT_DIR"

if [ "$STRIDE" = "1" ]; then SFX=""; else SFX="_s${STRIDE}"; fi
case "$D_EXT" in
    8)  PT_DIR="$OUT_DIR/model10_seeded${SFX}" ;;
    12) PT_DIR="$OUT_DIR/model10_seeded_d12${SFX}" ;;
    *)  echo "$TAG d_ext ${D_EXT} not supported (8 | 12)"; exit 1 ;;
esac

# wait for earlier model10 jobs that write into the same folders
for name in m10_seed m10_seed_collect m10_seed_qref m10_d12 m10_d12_qref m10_margin; do
    while :; do
        n=$(squeue -h -u "$USER" -n "$name" -o '%r' 2>/dev/null \
            | grep -vc 'DependencyNeverSatisfied' || true)
        [ "${n:-0}" -eq 0 ] && break
        echo "$TAG waiting for ${n} queued ${name} job(s)..."
        sleep 300
    done
done

if [ -z "${START_ROUND:-}" ]; then
    last=$(find "$PT_DIR" -maxdepth 1 -name '*_rrefine_r[0-9][0-9].npz' 2>/dev/null \
           | sed 's/.*_rrefine_r\([0-9][0-9]\)\.npz$/\1/' | sort -n | tail -1 || true)
    START_ROUND=$(( 10#${last:-0} + 1 ))
fi
END_ROUND=$(( START_ROUND + MAX_ROUNDS - 1 ))
echo "$TAG randomised rounds ${START_ROUND}..${END_ROUND} (stop after ${PATIENCE}" \
     "round(s) without improvement), targets=${TARGETS:-all}"

idle=0
status="not exhausted (MAX_ROUNDS reached)"
for round in $(seq "$START_ROUND" "$END_ROUND"); do
    jid=$(ROUND="$round" sbatch --parsable --wait scripts/model10_rrefine.slurm.sh) \
        || echo "$TAG warning: some round-${round} tasks failed; continuing"
    jid="${jid%%;*}"
    rr=$(printf '%02d' "$round")
    n_new=$(find "$PT_DIR" -maxdepth 1 -name "*_rrefine_r${rr}.npz" 2>/dev/null \
            | wc -l || true)
    logs="logs/m10rref_m10_rrefine_${jid}_*.out"
    n_zero=$(cat $logs 2>/dev/null | grep -c 'saved .*(rate 0' || true)
    n_tried=$(cat $logs 2>/dev/null | grep -c 'saved \|no gain' || true)
    echo "$TAG round ${round} (job ${jid}): ${n_tried} point(s) searched," \
         "${n_new} improved, ${n_zero} of them now at rate 0"
    if [ "$n_new" -eq 0 ]; then
        idle=$(( idle + 1 ))
        if [ "$idle" -ge "$PATIENCE" ]; then
            status="EXHAUSTED: ${idle} randomised round(s) in a row improved nothing"
            break
        fi
    else
        idle=0
    fi
done
echo "$TAG ${status}"

pid=$(sbatch --parsable scripts/model10_seeded_panels_collect.slurm.sh)
echo "$TAG figure job ${pid%%;*} submitted -> " \
     "${OUT_PNG:-results_model4_rate/model10_rate_panels.png}"
