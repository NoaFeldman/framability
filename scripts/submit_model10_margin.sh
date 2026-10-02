#!/bin/bash
# ============================================================
#  Margin rounds for the seeded model10 rates, d_ext = 8 then 12, then the
#  figure results_model4_rate/model10_rate_panels.png.
#
#  Each round (scripts/model10_margin.slurm.sh, array 0-199):
#    * rate-0 points get their frame optimised for MARGIN (all non-identity
#      columns strictly decaying), so it keeps rate 0 at nearby points;
#    * positive points try every margin frame within RADIUS grid steps and,
#      near the rate-0 region, a margin/rate polish from the best of them.
#  New rate-0 points are margin-optimised in the next round, so the region
#  grows by up to RADIUS steps per round.  d_ext = 12 also tries the padded
#  d_ext = 8 margin frames, which is why it runs second.
#
#  Rounds are sequential and at most one 200-task array may be queued at a
#  time, so every stage uses `sbatch --wait`; run on the login node:
#
#      nohup bash scripts/submit_model10_margin.sh > logs/submit_m10_margin.log 2>&1 &
#
#  Knobs (env): D_EXTS ("8 12"), N_ROUNDS (6 per d_ext), START_ROUND (default
#  one past the highest margin round on disk, per d_ext), EARLY_STOP=1 (stop a
#  d_ext after a round that writes nothing), RADIUS, MARGIN_ITERS, PUSH_ITERS,
#  NO_PUSH=1, STRIDE, OUT_DIR, OUT_PNG.
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."               # repo root
export OUT_DIR="${OUT_DIR:-results_model10_rate}"
export STRIDE="${STRIDE:-1}"
D_EXTS="${D_EXTS:-8 12}"
N_ROUNDS="${N_ROUNDS:-6}"
TAG='[m10 margin]'
mkdir -p logs "$OUT_DIR"

if [ "$STRIDE" = "1" ]; then SFX=""; else SFX="_s${STRIDE}"; fi

# wait for earlier model10 seeded / d12 jobs that write into the same folders
for name in m10_seed m10_seed_collect m10_seed_qref m10_d12 m10_d12_qref; do
    while :; do
        n=$(squeue -h -u "$USER" -n "$name" -o '%r' 2>/dev/null \
            | grep -vc 'DependencyNeverSatisfied' || true)
        [ "${n:-0}" -eq 0 ] && break
        echo "$TAG waiting for ${n} queued ${name} job(s)..."
        sleep 300
    done
done

for D_EXT in $D_EXTS; do
    case "$D_EXT" in
        8)  PT_DIR="$OUT_DIR/model10_seeded${SFX}" ;;
        12) PT_DIR="$OUT_DIR/model10_seeded_d12${SFX}" ;;
        *)  echo "$TAG d_ext ${D_EXT} not supported (8 | 12)"; exit 1 ;;
    esac
    if [ -n "${START_ROUND:-}" ]; then
        start="$START_ROUND"
    else
        last=$(find "$PT_DIR" -maxdepth 1 -name '*_margin_r[0-9][0-9].npz' 2>/dev/null \
               | sed 's/.*_margin_r\([0-9][0-9]\)\.npz$/\1/' | sort -n | tail -1 || true)
        start=$(( 10#${last:-0} + 1 ))
    fi
    end=$(( start + N_ROUNDS - 1 ))
    echo "$TAG d_ext=${D_EXT}: margin rounds ${start}..${end}"
    for round in $(seq "$start" "$end"); do
        jid=$(D_EXT="$D_EXT" ROUND="$round" \
              sbatch --parsable --wait --job-name=m10_margin \
                     scripts/model10_margin.slurm.sh) \
            || echo "$TAG warning: some d${D_EXT} round-${round} tasks failed; continuing"
        jid="${jid%%;*}"
        rr=$(printf '%02d' "$round")
        n_new=$(find "$PT_DIR" -maxdepth 1 -name "*_margin_r${rr}.npz" 2>/dev/null \
                | wc -l || true)
        logs="logs/m10margin_m10_margin_${jid}_*.out"
        n_mg=$(cat $logs 2>/dev/null | grep -c 'saved .*\[margin\]' || true)
        n_tr=$(cat $logs 2>/dev/null | grep -c 'saved .*\[transfer\]' || true)
        n_pu=$(cat $logs 2>/dev/null | grep -c 'saved .*\[push\]' || true)
        echo "$TAG d_ext=${D_EXT} round ${round} (job ${jid}): ${n_new} file(s):" \
             "${n_mg} margin-optimised, ${n_tr} new rate-0 by transfer, ${n_pu} pushed"
        if [ "${EARLY_STOP:-0}" = "1" ] && [ "$n_new" -eq 0 ]; then
            echo "$TAG d_ext=${D_EXT}: round ${round} wrote nothing -- stopping (EARLY_STOP=1)"
            break
        fi
    done
done

pid=$(sbatch --parsable scripts/model10_seeded_panels_collect.slurm.sh)
echo "$TAG figure job ${pid%%;*} submitted -> " \
     "${OUT_PNG:-results_model4_rate/model10_rate_panels.png}"
