#!/bin/bash
# ============================================================
#  Seeded framability-rate pipeline for one model of
#  trotter_lindbladian_scan.MODELS, on that model's own grid (models 3 / 4:
#  gamma, gamma' in [0, 10], step 0.2, 51 x 51 -- the range of the existing
#  figures), from scratch to the figure, in one go:
#
#    row 1 | stabilizer-3 | Pauli | opt Heisenberg d_ext=4 | 8 | 12
#    row 2 | product-state chi=10 | chi=40 | 8q osc rate | 8q gap
#
#  Stages (arrays 0-199 of scripts/rate_seeded.slurm.sh unless noted):
#    1. seed        product chi=10/40, stabilizer-3, Pauli, seeded d_ext=4/8
#                   (exact rescaled-Pauli optimum, families, stored frames,
#                   global search); resubmitted once if a wall-time kill left
#                   points out
#    1b. 8q         osc rate + gap of the full 8-qubit Liouvillian (periodic
#                   ring for the 1D models) on the grid strided by MB_STRIDE
#                   (scripts/model4_manybody.slurm.sh, <= 200 tasks);
#                   resubmitted once like stage 1.  Off (SKIP_MB=1) by default
#                   for model3 / model4, whose 8q panels live in
#                   results_model4_rate; skipped for models the many-body
#                   worker does not support
#    2. xeval       d_ext=4/8 neighbour cross-evaluation
#                   (scripts/rate_seeded_collect.slurm.sh MODE=xeval, 1 job)
#    3. qrefine     d_ext=4/8 quick-refine rounds
#    4. d12 seed    seeded with this run's d_ext=8 frames
#    5. d12 refine  rounds
#    6. margin      rounds at d_ext=8, then at d_ext=12
#    7. rrefine     randomised refine rounds at d_ext=8, then at d_ext=12
#    8. plot        scripts/rate_seeded_collect.slurm.sh MODE=plot (1 job),
#                   compared with the earlier figure's data ($REF_NPZ)
#  Every round loop runs until it stops by itself (N_* = 99, the 2-digit
#  round limit): after PATIENCE (2) rounds in a row without a new rate-0 point
#  or a drop above MIN_GAIN (1e-3); the deterministic loops (d12 refine,
#  margin) also after a single round that writes nothing.  Every stage skips
#  finished work and round numbers continue from the disk, so re-running the
#  script picks up where it stopped.
#
#  At most one 200-task array may be queued at a time, so every stage is
#  submitted with `sbatch --wait`; run on the login node, one model at a time:
#
#      MODEL=model4 nohup bash scripts/submit_rate_seeded.sh > logs/submit_rate_seeded_model4.log 2>&1 &
#
#  Per-model defaults (override with the env variables):
#    model4  STORED_DIRS=results_model4_rate   (base scan + refine + gopt frames)
#            REF_NPZ=results_model4_rate/model4_rate_panels.npz
#    model3  DTBASE_DIRS=results_dtbase_line   (finite-dt optimised frames)
#            REF_NPZ=results_dtbase_line/model3_dtbase_extrap.npz
#
#  Knobs (env): MODEL (required), OUT_DIR (results_<model>_seeded), OUT_PNG
#  (results_model4_rate/<model>_seeded_rate_panels.png), STORED_DIRS,
#  DTBASE_DIRS, XFER_DIRS, DIM, STRIDE (1; 5 = 11x11 preview), N_QREF, N_D12_REF,
#  N_MARGIN, N_RREF (99 each), MIN_GAIN (1e-3), PATIENCE (2), EARLY_STOP (1),
#  SKIP_SEED=1, SKIP_MB=1, SKIP_XEVAL=1, SKIP_QREF=1, SKIP_D12=1, SKIP_MARGIN=1,
#  SKIP_RREF=1, MB_STRIDE (1), MB_TIME (24:00:00),
#  SEED_TIME (16:00:00), D12_TIME (24:00:00), HOPS (20), MAX_SECONDS (5400),
#  TARGETS (nonmono), PY (.venv/bin/python), and every variable of the two
#  slurm scripts.
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."               # repo root
export MODEL="${MODEL:?set MODEL (e.g. model3 | model4)}"
case "$MODEL" in
    model4) def_stored="results_model4_rate"; def_dtbase=""
            def_ref="results_model4_rate/model4_rate_panels.npz"; def_skip_mb=1 ;;
    model3) def_stored=""; def_dtbase="results_dtbase_line"
            def_ref="results_dtbase_line/model3_dtbase_extrap.npz"; def_skip_mb=1 ;;
    *)      def_stored="results_${MODEL}_rate"; def_dtbase=""; def_ref=""
            def_skip_mb=0 ;;
esac
SKIP_MB="${SKIP_MB:-$def_skip_mb}"
MB_STRIDE="${MB_STRIDE:-1}"
export OUT_DIR="${OUT_DIR:-results_${MODEL}_seeded}"
export OUT_PNG="${OUT_PNG:-results_model4_rate/${MODEL}_seeded_rate_panels.png}"
export STORED_DIRS="${STORED_DIRS-$def_stored}"
export DTBASE_DIRS="${DTBASE_DIRS-$def_dtbase}"
export XFER_DIRS="${XFER_DIRS:-}"
export REF_NPZ="${REF_NPZ-$def_ref}"
export STRIDE="${STRIDE:-1}"
export PROD_STRIDE="${PROD_STRIDE:-$STRIDE}"
export HOPS="${HOPS:-20}"
export MAX_SECONDS="${MAX_SECONDS:-5400}"
N_QREF="${N_QREF:-99}"
N_D12_REF="${N_D12_REF:-99}"
N_MARGIN="${N_MARGIN:-99}"
N_RREF="${N_RREF:-99}"
EARLY_STOP="${EARLY_STOP:-1}"
MIN_GAIN="${MIN_GAIN:-1e-3}"          # a round "improves" only above this drop ...
PATIENCE="${PATIENCE:-2}"             # ... and this many quiet rounds in a row stop a loop
PY="${PY:-.venv/bin/python}"          # for the stop rule (login node)
JP="rs_${MODEL}"                      # job-name prefix
TAG="[${MODEL} seeded]"
mkdir -p logs "$OUT_DIR"

grid_size() {     # grid_size <stride>: "nx ny" of the model's grid
    "$PY" -c "
import sys; sys.path.insert(0, '.')
from trotter_lindbladian_scan import MODELS
s = MODELS['$MODEL']; k = $1
print(len(s.p1_vals[::k]), len(s.p2_vals[::k]))"
}
read -r NX NY < <(grid_size "$STRIDE") \
    || { echo "$TAG ERROR: cannot read the ${MODEL} grid with ${PY}"; exit 1; }
read -r PNX PNY < <(grid_size "$PROD_STRIDE") \
    || { echo "$TAG ERROR: cannot read the ${MODEL} grid with ${PY}"; exit 1; }
read -r MNX MNY < <(grid_size "$MB_STRIDE") \
    || { echo "$TAG ERROR: cannot read the ${MODEL} grid with ${PY}"; exit 1; }
N_TOTAL=$(( NX * NY ))
N_PROD=$(( PNX * PNY ))
N_MB=$(( MNX * MNY ))
if [ "$STRIDE" = "1" ]; then SFX=""; else SFX="_s${STRIDE}"; fi
D8_DIR="$OUT_DIR/${MODEL}_seeded${SFX}"
D12_DIR="$OUT_DIR/${MODEL}_seeded_d12${SFX}"
PROD_DIR="$OUT_DIR/${MODEL}_product"
MB_DIR="$OUT_DIR/${MODEL}_8q"                 # one stride per folder (stored per file)
BASE_PAT='pt_[0-9][0-9][0-9]_[0-9][0-9][0-9].npz'

count() { find "$1" -maxdepth 1 -name "$2" 2>/dev/null | wc -l || true; }

last_round() {    # highest round NN of files *<tag>NN.npz in dir $1 (0 if none)
    local r
    r=$(find "$1" -maxdepth 1 -name "*$2[0-9][0-9].npz" 2>/dev/null \
        | sed "s/.*$2\([0-9][0-9]\)\.npz\$/\1/" | sort -n | tail -1 || true)
    echo $(( 10#${r:-0} ))
}

run_array() {     # run_array <job name> <time> VAR=value ...   (blocks)
    local name="$1" time="$2"; shift 2
    env "$@" sbatch --wait --job-name="$name" --time="$time" \
        scripts/rate_seeded.slurm.sh \
        || echo "$TAG warning: some ${name} tasks failed; continuing"
}

round_loop() {    # round_loop <label> <dir> <file tag> <max rounds> <job name> <time> <det|rand> VAR=value ...
    # det : a round that writes nothing ends the loop (the next one would repeat it)
    # rand: fresh random seeds every round, so an empty round only counts as quiet
    local label="$1" dir="$2" ftag="$3" n="$4" name="$5" time="$6" kind="$7"; shift 7
    [ "$n" -gt 0 ] || return 0
    local start end round mx nz nf quiet=0
    start=$(( $(last_round "$dir" "$ftag") + 1 ))
    end=$(( start + n - 1 ))
    [ "$end" -le 99 ] || end=99                       # 2-digit round tags
    if [ "$start" -gt 99 ]; then
        echo "$TAG ${label}: round limit 99 reached"; return 0
    fi
    echo "$TAG ${label}: rounds ${start}..${end}"
    for round in $(seq "$start" "$end"); do
        run_array "$name" "$time" ROUND="$round" "$@"
        # (if the check itself fails: count the files and keep going)
        read -r mx nz nf < <("$PY" scripts/model10_refine_status.py \
                                 --round_gain "$dir" "$ftag" "$round" \
            || echo "nan 0 $(count "$dir" "*${ftag}$(printf '%02d' "$round").npz")") || true
        mx=${mx:-nan}; nz=${nz:-0}; nf=${nf:-0}
        echo "$TAG ${label} round ${round}: ${nf} file(s), largest drop ${mx}," \
             "${nz} new rate-0 point(s)"
        [ "$EARLY_STOP" = "1" ] || continue
        if [ "$nf" -eq 0 ] && [ "$kind" = "det" ]; then
            echo "$TAG ${label}: round ${round} wrote nothing -- stopping"
            break
        fi
        if [ "$nz" -gt 0 ] || awk -v a="$mx" -v b="$MIN_GAIN" 'BEGIN { exit !(a > b) }'; then
            quiet=0
        else
            quiet=$(( quiet + 1 ))
            if [ "$quiet" -ge "$PATIENCE" ]; then
                echo "$TAG ${label}: ${quiet} round(s) in a row without a new rate-0" \
                     "point or a drop above ${MIN_GAIN} -- stopping"
                break
            fi
        fi
    done
}

echo "$TAG grid ${NX}x${NY} (stride ${STRIDE}); out ${OUT_DIR}; stored frames:" \
     "rate scans '${STORED_DIRS}', finite-dt '${DTBASE_DIRS}', transfer '${XFER_DIRS}'"
for d in $STORED_DIRS; do
    echo "$TAG   ${d}/${MODEL}: $(count "$d/$MODEL" "$BASE_PAT") stored rate points"
done
for d in $DTBASE_DIRS; do
    echo "$TAG   ${d}: $(find "$d" -maxdepth 2 -path "*/${MODEL}_p1_*/base_000.npz" 2>/dev/null | wc -l) finite-dt points"
done

# ---- 1. seed: product, stabilizer-3, Pauli, d_ext = 4 / 8 ------------------
if [ "${SKIP_SEED:-0}" != "1" ]; then
    for attempt in 1 2; do
        n8=$(count "$D8_DIR" "$BASE_PAT"); npr=$(count "$PROD_DIR" "$BASE_PAT")
        [ "$n8" -ge "$N_TOTAL" ] && [ "$npr" -ge "$N_PROD" ] && break
        echo "$TAG stage 1 (pass ${attempt}): seed (${n8}/${N_TOTAL} seeded," \
             "${npr}/${N_PROD} product points on disk)"
        run_array "${JP}_seed" "${SEED_TIME:-16:00:00}" STAGE=seed
    done
fi
echo "$TAG stage 1: $(count "$D8_DIR" "$BASE_PAT")/${N_TOTAL} seeded," \
     "$(count "$PROD_DIR" "$BASE_PAT")/${N_PROD} product points"

# ---- 1b. 8-qubit many-body panels: osc rate, Lindbladian gap ---------------
if [ "$SKIP_MB" != "1" ]; then
    mb_ok=$("$PY" -c "
import sys; sys.path[:0] = ['scripts', '.']
from model4_manybody_worker import SUPPORTED_MODELS
print(int('$MODEL' in SUPPORTED_MODELS))" 2>/dev/null || echo 0)
    if [ "$mb_ok" != "1" ]; then
        echo "$TAG stage 1b: ${MODEL} is not a many-body worker model; 8q panels skipped"
    else
        MB_ARRAY="0-$(( (N_MB < 200 ? N_MB : 200) - 1 ))"
        for attempt in 1 2; do
            nmb=$(count "$MB_DIR" "$BASE_PAT")
            [ "$nmb" -ge "$N_MB" ] && break
            echo "$TAG stage 1b (pass ${attempt}): 8q panels, ${MNX}x${MNY} grid" \
                 "(stride ${MB_STRIDE}; ${nmb}/${N_MB} points on disk)"
            env MODEL="$MODEL" OUT_DIR="$OUT_DIR" STRIDE="$MB_STRIDE" \
                N_CHUNKS="$(( N_MB < 200 ? N_MB : 200 ))" \
                sbatch --wait --array="$MB_ARRAY" --job-name="${JP}_8q" \
                       --time="${MB_TIME:-24:00:00}" scripts/model4_manybody.slurm.sh \
                || echo "$TAG warning: some ${JP}_8q tasks failed; continuing"
        done
        echo "$TAG stage 1b: $(count "$MB_DIR" "$BASE_PAT")/${N_MB} 8q points"
    fi
fi

# ---- 2. neighbour cross-evaluation (once, before any refine round) ---------
if [ "${SKIP_XEVAL:-0}" != "1" ] && [ "$(last_round "$D8_DIR" _qrefine_r)" -eq 0 ]; then
    echo "$TAG stage 2: d_ext=4/8 neighbour cross-evaluation"
    MODE=xeval sbatch --wait --job-name="${JP}_xeval" \
        scripts/rate_seeded_collect.slurm.sh \
        || echo "$TAG warning: xeval job failed; continuing"
    echo "$TAG stage 2: $(count "$D8_DIR" '*_xeval.npz') point(s) improved"
fi

# ---- 3. d_ext = 4 / 8 quick-refine rounds ----------------------------------
if [ "${SKIP_QREF:-0}" != "1" ]; then
    round_loop "d_ext=4/8 qrefine" "$D8_DIR" _qrefine_r "$N_QREF" \
        "${JP}_qref" 08:00:00 rand STAGE=qrefine
fi

# ---- 4-5. d_ext = 12: seed stage, refine rounds ----------------------------
if [ "${SKIP_D12:-0}" != "1" ]; then
    for attempt in 1 2; do
        n12=$(count "$D12_DIR" "$BASE_PAT")
        [ "$n12" -ge "$N_TOTAL" ] && break
        echo "$TAG stage 4 (pass ${attempt}): d_ext=12 seed (${n12}/${N_TOTAL} on disk)"
        run_array "${JP}_d12" "${D12_TIME:-24:00:00}" STAGE=d12_seed
    done
    echo "$TAG stage 4: $(count "$D12_DIR" "$BASE_PAT")/${N_TOTAL} d_ext=12 points"
    round_loop "d_ext=12 refine" "$D12_DIR" _qrefine_r "$N_D12_REF" \
        "${JP}_d12_qref" 08:00:00 det STAGE=d12_refine
fi

# ---- 6. margin rounds, d_ext = 8 then 12 -----------------------------------
if [ "${SKIP_MARGIN:-0}" != "1" ]; then
    round_loop "d_ext=8 margin" "$D8_DIR" _margin_r "$N_MARGIN" \
        "${JP}_margin" 12:00:00 det STAGE=margin D_EXT=8
    round_loop "d_ext=12 margin" "$D12_DIR" _margin_r "$N_MARGIN" \
        "${JP}_margin" 12:00:00 det STAGE=margin D_EXT=12
fi

# ---- 7. randomised refine, d_ext = 8 then 12 -------------------------------
if [ "${SKIP_RREF:-0}" != "1" ]; then
    round_loop "d_ext=8 rrefine" "$D8_DIR" _rrefine_r "$N_RREF" \
        "${JP}_rref" 24:00:00 rand STAGE=rrefine D_EXT=8
    round_loop "d_ext=12 rrefine" "$D12_DIR" _rrefine_r "$N_RREF" \
        "${JP}_rref" 24:00:00 rand STAGE=rrefine D_EXT=12
fi

# ---- 8. figure ---------------------------------------------------------------
pid=$(MODE=plot sbatch --parsable --cpus-per-task=1 --time=01:00:00 \
          --job-name="${JP}_plot" scripts/rate_seeded_collect.slurm.sh)
echo "$TAG stage 8: figure job ${pid%%;*} submitted -> ${OUT_PNG}"
