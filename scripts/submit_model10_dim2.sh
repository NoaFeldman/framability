#!/bin/bash
# ============================================================
#  model10 (Shibata-Katsura) on the 2D square lattice, dim = 2: the whole
#  pipeline of the framability-rate figure
#  results_model4_rate/model10_dim2_rate_panels.png in one go
#
#    row 1 | stabilizer-3 | Pauli | opt Heisenberg d_ext=4 | 8 | 12
#    row 2 | product-state chi=10 | chi=40
#
#  (the framability panels of model10_rate_panels.png; the bond generator
#  takes the one-site terms h X, Delta1 X with weight 1/(2 dim) = 1/4)
#
#  Stages (arrays 0-199 of scripts/model10_dim2.slurm.sh unless noted):
#    1. seed        product chi=10/40, stabilizer-3, Pauli, seeded d_ext=4/8;
#                   every point is seeded with the dim=1 optima ($XFER_DIRS)
#                   at the point and its 8 neighbours.  Resubmitted once if a
#                   wall-time kill left points out.
#    2. xeval       d_ext=4/8 neighbour cross-evaluation
#                   (scripts/model10_dim2_collect.slurm.sh MODE=xeval, 1 job)
#    3. qrefine     up to N_QREF d_ext=4/8 quick-refine rounds
#    4. d12 seed    seeded with this run's d_ext=8 and the dim=1 d_ext=8/12
#    5. d12 refine  up to N_D12_REF rounds
#    6. margin      up to N_MARGIN rounds at d_ext=8, then at d_ext=12
#    7. rrefine     up to N_RREF randomised refine rounds at d_ext=8, then at
#                   d_ext=12 (TARGETS=nonmono: points above a less-noisy
#                   neighbour or next to rate 0, which quick refine / margin
#                   cannot reach away from the rate-0 region)
#    8. plot        scripts/model10_dim2_collect.slurm.sh MODE=plot (1 job)
#  EARLY_STOP=1 (default) ends a round loop after PATIENCE (2) rounds in a row
#  that made no new rate-0 point and no drop above MIN_GAIN (1e-3, invisible
#  on the colour scale); the deterministic loops (d12 refine, margin) also
#  end after a single round that writes nothing.  MIN_GAIN=0 runs
#  until no round lowers anything.  With N_* = 99 every loop runs until it
#  stops this way (99 = the 2-digit round limit).
#  scripts/model10_refine_status.py tells which loops are still moving.
#  Every stage skips finished work and round numbers continue from the disk,
#  so re-running the script picks up where it stopped.
#
#  At most one 200-task array may be queued at a time, so every stage is
#  submitted with `sbatch --wait`; run the script on the login node:
#
#      nohup bash scripts/submit_model10_dim2.sh > logs/submit_m10_dim2.log 2>&1 &
#
#  Needs the dim=1 seeded run on disk (results_model10_rate/model10_seeded,
#  model10_seeded_d12): read-only, the script aborts if it finds none.
#
#  Knobs (env): DIM (2), OUT_DIR (results_model10_rate_dim$DIM), XFER_DIRS
#  (results_model10_rate), XFER_RADIUS (1), STRIDE (1; 5 = 11x11 preview),
#  N_QREF (12), N_D12_REF (4), N_MARGIN (4), N_RREF (6), EARLY_STOP (1),
#  MIN_GAIN (1e-3), PATIENCE (2), PY (.venv/bin/python),
#  SKIP_SEED=1, SKIP_XEVAL=1, SKIP_QREF=1, SKIP_D12=1, SKIP_MARGIN=1,
#  SKIP_RREF=1, TARGETS (nonmono | all), OUT_PNG,
#  SEED_TIME (16:00:00), D12_TIME (24:00:00), and every variable of the two
#  slurm scripts (OPTIMIZER, DE_MAXITER, BUNDLE_ITERS, RADIUS, N_PROC, ...).
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."               # repo root
export DIM="${DIM:-2}"
export OUT_DIR="${OUT_DIR:-results_model10_rate_dim${DIM}}"
export XFER_DIRS="${XFER_DIRS:-results_model10_rate}"
export STRIDE="${STRIDE:-1}"
export PROD_STRIDE="${PROD_STRIDE:-$STRIDE}"
export OUT_PNG="${OUT_PNG:-results_model4_rate/model10_dim${DIM}_rate_panels.png}"
N_QREF="${N_QREF:-12}"
N_D12_REF="${N_D12_REF:-4}"
N_MARGIN="${N_MARGIN:-4}"
N_RREF="${N_RREF:-6}"
EARLY_STOP="${EARLY_STOP:-1}"
MIN_GAIN="${MIN_GAIN:-1e-3}"          # a round "improves" only above this drop ...
PATIENCE="${PATIENCE:-2}"             # ... and this many quiet rounds in a row stop a loop
PY="${PY:-.venv/bin/python}"          # for the stop rule (login node)
JP="m10d${DIM}"                       # job names, distinct from the dim=1 jobs
TAG="[m10 dim=${DIM}]"
mkdir -p logs "$OUT_DIR"

side() { echo $(( (51 + $1 - 1) / $1 )); }        # model10 grid: 0..2.5 step 0.05
N_TOTAL=$(( $(side "$STRIDE") ** 2 ))
N_PROD=$(( $(side "$PROD_STRIDE") ** 2 ))
if [ "$STRIDE" = "1" ]; then SFX=""; else SFX="_s${STRIDE}"; fi
D8_DIR="$OUT_DIR/model10_seeded${SFX}"
D12_DIR="$OUT_DIR/model10_seeded_d12${SFX}"
PROD_DIR="$OUT_DIR/model10_product"
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
        scripts/model10_dim2.slurm.sh \
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

# ---- 0. dim=1 transfer source ----------------------------------------------
n_src=0
for d in $XFER_DIRS; do
    a=$(count "$d/model10_seeded" "$BASE_PAT")
    b=$(count "$d/model10_seeded_d12" "$BASE_PAT")
    echo "$TAG transfer source ${d}: ${a} d_ext=4/8 points, ${b} d_ext=12 points"
    n_src=$(( n_src + a ))
done
if [ "$n_src" -eq 0 ]; then
    echo "$TAG ERROR: no dim=1 seeded frames under XFER_DIRS='${XFER_DIRS}'"
    exit 1
fi

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

# ---- 2. neighbour cross-evaluation (once, before any refine round) ---------
if [ "${SKIP_XEVAL:-0}" != "1" ] && [ "$(last_round "$D8_DIR" _qrefine_r)" -eq 0 ]; then
    echo "$TAG stage 2: d_ext=4/8 neighbour cross-evaluation"
    MODE=xeval sbatch --wait --job-name="${JP}_xeval" \
        scripts/model10_dim2_collect.slurm.sh \
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
        "${JP}_rref" 12:00:00 rand STAGE=rrefine D_EXT=8
    round_loop "d_ext=12 rrefine" "$D12_DIR" _rrefine_r "$N_RREF" \
        "${JP}_rref" 12:00:00 rand STAGE=rrefine D_EXT=12
fi

# ---- 8. figure ---------------------------------------------------------------
pid=$(MODE=plot sbatch --parsable --cpus-per-task=1 --time=01:00:00 \
          --job-name="${JP}_plot" scripts/model10_dim2_collect.slurm.sh)
echo "$TAG stage 8: figure job ${pid%%;*} submitted -> ${OUT_PNG}"
