#!/bin/bash
# ============================================================
#  Complex-frame (non-Hermitian elements, complex coefficients) framability-
#  rate panels for models 4, 10, 14, 15 and the two uniform Dicke gamma_down
#  families (kappa = 1, 0.25; N = 100), at HALF the resolution of the
#  existing panels (STRIDE=2: every second grid point = twice the step), from
#  scratch to the figures, in one go.  One flat point list over all models,
#  one 200-task array per stage (scripts/rate_complex.slurm.sh):
#
#    1. seed     Pauli, stabilizer-3, complex d_ext 4 (closed form + DE) and 8
#                (multistart bundle polish), seeded with the Hermitian optima
#                of the earlier runs and clock-ladder frames; resubmitted once
#                if a wall-time kill left points out
#    2. d12      complex d_ext 12, seeded with this run's d_ext 8; resubmitted
#                once
#    3. refine   rounds per d_ext in REFINE_DEXTS (default "8 12"): neighbour
#                transfer + complex basin hopping; each loop runs until PATIENCE
#                (2) rounds in a row bring no new rate-0 point and no drop above
#                MIN_GAIN (1e-3), or N_REF rounds
#    4. figures  scripts/rate_complex_collect.slurm.sh (1 job): panel npz +
#                comparison figure per model in $OUT_ROOT, paper-style figure
#                per model in $PAPER_DIR
#  Every stage skips finished work and round numbers continue from the disk,
#  so re-running the script picks up where it stopped.
#
#  Grids at STRIDE=2 (points): model4 26x26, model10 26x26, model14/15 8x7
#  (gamma_1 0..2.8 step 0.4, h 0..1.2 step 0.2), Dicke 11x8 (lam/lam_c 0..2
#  step 0.2, gamma_down 0..0.28 step 0.04): 1640 points, ~8 per task.
#
#  Run on the login node (stages use sbatch --wait):
#      nohup bash scripts/submit_rate_complex.sh > logs/submit_rate_complex.log 2>&1 &
#
#  Knobs (env): MODELS, STRIDE (2), OUT_ROOT (results_rate_complex), OLD_ROOTS
#  ("model:root ..." overrides of the earlier Hermitian runs), N_TASKS (200),
#  REFINE_DEXTS ("8 12"), N_REF (99), MIN_GAIN (1e-3), PATIENCE (2),
#  SEED_TIME (24:00:00), D12_TIME (24:00:00), REF_TIME (12:00:00),
#  SKIP_SEED=1, SKIP_D12=1, SKIP_REF=1, SKIP_PLOT=1, PAPER_DIR, PY
#  (.venv/bin/python), and every variable of scripts/rate_complex.slurm.sh.
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."               # repo root
export MODELS="${MODELS:-model4 model10 model14 model15 dicke_dec_N100 dicke_dec_k025_N100}"
export STRIDE="${STRIDE:-2}"
export OUT_ROOT="${OUT_ROOT:-results_rate_complex}"
export OLD_ROOTS="${OLD_ROOTS:-}"
N_TASKS="${N_TASKS:-200}"
[ "$N_TASKS" -ge 1 ] && [ "$N_TASKS" -le 200 ] \
    || { echo "[rate complex] ERROR: N_TASKS=${N_TASKS} not in 1..200"; exit 1; }
export N_CHUNKS="$N_TASKS"
REFINE_DEXTS="${REFINE_DEXTS:-8 12}"
N_REF="${N_REF:-99}"
MIN_GAIN="${MIN_GAIN:-1e-3}"
PATIENCE="${PATIENCE:-2}"
PY="${PY:-.venv/bin/python}"
JP="rc"
TAG="[rate complex]"
mkdir -p logs "$OUT_ROOT"

# a second driver on the same jobs would write the same files
if squeue -h -u "$USER" -o '%j' 2>/dev/null | grep -q "^${JP}_"; then
    echo "$TAG ERROR: ${JP}_* jobs are already queued (another driver running?)"
    exit 1
fi

collect() { "$PY" scripts/rate_complex_collect.py --models $MODELS --stride "$STRIDE" \
                --out_root "$OUT_ROOT" "$@"; }

count() {         # count seed|d12 -> "done total" ("0 1" if the query fails)
    local out; out=$(collect --count "$1" 2>/dev/null || true)
    [ -n "$out" ] && echo "$out" || echo "0 1"
}

run_array() {     # run_array <job name> <time> VAR=value ...   (blocks)
    local name="$1" time="$2"; shift 2
    env "$@" sbatch --wait --job-name="$name" --time="$time" \
        --array="0-$(( N_TASKS - 1 ))" scripts/rate_complex.slurm.sh \
        || echo "$TAG warning: some ${name} tasks failed; continuing"
}

echo "$TAG models: ${MODELS}; stride ${STRIDE}; out ${OUT_ROOT}; ${N_TASKS} tasks"
for m in $MODELS; do
    if [ "$m" = "model10" ]; then root="results_model10_rate"; else root="results_${m}_seeded"; fi
    for o in $OLD_ROOTS; do [ "${o%%:*}" = "$m" ] && root="${o#*:}"; done
    n8=$(find "$root/${m}_seeded" -maxdepth 1 -name 'pt_[0-9][0-9][0-9]_[0-9][0-9][0-9].npz' 2>/dev/null | wc -l || true)
    n12=$(find "$root/${m}_seeded_d12" -maxdepth 1 -name 'pt_[0-9][0-9][0-9]_[0-9][0-9][0-9].npz' 2>/dev/null | wc -l || true)
    echo "$TAG   ${m}: Hermitian seeds from ${root} (${n8} d4/d8 points, ${n12} d12 points)"
done

# ---- 1. seed: fixed frames, complex d_ext 4 / 8 ------------------------------
if [ "${SKIP_SEED:-0}" != "1" ]; then
    for attempt in 1 2; do
        read -r done total < <(count seed)
        [ "$done" -ge "$total" ] && break
        echo "$TAG stage 1 (pass ${attempt}): seed (${done}/${total} points on disk)"
        run_array "${JP}_seed" "${SEED_TIME:-24:00:00}" STAGE=seed
    done
fi
read -r done total < <(count seed)
echo "$TAG stage 1: ${done}/${total} points seeded"

# ---- 2. complex d_ext 12 -----------------------------------------------------
if [ "${SKIP_D12:-0}" != "1" ]; then
    for attempt in 1 2; do
        read -r done total < <(count d12)
        [ "$done" -ge "$total" ] && break
        echo "$TAG stage 2 (pass ${attempt}): d_ext=12 (${done}/${total} points on disk)"
        run_array "${JP}_d12" "${D12_TIME:-24:00:00}" STAGE=d12
    done
fi
read -r done total < <(count d12)
echo "$TAG stage 2: ${done}/${total} d_ext=12 points"

# ---- 3. refine rounds per d_ext ----------------------------------------------
if [ "${SKIP_REF:-0}" != "1" ]; then
    for m in $REFINE_DEXTS; do
        start=$(( $(collect --last_round "$m") + 1 ))
        end=$(( start + N_REF - 1 )); [ "$end" -le 99 ] || end=99
        quiet=0
        for round in $(seq "$start" "$end"); do
            run_array "${JP}_ref${m}" "${REF_TIME:-12:00:00}" STAGE=refine D_EXT="$m" ROUND="$round"
            read -r mx nz nf < <(collect --round_gain "$m" "$round" || echo "nan 0 0")
            echo "$TAG d_ext=${m} refine round ${round}: ${nf} file(s), largest drop ${mx}," \
                 "${nz} new rate-0 point(s)"
            if [ "$nz" -gt 0 ] || awk -v a="$mx" -v b="$MIN_GAIN" 'BEGIN { exit !(a > b) }'; then
                quiet=0
            else
                quiet=$(( quiet + 1 ))
                if [ "$quiet" -ge "$PATIENCE" ]; then
                    echo "$TAG d_ext=${m}: ${quiet} quiet round(s) in a row -- stopping"
                    break
                fi
            fi
        done
    done
fi

# ---- 4. panels and figures ---------------------------------------------------
if [ "${SKIP_PLOT:-0}" != "1" ]; then
    pid=$(sbatch --parsable scripts/rate_complex_collect.slurm.sh)
    echo "$TAG stage 4: collect / figure job ${pid%%;*} -> ${OUT_ROOT}/<model>_complex_rate_panels.png," \
         "${PAPER_DIR:-results_plots/paper}/<model>_complex_rate_panels.pdf"
fi
