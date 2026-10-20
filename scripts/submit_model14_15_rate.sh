#!/bin/bash
# ============================================================
#  model14 / model15: the transverse-field Ising chain (1D, dim = 1) with
#  S^- decay and Z dephasing,
#     H = J sum_i Z_i Z_{i+1} + h sum_i X_i                     (J = 1)
#     jumps sqrt(gamma_1) S^- = sqrt(gamma_1) |0><1|  and  sqrt(gamma_2) Z
#  model14: gamma_2 = 0, model15: gamma_2 = 1 (fixed);
#  grid v3 (2026-10-20): gamma_1 in [0, 3] step 0.2 (x) by h in [0, 1.3]
#  step 0.1 (y): 16 x 14 = 224 points.
#
#  Every panel of results_model4_rate/model10_rate_panels.png, per model:
#     row 1 | stabilizer-3 | Pauli | opt Heisenberg d_ext = 4 | 8 | 12
#     row 2 | product-state chi = 10 | 40 | 8q osc rate | 8q gap (periodic ring)
#  through scripts/submit_rate_seeded.sh (seed -> 8q -> xeval -> qrefine ->
#  d12 seed/refine -> margin -> rrefine -> figure).
#
#  PRELIMINARY-TEST settings (override with the env variables):
#    * the two models run SIDE BY SIDE, N_TASKS = 99 array tasks each (two
#      arrays + one single collect / plot job per model stay <= 200);
#    * light randomised refine: N_RREF = 2 rounds per d_ext, HOPS = 6,
#      MAX_SECONDS = 1800 per point, TARGETS = all (the 'nonmono' rule reads
#      both axes as noise axes; h is not one -- the rate grows with h -- so it
#      would flag almost every point anyway).
#    For a full-power run later: N_RREF=99 HOPS=20 MAX_SECONDS=5400, and
#    MIN_GAIN of order 1e-2: the rates here are O(1) everywhere, and the
#    default absolute 1e-3 never let the v2 run's stop rule fire (its d_ext = 8
#    randomised refine ran to the round cap 99).
#
#  Stage 0 (login node, before the drivers): the v2 model14 run (grid
#  gamma_1 in [0, 2] step 0.1 x h in [0, 4.8] step 0.4) is set aside as
#  results_model14_v2_seeded, and its 44 points on the v3 grid (gamma_1 =
#  0, 0.2, .., 2 at h = 0, 0.4, 0.8, 1.2) are imported with their best known
#  frames (scripts/rate_seeded_remap_import.py); the xeval / refine stages
#  spread them to the new points.  model15 never ran on v2: nothing to import.
#
#  Folder rules for results_<model>_seeded (checked for both models before
#  anything is submitted):
#    holds the v3 marker .grid_v3            -> resumed
#    holds the v2 marker .grid_gamma1_h_v2   -> moved to results_<model>_v2_seeded
#    exists, no marker, no pt_*.npz files    -> adopted (e.g. a partial plot's npz)
#    exists, no marker, with pt_*.npz files  -> refused (unknown old run)
#  and nothing starts while an rs_model14_* / rs_model15_* job is queued (stop
#  an old run first: its driver process, then scancel its jobs).
#
#  Hand prediction (not computed; bond shares 1/2 included): Pauli rate
#     2h + 2J - gamma_1/4 - gamma_2    (column Y(x)Z; whole grid)
#  i.e. 1.25 .. 4.6 for model14, 0.25 .. 3.6 for model15 -- smooth in h,
#  nothing at the 1D critical field h = J.
#
#  Output: results_<model>_seeded/{<model>_seeded, <model>_seeded_d12,
#                                  <model>_product, <model>_8q}/pt_*.npz
#          results_<model>_seeded/<model>_seeded_rate_panels.npz
#          results_model4_rate/<model>_seeded_rate_panels.png
#  Logs:   logs/submit_<model>_v3.log (one per model)
#
#  Run on the login node:
#      nohup bash scripts/submit_model14_15_rate.sh > logs/submit_model14_15_v3.log 2>&1 &
#  MODELS="model15" runs one model; every knob of submit_rate_seeded.sh
#  passes through.  Re-running picks up where it stopped.
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."               # repo root
mkdir -p logs
RUN_MODELS="${MODELS:-model14 model15}"
PY="${PY:-.venv/bin/python}"
MARK=".grid_v3"
OLD_MARK=".grid_gamma1_h_v2"          # written by the v2 version of this script
export N_TASKS="${N_TASKS:-99}"
export N_RREF="${N_RREF:-2}"
export HOPS="${HOPS:-6}"
export MAX_SECONDS="${MAX_SECONDS:-1800}"
export TARGETS="${TARGETS:-all}"

# ---- checks, for every model before anything is submitted ------------------
if squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -Eq '^rs_model1[45]_'; then
    echo "ERROR: rs_model14_* / rs_model15_* jobs are still queued or running:"
    squeue -u "$USER" -o '%i %j %T' | grep -E 'rs_model1[45]_' || true
    echo "       stop the old run first (kill its driver process, then scancel)"
    exit 1
fi
for MODEL in $RUN_MODELS; do
    d="results_${MODEL}_seeded"
    [ -d "$d" ] || continue
    [ -f "$d/$MARK" ] && continue
    if [ -f "$d/$OLD_MARK" ]; then
        if [ -e "results_${MODEL}_v2_seeded" ]; then
            echo "ERROR: $d is the v2 run, but results_${MODEL}_v2_seeded exists already"
            exit 1
        fi
    elif find "$d" -name 'pt_*.npz' -print -quit | grep -q .; then
        echo "ERROR: $d holds per-point files of an unknown earlier run (no marker);"
        echo "       move or delete it first"
        exit 1
    fi
done

# ---- stage 0: set the v2 runs aside, import their points on the v3 grid ----
for MODEL in $RUN_MODELS; do
    d="results_${MODEL}_seeded"
    v2="results_${MODEL}_v2_seeded"
    if [ -d "$d" ] && [ ! -f "$d/$MARK" ] && [ -f "$d/$OLD_MARK" ]; then
        echo "[${MODEL}] moving the v2 run ${d} -> ${v2}"
        mv "$d" "$v2"
    fi
    mkdir -p "$d"
    touch "$d/$MARK"
    if [ -d "$v2" ] && [ "${SKIP_IMPORT:-0}" != "1" ]; then
        echo "[${MODEL}] stage 0: importing the v2 points on the v3 grid from ${v2}"
        "$PY" scripts/rate_seeded_remap_import.py --model "$MODEL" \
            --src_dir "$v2" --out_dir "$d" \
            --mb_stride "${MB_STRIDE:-1}" --stride "${STRIDE:-1}" \
            || { echo "[${MODEL}] ERROR: import failed"; exit 1; }
    fi
done

# ---- both drivers side by side ----------------------------------------------
pids=()
for MODEL in $RUN_MODELS; do
    echo "===== ${MODEL}: start $(date); log logs/submit_${MODEL}_v3.log ====="
    MODEL="$MODEL" bash scripts/submit_rate_seeded.sh \
        > "logs/submit_${MODEL}_v3.log" 2>&1 &
    pids+=("$!")
done
rc=0
for p in "${pids[@]}"; do
    wait "$p" || rc=1
done
echo "===== done: $(date) (exit ${rc}) ====="
exit "$rc"
