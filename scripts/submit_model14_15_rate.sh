#!/bin/bash
# ============================================================
#  model14 / model15: the transverse-field Ising chain (1D, dim = 1) with
#  S^- decay and Z dephasing,
#     H = J sum_i Z_i Z_{i+1} + h sum_i X_i                     (J = 1)
#     jumps sqrt(gamma_1) S^- = sqrt(gamma_1) |0><1|  and  sqrt(gamma_2) Z
#  model14: gamma_2 = 0, model15: gamma_2 = 1 (fixed);
#  grid gamma_1 in [0, 2] step 0.1 (x) by h in [0, 4.8] step 0.4 (y):
#  21 x 13 = 273 points.
#  (Redefined 2026-10-16: the first version scanned (gamma_1, gamma_2) at fixed
#  h = 1 / 0.5 and was scrapped.)
#
#  Every panel of results_model4_rate/model10_rate_panels.png, per model:
#     row 1 | stabilizer-3 | Pauli | opt Heisenberg d_ext = 4 | 8 | 12
#     row 2 | product-state chi = 10 | 40 | 8q osc rate | 8q gap (periodic ring)
#  through scripts/submit_rate_seeded.sh (seed -> 8q -> xeval -> qrefine ->
#  d12 seed/refine -> margin -> rrefine -> figure), model14 first, then
#  model15: one model at a time, since at most one 200-task array may be
#  queued at a time.
#
#  Hand prediction (not computed; bond shares 1/2 included):
#     Pauli rate >= max(2h, 2J - gamma_1/4 - gamma_2)  > 0 on the whole window
#  -- the ZZ column: the S^- damping is cancelled exactly by its drift into I
#  and the field rotates it into YZ + ZY at total weight 2h; the X(x)I column:
#  the coupling rotates it into Y(x)Z at 2J against damping gamma_1/4 + gamma_2.
#  So the Pauli panel has no zero (>= 1.5 for model14, >= 0.5 for model15), and
#  any mu* = 0 region must come from the optimised frames.
#
#  Output: results_<model>_seeded/{<model>_seeded, <model>_seeded_d12,
#                                  <model>_product, <model>_8q}/pt_*.npz
#          results_<model>_seeded/<model>_seeded_rate_panels.npz
#          results_model4_rate/<model>_seeded_rate_panels.png
#
#  The workers skip every pt_<ix>_<iy>.npz already on disk, and the scrapped
#  21 x 21 run used the same file names, so this script refuses to start while
#  a results_<model>_seeded/ without this version's marker file exists:
#  delete the old folder first (rm -rf results_<model>_seeded).
#
#  Run on the login node:
#      nohup bash scripts/submit_model14_15_rate.sh > logs/submit_model14_15.log 2>&1 &
#  MODELS="model15" runs one model; every knob of submit_rate_seeded.sh
#  passes through (e.g. MB_STRIDE=2 for a coarser 8q grid, SKIP_MB=1).
#  Re-running picks up where it stopped (finished points / rounds are skipped).
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."               # repo root
mkdir -p logs
RUN_MODELS="${MODELS:-model14 model15}"
MARK=".grid_gamma1_h_v2"              # written by this version into every out dir

for MODEL in $RUN_MODELS; do          # check every model before submitting anything
    d="results_${MODEL}_seeded"
    if [ -d "$d" ] && [ ! -f "$d/$MARK" ]; then
        echo "ERROR: $d holds the scrapped (gamma_1, gamma_2) run; delete it first:"
        echo "       rm -rf $d"
        exit 1
    fi
done

for MODEL in $RUN_MODELS; do
    echo "===== ${MODEL}: $(date) ====="
    mkdir -p "results_${MODEL}_seeded"
    touch "results_${MODEL}_seeded/$MARK"
    MODEL="$MODEL" bash scripts/submit_rate_seeded.sh \
        || echo "[${MODEL}] submit_rate_seeded.sh failed; going on"
done
echo "===== done: $(date) ====="
