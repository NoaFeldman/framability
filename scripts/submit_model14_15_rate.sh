#!/bin/bash
# ============================================================
#  model14 / model15: the transverse-field Ising chain (1D, dim = 1) with
#  S^- decay and Z dephasing,
#     H = J sum_i Z_i Z_{i+1} + h sum_i X_i                     (J = 1)
#     jumps sqrt(gamma_1) S^- = sqrt(gamma_1) |0><1|  and  sqrt(gamma_2) Z
#  model14: h = 1 (the 1D critical field h = J), model15: h = 0.5;
#  grid (gamma_1, gamma_2) in [0, 2]^2, step 0.1 (21 x 21 = 441 points).
#
#  Every panel of results_model4_rate/model10_rate_panels.png, per model:
#     row 1 | stabilizer-3 | Pauli | opt Heisenberg d_ext = 4 | 8 | 12
#     row 2 | product-state chi = 10 | 40 | 8q osc rate | 8q gap (periodic ring)
#  through scripts/submit_rate_seeded.sh (seed -> 8q -> xeval -> qrefine ->
#  d12 seed/refine -> margin -> rrefine -> figure), model14 first, then
#  model15: one model at a time, since at most one 200-task array may be
#  queued at a time.
#
#  Hand prediction (not computed; bond shares 1/2 included): Pauli rate
#  >= 2h everywhere (2 for model14, 1 for model15) -- on the ZZ column the
#  S^- damping is cancelled exactly by its drift into I and the field rotates
#  it into YZ + ZY at total weight 2h, so any mu* = 0 region must come from
#  the optimised frames.
#
#  Output: results_<model>_seeded/{<model>_seeded, <model>_seeded_d12,
#                                  <model>_product, <model>_8q}/pt_*.npz
#          results_<model>_seeded/<model>_seeded_rate_panels.npz
#          results_model4_rate/<model>_seeded_rate_panels.png
#
#  Run on the login node:
#      nohup bash scripts/submit_model14_15_rate.sh > logs/submit_model14_15.log 2>&1 &
#  MODELS="model15" runs one model; every knob of submit_rate_seeded.sh
#  passes through (e.g. MB_STRIDE=2 for an 11 x 11 8q grid, SKIP_MB=1).
#  Re-running picks up where it stopped (finished points / rounds are skipped).
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."               # repo root
mkdir -p logs

for MODEL in ${MODELS:-model14 model15}; do
    echo "===== ${MODEL}: $(date) ====="
    MODEL="$MODEL" bash scripts/submit_rate_seeded.sh \
        || echo "[${MODEL}] submit_rate_seeded.sh failed; going on"
done
echo "===== done: $(date) ====="
