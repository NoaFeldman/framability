#!/bin/bash
# ============================================================
#  model3x: model3 with gamma extended to 16,
#     H = J sum_i Z_i Z_{i+1}                       (J = 1, h = 0)
#     jumps sqrt(gamma) |-><+|  and  sqrt(gamma') Z
#  grid gamma in [0, 16] x gamma' in [0, 10], step 0.2 (81 x 51 = 4131 points;
#  trotter_lindbladian_scan.MODELS['model3x']).  Its gamma <= 10 block is the
#  model3 grid, point for point (same indices, same file names).
#
#  The panels of results_model4_rate/model3_seeded_rate_panels.png on the
#  extended grid,
#     row 1 | stabilizer-3 | Pauli | opt Heisenberg d_ext = 4 | 8 | 12
#     row 2 | product-state chi = 10 | 40
#  through scripts/submit_rate_seeded.sh with MODEL=model3x:
#     0. import   the finished model3 run (results_model3_seeded: worker,
#                 xeval and every round file at d_ext = 4 / 8 / 12, product
#                 rates) is copied into results_model3x_seeded
#                 (scripts/rate_seeded_import.py; model3's run is untouched)
#     1. seed     only the 30 x 51 = 1530 new points gamma = 10.2 .. 16
#                 (7-8 per array task); the 2601 imported points are skipped.
#                 No finite-dt seeds exist there (the DT_BASE line covers
#                 gamma <= 10): rescaled-Pauli optimum, families, global search
#     2-7.        xeval / qrefine / d12 / margin / rrefine on the whole grid:
#                 the gamma = 10 column seeds the new strip by neighbour
#                 transfer, and imported points can only improve
#     8. plot     results_model4_rate/model3x_seeded_rate_panels.png; the log
#                 compares the gamma <= 10 block with
#                 results_model3_seeded/model3_seeded_rate_panels.npz
#  No 8q panels (as for model3).
#
#  The import must see a FINISHED model3 run, so this waits until no
#  rs_model3_* job (the model3 seeded pipeline) is queued or running.  It
#  also queues a 200-task array itself, so start it only when no other
#  200-task array is queued.
#
#  Output: results_model3x_seeded/{model3x_seeded, model3x_seeded_d12,
#                                  model3x_product}/pt_*.npz
#          results_model3x_seeded/model3x_seeded_rate_panels.npz
#          results_model4_rate/model3x_seeded_rate_panels.png
#
#  Run on the login node:
#      nohup bash scripts/submit_model3x_rate.sh > logs/submit_model3x.log 2>&1 &
#  Every knob of submit_rate_seeded.sh passes through (e.g. SKIP_RREF=1,
#  STRIDE=5 for a 17 x 11 preview -- needs a model3 run at the same stride
#  for the import).  Re-running picks up where it stopped (the import never
#  overwrites; finished points / rounds are skipped).
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."               # repo root
mkdir -p logs
TAG="[model3x]"

while squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -q '^rs_model3_'; do
    echo "$TAG waiting for the model3 seeded pipeline (rs_model3_* jobs): $(date)"
    sleep 600
done
src_npz="${IMPORT_DIR:-results_model3_seeded}/model3_seeded_rate_panels.npz"
[ -f "$src_npz" ] \
    || echo "$TAG warning: ${src_npz} missing -- has the model3 run reached its figure stage?"

echo "===== model3x: $(date) ====="
MODEL=model3x bash scripts/submit_rate_seeded.sh
echo "===== done: $(date) ====="
