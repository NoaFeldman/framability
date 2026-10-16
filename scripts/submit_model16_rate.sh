#!/bin/bash
# ============================================================
#  model16: the driven Rydberg gas of Xiang, Li, Bai & Ma, "Switching Dynamics
#  of Metastable Open Quantum Systems", arXiv:2505.05202 (Eq. 2), with an added
#  Z dephasing,
#     H = sum_l [ (Omega/2) X_l - Delta n_l ] + V/(N-1) sum_{l<k} n_l n_k
#     jumps sqrt(gamma) S^-_l  and  sqrt(gamma_phi) Z_l
#  Omega = 1.5, V = 10, gamma = 1 (the time unit), n = |1><1| (|1> = Rydberg).
#  Grid: Delta in [2, 5], step 0.2 (16 values) across the mean-field phases
#      monostable I (Delta ~ 2.4) | bistable (3 <~ Delta <~ 4.2) | monostable II (~4.4)
#  against gamma_phi in [0, 1], step 0.1 (11 values): 176 points, one array
#  task each (200 tasks, the last 24 idle).
#
#  Every panel of results_model4_rate/model10_rate_panels.png:
#     row 1 | stabilizer-3 | Pauli | opt Heisenberg d_ext = 4 | 8 | 12
#     row 2 | product-state chi = 10 | 40 | 8q osc rate | 8q gap
#  through scripts/submit_rate_seeded.sh (seed -> 8q -> xeval -> qrefine ->
#  d12 seed/refine -> margin -> rrefine -> figure): data generation, collect
#  and plot jobs in one go.  At most one 200-task array is queued at a time
#  (every stage is an `sbatch --wait`), so the script stays alive on the login
#  node until the figure job has been submitted.
#
#  How the all-to-all model enters the (nearest-neighbour) bond machinery
#  (trotter_lindbladian_scan._build_model16):
#   * rate panels: every pair of atoms is a bond carrying V/(N-1) and 1/(N-1)
#     of each one-site term, i.e. 1/(N-1) times the DIMER generator (V n n + the
#     one-site terms at full weight).  The dimer is what is scanned (dim = 1/2,
#     one-site share 1/(2 dim) = 1), so the rates are those of the pair
#     generator times (N-1); the mu* = 0 regions do not depend on N.
#   * 8q panels: the real model, N = 8 on the complete graph with V/7 per pair
#     (n_qubit_lindbladian.ALL_TO_ALL_MODELS).  The Liouvillian is invariant
#     under atom permutations, so its low modes are highly degenerate; the gap
#     (smallest non-zero decay rate) is unaffected, the osc rate is, as always,
#     a lower bound.
#  Conventions (paper vs here): Omega/2 X, not Omega X (the paper's mean-field
#  equations 4a-4c are those of Omega/2); standard jump sqrt(gamma)|0><1| (the
#  paper's sigma^- = sigma^x - i sigma^y would double it, but its population
#  decays at rate gamma).
#
#  Output: results_model16_seeded/{model16_seeded, model16_seeded_d12,
#                                  model16_product, model16_8q}/pt_*.npz
#          results_model16_seeded/model16_seeded_rate_panels.npz
#          results_model4_rate/model16_seeded_rate_panels.png
#
#  Run on the login node:
#      nohup bash scripts/submit_model16_rate.sh > logs/submit_model16.log 2>&1 &
#  Every knob of submit_rate_seeded.sh passes through (e.g. MB_STRIDE=2 for a
#  coarser 8q grid, SKIP_MB=1, N_RREF=3 to cap the randomised-refine rounds).
#  Re-running picks up where it stopped (finished points / rounds are skipped).
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."               # repo root
mkdir -p logs

for MODEL in ${MODELS:-model16}; do
    echo "===== ${MODEL}: $(date) ====="
    MODEL="$MODEL" bash scripts/submit_rate_seeded.sh \
        || echo "[${MODEL}] submit_rate_seeded.sh failed; going on"
done
echo "===== done: $(date) ====="
