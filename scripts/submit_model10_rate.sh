#!/bin/bash
# ============================================================
#  Full model10 panel pipeline: the figure of
#  results_model4_rate/model4_rate_panels.png, drawn by the same scripts for
#  model10 -> results_model10_rate/model10_rate_panels.{npz,png}.
#
#  model10 is the dissipative quantum Ising chain of
#     N. Shibata and H. Katsura, "Dissipative quantum Ising chain as a
#     non-Hermitian Ashkin-Teller model", Phys. Rev. B 99, 224432 (2019),
#     https://arxiv.org/abs/1904.12505
#  written in trotter_lindbladian_scan's axes (coupling along Z, field along X):
#     H = -J sum_i Z_i Z_{i+1} - h sum_i X_i        (J = h = 1, 1D chain, dim = 1)
#     jumps sqrt(Delta1) X_i  and  sqrt(Delta2) Z_i Z_{i+1}
#  on the (Delta1, Delta2) grid [0, 2.5]^2, step 0.05 (51 x 51).  The diagonal
#  is the paper's self-dual line (exact Liouvillian gap, cusp at 1/sqrt 3).
#
#  Hand predictions (rate_zero_lines analysis, not computed): the Pauli rate is
#  zero exactly on Delta1 >= 1, Delta2 >= 1; rescaled Pauli frames (d_ext = 4)
#  also reach zero on Delta1 >= 1, Delta2 (2 Delta2 + Delta1 - 1/Delta1) >= 2;
#  no frame reaches zero on the Delta2 = 0 edge.
#
#  Stages (MODEL=model10 through the model4 scripts):
#    1. rates      scripts/model4_rate_panels.slurm.sh          (array 0-199)
#                  six framability rates of the dim = 1 bond generator
#    2. refine     scripts/submit_model4_rate_quick_refine.sh   (<= N_ROUNDS rounds,
#                  each an array 0-199): neighbour cross-evaluation of the
#                  optimised Heisenberg rates
#    3. 8q panels  scripts/model4_manybody.slurm.sh             (array 0-120)
#                  osc rate + gap of the 8-site periodic-ring Liouvillian
#    4. Q          scripts/liouvillian_q.slurm.sh               (array 0-199)
#                  bond + 6-site-ring quality factor Q_max
#    5. Q_obs      scripts/observable_q.slurm.sh                (array 0-199)
#                  observable quality factor, Pauli + optimised basis
#    6. collect    scripts/model4_rate_panels_collect.slurm.sh
#
#  The stages run as ONE sequential chain (`sbatch --wait`), like
#  submit_model8_rate.sh, so at most one array (<= 200 tasks) is queued at a
#  time.  Run it on the login node inside tmux / nohup:
#
#      nohup bash scripts/submit_model10_rate.sh > logs/submit_model10.log 2>&1 &
#
#  Every worker skips points already on disk, so re-running the script after a
#  failure only fills holes; the refine stage resumes at the next round number.
#  SKIP_RATES=1 / SKIP_REFINE=1 / SKIP_MB=1 / SKIP_Q=1 / SKIP_OBS=1 skip a stage.
# ============================================================
set -euo pipefail

MODEL=model10
OUT_DIR="${OUT_DIR:-results_model10_rate}"
Q_DIR="${Q_DIR:-results_liouvillian_q}"
OBS_DIR="${OBS_DIR:-results_observable_q}"
Q_LEVELS="${Q_LEVELS:-1}"
N_ROUNDS="${N_ROUNDS:-10}"

cd "$(dirname "$0")/.."               # repo root
mkdir -p logs "$OUT_DIR" "$Q_DIR" "$OBS_DIR"

if [ "${SKIP_RATES:-0}" != "1" ]; then
    echo "[model10] stage 1/6: framability rates"
    MODEL=$MODEL OUT_DIR="$OUT_DIR" \
        sbatch --wait --job-name=m10_rate scripts/model4_rate_panels.slurm.sh
fi

if [ "${SKIP_REFINE:-0}" != "1" ]; then
    echo "[model10] stage 2/6: quick neighbour refine of the optimised rates"
    MODEL=$MODEL OUT_DIR="$OUT_DIR" N_ROUNDS="$N_ROUNDS" \
        bash scripts/submit_model4_rate_quick_refine.sh
fi

if [ "${SKIP_MB:-0}" != "1" ]; then
    echo "[model10] stage 3/6: 8-qubit ring many-body panels"
    MODEL=$MODEL OUT_DIR="$OUT_DIR" \
        sbatch --wait --job-name=m10_8q scripts/model4_manybody.slurm.sh
fi

if [ "${SKIP_Q:-0}" != "1" ]; then
    echo "[model10] stage 4/6: Lindbladian quality factor"
    MODEL=$MODEL OUT_DIR="$Q_DIR" \
        sbatch --wait --job-name=m10_q scripts/liouvillian_q.slurm.sh
fi

if [ "${SKIP_OBS:-0}" != "1" ]; then
    echo "[model10] stage 5/6: observable quality factor"
    MODEL=$MODEL OUT_DIR="$OBS_DIR" \
        sbatch --wait --job-name=m10_obs scripts/observable_q.slurm.sh
fi

echo "[model10] stage 6/6: collect + figure"
MODEL=$MODEL IN_DIR="$OUT_DIR" OUT_DIR="$OUT_DIR" Q_DIR="$Q_DIR" \
    Q_LEVELS="$Q_LEVELS" OBS_DIR="$OBS_DIR" \
    sbatch --wait --job-name=m10_collect scripts/model4_rate_panels_collect.slurm.sh

echo "[model10] done: $OUT_DIR/model10_rate_panels.png"
