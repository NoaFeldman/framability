#!/bin/bash
# ============================================================
#  Framability-rate panels of the atom-only open Dicke model
#  (dicke_lindbladian.md; trotter_lindbladian_scan._make_model_dicke) at
#  N = 10, 100, 1000 -- the registry entries dicke_N10 / dicke_N100 /
#  dicke_N1000 -- then the cross-N comparison figure.
#
#     L = (lam/sqrt N) sum_i (c_- s^-_i + c_+ s^+_i),  kappa D[L]  (cavity eliminated)
#     H = sum h Z + (lam^2/N) sum_{i<j} [Im A XX + Im B/2 (XY + YX)]
#     + gamma_phi sum D[Z]
#  omega = kappa = 1, omega0 = 0.1; grid lam/lam_c in [0, 2] step 0.1 (21)
#  x gamma_phi in [0, 1] step 0.1 (11) = 231 points per N.
#
#  Every N goes through scripts/submit_rate_seeded.sh (seed -> xeval ->
#  qrefine -> d12 seed/refine -> margin -> rrefine -> figure); the 8q stage is
#  skipped by itself (the many-body worker does not know these models).  The
#  rate scanned is that of the DIMER patch (N-1) x pair generator, whose
#  coefficients carry the explicit N: (N-1)/N on the pair Hamiltonian and the
#  collective pair jump, -(N-2)/N on the non-CP single-site counterterm, 1/N
#  on the Lamb shift.
#
#  The three drivers run SIDE BY SIDE with N_TASKS = 65 array tasks each
#  (3 x 65 = 195, plus <= 3 single collect / plot jobs, under the 200 limit).
#  When all three have finished, scripts/dicke_n_compare_plot.py draws
#      rows: N = 10 / 100 / 1000 / (N=10) - (N=1000)
#      cols: Pauli | opt d_ext = 4 | 8 | 12
#  and the gamma_phi = 0 line cuts, as one cluster job.
#
#  Output: results_dicke_N<N>_seeded/{dicke_N<N>_seeded, _seeded_d12, _product}/pt_*.npz
#          results_dicke_N<N>_seeded/dicke_N<N>_seeded_rate_panels.npz
#          results_model4_rate/dicke_N<N>_seeded_rate_panels.png
#          results_model4_rate/dicke_n_compare_panels.png  (+ _cuts.png)
#
#  Run on the login node:
#      nohup bash scripts/submit_dicke_rate.sh > logs/submit_dicke.log 2>&1 &
#  NS="10 100" runs a subset; every knob of submit_rate_seeded.sh passes
#  through (defaults here: N_TASKS=65, N_RREF=3).  Re-running picks up where
#  it stopped (finished points / rounds are skipped).
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."               # repo root
mkdir -p logs
NS="${NS:-10 100 1000}"
PY="${PY:-.venv/bin/python}"
export N_TASKS="${N_TASKS:-65}"
export N_RREF="${N_RREF:-3}"

n_models=$(echo $NS | wc -w)
if [ $(( n_models * (N_TASKS + 1) )) -gt 200 ]; then
    echo "ERROR: ${n_models} models x (N_TASKS=${N_TASKS} + 1) > 200 queued jobs"
    exit 1
fi
if squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -Eq '^rs_dicke_N'; then
    echo "ERROR: rs_dicke_N* jobs are still queued or running:"
    squeue -u "$USER" -o '%i %j %T' | grep -E 'rs_dicke_N' || true
    echo "       stop the old run first (kill its driver process, then scancel)"
    exit 1
fi
for N in $NS; do
    "$PY" -c "
import sys; sys.path.insert(0, '.')
from trotter_lindbladian_scan import MODELS
assert 'dicke_N$N' in MODELS, 'dicke_N$N is not registered (MODEL_DICKE_N)'" \
        || { echo "ERROR: dicke_N${N} is not in trotter_lindbladian_scan.MODELS"; exit 1; }
done

# ---- the drivers side by side ------------------------------------------------
pids=()
for N in $NS; do
    MODEL="dicke_N${N}"
    echo "===== ${MODEL}: start $(date); log logs/submit_${MODEL}.log ====="
    MODEL="$MODEL" bash scripts/submit_rate_seeded.sh \
        > "logs/submit_${MODEL}.log" 2>&1 &
    pids+=("$!")
done
rc=0
for p in "${pids[@]}"; do
    wait "$p" || rc=1
done
[ "$rc" = "0" ] || echo "warning: at least one driver failed; see logs/submit_dicke_N*.log"

# ---- cross-N figure (after every per-N figure job) ---------------------------
# the per-N plot jobs write the panel npz files the comparison reads
deps=$(squeue -u "$USER" -h -o '%i %j' 2>/dev/null \
       | awk '$2 ~ /^rs_dicke_N[0-9]+_plot$/ {print $1}' | paste -sd: - || true)
NS_ARG="$NS"
pid=$(sbatch --parsable --cpus-per-task=1 --mem=4G --time=00:30:00 \
          --job-name=dicke_n_compare ${deps:+--dependency=afterany:$deps} \
          --output=logs/dicke_n_compare_%j.out --error=logs/dicke_n_compare_%j.err \
          --wrap="${PY} scripts/dicke_n_compare_plot.py --ns ${NS_ARG}")
echo "===== cross-N figure job ${pid%%;*} submitted (after: ${deps:-none}) $(date) ====="
exit "$rc"
