#!/bin/bash
# ============================================================
#  Framability-rate panels of the atom-only open Dicke model
#  (dicke_lindbladian.md; trotter_lindbladian_scan._make_model_dicke) at
#  the N of NS -- registry entries dicke_N<N>, N in MODEL_DICKE_N
#  (5, 10, 20, 100, 1000) -- then the cross-N comparison figure.
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
#  The drivers of NS run SIDE BY SIDE with N_TASKS = 65 array tasks each.
#  Runs of OTHER N may still be going (e.g. N = 1000 while N = 5, 20 start):
#  only an N of NS that still has rs_dicke_N<N>_* jobs queued is refused, and
#  the 200-task budget reserves RESERVE_PER_RUN (66 = 65 array tasks + the
#  plot job) for every other dicke N seen in squeue, plus the user's queued
#  tasks that are not dicke jobs:
#      n(NS) x (N_TASKS + 1) + 66 x n(other dicke N) + other tasks <= 200
#  (each driver keeps at most one array stage queued, `sbatch --wait`).
#  When the drivers of NS have finished, scripts/dicke_n_compare_plot.py draws
#      rows: every N in COMPARE_NS with a panel file / (N_min) - (N_max)
#      cols: Pauli | opt d_ext = 4 | 8 | 12
#  and the gamma_phi line cuts, as one cluster job (after every queued
#  rs_dicke_N*_plot job; COMPARE_NS defaults to all registered N, missing
#  files are skipped, so it can be re-run any time:
#      .venv/bin/python scripts/dicke_n_compare_plot.py).
#
#  Output: results_dicke_N<N>_seeded/{dicke_N<N>_seeded, _seeded_d12, _product}/pt_*.npz
#          results_dicke_N<N>_seeded/dicke_N<N>_seeded_rate_panels.npz
#          results_model4_rate/dicke_N<N>_seeded_rate_panels.png
#          results_model4_rate/dicke_n_compare_panels.png  (+ _cuts.png)
#
#  Run on the login node:
#      nohup bash scripts/submit_dicke_rate.sh > logs/submit_dicke.log 2>&1 &
#      NS="5 20" nohup bash scripts/submit_dicke_rate.sh > logs/submit_dicke_5_20.log 2>&1 &
#  every knob of submit_rate_seeded.sh passes through (defaults here:
#  N_TASKS=65, N_RREF=3).  Re-running picks up where it stopped (finished
#  points / rounds are skipped).
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."               # repo root
mkdir -p logs
NS="${NS:-10 100 1000}"
PY="${PY:-.venv/bin/python}"
export N_TASKS="${N_TASKS:-65}"
export N_RREF="${N_RREF:-3}"
RESERVE_PER_RUN="${RESERVE_PER_RUN:-66}"
LIMIT=200

# ---- registered? -------------------------------------------------------------
for N in $NS; do
    "$PY" -c "
import sys; sys.path.insert(0, '.')
from trotter_lindbladian_scan import MODELS
assert 'dicke_N$N' in MODELS, 'dicke_N$N is not registered (MODEL_DICKE_N)'" \
        || { echo "ERROR: dicke_N${N} is not in trotter_lindbladian_scan.MODELS"; exit 1; }
done
COMPARE_NS="${COMPARE_NS:-$("$PY" -c "
import sys; sys.path.insert(0, '.')
from trotter_lindbladian_scan import MODEL_DICKE_N
print(' '.join(str(n) for n in sorted(MODEL_DICKE_N.values())))")}"

# ---- queue: same N refused, other N reserved ---------------------------------
# job names are rs_dicke_N<N>_<stage>; the trailing _ keeps N10 apart from N100
queued_names=$(squeue -u "$USER" -h -r -o '%j' 2>/dev/null || true)
for N in $NS; do
    if grep -Eq "^rs_dicke_N${N}_" <<< "$queued_names"; then
        echo "ERROR: rs_dicke_N${N}_* jobs are still queued or running:"
        squeue -u "$USER" -o '%i %j %T' | grep -E "rs_dicke_N${N}_" || true
        echo "       stop that run first (kill its driver process, then scancel)"
        exit 1
    fi
done
other_ns=$(sed -n 's/^rs_dicke_N\([0-9][0-9]*\)_.*/\1/p' <<< "$queued_names" | sort -un \
           | paste -sd' ' - || true)
n_other=$(echo $other_ns | wc -w)
n_foreign=$(grep -cvE '^(rs_dicke_N[0-9]+_|$)' <<< "$queued_names" || true)
n_models=$(echo $NS | wc -w)
need=$(( n_models * (N_TASKS + 1) ))
budget=$(( LIMIT - n_other * RESERVE_PER_RUN - n_foreign ))
echo "===== queue: other dicke runs N = ${other_ns:-none} (reserve ${RESERVE_PER_RUN} each), ${n_foreign} other queued task(s); this run needs ${need} of ${budget} ====="
if [ "$need" -gt "$budget" ]; then
    max_tasks=$(( budget / n_models - 1 ))
    echo "ERROR: ${n_models} models x (N_TASKS=${N_TASKS} + 1) = ${need} > ${budget} free of ${LIMIT}"
    [ "$max_tasks" -ge 1 ] && echo "       set N_TASKS=${max_tasks} (or fewer N in NS)"
    exit 1
fi

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

# ---- cross-N figure (after every queued per-N figure job) --------------------
# the per-N plot jobs write the panel npz files the comparison reads
deps=$(squeue -u "$USER" -h -o '%i %j' 2>/dev/null \
       | awk '$2 ~ /^rs_dicke_N[0-9]+_plot$/ {print $1}' | paste -sd: - || true)
pid=$(sbatch --parsable --cpus-per-task=1 --mem=4G --time=00:30:00 \
          --job-name=dicke_n_compare ${deps:+--dependency=afterany:$deps} \
          --output=logs/dicke_n_compare_%j.out --error=logs/dicke_n_compare_%j.err \
          --wrap="${PY} scripts/dicke_n_compare_plot.py --ns ${COMPARE_NS}")
echo "===== cross-N figure job ${pid%%;*} (N = ${COMPARE_NS}) submitted (after: ${deps:-none}) $(date) ====="
exit "$rc"
