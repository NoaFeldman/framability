#!/bin/bash
# ============================================================
#  Framability-rate panels of the atom-only open Dicke model
#  (dicke_lindbladian.md; trotter_lindbladian_scan._make_model_dicke) at
#  the N of NS -- registry entries <FAMILY>_N<N>, N in 5, 10, 20, 100, 1000
#  (MODEL_DICKE_PARAMS) -- then the cross-N comparison figure.
#
#     L = (lam/sqrt N) sum_i (c_- s^-_i + c_+ s^+_i),  kappa D[L]  (cavity eliminated)
#     H = sum h Z + (lam^2/N) sum_{i<j} [Im A XX + Im B/2 (XY + YX)]
#  FAMILY (env, default dicke):
#     dicke           + gamma_phi sum D[Z];  kappa = 1
#                     grid lam/lam_c [0, 2] / 0.1 x gamma_phi [0, 1] / 0.1 (231 pts)
#     dicke_dec       + gamma_down sum D[s^-], gamma_phi = 0;  kappa = 1
#     dicke_dec_k025  the same, good cavity kappa = 0.25
#     dicke_dec_deph / dicke_dec_k025_deph
#                     the two above + on-site dephasing tied to the decay,
#                     4 gamma_phi = 0.5 gamma_down (T2 = 4/3 T1)
#                     grid lam/lam_c [0, 2] / 0.1 x gamma_down [0, 0.3] / 0.02 (336 pts)
#  omega = 1, omega0 = 0.1 throughout (see _make_model_dicke_dec for why decay).
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
#  Runs of OTHER models (other N, other family) may still be going: only a
#  model of this run that still has rs_<FAMILY>_N<N>_* jobs queued is refused.
#  N_TASKS and NS are used as given (each driver keeps at most one array
#  stage queued, `sbatch --wait`); if, together with the other queued work (RESERVE_PER_RUN = 66 per other running Dicke model +
#  every non-Dicke task in squeue), more than 200 tasks would be queued, the
#  script only notes it (the new tasks pend) -- STRICT_BUDGET=1 refuses instead.
#  When the drivers of NS have finished, scripts/dicke_n_compare_plot.py draws
#      rows: every N in COMPARE_NS with a panel file / (N_min) - (N_max)
#      cols: Pauli | opt d_ext = 4 | 8 | 12
#  (+ floor-zero contour, mean-field transition) and the line cuts, as one
#  cluster job (after every queued rs_<FAMILY>_N*_plot job; COMPARE_NS
#  defaults to all registered N of the family, missing files are skipped, so
#  it can be re-run any time:
#      .venv/bin/python scripts/dicke_n_compare_plot.py --family <FAMILY>).
#
#  Output: results_<FAMILY>_N<N>_seeded/{<FAMILY>_N<N>_seeded, _seeded_d12, _product}/pt_*.npz
#          results_<FAMILY>_N<N>_seeded/<FAMILY>_N<N>_seeded_rate_panels.npz
#          results_model4_rate/<FAMILY>_N<N>_seeded_rate_panels.png
#          results_model4_rate/<FAMILY>_n_compare_panels.png  (+ _cuts.png)
#
#  Run on the login node:
#      nohup bash scripts/submit_dicke_rate.sh > logs/submit_dicke.log 2>&1 &
#      NS="5 20" nohup bash scripts/submit_dicke_rate.sh > logs/submit_dicke_5_20.log 2>&1 &
#      FAMILY=dicke_dec_k025 NS="10 100" nohup bash scripts/submit_dicke_rate.sh \
#          > logs/submit_dicke_dec_k025.log 2>&1 &
#      FAMILY=dicke_dec_k025_deph NS="10 100" nohup bash scripts/submit_dicke_rate.sh \
#          > logs/submit_dicke_dec_k025_deph.log 2>&1 &
#  every knob of submit_rate_seeded.sh passes through (defaults here:
#  N_TASKS=65, N_RREF=3).  Re-running picks up where it stopped (finished
#  points / rounds are skipped).
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."               # repo root
mkdir -p logs
FAMILY="${FAMILY:-dicke}"
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
assert '${FAMILY}_N$N' in MODELS, '${FAMILY}_N$N is not registered (MODEL_DICKE_PARAMS)'" \
        || { echo "ERROR: ${FAMILY}_N${N} is not in trotter_lindbladian_scan.MODELS"; exit 1; }
done
COMPARE_NS="${COMPARE_NS:-$("$PY" -c "
import sys; sys.path.insert(0, '.')
from trotter_lindbladian_scan import MODEL_DICKE_PARAMS
print(' '.join(str(n) for n in sorted(p['N'] for p in MODEL_DICKE_PARAMS.values()
                                      if p['family'] == '$FAMILY')))")}"

# ---- queue: same model refused, other Dicke models reserved ------------------
# job names are rs_<family>_N<N>_<stage>; the trailing _ keeps N10 apart from
# N100 and dicke_N10 apart from dicke_dec_N10
DICKE_JOB='^rs_(dicke[a-z0-9_]*_N[0-9]+)_'
queued_names=$(squeue -u "$USER" -h -r -o '%j' 2>/dev/null || true)
for N in $NS; do
    if grep -Eq "^rs_${FAMILY}_N${N}_" <<< "$queued_names"; then
        echo "ERROR: rs_${FAMILY}_N${N}_* jobs are still queued or running:"
        squeue -u "$USER" -o '%i %j %T' | grep -E "rs_${FAMILY}_N${N}_" || true
        echo "       stop that run first (kill its driver process, then scancel)"
        exit 1
    fi
done
others=$(sed -nE "s/${DICKE_JOB}.*/\1/p" <<< "$queued_names" | sort -u \
         | paste -sd' ' - || true)
n_other=$(echo $others | wc -w)
n_foreign=$(grep -cvE "(${DICKE_JOB}|^\$)" <<< "$queued_names" || true)
n_models=$(echo $NS | wc -w)
need=$(( n_models * (N_TASKS + 1) ))
busy=$(( n_other * RESERVE_PER_RUN + n_foreign ))
echo "===== ${FAMILY}: queue: other Dicke runs ${others:-none} (up to ${RESERVE_PER_RUN} each), ${n_foreign} other queued task(s); this run adds up to ${need} ====="
if [ "$n_foreign" -gt 0 ]; then
    echo "      other queued tasks by job name:"
    grep -vE "(${DICKE_JOB}|^\$)" <<< "$queued_names" | sort | uniq -c | sort -rn | head -5 \
        | sed 's/^/      /' || true
fi
# (no refusal on the size of this run itself: removed 2026-11-13 at the
#  user's request -- N_TASKS and NS are used as given)
# with the other queued work: the cluster lets more than LIMIT tasks sit in the
# queue (2026-11-12: 305 + the N=1000 run queued at once), so only a note --
# the new tasks pend until slots free up.  STRICT_BUDGET=1 refuses instead.
if [ $(( need + busy )) -gt "$LIMIT" ]; then
    echo "note: ${need} + ${busy} queued > ${LIMIT}: the new tasks will wait for free slots"
    if [ "${STRICT_BUDGET:-0}" = "1" ]; then
        echo "ERROR: STRICT_BUDGET=1 and only $(( LIMIT - busy )) of ${LIMIT} free"
        exit 1
    fi
fi

# ---- the drivers side by side ------------------------------------------------
pids=()
for N in $NS; do
    MODEL="${FAMILY}_N${N}"
    echo "===== ${MODEL}: start $(date); log logs/submit_${MODEL}.log ====="
    MODEL="$MODEL" bash scripts/submit_rate_seeded.sh \
        > "logs/submit_${MODEL}.log" 2>&1 &
    pids+=("$!")
done
rc=0
for p in "${pids[@]}"; do
    wait "$p" || rc=1
done
[ "$rc" = "0" ] || echo "warning: at least one driver failed; see logs/submit_${FAMILY}_N*.log"

# ---- cross-N figure (after every queued per-N figure job of the family) ------
# the per-N plot jobs write the panel npz files the comparison reads
deps=$(squeue -u "$USER" -h -o '%i %j' 2>/dev/null \
       | awk -v re="^rs_${FAMILY}_N[0-9]+_plot\$" '$2 ~ re {print $1}' | paste -sd: - || true)
pid=$(sbatch --parsable --cpus-per-task=1 --mem=4G --time=00:30:00 \
          --job-name="${FAMILY}_n_compare" ${deps:+--dependency=afterany:$deps} \
          --output="logs/${FAMILY}_n_compare_%j.out" --error="logs/${FAMILY}_n_compare_%j.err" \
          --wrap="${PY} scripts/dicke_n_compare_plot.py --family ${FAMILY} --ns ${COMPARE_NS}")
echo "===== cross-N figure job ${pid%%;*} (${FAMILY}, N = ${COMPARE_NS}) submitted (after: ${deps:-none}) $(date) ====="
exit "$rc"
