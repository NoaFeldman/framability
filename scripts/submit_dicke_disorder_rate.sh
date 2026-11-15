#!/bin/bash
# ============================================================
#  Pair-averaged framability rate of the open Dicke model with UNEQUAL atomic
#  frequencies omega0 (framability_rate_disorder.py): one worker array
#  (scripts/dicke_disorder_rate.slurm.sh) over every (family, N, width, grid
#  point), then the collect / figure job (--dependency=afterany).
#
#  Model: N atoms in M_SPECIES species of equal abundance at
#  omega0_s = 0.1 (1 + w (2s+1-M)/M) (bin midpoints of a uniform distribution
#  of relative half-width w), each atom with its own cavity response
#  c_pm = 1/(kappa + i(omega pm omega0_s)); grid and seeds of the uniform runs
#  (results_<family>_N<N>_seeded, lam/lam_c of the mean omega0).  Per point:
#  Pauli, one SHARED frame on every atom, and one frame PER SPECIES (started
#  from the shared optimum), at every d_ext of D_EXTS; w = 0 is the uniform
#  control.  Nothing here checks job counts or the queue (Slurm schedules).
#
#  Usage (login node, repo root):
#      bash scripts/submit_dicke_disorder_rate.sh
#      NS="20 100" WIDTHS="0 0.5" bash scripts/submit_dicke_disorder_rate.sh
#      FAMILIES=dicke_dec_k025 D_EXTS="4 8 12" bash scripts/submit_dicke_disorder_rate.sh
#      FORCE=1 bash scripts/submit_dicke_disorder_rate.sh     # recompute everything
#  Every knob of dicke_disorder_rate.slurm.sh passes through the environment
#  (FAMILIES, NS, WIDTHS, M_SPECIES, D_EXTS, N_TASKS, N_STARTS, BUNDLE_ITERS,
#  N_KICKS, MAX_SEED_EVAL, SEED_NS, OUT_ROOT, NO_CERTIFY, FORCE; FIG_DIR for
#  the collect).  Re-running resumes (finished points / d_ext are skipped).
#
#  Output: results_dicke_disorder/<family>_N<N>_M<M>_w<w>/pt_<ix>_<iy>.npz
#          results_dicke_disorder/<family>_N<N>_M<M>_panels.npz
#          results_model4_rate/<family>_N<N>_disorder_{panels,changes,cuts}.png
#  Logs:   logs/dicke_dis_<array id>_<task>.out, logs/dicke_dis_collect_<id>.out
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."               # repo root
mkdir -p logs

export FAMILIES="${FAMILIES:-dicke_dec dicke_dec_k025}"
export NS="${NS:-100}"
export WIDTHS="${WIDTHS:-0 0.2 0.5 0.8}"
export M_SPECIES="${M_SPECIES:-4}"
export D_EXTS="${D_EXTS:-4 8}"
export N_TASKS="${N_TASKS:-200}"
export OUT_ROOT="${OUT_ROOT:-results_dicke_disorder}"

aid=$(sbatch --parsable --array=0-$((N_TASKS - 1)) scripts/dicke_disorder_rate.slurm.sh)
aid="${aid%%;*}"
echo "===== worker array ${aid}: ${FAMILIES} | N = ${NS} | w = ${WIDTHS} | M = ${M_SPECIES} | d = ${D_EXTS} | ${N_TASKS} tasks ====="

cid=$(sbatch --parsable --dependency=afterany:"${aid}" scripts/dicke_disorder_rate_collect.slurm.sh)
echo "===== collect job ${cid%%;*} (after ${aid}) ====="
