#!/bin/bash
# ============================================================
#  One-shot submission of the d_ext = 12 seed probe (returns at once):
#    1. prepare   scripts/model10_d12_seed_probe_collect.slurm.sh MODE=prepare
#                 (1 job, N_PROC readers) -> targets.npz
#    2. probe     scripts/model10_d12_seed_probe.slurm.sh, array 0-199,
#                 --dependency=afterok:<prepare>
#    3. collect   MODE=collect, --dependency=afterok:<array>  -> summary in
#                 logs/m10probe_io_m10_probe_io_<id>.out and
#                 $OUT_DIR/model10_seeded_d12_probe/probe_summary.npz
#
#  Usage (from anywhere; the script moves to the repo root):
#      bash scripts/submit_model10_d12_seed_probe.sh                       # 1D run
#      OUT_DIR=results_model10_rate_dim2 DIM=2 bash scripts/submit_model10_d12_seed_probe.sh
#      TARGETS=near bash scripts/submit_model10_d12_seed_probe.sh          # radius-2 band
#  Every variable of the two slurm scripts passes through (TARGETS, RADIUS,
#  GROUPS, CONTROL=1, FAM_*, BUNDLE_ITERS, MARGIN_ITERS, N_POLISH_CANDS,
#  N_PROC, ADOPT=1, SEED, STRIDE), since sbatch exports the environment.
#
#  afterok needs every task to exit 0.  The worker catches per-point errors
#  and exits 0, so only a wall-time kill holds the collect back; then
#  resubmit (finished points are skipped; the prepare is redone, which is
#  harmless) or run the collect by hand:
#      MODE=collect sbatch scripts/model10_d12_seed_probe_collect.slurm.sh
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."               # repo root
export OUT_DIR="${OUT_DIR:-results_model10_rate}"
export STRIDE="${STRIDE:-1}"
export TARGETS="${TARGETS:-boundary}"
mkdir -p logs "$OUT_DIR"
TAG="[m10 d12 probe]"

pid=$(MODE=prepare sbatch --parsable scripts/model10_d12_seed_probe_collect.slurm.sh)
pid="${pid%%;*}"                      # --parsable may print "<id>;<cluster>"
echo "$TAG prepare job ${pid} submitted (targets=${TARGETS}, ${OUT_DIR})"

aid=$(sbatch --parsable --dependency=afterok:"$pid" scripts/model10_d12_seed_probe.slurm.sh)
aid="${aid%%;*}"
echo "$TAG probe array ${aid} queued (0-199) with --dependency=afterok:${pid}"

cid=$(MODE=collect sbatch --parsable --dependency=afterok:"$aid" \
          scripts/model10_d12_seed_probe_collect.slurm.sh)
cid="${cid%%;*}"
echo "$TAG collect job ${cid} queued with --dependency=afterok:${aid}"
echo "  targets:   logs/m10probe_io_m10_probe_io_${pid}.out"
echo "  progress:  tail -f logs/m10probe_m10_probe_${aid}_0.out"
echo "  summary:   logs/m10probe_io_m10_probe_io_${cid}.out"
