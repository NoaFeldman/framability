#!/bin/bash
# ============================================================
#  SLURM job (one task) bracketing the d_ext = 12 seed-probe array:
#    MODE=prepare   scripts/model10_d12_seed_probe_worker.py --prepare
#                   reads every point file of the d_ext 4/8/12 runs with
#                   N_PROC processes and writes
#                   $OUT_DIR/model10_seeded_d12_probe/targets.npz
#    MODE=collect   scripts/model10_d12_seed_probe_collect.py: the summary
#                   (ADOPT=1 also promotes improvements into the pipeline dir)
#
#  Chained by scripts/submit_model10_d12_seed_probe.sh; by hand:
#      MODE=prepare sbatch scripts/model10_d12_seed_probe_collect.slurm.sh
#      MODE=collect sbatch scripts/model10_d12_seed_probe_collect.slurm.sh
# ============================================================

#SBATCH --job-name=m10_probe_io
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/m10probe_io_%x_%A.out
#SBATCH --error=logs/m10probe_io_%x_%A.err

MODE=${MODE:?set MODE (prepare | collect)}
OUT_DIR=${OUT_DIR:-results_model10_rate}
STRIDE=${STRIDE:-1}
TARGETS=${TARGETS:-boundary}       # boundary | near | all
RADIUS=${RADIUS:-2}
RATE_TOL=${RATE_TOL:-1e-6}
N_PROC=${N_PROC:-${SLURM_CPUS_PER_TASK:-1}}
ADOPT=${ADOPT:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[model10 d12 probe] ${MODE} (${OUT_DIR}): starting"
case "$MODE" in
prepare)
    python scripts/model10_d12_seed_probe_worker.py \
        --prepare \
        --out_dir  "$OUT_DIR" \
        --stride   "$STRIDE" \
        ${DIM:+--dim "$DIM"} \
        --targets  "$TARGETS" \
        --radius   "$RADIUS" \
        --tol      "$RATE_TOL" \
        --n_proc   "$N_PROC"
    ;;
collect)
    python scripts/model10_d12_seed_probe_collect.py \
        --out_dir  "$OUT_DIR" \
        --stride   "$STRIDE" \
        --tol      "$RATE_TOL" \
        $([ "$ADOPT" = "1" ] && echo --adopt)
    ;;
*)
    echo "unknown MODE '$MODE'"; exit 1 ;;
esac
echo "[model10 d12 probe] ${MODE}: done"
