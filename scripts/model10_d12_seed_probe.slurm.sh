#!/bin/bash
# ============================================================
#  SLURM job-array: d_ext = 12 seed probe (scripts/model10_d12_seed_probe_worker.py)
#  -- new structured families (warped YZ polygon, latitude rings about X,
#  other plane splits, P + C) against the best known d_ext = 12 rate at the
#  boundary of the rate-0 region.  Targets come from targets.npz, written on
#  the login node first:
#
#      python scripts/model10_d12_seed_probe_worker.py --prepare [--out_dir ... --dim ...]
#      sbatch scripts/model10_d12_seed_probe.slurm.sh
#      python scripts/model10_d12_seed_probe_collect.py [--out_dir ...]
#
#  ~14 families x ~170 LPs (~1 s each at 144 columns) + two polishes
#  => 30-60 min per point; the boundary holds a few hundred points, so
#  1-3 points per task.  TARGETS=near (radius 2) roughly triples that.
#  Output: $OUT_DIR/model10_seeded_d12_probe/pt_<ix>_<iy>.npz  (never the
#  pipeline directories; the collect script's --adopt promotes improvements)
#
#  Knobs (env): OUT_DIR (results_model10_rate), DIM (model default; the 2D
#  run needs OUT_DIR=results_model10_rate_dim2 DIM=2), GROUPS ("warp lat
#  split pc control"), CONTROL=1 (also polish the incumbent), FAM_*,
#  BUNDLE_ITERS, MARGIN_ITERS, N_POLISH_CANDS, SEED, N_CHUNKS.
# ============================================================

#SBATCH --job-name=m10_probe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/m10probe_%x_%A_%a.out
#SBATCH --error=logs/m10probe_%x_%A_%a.err

OUT_DIR=${OUT_DIR:-results_model10_rate}
N_CHUNKS=${N_CHUNKS:-200}          # must match the --array size above
STRIDE=${STRIDE:-1}
GROUPS=${GROUPS:-"warp lat split pc control"}
MAX_STARTS=${MAX_STARTS:-12}
FAM_RANDOM=${FAM_RANDOM:-4}
FAM_POLISH=${FAM_POLISH:-2}
FAM_MAXFEV=${FAM_MAXFEV:-80}
N_POLISH_CANDS=${N_POLISH_CANDS:-2}
BUNDLE_ITERS=${BUNDLE_ITERS:-25}
MARGIN_ITERS=${MARGIN_ITERS:-40}
CONTROL=${CONTROL:-0}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[model10 d12 probe] chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS} (${OUT_DIR}): starting"

# shellcheck disable=SC2086
python scripts/model10_d12_seed_probe_worker.py \
    --task_id        "$SLURM_ARRAY_TASK_ID" \
    --n_chunks       "$N_CHUNKS" \
    --out_dir        "$OUT_DIR" \
    --stride         "$STRIDE" \
    ${DIM:+--dim "$DIM"} \
    --groups         $GROUPS \
    --max_starts     "$MAX_STARTS" \
    --fam_random     "$FAM_RANDOM" \
    --fam_polish     "$FAM_POLISH" \
    --fam_maxfev     "$FAM_MAXFEV" \
    --n_polish_cands "$N_POLISH_CANDS" \
    --bundle_iters   "$BUNDLE_ITERS" \
    --margin_iters   "$MARGIN_ITERS" \
    $([ "$CONTROL" = "1" ] && echo --control) \
    --seed           "$SEED"

echo "[model10 d12 probe] chunk ${SLURM_ARRAY_TASK_ID}: done"
