#!/bin/bash
# ============================================================
#  SLURM job-array: model10 framability rates on the 2D square lattice
#  (bond generator with dim = DIM = 2: one-site terms h X, Delta1 X enter a
#  bond with weight 1/4), 51 x 51 = 2601 points over 200 tasks (~13 each,
#  strided so a dead task thins the grid evenly).  One script, five stages:
#
#    STAGE=seed               product-state rates chi = 10, 40
#                             (model4_product_rate_worker), then stabilizer-3,
#                             Pauli (--fixed_frames) and the seeded d_ext = 4 / 8
#                             rates (model10_seeded_worker), seeded with the
#                             dim = 1 optima of $XFER_DIRS at the point and its
#                             8 neighbours, closed forms, families, global search
#    STAGE=qrefine ROUND=r    one d_ext = 4 / 8 quick neighbour-refine round
#    STAGE=d12_seed           d_ext = 12 (model10_d12_worker --stage seed), seeded
#                             with this run's d_ext = 8 and the dim = 1 d_ext = 8 / 12
#    STAGE=d12_refine ROUND=r one d_ext = 12 refine round
#    STAGE=margin D_EXT=8|12 ROUND=r   one margin round (model10_margin_worker)
#    STAGE=rrefine D_EXT=8|12 ROUND=r  one randomised refine round
#                             (model10_rrefine_worker, TARGETS=nonmono: points
#                             above a less-noisy neighbour or next to rate 0)
#
#  Driven by scripts/submit_model10_dim2.sh (which sets --job-name / --time per
#  stage); by hand, e.g.
#      STAGE=seed sbatch scripts/model10_dim2.slurm.sh
#      STAGE=qrefine ROUND=1 sbatch --time=08:00:00 scripts/model10_dim2.slurm.sh
#
#  Output: $OUT_DIR/{model10_product, model10_seeded, model10_seeded_d12}/
#          pt_<ix>_<iy>[_qrefine_rNN | _margin_rNN].npz
#  Finished points are skipped, so resubmitting a stage fills holes.
# ============================================================

#SBATCH --job-name=m10d2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=16:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/m10dim2_%x_%A_%a.out
#SBATCH --error=logs/m10dim2_%x_%A_%a.err

STAGE=${STAGE:?set STAGE (seed | qrefine | d12_seed | d12_refine | margin | rrefine)}
ROUND=${ROUND:-0}
D_EXT=${D_EXT:-8}
DIM=${DIM:-2}
OUT_DIR=${OUT_DIR:-results_model10_rate_dim${DIM}}
XFER_DIRS=${XFER_DIRS:-results_model10_rate}   # dim = 1 seeded run (read-only)
XFER_RADIUS=${XFER_RADIUS:-1}
N_CHUNKS=${N_CHUNKS:-200}          # must match the --array size above
STRIDE=${STRIDE:-1}
PROD_STRIDE=${PROD_STRIDE:-$STRIDE}
RATE_TOL=${RATE_TOL:-1e-6}
SEED=${SEED:-0}
# seed stage (defaults of scripts/model10_seeded.slurm.sh)
FAM_RANDOM=${FAM_RANDOM:-12}
FAM_POLISH=${FAM_POLISH:-3}
FAM_MAXFEV=${FAM_MAXFEV:-200}
DE_POPSIZE=${DE_POPSIZE:-6}
DE_MAXITER=${DE_MAXITER:-20}
BUNDLE_ITERS=${BUNDLE_ITERS:-80}
NM_MAXFEV=${NM_MAXFEV:-400}
# quick refine (defaults of scripts/model10_seeded_qrefine.slurm.sh)
OPTIMIZER=${OPTIMIZER:-rate}
N_RESTARTS=${N_RESTARTS:-3}
MAXFEV_4=${MAXFEV_4:-1000}
MAXFEV_8=${MAXFEV_8:-500}
POLISH=${POLISH:-150}
# d_ext = 12 (defaults of scripts/model10_d12.slurm.sh)
D12_FAM_RANDOM=${D12_FAM_RANDOM:-4}
D12_FAM_POLISH=${D12_FAM_POLISH:-2}
D12_FAM_MAXFEV=${D12_FAM_MAXFEV:-80}
D12_BUNDLE_ITERS=${D12_BUNDLE_ITERS:-40}
D12_NM_MAXFEV=${D12_NM_MAXFEV:-150}
REFINE_BUNDLE=${REFINE_BUNDLE:-25}
# margin (defaults of scripts/model10_margin.slurm.sh)
RADIUS=${RADIUS:-2}
MARGIN_ITERS=${MARGIN_ITERS:-40}
PUSH_ITERS=${PUSH_ITERS:-40}
# randomised refine (defaults of scripts/model10_rrefine.slurm.sh, but nonmono targets)
TARGETS=${TARGETS:-nonmono}
RR_POLISH_ITERS=${RR_POLISH_ITERS:-100}
HOPS=${HOPS:-6}
SCALES=${SCALES:-"0.02 0.06 0.15"}
MAX_SECONDS=${MAX_SECONDS:-1800}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"
TASK="$SLURM_ARRAY_TASK_ID"

echo "[model10 dim=${DIM}] stage ${STAGE} round ${ROUND}, chunk ${TASK}/${N_CHUNKS}: starting"

case "$STAGE" in
seed)
    # cheap product-state rates first, so they exist even if the seeded part times out
    python scripts/model4_product_rate_worker.py \
        --model    model10 \
        --dim      "$DIM" \
        --task_id  "$TASK" \
        --n_chunks "$N_CHUNKS" \
        --out_dir  "$OUT_DIR" \
        --stride   "$PROD_STRIDE"
    # --stored_dirs left empty: the dim = 1 frames come in through --xfer_dirs
    python scripts/model10_seeded_worker.py \
        --task_id      "$TASK" \
        --n_chunks     "$N_CHUNKS" \
        --dim          "$DIM" \
        --out_dir      "$OUT_DIR" \
        --stored_dirs \
        --xfer_dirs    $XFER_DIRS \
        --xfer_radius  "$XFER_RADIUS" \
        --fixed_frames \
        --stride       "$STRIDE" \
        --d_exts       4 8 \
        --fam_random   "$FAM_RANDOM" \
        --fam_polish   "$FAM_POLISH" \
        --fam_maxfev   "$FAM_MAXFEV" \
        --de_popsize   "$DE_POPSIZE" \
        --de_maxiter   "$DE_MAXITER" \
        --bundle_iters "$BUNDLE_ITERS" \
        --nm_maxfev    "$NM_MAXFEV" \
        --seed         "$SEED"
    ;;
qrefine)
    python scripts/model10_seeded_qrefine_worker.py \
        --round      "$ROUND" \
        --task_id    "$TASK" \
        --n_chunks   "$N_CHUNKS" \
        --dim        "$DIM" \
        --out_dir    "$OUT_DIR" \
        --stride     "$STRIDE" \
        --rate_tol   "$RATE_TOL" \
        --optimizer  "$OPTIMIZER" \
        --n_restarts "$N_RESTARTS" \
        --maxfev_4   "$MAXFEV_4" \
        --maxfev_8   "$MAXFEV_8" \
        --polish     "$POLISH" \
        --seed       "$SEED"
    ;;
d12_seed|d12_refine)
    python scripts/model10_d12_worker.py \
        --stage         "${STAGE#d12_}" \
        --round         "$ROUND" \
        --task_id       "$TASK" \
        --n_chunks      "$N_CHUNKS" \
        --dim           "$DIM" \
        --out_dir       "$OUT_DIR" \
        --stride        "$STRIDE" \
        --xfer_dirs     $XFER_DIRS \
        --xfer_radius   "$XFER_RADIUS" \
        --fam_random    "$D12_FAM_RANDOM" \
        --fam_polish    "$D12_FAM_POLISH" \
        --fam_maxfev    "$D12_FAM_MAXFEV" \
        --bundle_iters  "$D12_BUNDLE_ITERS" \
        --nm_maxfev     "$D12_NM_MAXFEV" \
        --refine_bundle "$REFINE_BUNDLE" \
        --rate_tol      "$RATE_TOL" \
        --seed          "$SEED"
    ;;
margin)
    python scripts/model10_margin_worker.py \
        --d_ext        "$D_EXT" \
        --round        "$ROUND" \
        --task_id      "$TASK" \
        --n_chunks     "$N_CHUNKS" \
        --dim          "$DIM" \
        --out_dir      "$OUT_DIR" \
        --stride       "$STRIDE" \
        --radius       "$RADIUS" \
        --tol          "$RATE_TOL" \
        --margin_iters "$MARGIN_ITERS" \
        --push_iters   "$PUSH_ITERS"
    ;;
rrefine)
    python scripts/model10_rrefine_worker.py \
        --d_ext        "$D_EXT" \
        --round        "$ROUND" \
        --task_id      "$TASK" \
        --n_chunks     "$N_CHUNKS" \
        --dim          "$DIM" \
        --out_dir      "$OUT_DIR" \
        --stride       "$STRIDE" \
        --targets      "$TARGETS" \
        --polish_iters "$RR_POLISH_ITERS" \
        --hops         "$HOPS" \
        --scales       $SCALES \
        --max_seconds  "$MAX_SECONDS" \
        --tol          "$RATE_TOL" \
        --seed         "$SEED"
    ;;
*)
    echo "unknown STAGE=${STAGE}"; exit 1 ;;
esac

echo "[model10 dim=${DIM}] stage ${STAGE} round ${ROUND}, chunk ${TASK}: done"
