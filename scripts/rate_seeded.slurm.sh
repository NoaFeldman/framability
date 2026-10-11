#!/bin/bash
# ============================================================
#  SLURM job-array: seeded framability-rate pipeline for any model of
#  trotter_lindbladian_scan.MODELS (MODEL=model3 | model4 | ...), on the model's
#  own grid (51 x 51 = 2601 points for models 3 / 4, ~13 per task over 200
#  tasks, strided so a dead task thins the grid evenly).  One script, six stages:
#
#    STAGE=seed               product-state rates chi = 10, 40
#                             (model4_product_rate_worker), then stabilizer-3,
#                             Pauli (--fixed_frames) and the seeded d_ext = 4 / 8
#                             rates (model10_seeded_worker --model): exact
#                             rescaled-Pauli optimum, structured families,
#                             stored frames ($STORED_DIRS: earlier rate scans;
#                             $DTBASE_DIRS: finite-dt optimised frames), global search
#    STAGE=qrefine ROUND=r    one d_ext = 4 / 8 quick neighbour-refine round
#    STAGE=d12_seed           d_ext = 12 seed stage (model10_d12_worker --model)
#    STAGE=d12_refine ROUND=r one d_ext = 12 refine round
#    STAGE=margin D_EXT=8|12 ROUND=r   one margin round
#    STAGE=rrefine D_EXT=8|12 ROUND=r  one randomised refine round (TARGETS)
#
#  Driven by scripts/submit_rate_seeded.sh (which sets --job-name / --time per
#  stage); by hand, e.g.
#      MODEL=model4 STAGE=seed sbatch scripts/rate_seeded.slurm.sh
#
#  Output: $OUT_DIR/{<model>_product, <model>_seeded, <model>_seeded_d12}/
#          pt_<ix>_<iy>[_qrefine_rNN | _margin_rNN | _rrefine_rNN].npz
#  Finished points are skipped, so resubmitting a stage fills holes.
# ============================================================

#SBATCH --job-name=rate_seed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=16:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/rateseed_%x_%A_%a.out
#SBATCH --error=logs/rateseed_%x_%A_%a.err

MODEL=${MODEL:?set MODEL (e.g. model3 | model4)}
STAGE=${STAGE:?set STAGE (seed | qrefine | d12_seed | d12_refine | margin | rrefine)}
ROUND=${ROUND:-0}
D_EXT=${D_EXT:-8}
OUT_DIR=${OUT_DIR:-results_${MODEL}_seeded}
STORED_DIRS=${STORED_DIRS:-}       # earlier rate scans holding <dir>/<model>/ (read-only)
DTBASE_DIRS=${DTBASE_DIRS:-}       # DT_BASE-line roots with finite-dt frames (read-only)
XFER_DIRS=${XFER_DIRS:-}           # other full-grid seeded runs to transfer from (read-only)
XFER_RADIUS=${XFER_RADIUS:-1}
DIM_ARG=${DIM:+--dim $DIM}         # empty: the model's own ModelSpec.dim
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
# randomised refine
TARGETS=${TARGETS:-nonmono}
RR_POLISH_ITERS=${RR_POLISH_ITERS:-100}
HOPS=${HOPS:-6}
SCALES=${SCALES:-"0.02 0.06 0.15"}
MAX_SECONDS=${MAX_SECONDS:-1800}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"
TASK="$SLURM_ARRAY_TASK_ID"

echo "[${MODEL} seeded] stage ${STAGE} round ${ROUND} d_ext ${D_EXT}, chunk ${TASK}/${N_CHUNKS}: starting"

case "$STAGE" in
seed)
    # cheap product-state rates first, so they exist even if the seeded part times out
    python scripts/model4_product_rate_worker.py \
        --model    "$MODEL" \
        $DIM_ARG \
        --task_id  "$TASK" \
        --n_chunks "$N_CHUNKS" \
        --out_dir  "$OUT_DIR" \
        --stride   "$PROD_STRIDE"
    python scripts/model10_seeded_worker.py \
        --model        "$MODEL" \
        $DIM_ARG \
        --task_id      "$TASK" \
        --n_chunks     "$N_CHUNKS" \
        --out_dir      "$OUT_DIR" \
        --stored_dirs  $STORED_DIRS \
        --dtbase_dirs  $DTBASE_DIRS \
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
        --model      "$MODEL" \
        $DIM_ARG \
        --round      "$ROUND" \
        --task_id    "$TASK" \
        --n_chunks   "$N_CHUNKS" \
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
        --model         "$MODEL" \
        $DIM_ARG \
        --stage         "${STAGE#d12_}" \
        --round         "$ROUND" \
        --task_id       "$TASK" \
        --n_chunks      "$N_CHUNKS" \
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
        --model        "$MODEL" \
        $DIM_ARG \
        --d_ext        "$D_EXT" \
        --round        "$ROUND" \
        --task_id      "$TASK" \
        --n_chunks     "$N_CHUNKS" \
        --out_dir      "$OUT_DIR" \
        --stride       "$STRIDE" \
        --radius       "$RADIUS" \
        --tol          "$RATE_TOL" \
        --margin_iters "$MARGIN_ITERS" \
        --push_iters   "$PUSH_ITERS"
    ;;
rrefine)
    python scripts/model10_rrefine_worker.py \
        --model        "$MODEL" \
        $DIM_ARG \
        --d_ext        "$D_EXT" \
        --round        "$ROUND" \
        --task_id      "$TASK" \
        --n_chunks     "$N_CHUNKS" \
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

echo "[${MODEL} seeded] stage ${STAGE} round ${ROUND} d_ext ${D_EXT}, chunk ${TASK}: done"
