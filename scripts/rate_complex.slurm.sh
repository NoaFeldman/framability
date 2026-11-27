#!/bin/bash
# ============================================================
#  SLURM job array: complex-frame (non-Hermitian) framability-rate panels,
#  scripts/rate_complex_worker.py.  All points of all MODELS (each model's
#  grid at STRIDE, default 2 = twice the step) form one flat list; task t of
#  N_CHUNKS takes points t, t + N_CHUNKS, ... (mixed models, balanced load).
#
#    STAGE=seed                    Pauli, stabilizer-3, complex d_ext 4 and 8
#    STAGE=d12                     complex d_ext 12 (needs the seed stage)
#    STAGE=refine D_EXT=m ROUND=r  one neighbour-transfer + basin-hopping round
#
#  Driven by scripts/submit_rate_complex.sh (sets --job-name / --time per
#  stage); by hand, e.g.
#      STAGE=seed sbatch scripts/rate_complex.slurm.sh
#
#  Output: $OUT_ROOT/<model>_s<STRIDE>/pt_<ix>_<iy>[_d12 | _refine_d<m>_r<NN>].npz
#  Finished points are skipped, so resubmitting a stage fills holes.
# ============================================================

#SBATCH --job-name=rate_complex
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/ratecplx_%x_%A_%a.out
#SBATCH --error=logs/ratecplx_%x_%A_%a.err

STAGE=${STAGE:?set STAGE (seed | d12 | refine)}
MODELS=${MODELS:-"model4 model10 model14 model15 dicke_dec_N100 dicke_dec_k025_N100"}
STRIDE=${STRIDE:-2}
OUT_ROOT=${OUT_ROOT:-results_rate_complex}
OLD_ROOTS=${OLD_ROOTS:-}              # model:root overrides of the earlier Hermitian runs
N_CHUNKS=${N_CHUNKS:-200}             # must match the --array size
ROUND=${ROUND:-1}
D_EXT=${D_EXT:-8}
SEED=${SEED:-0}
# seed stage
DE_POPSIZE4=${DE_POPSIZE4:-24}
DE_MAXITER4=${DE_MAXITER4:-300}
NM_MAXFEV4=${NM_MAXFEV4:-20000}
N_STARTS=${N_STARTS:-3}
N_KICKS=${N_KICKS:-2}
KICK0=${KICK0:-0.05}
BUNDLE_ITERS=${BUNDLE_ITERS:-60}
NM_MAXFEV=${NM_MAXFEV:-400}
MAX_SECONDS_SEED=${MAX_SECONDS_SEED:-3600}
# d12 stage
N_STARTS12=${N_STARTS12:-2}
N_KICKS12=${N_KICKS12:-1}
BUNDLE_ITERS12=${BUNDLE_ITERS12:-30}
NM_MAXFEV12=${NM_MAXFEV12:-0}
MAX_SECONDS_D12=${MAX_SECONDS_D12:-2400}
# refine stage
RADIUS=${RADIUS:-1}
HOPS=${HOPS:-12}
SCALES=${SCALES:-"0.02 0.06 0.15"}
POLISH_ITERS=${POLISH_ITERS:-40}
MAX_SECONDS=${MAX_SECONDS:-1800}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"
TASK="$SLURM_ARRAY_TASK_ID"

echo "[rate complex] stage ${STAGE} d_ext ${D_EXT} round ${ROUND}, chunk ${TASK}/${N_CHUNKS}: starting"

python scripts/rate_complex_worker.py \
    --stage            "$STAGE" \
    --models           $MODELS \
    --stride           "$STRIDE" \
    --out_root         "$OUT_ROOT" \
    --old_roots        $OLD_ROOTS \
    --task_id          "$TASK" \
    --n_chunks         "$N_CHUNKS" \
    --seed             "$SEED" \
    --de_popsize4      "$DE_POPSIZE4" \
    --de_maxiter4      "$DE_MAXITER4" \
    --nm_maxfev4       "$NM_MAXFEV4" \
    --n_starts         "$N_STARTS" \
    --n_kicks          "$N_KICKS" \
    --kick0            "$KICK0" \
    --bundle_iters     "$BUNDLE_ITERS" \
    --nm_maxfev        "$NM_MAXFEV" \
    --max_seconds_seed "$MAX_SECONDS_SEED" \
    --n_starts12       "$N_STARTS12" \
    --n_kicks12        "$N_KICKS12" \
    --bundle_iters12   "$BUNDLE_ITERS12" \
    --nm_maxfev12      "$NM_MAXFEV12" \
    --max_seconds_d12  "$MAX_SECONDS_D12" \
    --d_ext            "$D_EXT" \
    --round            "$ROUND" \
    --radius           "$RADIUS" \
    --hops             "$HOPS" \
    --scales           $SCALES \
    --polish_iters     "$POLISH_ITERS" \
    --max_seconds      "$MAX_SECONDS"

echo "[rate complex] stage ${STAGE} d_ext ${D_EXT} round ${ROUND}, chunk ${TASK}: done"
