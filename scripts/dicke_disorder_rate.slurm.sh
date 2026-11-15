#!/bin/bash
# ============================================================
#  SLURM job-array: pair-averaged framability rate of the open Dicke model
#  with unequal atomic frequencies (scripts/dicke_disorder_rate_worker.py,
#  framability_rate_disorder.py).  One array covers every
#  (family, N, width, grid point) combo; task t takes combos t, t + N_TASKS,
#  ... (one file per combo, written atomically, finished d_ext skipped), so
#  tasks never collide and a re-submit resumes.
#
#  Submitted by scripts/submit_dicke_disorder_rate.sh (sets --array and the
#  dependent collect job); by hand:
#    mkdir -p logs
#    N_TASKS=200 sbatch --array=0-199 scripts/dicke_disorder_rate.slurm.sh
#
#  Output: results_dicke_disorder/<family>_N<N>_M<M>_w<width>/pt_<ix>_<iy>.npz
#
#  Knobs (env): FAMILIES ("dicke_dec dicke_dec_k025"), NS ("100"),
#  WIDTHS ("0 0.2 0.5 0.8"), M_SPECIES (4), D_EXTS ("4 8"; add 12 for d12,
#  ~10x slower), N_TASKS (= array size, 200), N_STARTS (3), BUNDLE_ITERS (60),
#  N_KICKS (1), MAX_SEED_EVAL (40), SEED_NS (unset: the run's N first, then
#  every N of the family), OUT_ROOT (results_dicke_disorder), NO_CERTIFY, FORCE.
#  Default load: 2 families x 4 widths x 336 points = 2688 combos, ~14 per
#  task; w = 0 points take seconds, w > 0 ~2-8 min at d 4 + 8 (10 pair
#  patches per evaluation) -> ~1-2 h per task.
# ============================================================

#SBATCH --job-name=dicke_dis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/dicke_dis_%A_%a.out
#SBATCH --error=logs/dicke_dis_%A_%a.err

FAMILIES=${FAMILIES:-"dicke_dec dicke_dec_k025"}
NS=${NS:-100}
WIDTHS=${WIDTHS:-"0 0.2 0.5 0.8"}
M_SPECIES=${M_SPECIES:-4}
D_EXTS=${D_EXTS:-"4 8"}
N_TASKS=${N_TASKS:-200}            # must match the --array size
N_STARTS=${N_STARTS:-3}
BUNDLE_ITERS=${BUNDLE_ITERS:-60}
N_KICKS=${N_KICKS:-1}
MAX_SEED_EVAL=${MAX_SEED_EVAL:-40}
SEED_NS=${SEED_NS:-}
OUT_ROOT=${OUT_ROOT:-results_dicke_disorder}
NO_CERTIFY=${NO_CERTIFY:-}
FORCE=${FORCE:-}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[dicke disorder] task ${SLURM_ARRAY_TASK_ID}/${N_TASKS}: ${FAMILIES} N=${NS} w=${WIDTHS} M=${M_SPECIES} d=${D_EXTS}"

python scripts/dicke_disorder_rate_worker.py \
    --task_id       "$SLURM_ARRAY_TASK_ID" \
    --n_tasks       "$N_TASKS" \
    --families      $FAMILIES \
    --ns            $NS \
    --widths        $WIDTHS \
    --M             "$M_SPECIES" \
    --d_exts        $D_EXTS \
    --n_starts      "$N_STARTS" \
    --bundle_iters  "$BUNDLE_ITERS" \
    --n_kicks       "$N_KICKS" \
    --max_seed_eval "$MAX_SEED_EVAL" \
    --out_root      "$OUT_ROOT" \
    ${SEED_NS:+--seed_ns $SEED_NS} \
    ${NO_CERTIFY:+--no_certify} \
    ${FORCE:+--force}

echo "[dicke disorder] task ${SLURM_ARRAY_TASK_ID}: done"
