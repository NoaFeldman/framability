#!/bin/bash
# ============================================================
#  SLURM job-array: d_ext scan of one open-Dicke model (MODEL=dicke_N<N>)
#  at zero local dephasing -- scripts/model10_dext_scan_worker.py --model
#  $MODEL --p1 $P1S --p2 $P2S --d_exts $D_EXTS (any-model mode).
#
#  combos = points (lam/lam_c in P1S x gamma_phi in P2S) x d_ext, point-major;
#  task t works on combo t % n_combos as replica t // n_combos
#  (4 points x 7 d_ext = 28 combos, N_TASKS = 196 -> 7 replicas per combo,
#  replicas 2 and 5 cold: no pipeline frames, an independent check).
#    STAGE=search          one independent restart per task
#    STAGE=hop ROUND=R     one cross-restart basin-hopping replica per task,
#                          started from the search + hop rounds < R
#
#  Driven by scripts/submit_dicke_dext_scan.sh (which sets --job-name,
#  --time and --array=0-(N_TASKS-1)); by hand:
#      MODEL=dicke_N10 STAGE=search sbatch --job-name=dx_dicke_N10 scripts/dicke_dext_scan.slurm.sh
#
#  Cost per task (1 CPU; LP ~0.04 s at d_ext 6, ~0.5 s at 10, ~2 s at 16):
#  search = structured families (capped at FAM_SECONDS) + seeded global
#  search (DE budget shrinks with d_ext, ~40 min at 14 / 16) +
#  SEARCH_HOP_SECONDS of basin hopping: ~0.5 h at d_ext 4 .. ~3-4 h at 16;
#  a hop task ~HOP_SECONDS.
#  Output (one file per task, unique per stage / round / combo / replica):
#      $OUT_DIR/${MODEL}_dext_scan/m<MM>/pt_<ix>_<iy>/search_r<RRR>.npz
#      $OUT_DIR/${MODEL}_dext_scan/m<MM>/pt_<ix>_<iy>/hop<NN>_r<RRR>.npz
#  The seeded pipeline's frames are read (never written) from
#      $OUT_DIR/${MODEL}_seeded (d_ext 4, 8), $OUT_DIR/${MODEL}_seeded_d12 (12)
# ============================================================

#SBATCH --job-name=dx_dicke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-195
#SBATCH --output=logs/dxdicke_%x_%A_%a.out
#SBATCH --error=logs/dxdicke_%x_%A_%a.err

MODEL=${MODEL:?set MODEL (dicke_N5 | dicke_N10 | ...)}
STAGE=${STAGE:?set STAGE (search | hop)}
ROUND=${ROUND:-0}
OUT_DIR=${OUT_DIR:-results_${MODEL}_seeded}
N_TASKS=${N_TASKS:-196}            # must match the --array size
P1S=${P1S:-"0.1 0.5 1.0 1.5"}      # lam / lam_c
P2S=${P2S:-"0"}                    # gamma_phi
D_EXTS=${D_EXTS:-"4 6 8 10 12 14 16"}
COLD_EVERY=${COLD_EVERY:-3}
FAMILIES=${FAMILIES:-"base split probe"}
FAM_SECONDS=${FAM_SECONDS:-3600}
SEARCH_HOP_SECONDS=${SEARCH_HOP_SECONDS:-3600}
HOP_SECONDS=${HOP_SECONDS:-5400}
POLISH_ITERS=${POLISH_ITERS:-100}
SCALES=${SCALES:-"0.02 0.06 0.15 0.3"}
HOP_TOP=${HOP_TOP:-5}
IMPROVE_TOL=${IMPROVE_TOL:-1e-5}
SEED=${SEED:-0}

if [ "$STAGE" = "search" ]; then SECS=$SEARCH_HOP_SECONDS; else SECS=$HOP_SECONDS; fi

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "[${MODEL} dext scan] ${STAGE} round ${ROUND}, task ${SLURM_ARRAY_TASK_ID}/${N_TASKS}: starting"

# shellcheck disable=SC2086
python scripts/model10_dext_scan_worker.py \
    --model        "$MODEL" \
    --stage        "$STAGE" \
    --round        "$ROUND" \
    --task_id      "$SLURM_ARRAY_TASK_ID" \
    --n_tasks      "$N_TASKS" \
    --p1           $P1S \
    --p2           $P2S \
    --d_exts       $D_EXTS \
    --out_dir      "$OUT_DIR" \
    --stored_dirs \
    ${DIM:+--dim "$DIM"} \
    --cold_every   "$COLD_EVERY" \
    --families     $FAMILIES \
    --fam_seconds  "$FAM_SECONDS" \
    ${DE_POPSIZE:+--de_popsize "$DE_POPSIZE"} \
    ${DE_MAXITER:+--de_maxiter "$DE_MAXITER"} \
    --polish_iters "$POLISH_ITERS" \
    --scales       $SCALES \
    --hop_seconds  "$SECS" \
    --hop_top      "$HOP_TOP" \
    --improve_tol  "$IMPROVE_TOL" \
    --seed         "$SEED"

echo "[${MODEL} dext scan] ${STAGE} round ${ROUND}, task ${SLURM_ARRAY_TASK_ID}: done"
