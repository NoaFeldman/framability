#!/bin/bash
# ============================================================
#  SLURM job-array: stabilizer-3, product chi = 10 / 40 and optimised
#  Schrodinger pair-averaged rates of the disordered open Dicke model
#  (scripts/dicke_disorder_extra_worker.py).  Same combo layout as
#  dicke_disorder_rate.slurm.sh (task t takes combos t, t + N_TASKS, ...;
#  one file per combo, finished measures skipped), separate folders
#  <OUT_ROOT>/<family>_N<N>_M<M>_w<w>_extra/, so it can run next to that array.
#
#  Submitted by scripts/submit_paper_extra_rates.sh; by hand:
#    mkdir -p logs
#    N_TASKS=200 sbatch --array=0-199 scripts/dicke_disorder_extra.slurm.sh
#
#  Knobs (env): FAMILIES ("dicke_dec dicke_dec_k025"), NS ("100"),
#  WIDTHS ("0 0.2 0.5 0.8"), M_SPECIES (4), MEASURES ("stab3 prod schro"),
#  SCHRO_DEXTS ("6 8"), N_STARTS (3), N_DEFAULT_SEEDS (4), MAXFEV (2000),
#  N_TASKS (= array size, 200), OUT_ROOT (results_dicke_disorder), FORCE.
#  Load: 2 families x 4 widths x 336 points = 2688 combos, ~14 per task; at
#  w > 0 every evaluation covers 10 pair patches (w = 0: one).
# ============================================================

#SBATCH --job-name=dicke_disx
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/dicke_disx_%A_%a.out
#SBATCH --error=logs/dicke_disx_%A_%a.err

FAMILIES=${FAMILIES:-"dicke_dec dicke_dec_k025"}
NS=${NS:-100}
WIDTHS=${WIDTHS:-"0 0.2 0.5 0.8"}
M_SPECIES=${M_SPECIES:-4}
MEASURES=${MEASURES:-"stab3 prod schro"}
SCHRO_DEXTS=${SCHRO_DEXTS:-"6 8"}
N_STARTS=${N_STARTS:-3}
N_DEFAULT_SEEDS=${N_DEFAULT_SEEDS:-4}
MAXFEV=${MAXFEV:-2000}
N_TASKS=${N_TASKS:-200}            # must match the --array size
OUT_ROOT=${OUT_ROOT:-results_dicke_disorder}
FORCE=${FORCE:-}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[dicke disorder extra] task ${SLURM_ARRAY_TASK_ID}/${N_TASKS}: ${FAMILIES} N=${NS} w=${WIDTHS} M=${M_SPECIES} measures=${MEASURES} schro d=${SCHRO_DEXTS}"

python scripts/dicke_disorder_extra_worker.py \
    --task_id         "$SLURM_ARRAY_TASK_ID" \
    --n_tasks         "$N_TASKS" \
    --families        $FAMILIES \
    --ns              $NS \
    --widths          $WIDTHS \
    --M               "$M_SPECIES" \
    --measures        $MEASURES \
    --schro_dexts     $SCHRO_DEXTS \
    --n_starts        "$N_STARTS" \
    --n_default_seeds "$N_DEFAULT_SEEDS" \
    --maxfev          "$MAXFEV" \
    --out_root        "$OUT_ROOT" \
    ${FORCE:+--force}

echo "[dicke disorder extra] task ${SLURM_ARRAY_TASK_ID}: done"
