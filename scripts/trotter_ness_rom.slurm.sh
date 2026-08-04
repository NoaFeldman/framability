#!/bin/bash
# ============================================================
#  SLURM job-array: RoM of the 2x2 lattice NESS (trotter_ness_rom).
#
#  Per grid point: the steady state of L_full = build_full_lindbladian_model(...)
#  and the RoM of that (generally mixed) 4-qubit state.  The NESS depends on the
#  generator only, NOT on the Trotter step, so there is no DT_BASE sweep here --
#  one steady state and one LP per point.  Points with no unique steady state
#  (every gamma = 0 edge) store NaN and appear blank on the colormap.
#
#  Runs on the SAME decimated grid and the SAME full-grid file naming as
#  trotter_rom_dtbase, so scripts/trotter_rom_dtbase_extrap.py can put the NESS
#  panel beside the extrapolated ones:
#    model1  11 x 26 =  286      model2  11 x 26 =  286
#    model3  26 x 26 =  676      model4  26 x 26 =  676
#    model5  11 x 51 =  561      model6  26 x 26 =  676     (total 3161)
#
#  This array is cheap (minutes, not hours) -- it is split over 200 tasks only
#  to match the rest of the pipeline.
#
#  Submit one model (default model1):
#    mkdir -p logs results_trotter_ness_rom
#    MODEL=model6 sbatch scripts/trotter_ness_rom.slurm.sh
#
#  Normally submitted together with the DT_BASE sweep and the collect job by
#    bash scripts/submit_trotter_rom_dtbase_all.sh
# ============================================================

#SBATCH --job-name=trot_ness
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/trot_ness_%A_%a.out
#SBATCH --error=logs/trot_ness_%A_%a.err

MODEL=${MODEL:-model1}
OUT_DIR=${OUT_DIR:-results_trotter_ness_rom}
STRIDE=${STRIDE:-2}      # must match the DT_BASE sweep
N_CHUNKS=${N_CHUNKS:-200}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"

export ROM_HANDBOOK_DIR="${ROM_HANDBOOK_DIR:-$PWD/RoM-handbook}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

if ! python -c "import numpy, scipy" 2>/dev/null; then
    echo "ERROR: .venv is missing numpy/scipy." >&2
    exit 1
fi
if [ ! -f "$ROM_HANDBOOK_DIR/data/Amat/Amat4.npz" ]; then
    echo "ERROR: $ROM_HANDBOOK_DIR/data/Amat/Amat4.npz not found." >&2
    exit 1
fi

echo "[$MODEL stride=$STRIDE] chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/trotter_ness_rom_worker.py \
    --model    "$MODEL" \
    --task_id  "$SLURM_ARRAY_TASK_ID" \
    --n_chunks "$N_CHUNKS" \
    --out_dir  "$OUT_DIR" \
    --stride   "$STRIDE"

echo "[$MODEL] chunk ${SLURM_ARRAY_TASK_ID}: done"
