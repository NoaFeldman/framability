#!/bin/bash
# ============================================================
#  SLURM collect job: dt->0 extrapolation + figures for the DT_BASE sweep.
#
#  Submitted by scripts/submit_trotter_rom_dtbase_all.sh with a dependency on
#  every data-generation array, so the whole thing runs in one go.  It writes,
#  for each model:
#
#    results_plots/trotter_rom_dtbase_extrap_<model>.png       2 panels:
#        stabilizer-3 framability^(1/dt) and RoM^(1/dt) at dt = 0
#    results_plots/trotter_rom_dtbase_extrap_raw_<model>.png   3 panels:
#        the same two as raw dt->0 values, plus the RoM of the 2x2 NESS
#
#  Run by hand at any time to see partial progress (missing points are NaN):
#    sbatch scripts/trotter_rom_dtbase_collect.slurm.sh
# ============================================================

#SBATCH --job-name=trot_romdtb_col
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/trot_romdtb_col_%j.out
#SBATCH --error=logs/trot_romdtb_col_%j.err

IN_DIR=${IN_DIR:-results_trotter_rom_dtbase}
NESS_DIR=${NESS_DIR:-results_trotter_ness_rom}
STRIDE=${STRIDE:-2}
FIT_N=${FIT_N:-15}
DEG=${DEG:-1}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"

export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"
mkdir -p results_plots

echo "[collect] extrapolating ${IN_DIR} (stride ${STRIDE}, fit_n ${FIT_N}, deg ${DEG})"

# the value^(1/dt) limits -- the physical result
python scripts/trotter_rom_dtbase_extrap.py --all --save_npz \
    --in_dir "$IN_DIR" --ness_dir "$NESS_DIR" \
    --stride "$STRIDE" --fit_n "$FIT_N" --deg "$DEG"

# the raw limits + the NESS RoM panel
python scripts/trotter_rom_dtbase_extrap.py --all --save_npz --raw \
    --in_dir "$IN_DIR" --ness_dir "$NESS_DIR" \
    --stride "$STRIDE" --fit_n "$FIT_N" --deg "$DEG"

echo "[collect] done; figures in results_plots/"
