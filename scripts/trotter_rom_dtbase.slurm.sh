#!/bin/bash
# ============================================================
#  SLURM job-array: DT_BASE sweep for the dt->0 limits (trotter_rom_dtbase).
#
#  For every grid point of one MODEL (model1..model6) the Trotter control
#  DT_BASE is swept over base_grid(MODE), and at each base both
#
#     * the stabilizer-3 framability of the two-qubit bond gate, and
#     * the RoM of the 2x2-lattice state after one application of
#       expm(L_full dt)
#
#  are evaluated at dt = DT_BASE / max(||H||_1, {gamma_k}).  The dt->0 limits
#  are then taken by scripts/trotter_rom_dtbase_extrap.py.
#
#  MODE=fit  (default) 20 bases 0.01..0.20 -- the whole range the degree-1 fit
#            over the 15 points nearest dt=0 ever consumes.
#  MODE=full 99 bases 0.01..0.99 -- the results_dtbase_line grid, 5x the cost
#            and the same extrapolant; use it only to inspect the full line.
#
#  Each model runs on its FULL trotter_lindbladian_scan grid:
#    model1  21 x  51 = 1071      model2  21 x  51 = 1071
#    model3  51 x  51 = 2601      model4  51 x  51 = 2601
#    model5  21 x 101 = 2121      model6  51 x  51 = 2601    (total 12066)
#
#  The unit of work is a grid POINT (the whole base line at once): L_bond and
#  L_full do not depend on DT_BASE, so they are built once per point and only
#  the two matrix exponentials are redone per base.  The grid is split across a
#  200-task array; each task skips any npz already complete on disk, so
#  resubmission after a timeout simply continues.
#
#  Cost is ~20x a trotter_rom_state point, i.e. roughly 60 points x 20 bases per
#  task at MODE=fit.  If tasks time out, resubmit -- finished points are skipped.
#
#  Submit one model (default model1):
#    mkdir -p logs results_trotter_rom_dtbase
#    MODEL=model6 sbatch scripts/trotter_rom_dtbase.slurm.sh
#
#  Submit all six:
#    bash scripts/submit_trotter_rom_dtbase.sh
#
#  ONE-TIME SETUP (login node): the RoM-handbook clone must exist for its
#  data/Amat/Amat4.npz -- no compilation and no pip install are needed:
#    git clone https://github.com/quantum-programming/RoM-handbook.git
# ============================================================

#SBATCH --job-name=trot_romdtb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/trot_romdtb_%A_%a.out
#SBATCH --error=logs/trot_romdtb_%A_%a.err

MODEL=${MODEL:-model1}
OUT_DIR=${OUT_DIR:-results_trotter_rom_dtbase}
MODE=${MODE:-fit}        # fit (20 bases) | full (99 bases)
N_CHUNKS=${N_CHUNKS:-200}
DIM=${DIM:-}             # empty -> the model's own dim (all models: 2)

# --- environment ---------------------------------------------------------------
# Repo-local venv, exactly like trotter_scan.slurm.sh.  Do NOT use $HOME (it is
# read-only on the compute nodes) and do NOT pip-install here (system python is
# PEP-668 externally managed).
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"

export ROM_HANDBOOK_DIR="${ROM_HANDBOOK_DIR:-$PWD/RoM-handbook}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"
# The worker pins the BLAS/LP thread counts to 1 itself (one cpu-per-task).

if ! python -c "import numpy, scipy" 2>/dev/null; then
    echo "ERROR: .venv is missing numpy/scipy." >&2
    exit 1
fi
if [ ! -f "$ROM_HANDBOOK_DIR/data/Amat/Amat4.npz" ]; then
    echo "ERROR: $ROM_HANDBOOK_DIR/data/Amat/Amat4.npz not found." \
         "Clone the RoM-handbook repo (see one-time setup above)." >&2
    exit 1
fi

echo "[$MODEL/$MODE] chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

EXTRA_ARGS=()
[ -n "$DIM" ] && EXTRA_ARGS+=(--dim "$DIM")

python scripts/trotter_rom_dtbase_worker.py \
    --model    "$MODEL" \
    --task_id  "$SLURM_ARRAY_TASK_ID" \
    --n_chunks "$N_CHUNKS" \
    --out_dir  "$OUT_DIR" \
    --mode     "$MODE" \
    "${EXTRA_ARGS[@]}"

echo "[$MODEL] chunk ${SLURM_ARRAY_TASK_ID}: done"
