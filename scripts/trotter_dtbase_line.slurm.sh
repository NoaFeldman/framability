#!/bin/bash
# ============================================================
#  SLURM job-array: framability vs Trotter DT_BASE for one model point.
#
#  For a fixed MODEL and (P1, P2) parameter point, the Trotter-step control
#  base is swept over the bottom 10 values of the historical grid,
#  base in [1e-2*i for i in range(1,11)]  (0.01 .. 0.10, 10 points -- item 1 of
#  the dt-extrapolation pipeline redesign).  For each base the bond gate is
#  expm(L_bond * dt) with
#      dt = base / max(||H||_1, {gamma_k})       (choose_dt(..., base=base))
#  and seven framabilities are evaluated: stabilizer-3, Pauli, opt Heisenberg
#  (d_ext=4 and 6), max Janek, optimal-Schroedinger (d_ext=6) and product-state
#  (chi=10).
#
#  The 10 base points map one-per-task onto a 0-9 array (well under the 200-job
#  limit); each task loads any existing npz and computes only the MEASURES keys
#  still missing from it (see trotter_dtbase_line_worker._missing_keys), so
#  re-running this after MEASURES grows only backfills the new keys and never
#  redoes already-stored ones.
#
#  Submit (choose the model and the x,y parameter values):
#    mkdir -p logs results_dtbase_line
#    MODEL=model3 P1=3 P2=3 sbatch scripts/trotter_dtbase_line.slurm.sh
#
#  Output goes to results_dtbase_line/<tag>/base_<idx>.npz with
#  tag = <model>_p1_<p1>_p2_<p2>.  Budgets are overridable via env vars, e.g.:
#    MODEL=model1 P1=2 P2=1 FRA_MAXFEV_4=2000 sbatch scripts/trotter_dtbase_line.slurm.sh
# ============================================================

#SBATCH --job-name=trot_dtbase
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/dtbase_%x_%A_%a.out
#SBATCH --error=logs/dtbase_%x_%A_%a.err

MODEL=${MODEL:-model3}
P1=${P1:?set P1 to the x-axis parameter value}
P2=${P2:?set P2 to the y-axis parameter value}
OUT_DIR=${OUT_DIR:-results_dtbase_line}
N_CHUNKS=${N_CHUNKS:-1}      # 1 -> task_id is the base index (10 tasks)
DIM=${DIM:-2}
FRA_RESTARTS=${FRA_RESTARTS:-5}
FRA_MAXFEV_4=${FRA_MAXFEV_4:-1000}
FRA_MAXFEV_6=${FRA_MAXFEV_6:-500}
CH1_RESTARTS=${CH1_RESTARTS:-15}
SCH_RESTARTS=${SCH_RESTARTS:-5}
SCH_MAXFEV_6=${SCH_MAXFEV_6:-500}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[$MODEL P1=$P1 P2=$P2] base ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/trotter_dtbase_line_worker.py \
    --model         "$MODEL" \
    --p1            "$P1" \
    --p2            "$P2" \
    --task_id       "$SLURM_ARRAY_TASK_ID" \
    --n_chunks      "$N_CHUNKS" \
    --out_dir       "$OUT_DIR" \
    --dim           "$DIM" \
    --fra_restarts  "$FRA_RESTARTS" \
    --fra_maxfev_4  "$FRA_MAXFEV_4" \
    --fra_maxfev_6  "$FRA_MAXFEV_6" \
    --ch1_restarts  "$CH1_RESTARTS" \
    --sch_restarts  "$SCH_RESTARTS" \
    --sch_maxfev_6  "$SCH_MAXFEV_6" \
    --seed          "$SEED"

echo "[$MODEL P1=$P1 P2=$P2] base ${SLURM_ARRAY_TASK_ID}: done"
