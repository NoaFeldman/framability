#!/bin/bash
# ============================================================
#  Collect + figures of the reweighted-Pauli w_min scan of the open Dicke
#  model (scripts/dicke_pauli_weight_collect.py).  Submitted by
#  scripts/submit_dicke_pauli_weight.sh after the worker array
#  (--dependency=afterany: missing points are simply absent from the figures;
#  re-run this by hand after a resubmit):
#    sbatch scripts/dicke_pauli_weight_collect.slurm.sh
#
#  Knobs (env): FAMILY (dicke_dec), N_ATOMS (100), M_SPECIES (4),
#  OUT_ROOT (results_dicke_pauli_weight), FIG_DIR (results_model4_rate),
#  T_VALUES ("10 100", total times of the sample-count figure).
#
#  Output: results_model4_rate/<family>_N<N>_pauli_weight_{curves,cost}.png
#          results_model4_rate/<family>_N<N>_pauli_weight_wstar.png (GRID=1 runs)
#          results_dicke_pauli_weight/<family>_N<N>_M<M>_pauli_weight_summary.npz
# ============================================================

#SBATCH --job-name=dicke_pw_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dicke_pw_collect_%j.out
#SBATCH --error=logs/dicke_pw_collect_%j.err

FAMILY=${FAMILY:-dicke_dec}
N_ATOMS=${N_ATOMS:-100}
M_SPECIES=${M_SPECIES:-4}
OUT_ROOT=${OUT_ROOT:-results_dicke_pauli_weight}
FIG_DIR=${FIG_DIR:-results_model4_rate}
T_VALUES=${T_VALUES:-"10 100"}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python scripts/dicke_pauli_weight_collect.py \
    --family  "$FAMILY" \
    --N       "$N_ATOMS" \
    --M       "$M_SPECIES" \
    --root    "$OUT_ROOT" \
    --fig_dir "$FIG_DIR" \
    --T       $T_VALUES
