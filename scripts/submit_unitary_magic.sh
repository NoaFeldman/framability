#!/bin/bash
# ============================================================
#  Submit all 5 unitary-magic-scan data arrays and the dependent collect-all
#  job in one go: each model's 200-task array (scripts/magic_scan.slurm.sh)
#  runs independently, and scripts/magic_collect_all.slurm.sh is submitted
#  with --dependency=afterok on all 5 array jobs, so it only starts once
#  every data-generation job has finished successfully.
#
#  Usage:
#    mkdir -p logs results_unitary_magic results_plots
#    bash scripts/submit_unitary_magic.sh
#
#  Optional overrides (apply to every model's data array):
#    GROUP=nc bash scripts/submit_unitary_magic.sh      # cheap maps only
#    FORCE=1  bash scripts/submit_unitary_magic.sh      # recompute everything
#
#  ONE-TIME SETUP (login node): the Choi state comes from rom_of_gate, which
#  needs the RoM-handbook clone present at import time (no compilation, no pip):
#    git clone https://github.com/quantum-programming/RoM-handbook.git
# ============================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p logs results_unitary_magic results_plots

# scan_module:model_key pairs, one per model of unitary_models_for_magic.tex.
MODELS=(
    "model1_ising_tilted_magic_scan:ising_tilted"
    "model2_xy_transverse_magic_scan:xy_transverse"
    "model3_xxz_transverse_magic_scan:xxz_transverse"
    "model4_xyz_magic_scan:xyz"
    "model5_xy_dm_magic_scan:xy_dm"
)

JOB_IDS=()
for ENTRY in "${MODELS[@]}"; do
    SCAN_MODULE="${ENTRY%%:*}"
    MODEL="${ENTRY##*:}"
    JID=$(SCAN_MODULE="$SCAN_MODULE" MODEL="$MODEL" \
          sbatch --parsable \
          --export=ALL,SCAN_MODULE="$SCAN_MODULE",MODEL="$MODEL" \
          scripts/magic_scan.slurm.sh)
    echo "submitted data array for ${SCAN_MODULE}/${MODEL}: job ${JID}"
    JOB_IDS+=("$JID")
done

DEP="afterok"
for JID in "${JOB_IDS[@]}"; do
    DEP="${DEP}:${JID}"
done

CJID=$(sbatch --parsable --dependency="$DEP" scripts/magic_collect_all.slurm.sh)
echo "submitted collect-all job ${CJID} (depends on: ${JOB_IDS[*]})"
