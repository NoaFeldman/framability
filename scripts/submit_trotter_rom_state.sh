#!/bin/bash
# Submit the state-RoM sub-pipeline (trotter_rom_state) for every model, one
# 200-task array per model.  Re-running is safe and cheap: current npz files are
# skipped, so this doubles as the "fill in what failed" command.
#
#   bash scripts/submit_trotter_rom_state.sh                 # all six models
#   MODELS="model6" bash scripts/submit_trotter_rom_state.sh # just one
set -euo pipefail

MODELS=${MODELS:-"model1 model2 model3 model4 model5 model6"}
OUT_DIR=${OUT_DIR:-results_trotter_rom_state}
SCAN_DIR=${SCAN_DIR:-results_trotter_v3}

mkdir -p logs "$OUT_DIR"

for m in $MODELS; do
    jid=$(MODEL="$m" OUT_DIR="$OUT_DIR" SCAN_DIR="$SCAN_DIR" \
          sbatch --parsable scripts/trotter_rom_state.slurm.sh)
    echo "submitted $m as job $jid"
done
