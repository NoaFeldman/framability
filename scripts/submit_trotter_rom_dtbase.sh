#!/bin/bash
# Submit the DT_BASE sweep (trotter_rom_dtbase) for every model, one 200-task
# array per model.  Re-running is safe and cheap: complete npz files are
# skipped, so this doubles as the "fill in what timed out" command.
#
#   bash scripts/submit_trotter_rom_dtbase.sh                  # all six models
#   MODELS="model6" bash scripts/submit_trotter_rom_dtbase.sh  # just one
#   MODE=full bash scripts/submit_trotter_rom_dtbase.sh        # 99-base grid
#   STRIDE=1 bash scripts/submit_trotter_rom_dtbase.sh         # full resolution
set -euo pipefail

MODELS=${MODELS:-"model1 model2 model3 model4 model5 model6"}
OUT_DIR=${OUT_DIR:-results_trotter_rom_dtbase}
MODE=${MODE:-fit}
STRIDE=${STRIDE:-2}

mkdir -p logs "$OUT_DIR"

for m in $MODELS; do
    jid=$(MODEL="$m" OUT_DIR="$OUT_DIR" MODE="$MODE" STRIDE="$STRIDE" \
          sbatch --parsable scripts/trotter_rom_dtbase.slurm.sh)
    echo "submitted $m ($MODE, stride $STRIDE) as job $jid"
done
