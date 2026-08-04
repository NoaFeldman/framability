#!/bin/bash
# One-shot submission of the whole DT_BASE / NESS pipeline:
#
#   1. six DT_BASE-sweep arrays   (trotter_rom_dtbase.slurm.sh)
#   2. six NESS-RoM arrays        (trotter_ness_rom.slurm.sh)
#   3. one collect job            (trotter_rom_dtbase_collect.slurm.sh), held by
#      a SLURM dependency until every array above has terminated
#
# so the figures appear without any further intervention.
#
#   bash scripts/submit_trotter_rom_dtbase_all.sh                  # everything
#   MODELS="model3 model4" bash scripts/submit_trotter_rom_dtbase_all.sh
#   STRIDE=1 bash scripts/submit_trotter_rom_dtbase_all.sh         # full grid
#   MODE=full bash scripts/submit_trotter_rom_dtbase_all.sh        # 99 bases
#
# The dependency is `afterany`, not `afterok`: a single failed point makes a
# worker exit non-zero, which fails the whole array and would leave an afterok
# collect job stuck in DependencyNeverSatisfied forever.  afterany runs the
# collect once the arrays have terminated however they terminated; missing
# points simply come out as NaN (blank cells), which is what you want to see.
#
# Everything is resumable: completed points are skipped, so re-running this
# script after a timeout continues rather than recomputing.
set -euo pipefail

MODELS=${MODELS:-"model1 model2 model3 model4 model5 model6"}
OUT_DIR=${OUT_DIR:-results_trotter_rom_dtbase}
NESS_DIR=${NESS_DIR:-results_trotter_ness_rom}
MODE=${MODE:-fit}
STRIDE=${STRIDE:-2}

mkdir -p logs results_plots "$OUT_DIR" "$NESS_DIR"

jids=()

for m in $MODELS; do
    jid=$(MODEL="$m" OUT_DIR="$OUT_DIR" MODE="$MODE" STRIDE="$STRIDE" \
          sbatch --parsable scripts/trotter_rom_dtbase.slurm.sh)
    jids+=("$jid")
    echo "submitted DT_BASE sweep $m ($MODE, stride $STRIDE) as job $jid"
done

for m in $MODELS; do
    jid=$(MODEL="$m" OUT_DIR="$NESS_DIR" STRIDE="$STRIDE" \
          sbatch --parsable scripts/trotter_ness_rom.slurm.sh)
    jids+=("$jid")
    echo "submitted NESS RoM     $m (stride $STRIDE) as job $jid"
done

dep=$(IFS=:; echo "${jids[*]}")
col=$(IN_DIR="$OUT_DIR" NESS_DIR="$NESS_DIR" STRIDE="$STRIDE" \
      sbatch --parsable --dependency=afterany:"$dep" \
      scripts/trotter_rom_dtbase_collect.slurm.sh)

echo
echo "submitted collect as job $col (waits for ${#jids[@]} arrays)"
echo "figures will appear in results_plots/trotter_rom_dtbase_extrap[_raw]_<model>.png"
