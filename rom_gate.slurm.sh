#!/bin/bash
#SBATCH --job-name=rom_gate
#SBATCH --output=logs/rom_gate_%A_%a.out
#SBATCH --error=logs/rom_gate_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-16

# Choi-state Robustness of Magic of unitary gates, one gate per array task.
# The array range MUST match the number of (non-comment) lines in rom_gates.txt:
#     grep -cvE '^\s*(#|$)' rom_gates.txt
# Tasks 1-10 (1-2 qubit gates) finish in seconds-minutes; tasks 11-14
# (3-qubit gates, 6-qubit Choi) in minutes; tasks 15-16 (4-qubit gates,
# 8-qubit Choi) may take hours and genuinely need Gurobi.
#
# NOTE: the logs/ directory must exist BEFORE sbatch (Slurm will not create it):
#   mkdir -p logs
#
# ONE-TIME SETUP (see also the deployment notes):
#   git clone https://github.com/quantum-programming/RoM-handbook.git
#   g++ RoM-handbook/exputils/dot/fast_dot_products.cpp \
#       -o RoM-handbook/exputils/dot/fast_dot_products.exe \
#       -std=c++17 -lz -O2 -DNDEBUG -mtune=native -march=native -fopenmp

# --- environment ------------------------------------------------------------
VENV="$HOME/venvs/framability"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet numpy scipy numba tqdm

# Gurobi: required for exact CG on the larger tasks. If your cluster provides
# it as a module, this picks it up; otherwise rom_of_gate.py (--solver auto)
# falls back to scipy automatically (fine for tasks 1-10, slow/fragile beyond).
module load gurobi 2>/dev/null || true
python -m pip install --quiet gurobipy 2>/dev/null || true

# --- paths (collision-safe across array tasks) --------------------------------
export ROM_HANDBOOK_DIR="${ROM_HANDBOOK_DIR:-$PWD/RoM-handbook}"
export ROM_TMPDIR="${SLURM_TMPDIR:-/tmp}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

if [ ! -x "$ROM_HANDBOOK_DIR/exputils/dot/fast_dot_products.exe" ]; then
    echo "WARNING: fast_dot_products.exe not compiled; only 1-2 qubit gates" \
         "(naive LP) will work. See one-time setup above." >&2
fi

# --- run ----------------------------------------------------------------------
# Each task writes its own results_rom/rom_<id>_<label>.json -> no collisions.
python rom_of_gate.py \
    --gates-file rom_gates.txt \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --out results_rom \
    --solver auto \
    --method auto \
    --verbose
