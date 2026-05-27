#!/bin/bash
# ============================================================
#  SLURM single-task: collect Trotter neighbor-refinement results.
#  Run after the refinement array has finished:
#
#    sbatch --dependency=afterok:<array_job_id> scripts/trotter_nb_refine_collect.sh
# ============================================================

#SBATCH --job-name=trotter_nb_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=logs/trotter_nb_collect_%j.out
#SBATCH --error=logs/trotter_nb_collect_%j.err

IN_DIR=${IN_DIR:-results_trotter}
OUT_DIR=${OUT_DIR:-results_trotter}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python scripts/trotter_nb_refine_collect.py --in_dir "$IN_DIR" --out_dir "$OUT_DIR"
