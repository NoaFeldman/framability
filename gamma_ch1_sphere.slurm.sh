#!/bin/bash
#SBATCH --job-name=gamma_ch1
#SBATCH --output=gamma_ch1_%j.out
#SBATCH --error=gamma_ch1_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# --- environment (edit to match your cluster) -------------------------------
# Option A: module system
# module load python/3.11
# Option B: a virtualenv you control
# source ~/venvs/framability/bin/activate

# Make sure deps are present (no-op if already installed)
python -m pip install --user --quiet numpy plotly

# --- run --------------------------------------------------------------------
python gamma_ch1_sphere.py --gate cnot --seed 0 --theta-step 0.07 --out-dir gamma_ch1_out
