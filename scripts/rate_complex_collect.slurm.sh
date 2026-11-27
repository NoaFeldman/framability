#!/bin/bash
# ============================================================
#  Single job: collect the complex-frame rate pipeline into
#  $OUT_ROOT/<model>_complex_rate_panels.{npz,png} (scripts/rate_complex_collect.py)
#  and draw the paper-style 2 x 4 figure of every model with the complex
#  Heisenberg panels (scripts/paper_rate_panels_plot.py --complex_root),
#  -> $PAPER_DIR/<model>_complex_rate_panels.{png,pdf}.
#
#      sbatch scripts/rate_complex_collect.slurm.sh
# ============================================================

#SBATCH --job-name=rc_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ratecplx_collect_%j.out
#SBATCH --error=logs/ratecplx_collect_%j.err

MODELS=${MODELS:-"model4 model10 model14 model15 dicke_dec_N100 dicke_dec_k025_N100"}
STRIDE=${STRIDE:-2}
OUT_ROOT=${OUT_ROOT:-results_rate_complex}
PAPER_DIR=${PAPER_DIR:-results_plots/paper}
ZERO_TOL=${ZERO_TOL:-1e-5}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python scripts/rate_complex_collect.py --models $MODELS --stride "$STRIDE" \
    --out_root "$OUT_ROOT" --zero_tol "$ZERO_TOL"
python scripts/paper_rate_panels_plot.py --cases $MODELS --complex_root "$OUT_ROOT" \
    --out_dir "$PAPER_DIR" --zero_tol "$ZERO_TOL"
