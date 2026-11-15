#!/bin/bash
# ============================================================
#  Collect + figures of the disordered-Dicke average-rate run
#  (scripts/dicke_disorder_rate_collect.py).  Submitted by
#  scripts/submit_dicke_disorder_rate.sh after the worker array
#  (--dependency=afterany: missing points are drawn blank and counted, so the
#  figures appear even if a task hit the wall time; re-run this by hand after
#  a resubmit).  Same knobs as dicke_disorder_rate.slurm.sh:
#  FAMILIES, NS, WIDTHS, M_SPECIES, D_EXTS, OUT_ROOT; FIG_DIR (results_model4_rate).
#    sbatch scripts/dicke_disorder_rate_collect.slurm.sh
#
#  Output: results_dicke_disorder/<family>_N<N>_M<M>_panels.npz
#          results_model4_rate/<family>_N<N>_disorder_{panels,changes,cuts}.png
# ============================================================

#SBATCH --job-name=dicke_dis_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dicke_dis_collect_%j.out
#SBATCH --error=logs/dicke_dis_collect_%j.err

FAMILIES=${FAMILIES:-"dicke_dec dicke_dec_k025"}
NS=${NS:-100}
WIDTHS=${WIDTHS:-"0 0.2 0.5 0.8"}
M_SPECIES=${M_SPECIES:-4}
D_EXTS=${D_EXTS:-"4 8"}
OUT_ROOT=${OUT_ROOT:-results_dicke_disorder}
FIG_DIR=${FIG_DIR:-results_model4_rate}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python scripts/dicke_disorder_rate_collect.py \
    --families $FAMILIES \
    --ns       $NS \
    --widths   $WIDTHS \
    --M        "$M_SPECIES" \
    --d_exts   $D_EXTS \
    --root     "$OUT_ROOT" \
    --fig_dir  "$FIG_DIR"
