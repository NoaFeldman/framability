#!/bin/bash
# ============================================================
#  SLURM job (one task): collect of the open-Dicke d_ext scans + figure
#    1. scripts/model10_dext_scan_collect.py --model dicke_N<N>, every N in
#       NS -> convergence table (this log) and
#       results_dicke_N<N>_seeded/dicke_N<N>_dext_scan/summary.npz
#    2. scripts/dicke_dext_scan_plot.py --ns $NS -> $OUT_PNG (rate vs d_ext,
#       one line per lam/lam_c, one panel per N)
#
#  Submitted at the end of scripts/submit_dicke_dext_scan.sh; by hand any
#  time (unfinished combos just have fewer restarts):
#      NS="10 100" sbatch scripts/dicke_dext_scan_collect.slurm.sh
#  P1S / P2S / D_EXTS must match the scan's.
# ============================================================

#SBATCH --job-name=dx_dicke_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dxdicke_io_%x_%A.out
#SBATCH --error=logs/dxdicke_io_%x_%A.err

NS=${NS:-"5 10 20 100 1000"}
P1S=${P1S:-"0.1 0.5 1.0 1.5"}
P2S=${P2S:-"0"}
D_EXTS=${D_EXTS:-"4 6 8 10 12 14 16"}
MIN_HITS=${MIN_HITS:-3}
MIN_COLD_HITS=${MIN_COLD_HITS:-1}
TOL_REL=${TOL_REL:-1e-4}
TOL_ABS=${TOL_ABS:-1e-5}
OUT_PNG=${OUT_PNG:-results_model4_rate/dicke_dext_scan.png}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

for N in $NS; do
    MODEL="dicke_N${N}"
    echo "===== ${MODEL} ====="
    # shellcheck disable=SC2086
    python scripts/model10_dext_scan_collect.py \
        --model         "$MODEL" \
        --out_dir       "results_${MODEL}_seeded" \
        --stored_dirs \
        --p1            $P1S \
        --p2            $P2S \
        --d_exts        $D_EXTS \
        --min_hits      "$MIN_HITS" \
        --min_cold_hits "$MIN_COLD_HITS" \
        --tol_rel       "$TOL_REL" \
        --tol_abs       "$TOL_ABS" \
        || echo "warning: collect of ${MODEL} failed"
done

# shellcheck disable=SC2086
python scripts/dicke_dext_scan_plot.py --ns $NS --out "$OUT_PNG"
python scripts/dicke_dext_scan_plot.py --ns $NS --out "${OUT_PNG%.png}_logy.png" --logy

echo "[dicke dext scan] collect + figure: done"
