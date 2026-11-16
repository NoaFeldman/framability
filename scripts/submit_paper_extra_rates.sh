#!/bin/bash
# ============================================================
#  Paper figure 1 (rate panels): the runs that are still missing, then the
#  figures.  Everything is submitted at once with dependency chains; only the
#  wait for still-running Dicke seeded runs (rs_<m>_*) blocks, so start it
#  with nohup:
#
#      nohup bash scripts/submit_paper_extra_rates.sh > logs/submit_paper_extra.log 2>&1 &
#
#  Per case in CASES:
#    Schrodinger (every case)   200-task array of scripts/model4_rate_panels.slurm.sh
#        with MEASURES=schro TAG=schro: optimised state frames at SCHRO_DEXTS,
#        seeded with great-circle (d-2)-gons + poles (SCHRO_SEEDS=circles; the
#        old model4 rate_schro_6 = 2.000 plateau is the stuck octahedron init)
#        and, for model4 / model10, warm-started from the stored x_schro_<d>.
#        -> <out>/<model>_schro/pt_*.npz
#        then scripts/paper_schro_collect.py (afterany) -> <out>/<model>_schro_panels.npz
#        (point-wise min with the old rate_schro_4 / 6 where they exist)
#    Product chi = 10 / 40 (Dicke cases only; model4 / model10 have them)
#        The seeded runs computed none: the committed product worker did not
#        accept Dicke models.  200-task array of scripts/model4_product_rate.slurm.sh
#        -> results_<m>_seeded/<m>_product/pt_*.npz (finished points skipped),
#        then scripts/rate_seeded_collect.slurm.sh MODE=plot (afterany) adds
#        rate_prod_10 / 40 to results_<m>_seeded/<m>_seeded_rate_panels.npz.
#        Both start after the model's seeded run (rs_<m>_*) has finished
#        (scripts/wait_seeded_run.sh; at once if none is running).
#
#  Disordered Dicke (unequal omega0, framability_rate_disorder; DIS_FAMILIES x
#  DIS_NS x DIS_WIDTHS, M = DIS_M species; skip with SKIP_DISORDER=1):
#    completion  scripts/dicke_disorder_rate.slurm.sh, D_EXTS = DIS_D_EXTS
#                (4 8 12): resumes, so only the missing points / d_ext run
#                -> results_dicke_disorder/<fam>_N<N>_M<M>_w<w>/, then
#                scripts/dicke_disorder_rate_collect.slurm.sh (afterany)
#                -> <fam>_N<N>_M<M>_panels.npz (+ its own figures)
#    extras      scripts/dicke_disorder_extra.slurm.sh: stabilizer-3, product
#                chi 10 / 40, optimised shared Schrodinger frame (SCHRO_DEXTS),
#                pair-averaged; after the uniform Schrodinger arrays of the
#                same models (afterany: their x_schro_<d> warm-start it)
#                -> results_dicke_disorder/<fam>_N<N>_M<M>_w<w>_extra/, then
#                scripts/dicke_disorder_extra_collect.py -> <fam>_N<N>_M<M>_extra_panels.npz
#  Finally (afterany of every collect): scripts/paper_rate_panels_plot.py
#  --cases $CASES <fam>_N<N>_M<M>_w<w> ... -> results_plots/paper/<case>_rate_panels.{png,pdf}
#  (one figure per uniform case and per disorder width).
#
#  Cases: model4 (out results_model4_rate), model10 (results_model10_rate),
#  model10_dim2 (results_model10_rate_dim2, DIM=2), and any Dicke model
#  (results_<m>_seeded; its seeded pipeline must have run for the
#  Pauli / stabilizer / Heisenberg panels).
#
#  Knobs (env): CASES (model4 model10 model10_dim2 $DICKE_MODELS),
#  DICKE_MODELS (dicke_dec_N100 dicke_dec_k025_N100: individual decay
#  gamma_down, kappa = 1 / 0.25, N = 100 as the disorder run),
#  SCHRO_DEXTS (6 8), SCHRO_RESTARTS (5), SCHRO_MAXFEV (2000), SCHRO_SEEDS
#  (circles), SKIP_SCHRO (0), SKIP_PROD (0), WAIT_SEEDED (1),
#  DIS_FAMILIES (dicke_dec dicke_dec_k025), DIS_NS (100), DIS_WIDTHS
#  (0 0.2 0.5 0.8), DIS_M (4), DIS_D_EXTS (4 8 12), DIS_EXTRA_MAXFEV (2000),
#  SKIP_DISORDER (0), PY (.venv/bin/python).
#  Re-running is safe: finished points are skipped everywhere.
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."               # repo root
mkdir -p logs
. scripts/wait_seeded_run.sh
DICKE_MODELS="${DICKE_MODELS:-dicke_dec_N100 dicke_dec_k025_N100}"
CASES="${CASES:-model4 model10 model10_dim2 ${DICKE_MODELS}}"
SCHRO_DEXTS="${SCHRO_DEXTS:-6 8}"
SCHRO_RESTARTS="${SCHRO_RESTARTS:-5}"
SCHRO_MAXFEV="${SCHRO_MAXFEV:-2000}"
SCHRO_SEEDS="${SCHRO_SEEDS:-circles}"
SKIP_SCHRO="${SKIP_SCHRO:-0}"
SKIP_PROD="${SKIP_PROD:-0}"
WAIT_SEEDED="${WAIT_SEEDED:-1}"
DIS_FAMILIES="${DIS_FAMILIES:-dicke_dec dicke_dec_k025}"
DIS_NS="${DIS_NS:-100}"
DIS_WIDTHS="${DIS_WIDTHS:-0 0.2 0.5 0.8}"
DIS_M="${DIS_M:-4}"
DIS_D_EXTS="${DIS_D_EXTS:-4 8 12}"
DIS_EXTRA_MAXFEV="${DIS_EXTRA_MAXFEV:-2000}"
SKIP_DISORDER="${SKIP_DISORDER:-0}"
PY="${PY:-.venv/bin/python}"

dep_ids=()          # every collect: the figure waits for all of them
schro_dicke=()      # uniform Dicke Schrodinger arrays: the disorder extras wait for them
dicke_cases=()
dep() { [ -n "$1" ] && echo "--dependency=afterany:$1" || true; }
jobid() { echo "${1%%;*}"; }

for case in $CASES; do
    model="$case"; out=""; dim=""; warm=""; ref=""; dicke=0
    case "$case" in
        model4)       out=results_model4_rate;  warm=$out; ref=$out/model4_rate_panels.npz ;;
        model10)      out=results_model10_rate; warm=$out; ref=$out/model10_rate_panels.npz ;;
        model10_dim2) model=model10; out=results_model10_rate_dim2; dim=2 ;;
        dicke_*)      out="results_${case}_seeded"; dicke=1; dicke_cases+=("$case") ;;
        *)            echo "ERROR: unknown case ${case}"; exit 1 ;;
    esac
    "$PY" -c "import sys; sys.path.insert(0, '.'); from trotter_lindbladian_scan import MODELS; MODELS['$model']" \
        || { echo "ERROR: ${model} is not registered"; exit 1; }
    tag="[${case}]"

    # ---- Schrodinger rerun + collect (independent of the seeded run) ------------
    if [ "$SKIP_SCHRO" != "1" ]; then
        sid=$(jobid "$(MODEL="$model" OUT_DIR="$out" DIM="$dim" MEASURES=schro TAG=schro \
              SCHRO_DEXTS="$SCHRO_DEXTS" SCHRO_SEEDS="$SCHRO_SEEDS" WARM_DIR="$warm" \
              SCHRO_RESTARTS="$SCHRO_RESTARTS" SCHRO_MAXFEV="$SCHRO_MAXFEV" \
              sbatch --parsable --job-name="sch_${case}" scripts/model4_rate_panels.slurm.sh)")
        cid=$(jobid "$(sbatch --parsable $(dep "$sid") --job-name="sch_${case}_collect" \
                  --time=00:30:00 --mem=4G --output="logs/paper_schro_collect_${case}_%j.out" \
                  --wrap "${PY} scripts/paper_schro_collect.py --model ${model} \
                          --out_dir ${out} --tag schro ${ref:+--ref_npz ${ref}}")")
        dep_ids+=("$cid")
        [ "$dicke" = "1" ] && schro_dicke+=("$sid")
        echo "$tag Schrodinger d_ext ${SCHRO_DEXTS}: array ${sid} -> ${out}/${model}_schro/," \
             "collect ${cid} -> ${out}/${model}_schro_panels.npz"
    fi
done

# ---- disordered Dicke: completion of the main run + the extra panels ----------------
dis_cases=""
if [ "$SKIP_DISORDER" != "1" ]; then
    denv=(FAMILIES="$DIS_FAMILIES" NS="$DIS_NS" WIDTHS="$DIS_WIDTHS" M_SPECIES="$DIS_M"
          OUT_ROOT=results_dicke_disorder N_TASKS=200)
    aid=$(jobid "$(env "${denv[@]}" D_EXTS="$DIS_D_EXTS" \
          sbatch --parsable --array=0-199 scripts/dicke_disorder_rate.slurm.sh)")
    acid=$(jobid "$(env "${denv[@]}" D_EXTS="$DIS_D_EXTS" \
           sbatch --parsable $(dep "$aid") scripts/dicke_disorder_rate_collect.slurm.sh)")
    dep_ids+=("$acid")
    echo "[disorder] completion d_ext ${DIS_D_EXTS}: array ${aid} (finished points skip)," \
         "collect ${acid} -> results_dicke_disorder/<fam>_N<N>_M${DIS_M}_panels.npz"
    xdeps=$(IFS=:; echo "${schro_dicke[*]:-}")
    xid=$(jobid "$(env "${denv[@]}" SCHRO_DEXTS="$SCHRO_DEXTS" MAXFEV="$DIS_EXTRA_MAXFEV" \
          sbatch --parsable --array=0-199 $(dep "$xdeps") scripts/dicke_disorder_extra.slurm.sh)")
    xcid=$(jobid "$(sbatch --parsable $(dep "$xid") --job-name=dicke_disx_collect \
           --time=00:30:00 --mem=4G --output="logs/dicke_disx_collect_%j.out" \
           --wrap "${PY} scripts/dicke_disorder_extra_collect.py --families ${DIS_FAMILIES} \
                   --ns ${DIS_NS} --widths ${DIS_WIDTHS} --M ${DIS_M} \
                   --schro_dexts ${SCHRO_DEXTS}")")
    dep_ids+=("$xcid")
    echo "[disorder] stabilizer / product / Schrodinger: array ${xid} (after ${xdeps:-nothing})," \
         "collect ${xcid} -> results_dicke_disorder/<fam>_N<N>_M${DIS_M}_extra_panels.npz"
    for fam in $DIS_FAMILIES; do
        for N in $DIS_NS; do
            for w in $DIS_WIDTHS; do
                dis_cases+=" ${fam}_N${N}_M${DIS_M}_w$(printf '%.2f' "$w")"
            done
        done
    done
fi

# ---- uniform Dicke: product chi 10 / 40 + seeded-panel recollect, after the seeded run --
for m in ${dicke_cases[@]+"${dicke_cases[@]}"}; do
    out="results_${m}_seeded"
    [ "$WAIT_SEEDED" = "1" ] && wait_seeded_run "$m"
    [ -f "${out}/${m}_seeded_rate_panels.npz" ] \
        || echo "[${m}] note: ${out}/${m}_seeded_rate_panels.npz missing -- run FAMILY=... scripts/submit_dicke_rate.sh"
    if [ "$SKIP_PROD" != "1" ]; then
        pid=$(jobid "$(MODEL="$m" OUT_DIR="$out" \
              sbatch --parsable --job-name="prd_${m}" scripts/model4_product_rate.slurm.sh)")
        rid=$(jobid "$(MODEL="$m" MODE=plot OUT_DIR="$out" \
              sbatch --parsable $(dep "$pid") scripts/rate_seeded_collect.slurm.sh)")
        dep_ids+=("$rid")
        echo "[${m}] product chi 10/40: array ${pid} -> ${out}/${m}_product/," \
             "recollect ${rid} -> ${out}/${m}_seeded_rate_panels.npz"
    fi
done

# ---- paper figures ---------------------------------------------------------------------
deps=$(IFS=:; echo "${dep_ids[*]:-}")
fid=$(jobid "$(sbatch --parsable $(dep "$deps") --job-name=paper_rate_panels --time=00:30:00 \
          --mem=8G --output="logs/paper_rate_panels_%j.out" \
          --wrap "${PY} scripts/paper_rate_panels_plot.py --cases ${CASES}${dis_cases}")")
echo "figure job ${fid} (after ${deps:-nothing}): results_plots/paper/<case>_rate_panels.{png,pdf}" \
     "for ${CASES}${dis_cases}; the log lists every panel's source file and MISSING panels"
