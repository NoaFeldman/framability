#!/bin/bash
# ============================================================
#  Paper figure 2 (rate vs d_ext = 4, 6, ..., 16): runs the d_ext scans of
#  every paper model SIDE BY SIDE, then draws results_plots/paper/rate_vs_dext
#  (.png/.pdf, + _logy) with scripts/paper_dext_plot.py.
#
#    model10   scripts/submit_model10_dext_scan.sh (diagonal points DELTAS,
#              Delta_1 = Delta_2) with D_EXTS = 4..16.  The d_ext 6 / 10
#              search files of the earlier scan are kept (skipped on disk),
#              the new hop rounds cover every d_ext.
#    model4    scripts/submit_dext_scan_model.sh, points M4_P1S x M4_P2S
#              (gamma x gamma'), seeded from results_model4_seeded and the
#              original scan in results_model4_rate
#    Dicke     scripts/submit_dext_scan_model.sh for every model in
#              DICKE_MODELS (default the individual-decay families at N = 100,
#              as the disorder run: dicke_dec_N100, kappa = 1, and
#              dicke_dec_k025_N100, kappa = 0.25), points DICKE_P1S x DICKE_P2S
#              = lam/lam_c x gamma_down.  Each Dicke driver first waits until
#              that model's seeded run (rs_<m>_*) is done, so its search is
#              seeded from the final d_ext 8 / 12 frames; model4 / model10
#              start at once.
#    disorder  the disordered Dicke model (unequal omega0, pair-averaged rate,
#              framability_rate_disorder) at the SAME points, families
#              DIS_FAMILIES, N = DIS_NS, widths DIS_DEXT_WIDTHS, M = DIS_M:
#              scripts/dicke_disorder_rate.slurm.sh with POINTS, one task per
#              (family, N, width, point), D_EXTS, N_STARTS 5 / N_KICKS 3,
#              -> results_dicke_disorder_dext/; every new file starts from the
#              main run's record (INIT_ROOT=results_dicke_disorder: d 4 / 8 / 12
#              reused, 6 / 10 / 14 / 16 seeded by the padded next-smaller
#              optimum).  Up to DIS_DEXT_PASSES array passes (a timed-out task
#              resumes at its last saved d_ext), then
#              scripts/dicke_disorder_dext_collect.py -> <fam>_N<N>_M<M>_w<w>_dext_summary.npz,
#              drawn dashed / dotted on the matching Dicke panel.
#
#  Start on the login node:
#      nohup bash scripts/submit_paper_dext_scan.sh > logs/submit_paper_dext.log 2>&1 &
#  Per-model driver logs: logs/submit_paper_dext_<model>.log.
#  More hop rounds later (searches kept):
#      SKIP_SEARCH=1 MAX_ROUNDS=2 nohup bash scripts/submit_paper_dext_scan.sh ... &
#
#  Knobs (env): RUN ("model10 model4 dicke"), D_EXTS (4 6 8 10 12 14 16),
#  DELTAS (0.5 0.6 0.7), M4_P1S (2.0 4.0), M4_P2S (2.0 3.0), DICKE_MODELS
#  (dicke_dec_N100 dicke_dec_k025_N100), DICKE_P1S (0.5 1.0 1.5), DICKE_P2S
#  (0.1), DIS_FAMILIES (dicke_dec dicke_dec_k025), DIS_NS (100),
#  DIS_DEXT_WIDTHS (0.5 0.8), DIS_M (4), DIS_DEXT_TIME (24:00:00),
#  DIS_DEXT_PASSES (3), FAM_SECONDS (3600), PY; everything else is passed
#  through to the per-model drivers (MAX_ROUNDS, PATIENCE, SKIP_SEARCH,
#  HOP_SECONDS, ...).  RUN takes "disorder" too (default: all four).
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."               # repo root
mkdir -p logs
RUN="${RUN:-model10 model4 dicke disorder}"
export D_EXTS="${D_EXTS:-4 6 8 10 12 14 16}"
export FAM_SECONDS="${FAM_SECONDS:-3600}"
DELTAS="${DELTAS:-0.5 0.6 0.7}"
M4_P1S="${M4_P1S:-2.0 4.0}"
M4_P2S="${M4_P2S:-2.0 3.0}"
DICKE_MODELS="${DICKE_MODELS:-dicke_dec_N100 dicke_dec_k025_N100}"
DICKE_P1S="${DICKE_P1S:-0.5 1.0 1.5}"
DICKE_P2S="${DICKE_P2S:-0.1}"
DIS_FAMILIES="${DIS_FAMILIES:-dicke_dec dicke_dec_k025}"
DIS_NS="${DIS_NS:-100}"
DIS_DEXT_WIDTHS="${DIS_DEXT_WIDTHS:-0.5 0.8}"
DIS_M="${DIS_M:-4}"
DIS_DEXT_TIME="${DIS_DEXT_TIME:-24:00:00}"
DIS_DEXT_PASSES="${DIS_DEXT_PASSES:-3}"
PY="${PY:-.venv/bin/python}"

pids=()
names=()
launch() {        # launch <name> <command...>  (background, own log)
    local name="$1"; shift
    echo "===== ${name}: start $(date); log logs/submit_paper_dext_${name}.log ====="
    ( "$@" ) > "logs/submit_paper_dext_${name}.log" 2>&1 &
    pids+=("$!")
    names+=("$name")
}

if [[ " $RUN " == *" model10 "* ]]; then
    launch model10 env DELTAS="$DELTAS" bash scripts/submit_model10_dext_scan.sh
fi
if [[ " $RUN " == *" model4 "* ]]; then
    launch model4 env MODEL=model4 OUT_DIR=results_model4_seeded \
        P1S="$M4_P1S" P2S="$M4_P2S" STORED_DIRS=results_model4_rate \
        bash scripts/submit_dext_scan_model.sh
fi
if [[ " $RUN " == *" dicke "* ]]; then
    for m in $DICKE_MODELS; do
        launch "$m" env MODEL="$m" OUT_DIR="results_${m}_seeded" \
            P1S="$DICKE_P1S" P2S="$DICKE_P2S" STORED_DIRS= \
            bash scripts/submit_dext_scan_model.sh
    done
fi

disorder_dext() {  # disordered Dicke at the Dicke points, every d_ext (blocks)
    local pts="" a b n pass jid left
    for a in $DICKE_P1S; do for b in $DICKE_P2S; do pts+=" ${a}:${b}"; done; done
    n=$(( $(echo $DIS_FAMILIES | wc -w) * $(echo $DIS_NS | wc -w) \
          * $(echo $DIS_DEXT_WIDTHS | wc -w) * $(echo $pts | wc -w) ))
    local denv=(FAMILIES="$DIS_FAMILIES" NS="$DIS_NS" WIDTHS="$DIS_DEXT_WIDTHS"
                M_SPECIES="$DIS_M" D_EXTS="$D_EXTS" POINTS="${pts# }" N_TASKS="$n"
                OUT_ROOT=results_dicke_disorder_dext INIT_ROOT=results_dicke_disorder
                N_STARTS="${DIS_N_STARTS:-5}" N_KICKS="${DIS_N_KICKS:-3}")
    for pass in $(seq 1 "$DIS_DEXT_PASSES"); do
        jid=$(env "${denv[@]}" sbatch --parsable --wait --array=0-$(( n - 1 )) \
                  --time="$DIS_DEXT_TIME" --job-name=dicke_dis_dext \
                  scripts/dicke_disorder_rate.slurm.sh) \
            || echo "[disorder dext] warning: some pass-${pass} tasks failed or timed out"
        echo "[disorder dext] pass ${pass}: array ${jid%%;*} (${n} tasks, points${pts}," \
             "widths ${DIS_DEXT_WIDTHS}, d_ext ${D_EXTS})"
        # (point, width) combos not finished at every d_ext (missing files count)
        left=$("$PY" -c "
import glob, numpy as np
ds = '$D_EXTS'.split()
fs = glob.glob('results_dicke_disorder_dext/*_w*/pt_*.npz')
done = sum(all(f'F_shared_cert_{d}' in np.load(f, allow_pickle=True).files for d in ds)
           for f in fs)
print($n - done)" 2>/dev/null || echo "?")
        echo "[disorder dext] combos with a missing d_ext: ${left}"
        [ "$left" = "0" ] && break
    done
    sbatch --parsable --wait --job-name=dicke_dis_dext_collect --time=00:30:00 --mem=4G \
        --output="logs/dicke_dis_dext_collect_%j.out" \
        --wrap "${PY} scripts/dicke_disorder_dext_collect.py --families ${DIS_FAMILIES} \
                --ns ${DIS_NS} --widths ${DIS_DEXT_WIDTHS} --M ${DIS_M} \
                --p1 ${DICKE_P1S} --p2 ${DICKE_P2S} --d_exts ${D_EXTS}" \
        || echo "[disorder dext] warning: collect failed"
}
if [[ " $RUN " == *" disorder "* ]]; then
    launch disorder disorder_dext
fi

rc=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "===== ${names[$i]}: driver finished $(date) ====="
    else
        echo "===== ${names[$i]}: driver FAILED (see logs/submit_paper_dext_${names[$i]}.log) ====="
        rc=1
    fi
done

# model10's own driver submits its collect without waiting: collect again
# (cheap, idempotent) so the summary is complete before the figure.
if [[ " $RUN " == *" model10 "* ]]; then
    # shellcheck disable=SC2086
    sbatch --parsable --wait --job-name=paper_m10_dext_collect --time=00:30:00 --mem=4G \
        --output="logs/paper_m10_dext_collect_%j.out" \
        --wrap "${PY} scripts/model10_dext_scan_collect.py --deltas ${DELTAS} \
                --d_exts ${D_EXTS} --out_dir results_model10_rate" \
        || echo "warning: model10 collect failed"
fi

# ---- paper figure ------------------------------------------------------------------
fig="${PY} scripts/paper_dext_plot.py --dicke_models ${DICKE_MODELS}"
sbatch --parsable --wait --job-name=paper_dext_fig --time=00:20:00 --mem=4G \
    --output="logs/paper_dext_fig_%j.out" \
    --wrap "${fig} && ${fig} --logy" \
    || { echo "warning: figure job failed"; rc=1; }
echo "===== done $(date): results_plots/paper/rate_vs_dext.{png,pdf} (+ _logy) ====="
exit "$rc"
